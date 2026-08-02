"""LiveKit room access JWT — shape, signature, reachability probe.

The LiveKit media plane is load-bearing for Connect calls (see spec
``docs/superpowers/specs/2026-06-06-livekit-media-plane-design.md``).
Each test here pins one production-affecting property:

* JWT identity = the Connect user DID (so the peer sees the right
  participant identity on ``room.remoteParticipants``)
* JWT room = ``call_<call_id>`` (both peers compute this without
  server coordination, so the value must be deterministic)
* JWT signature validates against the configured API secret (else
  LiveKit rejects every connect attempt at runtime)
* Grant flags grant publish + subscribe (else the peer can't talk
  or hear)
* Reachability probe caches its result and fails closed on errors
  (so the invite-time decision tree falls back to P2P when LiveKit
  is down rather than minting tokens for a dead SFU)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import jwt as pyjwt
import pytest

from augmentum.connect.livekit_tokens import (
    DEFAULT_TOKEN_TTL_SECONDS,
    ROOM_PREFIX,
    LiveKitToken,
    _reset_reachability_cache,
    delete_room,
    livekit_api_key,
    livekit_api_secret,
    livekit_health_url,
    livekit_reachable,
    livekit_url,
    mint_call_token,
    room_name_for,
)


@pytest.fixture(autouse=True)
def _isolate_reachability_cache() -> None:
    """Drop the cache before each test so the probe re-runs."""

    _reset_reachability_cache()


# ── Room naming ───────────────────────────────────────────────────


class TestRoomNameFor:
    def test_canonical_prefix(self) -> None:
        # Both peers compute this independently — drift here would
        # leave one peer joining the wrong room on every call.
        assert room_name_for("abc123") == f"{ROOM_PREFIX}abc123"

    def test_deterministic(self) -> None:
        # Pure function — repeated calls must agree.
        assert room_name_for("x") == room_name_for("x")

    def test_distinct_call_ids_distinct_rooms(self) -> None:
        assert room_name_for("a") != room_name_for("b")


# ── JWT minting ───────────────────────────────────────────────────


class TestMintCallToken:
    def test_returns_bundle_with_url_room_expires(self) -> None:
        import time as _time
        before = int(_time.time())
        bundle = mint_call_token(
            call_id="cid",
            user_did="alice@home",
            api_key="k",
            api_secret="s" * 32,
            url="wss://lk.example:7880",
        )
        after = int(_time.time())

        assert isinstance(bundle, LiveKitToken)
        assert bundle.url == "wss://lk.example:7880"
        assert bundle.room == "call_cid"
        # The bundle's expires_at is computed from time.time() inside
        # the helper — both bounds tolerate test-runner clock jitter.
        assert before + DEFAULT_TOKEN_TTL_SECONDS <= bundle.expires_at
        assert bundle.expires_at <= after + DEFAULT_TOKEN_TTL_SECONDS
        assert bundle.token  # JWT string

    def test_jwt_identity_is_user_did(self) -> None:
        # The peer sees this on room.remoteParticipants — drift here
        # would surface as "unknown caller" UX. The Connect DID IS
        # the LiveKit identity, no separate ID layer.
        bundle = mint_call_token(
            call_id="cid",
            user_did="alice@home",
            api_key="k",
            api_secret="s" * 32,
            url="wss://x",
        )

        claims = pyjwt.decode(
            bundle.token, "s" * 32, algorithms=["HS256"],
            options={"verify_aud": False},
        )
        assert claims["sub"] == "alice@home"

    def test_jwt_grants_room_join_publish_subscribe(self) -> None:
        # If any of these flip off, the peer either can't enter the
        # room, can't be heard, or can't hear. All three are required
        # for a working call.
        bundle = mint_call_token(
            call_id="cid",
            user_did="u",
            api_key="k",
            api_secret="s" * 32,
            url="wss://x",
        )

        claims = pyjwt.decode(
            bundle.token, "s" * 32, algorithms=["HS256"],
            options={"verify_aud": False},
        )
        video = claims["video"]
        assert video["room"] == "call_cid"
        assert video["roomJoin"] is True
        assert video["canPublish"] is True
        assert video["canSubscribe"] is True

    def test_jwt_exp_matches_ttl(self) -> None:
        import time as _time
        before = int(_time.time())
        bundle = mint_call_token(
            call_id="cid",
            user_did="u",
            api_key="k",
            api_secret="s" * 32,
            url="wss://x",
            ttl_seconds=600,
        )
        after = int(_time.time())

        claims = pyjwt.decode(
            bundle.token, "s" * 32, algorithms=["HS256"],
            options={"verify_aud": False, "verify_exp": False},
        )
        # The bundle's expires_at must match the JWT's actual exp.
        # If they diverge, UI shows "good for Xs" while the SFU
        # rejects on a different schedule.
        assert claims["exp"] == bundle.expires_at
        assert before + 600 <= claims["exp"] <= after + 600

    def test_signature_validates_with_correct_secret(self) -> None:
        bundle = mint_call_token(
            call_id="cid",
            user_did="u",
            api_key="k",
            api_secret="correct-secret-32-bytes-padding!!",
            url="wss://x",
        )

        # Should NOT raise.
        pyjwt.decode(
            bundle.token,
            "correct-secret-32-bytes-padding!!",
            algorithms=["HS256"],
            options={"verify_aud": False},
        )

    def test_signature_rejects_wrong_secret(self) -> None:
        # The whole point of the JWT — LiveKit validates by
        # recomputing the HS256 against its configured secret. If
        # ours disagrees, every call is rejected.
        bundle = mint_call_token(
            call_id="cid",
            user_did="u",
            api_key="k",
            api_secret="the-right-secret-32-bytes-padded!",
            url="wss://x",
        )

        with pytest.raises(pyjwt.InvalidSignatureError):
            pyjwt.decode(
                bundle.token,
                "wrong-secret-also-32-bytes-padded",
                algorithms=["HS256"],
                options={"verify_aud": False},
            )

    def test_distinct_call_ids_distinct_rooms_in_jwt(self) -> None:
        a = mint_call_token(
            call_id="A", user_did="u",
            api_key="k", api_secret="s" * 32, url="wss://x",
        )
        b = mint_call_token(
            call_id="B", user_did="u",
            api_key="k", api_secret="s" * 32, url="wss://x",
        )
        assert a.room != b.room

    def test_empty_call_id_rejected(self) -> None:
        # Empty call_id would mint a token for room "call_" — every
        # caller would land in the same room. Hard-fail at the
        # boundary.
        with pytest.raises(ValueError, match="call_id"):
            mint_call_token(
                call_id="", user_did="u",
                api_key="k", api_secret="s" * 32, url="wss://x",
            )

    def test_empty_user_did_rejected(self) -> None:
        with pytest.raises(ValueError, match="user_did"):
            mint_call_token(
                call_id="cid", user_did="",
                api_key="k", api_secret="s" * 32, url="wss://x",
            )

    def test_missing_key_or_secret_rejected(self) -> None:
        with pytest.raises(ValueError, match="API key"):
            mint_call_token(
                call_id="cid", user_did="u",
                api_key="", api_secret="s" * 32, url="wss://x",
            )
        with pytest.raises(ValueError, match="API key"):
            mint_call_token(
                call_id="cid", user_did="u",
                api_key="k", api_secret="", url="wss://x",
            )

    def test_zero_or_negative_ttl_rejected(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            mint_call_token(
                call_id="cid", user_did="u",
                api_key="k", api_secret="s" * 32, url="wss://x",
                ttl_seconds=0,
            )


# ── Env access ────────────────────────────────────────────────────


class TestEnvAccess:
    def test_dev_defaults_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Dogfood loop runs without any LIVEKIT_* env vars set —
        # falls back to localhost + dev secret so the system boots.
        monkeypatch.delenv("LIVEKIT_URL", raising=False)
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        monkeypatch.delenv("LIVEKIT_HEALTH_URL", raising=False)

        assert "localhost" in livekit_url()
        assert livekit_api_key()  # non-empty
        # Marked "dev" so an audit can recognize it; same posture
        # as the AUGMENTUM_TURN_SECRET dev fallback.
        assert "dev" in livekit_api_secret()

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LIVEKIT_URL", "wss://prod.example:7880")
        monkeypatch.setenv("LIVEKIT_API_KEY", "prod-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "prod-secret-32-bytes-padded-yo!!")

        assert livekit_url() == "wss://prod.example:7880"
        assert livekit_api_key() == "prod-key"
        assert livekit_api_secret() == "prod-secret-32-bytes-padded-yo!!"

    def test_health_url_derived_from_signaling_url(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # wss → https, ws → http. Lets the operator set one URL.
        monkeypatch.delenv("LIVEKIT_HEALTH_URL", raising=False)

        monkeypatch.setenv("LIVEKIT_URL", "wss://lk.example:7880")
        assert livekit_health_url() == "https://lk.example:7880"

        monkeypatch.setenv("LIVEKIT_URL", "ws://lk.example:7880")
        assert livekit_health_url() == "http://lk.example:7880"

    def test_health_url_explicit_override(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LIVEKIT_HEALTH_URL", "http://internal:9090")
        assert livekit_health_url() == "http://internal:9090"


# ── Reachability probe ────────────────────────────────────────────


class TestLivekitReachable:
    async def test_returns_true_on_200(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok = await livekit_reachable(health_url="http://lk:7880")
        assert ok is True

    async def test_returns_false_on_5xx(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok = await livekit_reachable(health_url="http://lk:7880")
        assert ok is False

    async def test_returns_false_on_connection_refused(self) -> None:
        # Anything that raises in the probe path flips closed —
        # we don't want to mint tokens for a dead SFU on a flaky
        # network.
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=OSError("connection refused"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok = await livekit_reachable(health_url="http://lk:7880")
        assert ok is False

    async def test_caches_result_within_cache_window(self) -> None:
        # The hot path is invite handling — we don't want to hit
        # LiveKit's HTTP port for every invite. Result is cached
        # for 30s.
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await livekit_reachable(health_url="http://lk:7880", now=1000.0)
            await livekit_reachable(health_url="http://lk:7880", now=1015.0)
            await livekit_reachable(health_url="http://lk:7880", now=1029.0)

        # All three calls returned, but only the first hit the wire.
        assert mock_client.get.call_count == 1

    async def test_cache_expires_after_window(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await livekit_reachable(health_url="http://lk:7880", now=1000.0)
            await livekit_reachable(health_url="http://lk:7880", now=1060.0)

        # Second probe past the 30s window hits the wire again.
        assert mock_client.get.call_count == 2


# ── Room teardown ─────────────────────────────────────────────────


class TestDeleteRoom:
    async def test_calls_delete_with_canonical_room_name(self) -> None:
        mock_client = MagicMock()
        mock_client.room = MagicMock()
        mock_client.room.delete_room = AsyncMock(return_value=MagicMock())
        mock_client.aclose = AsyncMock()

        with patch(
            "livekit.api.LiveKitAPI", return_value=mock_client,
        ), patch(
            "livekit.api.DeleteRoomRequest",
        ) as mock_req:
            ok = await delete_room(
                call_id="cid",
                api_key="k",
                api_secret="s" * 32,
                url="wss://lk:7880",
            )

        assert ok is True
        # The room name must match what mint_call_token put in the JWT.
        mock_req.assert_called_once_with(room="call_cid")

    async def test_treats_not_found_as_success(self) -> None:
        # The hangup path calls delete_room unconditionally — even
        # when the room never got created (declined before media
        # plane was set up). NotFound must NOT log a warning or
        # surface as failure.
        mock_client = MagicMock()
        mock_client.room = MagicMock()
        mock_client.room.delete_room = AsyncMock(
            side_effect=Exception("room not found"),
        )
        mock_client.aclose = AsyncMock()

        with patch("livekit.api.LiveKitAPI", return_value=mock_client):
            ok = await delete_room(
                call_id="cid",
                api_key="k",
                api_secret="s" * 32,
                url="wss://lk:7880",
            )
        assert ok is True

    async def test_returns_false_on_real_error(self) -> None:
        mock_client = MagicMock()
        mock_client.room = MagicMock()
        mock_client.room.delete_room = AsyncMock(
            side_effect=Exception("connection refused"),
        )
        mock_client.aclose = AsyncMock()

        with patch("livekit.api.LiveKitAPI", return_value=mock_client):
            ok = await delete_room(
                call_id="cid",
                api_key="k",
                api_secret="s" * 32,
                url="wss://lk:7880",
            )
        assert ok is False

    async def test_empty_call_id_returns_false_without_call(self) -> None:
        # Don't try to delete the room "call_" — that's a bug if
        # we get here. Return False so the caller can log it.
        with patch("livekit.api.LiveKitAPI") as mock_api:
            ok = await delete_room(call_id="")
        assert ok is False
        mock_api.assert_not_called()
