"""HMAC ephemeral TURN credentials — coturn contract.

These tests pin the exact HMAC-SHA1 shape coturn validates against
when run with ``--use-auth-secret``. If they ever start failing,
something has drifted away from coturn's expected format and the
relay will silently reject every cred at runtime.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

import pytest

from augmentum.calling.turn_credentials import (
    DEFAULT_TURN_CRED_TTL_SECONDS,
    TurnCredentials,
    mint_ephemeral,
    turn_secret_from_env,
)


def _expected_password(username: str, secret: str) -> str:
    """Recompute the HMAC the way coturn does. Keeps the test
    intentionally independent of the production helper's path."""

    digest = hmac.new(secret.encode("utf-8"), username.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


class TestMintEphemeral:
    def test_username_carries_expiry_and_hint(self) -> None:
        cred = mint_ephemeral("alice", secret="s", now=1_700_000_000, ttl_seconds=3600)

        # Format must be <unix_expiry>:<hint>. coturn parses this
        # by splitting on ':' and checking the leading integer.
        head, _, tail = cred.username.partition(":")
        assert head == str(1_700_000_000 + 3600)
        assert tail == "alice"
        assert cred.expires_at == 1_700_000_000 + 3600

    def test_password_matches_coturn_hmac(self) -> None:
        cred = mint_ephemeral("alex", secret="topsecret", now=1_700_000_000)

        # The whole point: coturn validates by recomputing this HMAC.
        # If our output ever disagrees, every cred is rejected.
        assert cred.password == _expected_password(cred.username, "topsecret")

    def test_default_ttl_is_24_hours(self) -> None:
        cred = mint_ephemeral("u", secret="s", now=0)
        assert cred.expires_at - 0 == DEFAULT_TURN_CRED_TTL_SECONDS == 86400

    def test_hint_sanitization_drops_separators(self) -> None:
        # ':' would break coturn's expiry/tail split; whitespace
        # would mangle its log format. Both must be stripped.
        cred = mint_ephemeral("u:ser name@host", secret="s", now=0)
        head, _, tail = cred.username.partition(":")
        assert ":" not in tail
        assert " " not in tail
        assert tail == "username" + "host"

    def test_empty_hint_falls_back_to_anon(self) -> None:
        # An empty identity hint is acceptable input but mustn't
        # produce a malformed username — fall back to a sentinel.
        cred = mint_ephemeral("", secret="s", now=0)
        head, _, tail = cred.username.partition(":")
        assert tail == "anon"

    def test_hint_truncated_to_32_chars(self) -> None:
        # Bounds the username length so coturn's log lines stay sane.
        long_hint = "x" * 100
        cred = mint_ephemeral(long_hint, secret="s", now=0)
        _, _, tail = cred.username.partition(":")
        assert len(tail) == 32

    def test_distinct_secrets_produce_distinct_passwords(self) -> None:
        a = mint_ephemeral("u", secret="alpha", now=0)
        b = mint_ephemeral("u", secret="beta", now=0)
        assert a.username == b.username
        assert a.password != b.password

    def test_zero_or_negative_ttl_rejected(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            mint_ephemeral("u", secret="s", ttl_seconds=0)
        with pytest.raises(ValueError, match="ttl_seconds"):
            mint_ephemeral("u", secret="s", ttl_seconds=-1)

    def test_empty_secret_rejected(self) -> None:
        with pytest.raises(ValueError, match="turn secret"):
            mint_ephemeral("u", secret="")


class TestSecretFromEnv:
    def test_reads_env_var_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUGMENTUM_TURN_SECRET", "from-env")
        assert turn_secret_from_env() == "from-env"

    def test_falls_back_to_dev_default_when_unset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The fallback exists so dogfood works without operator
        # setup. Production deployments override; this just keeps
        # the local loop unblocked.
        monkeypatch.delenv("AUGMENTUM_TURN_SECRET", raising=False)
        secret = turn_secret_from_env()
        assert secret  # non-empty
        assert "dev" in secret  # named so an audit can recognize it


class TestAsIceServer:
    def test_renders_browser_ready_dict(self) -> None:
        cred = TurnCredentials(username="u", password="p", expires_at=0)
        rendered = cred.as_ice_server("turn:relay.example:3478?transport=udp")

        # Shape browsers accept directly when handed to RTCPeerConnection.
        assert rendered == {
            "urls": ["turn:relay.example:3478?transport=udp"],
            "username": "u",
            "credential": "p",
        }
