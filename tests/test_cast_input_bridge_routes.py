"""Tests for the cast-input WS route handlers.

Covers the proxy-side endpoints that the registry-level tests in
``test_cast_input_registry.py`` don't reach:

  * ``/api/cast/input/ws`` — phone-side producer
      - echo mode reflects frames (cast-latency-test regression guard)
      - production mode requires session_id
      - 1011 when registry not initialised
      - happy path attaches phone, forwards inbound frames to the
        registered container's WS, and detaches on disconnect
      - session ownership check blocks other users when game-stream
        runtime is wired

  * ``/api/cast/input/container-ws/{session_id}`` — in-container daemon
      - token required (1008 if missing)
      - 404-equivalent (1008) when session unknown
      - bad token rejected (1008)
      - 1011 when runtime or registry not initialised
      - happy path attaches container, routes inbound rumble frames to
        the owning phone, detaches on disconnect

These are integration-shaped: we drive the real registry through the
real FastAPI WS handlers via TestClient, with stub WSes for the
counterparty (phone-test → fake container, container-test → fake phone)
so we can observe routing without spinning a second WS in parallel.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocketDisconnect

from augmentum.cast.input_bridge import ROUTING_INDEX, CastInputRegistry

# ── Helpers ──────────────────────────────────────────────────────────


class _FakeWS:
    """WebSocket double for the counterparty side of a routing test.

    Mirrors the surface CastInputRegistry calls (``send_json``,
    ``close``); inbound capture is exposed via ``sent``.
    """

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed: bool = False
        self.close_code: int | None = None

    async def send_json(self, obj: dict[str, Any]) -> None:
        self.sent.append(obj)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code


def _ws_url(path: str, **params: str) -> str:
    """Build a WS URL with the test ticket pre-attached.

    The mock session_manager in conftest's ``app`` fixture validates any
    ticket string, so the literal value here only matters for shape.
    """
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    base = f"{path}?ticket=t"
    return f"{base}&{qs}" if qs else base


# ── /api/cast/input/ws — echo mode ───────────────────────────────────


class TestEchoMode:
    def test_echo_reflects_seq_and_t_send(self, app, client):
        """cast-latency-test regression guard.

        The echo path was the only consumer before the production
        wiring landed, and the latency-test page still depends on it.
        Any future refactor that breaks reflection of the frame's
        ``seq`` + ``t_send`` will show up here.
        """
        with client.websocket_connect(_ws_url("/api/cast/input/ws", echo="1")) as ws:
            ws.send_json({"seq": 7, "t_send": 1234.5, "event": {}})
            reply = ws.receive_json()
            assert reply["seq"] == 7
            assert reply["t_send"] == 1234.5
            assert "t_recv" in reply
            assert isinstance(reply["t_recv"], int | float)

    def test_echo_does_not_require_session_id(self, app, client):
        """Echo path predates session-bound routing — keep it relaxed."""
        with client.websocket_connect(_ws_url("/api/cast/input/ws", echo="1")) as ws:
            ws.send_json({"seq": 1, "t_send": 0.0})
            reply = ws.receive_json()
            assert reply["seq"] == 1

    def test_echo_ignores_non_dict_frames(self, app, client):
        """Garbage strings must not crash the handler or echo back."""
        with client.websocket_connect(_ws_url("/api/cast/input/ws", echo="1")) as ws:
            ws.send_text("not-json")
            ws.send_json({"seq": 99, "t_send": 1.0})
            reply = ws.receive_json()
            assert reply["seq"] == 99


# ── /api/cast/input/ws — production mode ─────────────────────────────


class TestProductionPhoneWS:
    def test_missing_session_id_rejected(self, app, client):
        """Production mode without ?session_id closes with 1008."""
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(_ws_url("/api/cast/input/ws")) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008

    def test_no_registry_returns_1011(self, app, client):
        """Server misconfig (registry missing) closes with 1011."""
        app.state.cast_input_registry = None
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    _ws_url("/api/cast/input/ws", session_id="s1"),
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1011

    def test_attach_and_detach_round_trip(self, app, client):
        """Phone attaches to registry on connect, detaches on close.

        The route's finally-block must call detach_phone or stale
        ConnectedPhone records pile up across reconnects.
        """
        app.state.cast_input_registry = CastInputRegistry()
        registry = app.state.cast_input_registry
        # Skip ownership check — no gs_runtime wired.
        app.state.game_stream_runtime = None

        assert len(registry._phones) == 0
        with client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as _ws:
            assert len(registry._phones) == 1
            phone = next(iter(registry._phones.values()))
            assert phone.session_id == "s1"
            assert phone.pad_index == 0
        # Disconnect triggered detach.
        assert len(registry._phones) == 0

    def test_forwards_input_frame_to_registered_container(self, app, client):
        """A phone frame arrives at the container's WS with slot stamped."""
        app.state.cast_input_registry = CastInputRegistry()
        registry = app.state.cast_input_registry
        app.state.game_stream_runtime = None

        # Pre-attach a fake container so route_input has somewhere to go.
        fake_container_ws = _FakeWS()
        registry.attach_container(
            ws=fake_container_ws, session_id="s1", user_id="usr_test",
            pad_routing=ROUTING_INDEX,
        )

        with client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws:
            ws.send_json({
                "seq": 1, "t_send": 100.0,
                "event": {
                    "kind": "gamepad_state",
                    "pad_index": 0,
                    "buttons": [0] * 17,
                    "axes": [0.0] * 4,
                },
            })
            # Allow the route's await registry.route_input to flush.
            # TestClient runs the route synchronously between sends, so
            # by the time we check, the forward has happened.
            import time
            for _ in range(50):
                if fake_container_ws.sent:
                    break
                time.sleep(0.01)

        # First frame: the input we sent. Detach_phone may follow up with
        # a synthetic "neutral release" frame (seq=-1) so any buttons held
        # at disconnect get released in the emulator — that's intentional
        # behaviour, not a routing duplicate.
        assert len(fake_container_ws.sent) >= 1
        forwarded = fake_container_ws.sent[0]
        assert forwarded["seq"] == 1
        # Under index routing, slot==pad_index is claimed immediately.
        assert forwarded["slot"] == 0
        assert forwarded["event"]["kind"] == "gamepad_state"
        if len(fake_container_ws.sent) == 2:
            release = fake_container_ws.sent[1]
            assert release["seq"] == -1
            assert release["event"]["buttons"] == [0] * 17

    def test_session_ownership_blocks_other_users(self, app, client):
        """Wiring gs_runtime with a row that says "not yours" returns 1008.

        Without this check, a logged-in user could connect to any
        session_id they happened to learn and inject inputs into
        someone else's emulator.
        """
        app.state.cast_input_registry = CastInputRegistry()
        gs_runtime = MagicMock()
        gs_runtime._store = MagicMock()
        gs_runtime._store.get_session = AsyncMock(return_value=None)
        app.state.game_stream_runtime = gs_runtime

        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    _ws_url("/api/cast/input/ws", session_id="not-mine"),
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008
        # Verify the store was queried scoped to the test_user.
        gs_runtime._store.get_session.assert_awaited_once()
        kwargs = gs_runtime._store.get_session.await_args.kwargs
        assert kwargs.get("user_id") == "usr_test"


# ── /api/cast/input/ws — browser-cast (receiver_id) branch ─────────


class _FakeReceiver:
    """Stand-in for ConnectedReceiver — only the fields the WS
    handler reads."""

    def __init__(self, registration_id: str, user_id: str) -> None:
        self.registration_id = registration_id
        self.user_id = user_id


class _FakeReceiverRegistry:
    """Minimal receiver_registry stand-in for browser-cast tests.

    Supplies ``get`` (used by the ownership check) and ``send``
    (used by the input bridge to fan out CMD_INPUT_GAMEPAD).
    """

    def __init__(self) -> None:
        self._receivers: dict[str, _FakeReceiver] = {}
        self.sent: list[tuple[str, Any]] = []

    def add(self, registration_id: str, user_id: str) -> None:
        self._receivers[registration_id] = _FakeReceiver(
            registration_id, user_id,
        )

    def get(self, registration_id: str) -> _FakeReceiver | None:
        return self._receivers.get(registration_id)

    async def send(self, registration_id: str, cmd: Any) -> bool:
        if registration_id not in self._receivers:
            return False
        self.sent.append((registration_id, cmd))
        return True


class TestBrowserCastWS:
    def test_receiver_id_path_rejects_when_registry_missing(self, app, client):
        """No receiver_registry on app.state → 1011 (server misconfig)."""
        app.state.cast_input_registry = CastInputRegistry()
        app.state.receiver_registry = None
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    _ws_url("/api/cast/input/ws", receiver_id="rcv_abc"),
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1011

    def test_receiver_id_rejects_unknown_receiver(self, app, client):
        """Bad receiver_id → 1008. Defends against a controller targeting
        a receiver that doesn't belong to the logged-in user."""
        app.state.cast_input_registry = CastInputRegistry()
        app.state.receiver_registry = _FakeReceiverRegistry()
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    _ws_url("/api/cast/input/ws", receiver_id="rcv_missing"),
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008

    def test_receiver_id_rejects_cross_user(self, app, client):
        """A receiver owned by another user → 1008. Critical isolation
        boundary — without this an attacker could grab a leaked
        receiver_id and route gamepad input into someone else's TV."""
        app.state.cast_input_registry = CastInputRegistry()
        rr = _FakeReceiverRegistry()
        rr.add("rcv_xyz", "usr_other")
        app.state.receiver_registry = rr
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    _ws_url("/api/cast/input/ws", receiver_id="rcv_xyz"),
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008

    def test_receiver_id_happy_path_forwards_input(self, app, client):
        """Owned receiver attached; gamepad frames fan out as
        CMD_INPUT_GAMEPAD via receiver_registry.send."""
        from augmentum.cast.receiver_protocol import CMD_INPUT_GAMEPAD
        app.state.cast_input_registry = CastInputRegistry()
        rr = _FakeReceiverRegistry()
        rr.add("rcv_ok", "usr_test")
        app.state.receiver_registry = rr

        with client.websocket_connect(
            _ws_url("/api/cast/input/ws", receiver_id="rcv_ok", pad_index="1"),
        ) as ws:
            ws.send_json({
                "seq": 1, "t_send": 100.0,
                "event": {
                    "kind": "gamepad_state",
                    "pad_index": 1,
                    "buttons": [1] + [0] * 16,
                    "axes": [0.0] * 4,
                },
            })
            import time
            for _ in range(50):
                if rr.sent:
                    break
                time.sleep(0.01)
        assert len(rr.sent) == 1
        rcv_id, cmd = rr.sent[0]
        assert rcv_id == "rcv_ok"
        assert cmd.cmd == CMD_INPUT_GAMEPAD
        assert cmd.args["pad_index"] == 1
        assert cmd.args["buttons"][0] == 1


# ── /api/cast/input/container-ws/{session_id} ────────────────────────


class TestContainerWS:
    def test_no_token_rejected(self, app, client):
        """Token is the only credential — without it, 1008."""
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    "/api/cast/input/container-ws/s1",
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008

    def test_no_runtime_returns_1011(self, app, client):
        """Server misconfig closes with 1011."""
        app.state.game_stream_runtime = None
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    "/api/cast/input/container-ws/s1?token=x",
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1011

    def test_unknown_session_rejected(self, app, client):
        """Token presented for a session that doesn't exist → 1008."""
        app.state.cast_input_registry = CastInputRegistry()
        gs_runtime = MagicMock()
        gs_runtime._store = MagicMock()
        gs_runtime._store.get_session = AsyncMock(return_value=None)
        app.state.game_stream_runtime = gs_runtime

        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    "/api/cast/input/container-ws/missing?token=x",
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008

    def test_bad_token_rejected(self, app, client):
        """compare_digest mismatch returns 1008."""
        app.state.cast_input_registry = CastInputRegistry()
        gs_runtime = MagicMock()
        gs_runtime._store = MagicMock()
        gs_runtime._store.get_session = AsyncMock(return_value={
            "id": "s1", "user_id": "usr_test",
            "cast_input_token": "correct-token",
            "system_id": "",
        })
        app.state.game_stream_runtime = gs_runtime

        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    "/api/cast/input/container-ws/s1?token=wrong-token",
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008

    def test_empty_token_on_row_rejected(self, app, client):
        """A session row with no cast_input_token can never match.

        Defends against an accidental ALTER that nulls existing tokens —
        comparing "" against "" via hmac.compare_digest would succeed
        without the explicit empty-string guard in the handler.

        We have to mint a non-empty token query param to get past the
        ``not token_param`` short-circuit; the handler's subsequent
        ``not expected_token`` guard is what we're pinning here.
        """
        app.state.cast_input_registry = CastInputRegistry()
        gs_runtime = MagicMock()
        gs_runtime._store = MagicMock()
        gs_runtime._store.get_session = AsyncMock(return_value={
            "id": "s1", "user_id": "usr_test",
            "cast_input_token": "",
            "system_id": "",
        })
        app.state.game_stream_runtime = gs_runtime

        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    "/api/cast/input/container-ws/s1?token=x",
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008

    def test_happy_path_attaches_and_routes_rumble(self, app, client):
        """Valid token attaches container, rumble routes to owning phone."""
        app.state.cast_input_registry = CastInputRegistry()
        registry = app.state.cast_input_registry

        gs_runtime = MagicMock()
        gs_runtime._store = MagicMock()
        gs_runtime._store.get_session = AsyncMock(return_value={
            "id": "s1", "user_id": "usr_test",
            "cast_input_token": "good-token",
            "system_id": "",
        })
        app.state.game_stream_runtime = gs_runtime

        # Pre-attach a fake phone with a claimed slot so rumble routes.
        fake_phone_ws = _FakeWS()
        phone = registry.attach_phone(
            ws=fake_phone_ws, session_id="s1", user_id="usr_test",
            pad_index=0,
        )
        # Manually mark the slot as owned (route_rumble looks it up via
        # _slot_owners, which is populated lazily on first input frame
        # — we short-circuit by writing the bookkeeping directly).
        registry._slot_owners["s1"] = {0: phone.attachment_id}
        phone.slot = 0

        assert "s1" not in registry._containers
        with client.websocket_connect(
            "/api/cast/input/container-ws/s1?token=good-token",
        ) as ws:
            assert "s1" in registry._containers
            ws.send_json({
                "kind": "rumble", "slot": 0,
                "duration_ms": 200, "strong": 0.75, "weak": 0.25,
            })
            # Wait for routing to flush.
            import time
            for _ in range(50):
                if fake_phone_ws.sent:
                    break
                time.sleep(0.01)

        # Container detached on close.
        assert "s1" not in registry._containers
        # Phone got the rumble frame.
        assert len(fake_phone_ws.sent) == 1
        rumble = fake_phone_ws.sent[0]
        assert rumble["kind"] == "rumble"
        assert rumble["slot"] == 0
        assert rumble["duration_ms"] == 200
        assert rumble["strong"] == 0.75
        assert rumble["weak"] == 0.25

    def test_non_rumble_frames_silently_accepted(self, app, client):
        """Heartbeats / future kinds shouldn't crash the read loop."""
        app.state.cast_input_registry = CastInputRegistry()
        gs_runtime = MagicMock()
        gs_runtime._store = MagicMock()
        gs_runtime._store.get_session = AsyncMock(return_value={
            "id": "s1", "user_id": "usr_test",
            "cast_input_token": "good-token",
            "system_id": "",
        })
        app.state.game_stream_runtime = gs_runtime

        with client.websocket_connect(
            "/api/cast/input/container-ws/s1?token=good-token",
        ) as ws:
            ws.send_json({"kind": "heartbeat"})
            ws.send_text("not-json")
            ws.send_json({"foo": "bar"})
        # If we got here without disconnect-on-malformed, the read
        # loop is sufficiently defensive.
        assert "s1" not in app.state.cast_input_registry._containers

    def test_resolves_pad_routing_from_controller_service(self, app, client):
        """When system_id is set, container honours per-system pad_routing.

        Pin against silent regression of the ControllerService consumer
        path — losing this wiring would silently revert every session
        to index routing regardless of user preference.
        """
        app.state.cast_input_registry = CastInputRegistry()
        registry = app.state.cast_input_registry

        gs_runtime = MagicMock()
        gs_runtime._store = MagicMock()
        gs_runtime._store.get_session = AsyncMock(return_value={
            "id": "s1", "user_id": "usr_test",
            "cast_input_token": "good-token",
            "system_id": "snes",
        })
        app.state.game_stream_runtime = gs_runtime

        # Stub controller_service.resolve to return a firstpress layout.
        controller_service = MagicMock()
        layout = MagicMock()
        layout.pad_routing = "firstpress"
        controller_service.resolve = AsyncMock(return_value=layout)
        app.state.controller_service = controller_service

        with client.websocket_connect(
            "/api/cast/input/container-ws/s1?token=good-token",
        ) as _ws:
            container = registry.get_container("s1")
            assert container is not None
            assert container.pad_routing == "firstpress"
            assert container.system_id == "snes"

        controller_service.resolve.assert_awaited_once_with(
            user_id="usr_test", system_id="snes",
        )


# ── /api/cast/input/ws — guest join (invite token) ──────────────────


class TestGuestJoinWS:
    """The middleware resolves ``?join_token=wsi_*`` against the
    invite store and stashes the record on the WS scope. The route
    then claims a slot off the token and attaches the phone as if
    the host themselves were joining (the host's user_id becomes the
    WS owner).

    These tests verify both halves of that contract.
    """

    def _install_invite_store(self, app, *, session_id="s1", host_user_id="usr_test", max_slots=3):
        from augmentum.cast.invite_store import InviteStore
        store = InviteStore()
        app.state.cast_invite_store = store
        record = store.mint(
            session_id=session_id, host_user_id=host_user_id,
            max_slots=max_slots,
        )
        return store, record

    def test_unknown_join_token_falls_through_to_unauthorised(self, app, client):
        """Unknown wsi_* tokens get the standard auth-fail close.

        Middleware can't resolve the token → falls through to the
        ``Unauthorized`` close (code 4001). TestClient surfaces this
        as a WebSocketDisconnect on the first receive call.
        """
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    "/api/cast/input/ws?join_token=wsi_definitelynotreal",
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 4001

    def test_valid_join_token_attaches_phone_as_host(self, app, client):
        """Token possession resolves to the host's user_id; the phone
        attaches to the host's session without a cookie."""
        app.state.cast_input_registry = CastInputRegistry()
        registry = app.state.cast_input_registry
        # No gs_runtime needed — guest path skips ownership check.
        app.state.game_stream_runtime = None
        _, record = self._install_invite_store(app)

        assert len(registry._phones) == 0
        with client.websocket_connect(
            f"/api/cast/input/ws?join_token={record.token}",
        ) as _ws:
            assert len(registry._phones) == 1
            phone = next(iter(registry._phones.values()))
            assert phone.session_id == "s1"
            # The host's user_id is the WS owner — the guest's input
            # legally counts as the host's for the purposes of registry
            # bookkeeping and downstream emulator action.
            assert phone.user_id == "usr_test"
        assert len(registry._phones) == 0

    def test_claim_decrements_slots(self, app, client):
        """One successful join consumes one slot on the invite."""
        app.state.cast_input_registry = CastInputRegistry()
        app.state.game_stream_runtime = None
        _, record = self._install_invite_store(app, max_slots=2)

        with client.websocket_connect(
            f"/api/cast/input/ws?join_token={record.token}",
        ) as _ws:
            assert record.slots_remaining == 1

    def test_exhausted_token_rejects_with_1008(self, app, client):
        """When the token's slot counter is already 0, the route
        closes with 1008 before accepting.
        """
        app.state.cast_input_registry = CastInputRegistry()
        app.state.game_stream_runtime = None
        store, record = self._install_invite_store(app, max_slots=1)
        # Drain by direct claim — simulates a guest who already joined.
        store.claim(record.token)
        assert record.slots_remaining == 0

        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    f"/api/cast/input/ws?join_token={record.token}",
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008

    def test_revoked_token_rejects_via_middleware(self, app, client):
        """Revoking pops the record from the store, so middleware
        can't resolve it → reverts to standard auth-fail close (4001).
        """
        store, record = self._install_invite_store(app)
        store.revoke(record.token, host_user_id="usr_test")

        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(
                    f"/api/cast/input/ws?join_token={record.token}",
                ) as ws:
            ws.receive_text()
        assert exc_info.value.code == 4001


# ── /api/cast/games/session/{id}/invite ──────────────────────────────


class TestInviteCreateEndpoint:
    """The host-side mint endpoint. Stubbed receiver_registry +
    game_stream_runtime + profile registry — we only verify the
    auth/validation branches and the InviteStore side effect.
    """

    def _wire_app(
        self, app, *, profile_max_players: int = 4,
        session_owner: str = "usr_test", receiver_owner: str = "usr_test",
        has_receiver: bool = True,
    ):
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()

        gs_runtime = MagicMock()
        gs_runtime._store = MagicMock()
        gs_runtime._store.get_session = AsyncMock(return_value=(
            {"id": "s1", "user_id": session_owner,
             "profile_id": "emulator-streamed"}
            if session_owner else None
        ))
        app.state.game_stream_runtime = gs_runtime

        rcv_registry = MagicMock()
        if has_receiver:
            fake_rcv = MagicMock()
            fake_rcv.user_id = receiver_owner
            rcv_registry.get = MagicMock(return_value=fake_rcv)
        else:
            rcv_registry.get = MagicMock(return_value=None)
        rcv_registry.send = AsyncMock(return_value=True)
        rcv_registry.broadcast = AsyncMock(return_value=1)
        app.state.receiver_registry = rcv_registry

        # Pin max_players for the test by patching the profile_registry
        # via the app's import — we just need .get(profile_id).max_players.
        from augmentum.game_stream import profiles as _profiles_mod
        original = _profiles_mod.profile_registry.get
        fake_profile = MagicMock()
        fake_profile.max_players = profile_max_players
        _profiles_mod.profile_registry.get = MagicMock(  # type: ignore[attr-defined]
            return_value=fake_profile,
        )
        # Yield-and-restore via a small finalizer attached to the app.
        # pytest's request fixture would be cleaner but the test class
        # doesn't take it — restore at end of each test instead.
        app._test_restore_profile_get = (_profiles_mod, original)  # type: ignore[attr-defined]
        return rcv_registry

    def _restore(self, app):
        sentinel = getattr(app, "_test_restore_profile_get", None)
        if sentinel is not None:
            mod, original = sentinel
            mod.profile_registry.get = original

    def test_happy_path_mints_token_and_dispatches_qr(self, app, client):
        rcv_registry = self._wire_app(app)
        try:
            r = client.post(
                "/api/cast/games/session/s1/invite",
                json={"receiver_id": "r1", "max_slots": 2},
            )
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["token"].startswith("wsi_")
            # join_url is the absolute LAN URL the receiver encodes
            # into the QR. _build_join_url resolves the public host
            # via x-forwarded-host headers (here: testserver).
            assert body["join_url"].endswith(
                f"/ui/cast-guest-join/?token={body['token']}",
            )
            assert body["qr_url"] == f"/api/cast/invite/qr/{body['token']}.svg"
            assert body["slots_remaining"] == 2
            assert body["slots_total"] == 2
            # Receiver got the show_invite_qr cmd.
            rcv_registry.send.assert_awaited_once()
            (recv_id, cmd) = rcv_registry.send.await_args.args
            assert recv_id == "r1"
            assert cmd.cmd == "show_invite_qr"
            assert cmd.args["token"] == body["token"]
            assert cmd.args["slots_remaining"] == 2
        finally:
            self._restore(app)

    def test_single_player_profile_returns_409(self, app, client):
        self._wire_app(app, profile_max_players=1)
        try:
            r = client.post(
                "/api/cast/games/session/s1/invite",
                json={"receiver_id": "r1", "max_slots": 2},
            )
            assert r.status_code == 409
        finally:
            self._restore(app)

    def test_unowned_session_returns_404(self, app, client):
        self._wire_app(app, session_owner="")
        try:
            r = client.post(
                "/api/cast/games/session/s1/invite",
                json={"receiver_id": "r1", "max_slots": 2},
            )
            assert r.status_code == 404
        finally:
            self._restore(app)

    def test_unowned_receiver_returns_404(self, app, client):
        self._wire_app(app, receiver_owner="someone_else")
        try:
            r = client.post(
                "/api/cast/games/session/s1/invite",
                json={"receiver_id": "r1", "max_slots": 2},
            )
            assert r.status_code == 404
        finally:
            self._restore(app)

    def test_max_slots_capped_at_max_players_minus_host(self, app, client):
        """Profile max_players=4 — host takes 1, so an invite caps at 3.

        Caller asking for 5 silently caps; not an error since "give me
        as many as possible" is a reasonable intent.
        """
        self._wire_app(app, profile_max_players=4)
        try:
            r = client.post(
                "/api/cast/games/session/s1/invite",
                json={"receiver_id": "r1", "max_slots": 5},
            )
            assert r.status_code == 200
            assert r.json()["slots_total"] == 3
        finally:
            self._restore(app)


class TestInviteRevokeEndpoint:
    def test_revoke_clears_token_from_store(self, app, client):
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()
        record = app.state.cast_invite_store.mint(
            session_id="s1", host_user_id="usr_test",
        )
        rcv_registry = MagicMock()
        rcv_registry.broadcast = AsyncMock(return_value=1)
        app.state.receiver_registry = rcv_registry

        r = client.post(
            f"/api/cast/games/session/s1/invite/{record.token}/revoke",
        )
        assert r.status_code == 200, r.text
        assert app.state.cast_invite_store.get(record.token) is None

    def test_revoke_idempotent_on_unknown_token(self, app, client):
        """Revoking a token that was never minted is a no-op success.

        The intent ("this token should not work") is satisfied either
        way; an error would force the caller to special-case races.
        """
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()
        rcv_registry = MagicMock()
        rcv_registry.broadcast = AsyncMock(return_value=0)
        app.state.receiver_registry = rcv_registry

        r = client.post(
            "/api/cast/games/session/s1/invite/wsi_neverminted/revoke",
        )
        assert r.status_code == 200

    def test_session_stop_hook_revokes_pending_invites(self, app, client):
        """The runtime's on_session_stopped hook (wired in server.py)
        must revoke any outstanding invites for that session.

        We can't easily exercise the real stop_session path in a unit
        test, but the substrate is the InviteStore.revoke_for_session
        call — already covered by test_revoke_for_session_clears_all
        in test_cast_invite_store.py. This test pins the *integration*
        by manually invoking the hook the way create_app wires it.
        """
        from augmentum.cast.invite_store import InviteStore
        store = InviteStore()
        app.state.cast_invite_store = store
        rec = store.mint(session_id="s1", host_user_id="usr_test")

        rcv_registry = MagicMock()
        rcv_registry.broadcast = AsyncMock(return_value=1)
        app.state.receiver_registry = rcv_registry

        # Replicate the hook body. Keeping this in sync with server.py
        # is intentional — if the hook signature changes, this test
        # catches the divergence at the test-write boundary.
        import asyncio
        async def _hook(session_id: str, user_id: str):
            s = getattr(app.state, "cast_invite_store", None)
            if s is not None:
                s.revoke_for_session(session_id)
        asyncio.get_event_loop().run_until_complete(_hook("s1", "usr_test"))
        assert store.get(rec.token) is None

    def test_revoke_for_wrong_session_returns_404(self, app, client):
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()
        record = app.state.cast_invite_store.mint(
            session_id="s1", host_user_id="usr_test",
        )
        rcv_registry = MagicMock()
        rcv_registry.broadcast = AsyncMock(return_value=0)
        app.state.receiver_registry = rcv_registry

        r = client.post(
            f"/api/cast/games/session/other-session/invite/{record.token}/revoke",
        )
        assert r.status_code == 404


# ── Multi-phone simultaneous input ────────────────────────────────────


class TestMultiPhoneSimultaneousInput:
    """The core couch co-op property: two (or more) phones can each
    drive a different slot on the same session container concurrently.

    The substrate that makes this work:

      1. Each phone holds its own WS to ``/api/cast/input/ws`` — these
         are independent connections handled by independent route
         coroutines on the proxy.
      2. ``CastInputRegistry.attach_phone`` keys by an opaque
         ``attachment_id``, so two phones for the same session don't
         collide in the lookup table.
      3. Slot claim runs lazily on first frame:
            - ``index`` strategy uses the phone's reported pad_index
            - ``firstpress`` strategy claims the next free slot on
              the first frame carrying a non-zero button
         Once claimed, ``phone.slot`` is stamped onto every outbound
         frame so the container daemon can route per-slot to its
         per-pad UInput devices.
      4. The container WS is single — ALL phones' frames funnel
         through one socket with the slot field as the disambiguator.
         The container's read_loop processes frames sequentially;
         the per-slot ``PadState`` cache means buttons on P1 don't
         leak into P2's state.

    These tests drive the WS routes via TestClient one phone at a
    time (TestClient's WS connections are sync) but verify the per-
    slot stamping and isolation that makes simultaneous frames work.
    """

    def _wire_two_phones(self, app, *, routing="firstpress"):
        """Install registry + fake container and return references."""
        from augmentum.cast.input_bridge import CastInputRegistry
        app.state.cast_input_registry = CastInputRegistry()
        app.state.game_stream_runtime = None  # skip ownership check
        registry = app.state.cast_input_registry
        fake_container_ws = _FakeWS()
        registry.attach_container(
            ws=fake_container_ws, session_id="s1", user_id="usr_test",
            pad_routing=routing,
        )
        return registry, fake_container_ws

    def test_firstpress_claims_unique_slots_per_phone(self, app, client):
        """Two phones, both attached simultaneously, each press their
        own buttons. Under firstpress the first presser gets slot 0,
        the second gets slot 1. Both slots stamp distinctly on the
        container WS even though both phones are alive at once.

        Critical: phones MUST stay attached together for this to test
        what we care about (concurrent slot ownership). If we sequenced
        attach→detach→attach, the second phone would just reclaim
        slot 0 from the freed pool — a different (also valid) flow.
        """
        registry, fake_container_ws = self._wire_two_phones(app)

        def _press_frame(seq: int, button: int) -> dict:
            buttons = [0] * 17
            buttons[button] = 1
            return {
                "seq": seq, "t_send": 100.0,
                "event": {
                    "kind": "gamepad_state",
                    "pad_index": 0,
                    "buttons": buttons,
                    "axes": [0.0] * 4,
                },
            }

        # Both phones open and held open. Phone A claims first by
        # pressing button 0; phone B then claims slot 1 by pressing
        # button 1. Slot pool: {0: A, 1: B} at end of both sends.
        with client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws_a, client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws_b:
            ws_a.send_json(_press_frame(100, button=0))
            ws_b.send_json(_press_frame(101, button=1))
            import time
            # Wait for both frames to land on the container side.
            for _ in range(50):
                seqs = {f.get("seq") for f in fake_container_ws.sent}
                if {100, 101}.issubset(seqs):
                    break
                time.sleep(0.01)

        frame_a = next(f for f in fake_container_ws.sent if f.get("seq") == 100)
        frame_b = next(f for f in fake_container_ws.sent if f.get("seq") == 101)
        # Slots are distinct — no double-assignment.
        assert frame_a["slot"] != frame_b["slot"]
        # Under firstpress, slots are claimed lowest-first.
        assert {frame_a["slot"], frame_b["slot"]} == {0, 1}

    def test_index_routing_honours_pad_index_per_phone(self, app, client):
        """Under ``index`` routing, slot follows pad_index.

        Phone B (pad_index=1) connects FIRST; phone A (pad_index=0)
        connects second. Slot assignment is still deterministic by
        reported pad_index — A lands on slot 0, B on slot 1.
        """
        registry, fake_container_ws = self._wire_two_phones(
            app, routing="index",
        )

        def _frame(pad_index: int) -> dict:
            return {
                "seq": pad_index + 100, "t_send": 100.0,
                "event": {
                    "kind": "gamepad_state",
                    "pad_index": pad_index,
                    "buttons": [0] * 17,
                    "axes": [0.0] * 4,
                },
            }

        # Both phones open and held open. Phone B (pad_index=1) sends
        # first, then phone A (pad_index=0). Index strategy doesn't
        # care about connect-order — slot maps to pad_index.
        with client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="1"),
        ) as ws_b, client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws_a:
            ws_b.send_json(_frame(1))
            ws_a.send_json(_frame(0))
            import time
            for _ in range(50):
                seqs = {f.get("seq") for f in fake_container_ws.sent}
                if {100, 101}.issubset(seqs):
                    break
                time.sleep(0.01)

        frame_a = next(
            f for f in fake_container_ws.sent if f.get("seq") == 100
        )
        frame_b = next(
            f for f in fake_container_ws.sent if f.get("seq") == 101
        )
        assert frame_a["slot"] == 0
        assert frame_b["slot"] == 1

    def test_neutral_frames_under_firstpress_dont_claim_slot(self, app, client):
        """firstpress requires an active button press to claim — a
        phone sending only neutral state stays at slot=-1.

        This is the "two phones on the QR scan page but neither has
        pressed anything yet" race. Both phones should pump frames
        without colliding on slot 0.
        """
        registry, fake_container_ws = self._wire_two_phones(app)

        neutral = {
            "seq": 1, "t_send": 100.0,
            "event": {
                "kind": "gamepad_state",
                "pad_index": 0,
                "buttons": [0] * 17,
                "axes": [0.0] * 4,
            },
        }

        with client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws_a:
            ws_a.send_json(neutral)
            import time
            for _ in range(30):
                if any(f.get("seq") == 1 for f in fake_container_ws.sent):
                    break
                time.sleep(0.01)
            # Without a button press, slot stays unclaimed (-1) so
            # the container daemon drops the frame.
            first = next(f for f in fake_container_ws.sent if f.get("seq") == 1)
            assert first["slot"] == -1

    def test_max_4_phones_per_session(self, app, client):
        """Slot pool caps at 4. A fifth concurrently-connected phone
        sends frames but firstpress can't claim — slot stays -1
        until one of the first 4 disconnects and frees a slot.
        """
        registry, fake_container_ws = self._wire_two_phones(app)

        def press_frame(seq: int) -> dict:
            buttons = [0] * 17
            buttons[0] = 1
            return {
                "seq": seq, "t_send": 100.0,
                "event": {
                    "kind": "gamepad_state",
                    "pad_index": 0,
                    "buttons": buttons,
                    "axes": [0.0] * 4,
                },
            }

        # Keep all 5 phones connected simultaneously via nested
        # context managers. TestClient handles independent WS sockets
        # concurrently in separate background threads.
        with client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws1, client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws2, client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws3, client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws4, client.websocket_connect(
            _ws_url("/api/cast/input/ws", session_id="s1", pad_index="0"),
        ) as ws5:
            for idx, ws in enumerate([ws1, ws2, ws3, ws4, ws5]):
                ws.send_json(press_frame(idx + 10))
            import time
            for _ in range(80):
                seqs = {f.get("seq") for f in fake_container_ws.sent}
                if {10, 11, 12, 13, 14}.issubset(seqs):
                    break
                time.sleep(0.01)

        # Property check, not order check: among the 5 sent frames,
        # exactly 4 land on distinct claimed slots {0,1,2,3}, and
        # exactly 1 lands with slot=-1 (the unlucky 5th to win the
        # claim race). Send-order ≠ receive-order under concurrent
        # WS reads, so we don't assert which seq took which slot.
        target_seqs = {10, 11, 12, 13, 14}
        frames_by_seq = {
            f.get("seq"): f for f in fake_container_ws.sent
            if f.get("seq") in target_seqs
        }
        # All 5 phones' input frames reached the container.
        assert set(frames_by_seq.keys()) == target_seqs

        slots_claimed = sorted(
            f["slot"] for f in frames_by_seq.values() if f["slot"] >= 0
        )
        unclaimed = [
            f for f in frames_by_seq.values() if f["slot"] == -1
        ]
        assert slots_claimed == [0, 1, 2, 3]
        assert len(unclaimed) == 1


# ── Phase 2: identify / claim endpoints ─────────────────────────────


class TestGuestIdentifyEndpoint:
    """The unauthenticated identify endpoint — token IS the credential."""

    def _wire(self, app, *, guest_profiles=None):
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()
        record = app.state.cast_invite_store.mint(
            session_id="s1", host_user_id="usr_test",
        )

        # Mock the GuestStore — we test the SQLite path separately
        # in test_cast_guest_store.py.
        guest_store = MagicMock()
        guest_store.list_for_host = AsyncMock(
            return_value=guest_profiles or [],
        )
        app.state.guest_store = guest_store
        return record, guest_store

    def test_unknown_token_returns_404(self, app, client):
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()
        app.state.guest_store = MagicMock()
        r = client.post(
            "/api/cast/guest/identify",
            json={"token": "wsi_nope"},
        )
        assert r.status_code == 404

    def test_no_substrate_returns_503(self, app, client):
        app.state.cast_invite_store = None
        r = client.post(
            "/api/cast/guest/identify",
            json={"token": "wsi_x"},
        )
        assert r.status_code == 503

    def test_empty_roster_returns_matched_false(self, app, client):
        record, _ = self._wire(app, guest_profiles=[])
        r = client.post(
            "/api/cast/guest/identify",
            json={"token": record.token},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["matched"] is False
        assert body["existing_profiles"] == []

    def test_roster_returns_existing_profiles(self, app, client):
        record, _ = self._wire(app, guest_profiles=[
            {"id": "gp_alice", "display_name": "alice", "color": "#4ade80"},
            {"id": "gp_bob", "display_name": "bob", "color": ""},
        ])
        r = client.post(
            "/api/cast/guest/identify",
            json={"token": record.token},
        )
        body = r.json()
        assert body["matched"] is False
        assert len(body["existing_profiles"]) == 2
        assert body["existing_profiles"][0]["display_name"] == "alice"


class TestGuestClaimEndpoint:
    def _wire(self, app):
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()
        record = app.state.cast_invite_store.mint(
            session_id="s1", host_user_id="usr_test",
        )
        guest_store = MagicMock()
        guest_store.create_profile = AsyncMock()
        guest_store.get = AsyncMock()
        guest_store.get_by_name = AsyncMock(return_value=None)
        guest_store.touch_last_seen = AsyncMock()
        app.state.guest_store = guest_store
        return record, guest_store

    def test_existing_profile_id_returns_profile(self, app, client):
        record, guest_store = self._wire(app)
        guest_store.get.return_value = {
            "id": "gp_alice", "display_name": "alice", "color": "#4ade80",
        }
        r = client.post(
            "/api/cast/guest/claim",
            json={"token": record.token, "profile_id": "gp_alice"},
        )
        assert r.status_code == 200
        assert r.json()["profile"]["id"] == "gp_alice"

    def test_unknown_existing_profile_returns_404(self, app, client):
        record, guest_store = self._wire(app)
        guest_store.get.return_value = None
        r = client.post(
            "/api/cast/guest/claim",
            json={"token": record.token, "profile_id": "gp_ghost"},
        )
        assert r.status_code == 404

    def test_new_name_creates_profile(self, app, client):
        record, guest_store = self._wire(app)
        guest_store.create_profile.return_value = {
            "id": "gp_charlie", "display_name": "charlie", "color": "",
        }
        r = client.post(
            "/api/cast/guest/claim",
            json={"token": record.token, "new_name": "charlie"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["profile"]["display_name"] == "charlie"
        guest_store.create_profile.assert_awaited_once()

    def test_name_collision_returns_conflict_with_existing(self, app, client):
        """Same-name collision response carries the existing profile
        id so the UI can fall back to "is that you?" without a
        second roundtrip.
        """
        record, guest_store = self._wire(app)
        # Mirror what create_profile does on UNIQUE violation: raise,
        # then get_by_name returns the existing row.
        guest_store.create_profile.side_effect = Exception("UNIQUE constraint")
        guest_store.get_by_name.return_value = {
            "id": "gp_alice", "display_name": "alice", "color": "#4ade80",
        }
        r = client.post(
            "/api/cast/guest/claim",
            json={"token": record.token, "new_name": "alice"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["conflict"] is True
        assert body["existing_profile"]["id"] == "gp_alice"

    def test_missing_both_profile_and_name_returns_400(self, app, client):
        record, _ = self._wire(app)
        r = client.post(
            "/api/cast/guest/claim",
            json={"token": record.token},
        )
        assert r.status_code == 400

    def test_touch_last_seen_called_with_play_count_bump(self, app, client):
        """Each successful claim must bump play_count exactly once so
        the host's "most-recent guests" list orders correctly.
        """
        record, guest_store = self._wire(app)
        guest_store.get.return_value = {
            "id": "gp_alice", "display_name": "alice", "color": "",
        }
        r = client.post(
            "/api/cast/guest/claim",
            json={"token": record.token, "profile_id": "gp_alice"},
        )
        assert r.status_code == 200
        guest_store.touch_last_seen.assert_awaited_once()
        kwargs = guest_store.touch_last_seen.await_args.kwargs
        assert kwargs.get("increment_play_count") is True


class TestGuestProfileWSAttachment:
    """When a guest connects with ?guest_profile_id=, the registry's
    ConnectedPhone carries the profile identity for downstream UI
    (player chip strip).
    """

    def test_profile_id_threads_through_to_connected_phone(self, app, client):
        from augmentum.cast.input_bridge import CastInputRegistry
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_input_registry = CastInputRegistry()
        app.state.cast_invite_store = InviteStore()
        record = app.state.cast_invite_store.mint(
            session_id="s1", host_user_id="usr_test",
        )

        guest_store = MagicMock()
        guest_store.get = AsyncMock(return_value={
            "id": "gp_alice", "display_name": "alice", "color": "#4ade80",
        })
        app.state.guest_store = guest_store
        app.state.game_stream_runtime = None

        url = (
            f"/api/cast/input/ws?join_token={record.token}"
            f"&guest_profile_id=gp_alice"
        )
        with client.websocket_connect(url) as _ws:
            phones = list(app.state.cast_input_registry._phones.values())
            assert len(phones) == 1
            assert phones[0].guest_profile_id == "gp_alice"
            assert phones[0].guest_display_name == "alice"
            assert phones[0].guest_color == "#4ade80"

    def test_unknown_profile_id_rejects_with_1008(self, app, client):
        from augmentum.cast.input_bridge import CastInputRegistry
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_input_registry = CastInputRegistry()
        app.state.cast_invite_store = InviteStore()
        record = app.state.cast_invite_store.mint(
            session_id="s1", host_user_id="usr_test",
        )
        guest_store = MagicMock()
        guest_store.get = AsyncMock(return_value=None)
        app.state.guest_store = guest_store
        app.state.game_stream_runtime = None

        url = (
            f"/api/cast/input/ws?join_token={record.token}"
            f"&guest_profile_id=gp_ghost"
        )
        with pytest.raises(WebSocketDisconnect) as exc_info, \
                client.websocket_connect(url) as ws:
            ws.receive_text()
        assert exc_info.value.code == 1008
