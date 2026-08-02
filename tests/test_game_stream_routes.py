"""Behavior tests for the game-stream REST endpoints (/api/game-stream/*).

Focus areas:
* Master toggle -- when ``game_stream_enabled`` is False, every endpoint
  returns 503 (we treat the feature as administratively off).
* User isolation -- a user must never see another user's worlds or
  sessions, even if a session_id is guessed.
* Auth gate -- unauthenticated requests get 401 (after the master
  toggle, before any work).
* Lifecycle entry points -- POST /sessions starts a session with the
  stub adapter, DELETE stops + releases ports.
"""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest
from fastapi.testclient import TestClient

# Reuse the schema from the smoke-test file rather than duplicating SQL --
# the migration is the source of truth, this is a behavioural test of the
# routes layer.
from tests.test_game_stream_smoke import _SCHEMA_SQL  # noqa: E402


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def stream_client(app, monkeypatch):
    """TestClient with a real GameStreamStore + GameStreamRuntime wired."""
    from augmentum.config import settings as config_settings
    from augmentum.game_stream import GameStreamRuntime, PortPool
    from augmentum.state.game_stream_store import GameStreamStore

    # Enable the master toggle for the duration of the fixture.
    monkeypatch.setattr(config_settings, "game_stream_enabled", True)

    conn = _run(aiosqlite.connect(":memory:"))
    _run(conn.executescript(_SCHEMA_SQL))
    # The conftest test_user is "usr_test"; seed both that and an
    # "usr_other" so isolation tests can act as the second user.
    _run(conn.execute("INSERT INTO users (id) VALUES ('usr_test')"))
    _run(conn.execute("INSERT INTO users (id) VALUES ('usr_other')"))
    _run(conn.commit())

    store = GameStreamStore(conn)
    runtime = GameStreamRuntime(
        store=store,
        port_pool=PortPool(base=40000, count=4),
        max_concurrent_per_user=2,
    )
    app.state.game_stream_store = store
    app.state.game_stream_runtime = runtime

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, store, runtime
    _run(conn.close())


@pytest.fixture
def stream_client_disabled(app, monkeypatch):
    """Master toggle off -- every route should return 503."""
    from augmentum.config import settings as config_settings

    monkeypatch.setattr(config_settings, "game_stream_enabled", False)
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc


# ── Master toggle ────────────────────────────────────────────────────


class TestMasterToggle:
    def test_profiles_503_when_disabled(self, stream_client_disabled):
        r = stream_client_disabled.get("/api/game-stream/profiles")
        assert r.status_code == 503

    def test_worlds_list_503_when_disabled(self, stream_client_disabled):
        r = stream_client_disabled.get("/api/game-stream/worlds")
        assert r.status_code == 503

    def test_session_start_503_when_disabled(self, stream_client_disabled):
        r = stream_client_disabled.post(
            "/api/game-stream/sessions", json={"profile_id": "luanti"},
        )
        assert r.status_code == 503


# ── Profiles ─────────────────────────────────────────────────────────


class TestProfiles:
    def test_lists_built_in_profiles(self, stream_client):
        client, _, _ = stream_client
        r = client.get("/api/game-stream/profiles")
        assert r.status_code == 200
        body = r.json()
        ids = {p["id"] for p in body["profiles"]}
        assert "luanti" in ids

    def test_profiles_expose_input_capabilities(self, stream_client):
        client, _, _ = stream_client
        r = client.get("/api/game-stream/profiles")
        assert r.status_code == 200
        luanti = next(p for p in r.json()["profiles"] if p["id"] == "luanti")
        assert luanti["wants_gamepad"] is True
        assert luanti["input_capabilities"]["gamepad"]["supported"] is True
        assert luanti["input_capabilities"]["pointer"]["mode"] == "relative"


# ── Worlds ───────────────────────────────────────────────────────────


class TestWorlds:
    def test_create_returns_201_and_round_trips(self, stream_client):
        client, _, _ = stream_client
        r = client.post(
            "/api/game-stream/worlds",
            json={"profile_id": "luanti", "name": "My World"},
        )
        assert r.status_code == 201
        world = r.json()["world"]
        assert world["name"] == "My World"
        assert world["profile_id"] == "luanti"
        # And it's listed.
        listing = client.get("/api/game-stream/worlds").json()["worlds"]
        assert any(w["id"] == world["id"] for w in listing)

    def test_create_rejects_unknown_profile(self, stream_client):
        client, _, _ = stream_client
        r = client.post(
            "/api/game-stream/worlds",
            json={"profile_id": "no-such-game", "name": "X"},
        )
        assert r.status_code == 400

    def test_user_isolation_on_worlds(self, stream_client):
        """Core isolation: usr_other's world is invisible to usr_test."""
        client, store, _ = stream_client
        _run(store.create_world(
            user_id="usr_other", profile_id="luanti", name="Theirs",
        ))
        r = client.get("/api/game-stream/worlds")
        listing = r.json()["worlds"]
        assert listing == []  # usr_test owns nothing, isn't whitelisted

    def test_whitelist_makes_world_visible(self, stream_client):
        client, store, _ = stream_client
        wid = _run(store.create_world(
            user_id="usr_other",
            profile_id="luanti",
            name="Shared",
            whitelist=["usr_test"],
        ))
        listing = client.get("/api/game-stream/worlds").json()["worlds"]
        assert any(w["id"] == wid for w in listing)


# ── Sessions ─────────────────────────────────────────────────────────


class TestSessions:
    def test_start_session_assigns_ports(self, stream_client):
        client, _, _ = stream_client
        r = client.post(
            "/api/game-stream/sessions", json={"profile_id": "luanti"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["stream_port"] == 40000
        assert body["game_port"] == 40001
        assert body["session_id"]
        assert body["signaling_path"].endswith(body["session_id"])

    def test_start_session_threads_input_options(self, stream_client):
        class RecorderAdapter:
            host_network = False

            def __init__(self):
                self.kwargs = None

            async def start(self, **kwargs):
                self.kwargs = kwargs
                return "cnt_input"

            async def stop(self, container_id: str, *, timeout: int = 10):
                return None

            async def is_alive(self, container_id: str) -> bool:
                return True

        client, _, runtime = stream_client
        adapter = RecorderAdapter()
        runtime._adapter = adapter
        r = client.post(
            "/api/game-stream/sessions",
            json={
                "profile_id": "luanti",
                "input": {
                    "touch_mode": True,
                    "mouse_sensitivity": 0.08,
                    "gamepad_enabled": False,
                    "controller_deadzone": 0.22,
                },
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert body["input"] == {
            "touch_mode": True,
            "mouse_sensitivity": 0.08,
            "gamepad_enabled": False,
            "controller_deadzone": 0.22,
        }
        assert adapter.kwargs["touch_mode"] is True
        assert adapter.kwargs["mouse_sensitivity"] == 0.08
        assert adapter.kwargs["gamepad_enabled"] is False
        assert adapter.kwargs["controller_deadzone"] == 0.22

    def test_start_session_composes_companion_when_requested(
        self, stream_client, monkeypatch,
    ):
        """@example: companion=true creates a paired game-agent session
        and threads its bridge URL into the container env.

        Without this wiring the streamed-emulator container starts up
        with AUGMENTUM_AGENT_BRIDGE_URL unset, so entrypoint-base.sh
        skips agent-bridge.py and the AI session is dead on arrival.
        """

        import json as _json

        from augmentum.config import settings as config_settings

        async def _stub_llm(_prompt, _frame):
            return _json.dumps({
                "observations": ["stub"],
                "state_update": "",
                "actions": [],
                "confidence": 0.5,
                "next_check_in_ms": 500,
            })

        class RecorderAdapter:
            host_network = False

            def __init__(self):
                self.kwargs = None

            async def start(self, **kwargs):
                self.kwargs = kwargs
                return "cnt_companion"

            async def stop(self, container_id, *, timeout=10):
                return None

            async def is_alive(self, container_id):
                return True

        # Wire the game-agent dependencies the helper expects.
        client, _, runtime = stream_client
        client.app.state.game_agent_llm = _stub_llm
        monkeypatch.setattr(
            config_settings,
            "agent_bridge_base_url",
            "ws://augmentum-test:8080",
        )
        adapter = RecorderAdapter()
        runtime._adapter = adapter

        r = client.post(
            "/api/game-stream/sessions",
            json={
                "profile_id": "luanti",
                "companion": {
                    "objective": "advance dialog",
                    "log_schema": "emulator.v1",
                },
            },
        )
        assert r.status_code == 201, r.text
        body = r.json()

        # The route surfaces the paired agent_session_id so the browser
        # can subscribe to the log SSE stream.
        assert body.get("agent_session_id", "").startswith("s_")

        # The container env carries the bridge URL with the internal
        # host (the operator-configured base) AND a token. Without
        # either the daemon can't authenticate at the WS endpoint.
        url = adapter.kwargs["agent_bridge_url"]
        assert url.startswith("ws://augmentum-test:8080")
        assert "/surfaces/emulator/bridge/" in url
        assert "?token=" in url
        assert body["agent_session_id"] in url

    def test_companion_503_when_bridge_base_url_missing(
        self, stream_client, monkeypatch,
    ):
        """@example: companion=true without agent_bridge_base_url -> 503.

        ROOT CAUSE:
        Misconfigured deployments would otherwise launch an AI
        container with no AUGMENTUM_AGENT_BRIDGE_URL, agent-bridge.py
        would silently skip, and the user would see a passive
        container with no AI taking action. Failing the POST loudly
        forces the operator to set the env var.
        """

        import json as _json

        from augmentum.config import settings as config_settings

        async def _stub_llm(_prompt, _frame):
            return _json.dumps({
                "observations": [], "state_update": "",
                "actions": [], "confidence": 0.5, "next_check_in_ms": 500,
            })

        client, _, _ = stream_client
        client.app.state.game_agent_llm = _stub_llm
        monkeypatch.setattr(
            config_settings, "agent_bridge_base_url", "",
        )

        r = client.post(
            "/api/game-stream/sessions",
            json={
                "profile_id": "luanti",
                "companion": {"objective": "x", "log_schema": "emulator.v1"},
            },
        )
        assert r.status_code == 503
        assert "AGENT_BRIDGE" in r.json()["error"]

    def test_concurrent_cap_returns_429(self, stream_client):
        client, _, runtime = stream_client
        # cap is 2 per the fixture
        for _ in range(2):
            r = client.post(
                "/api/game-stream/sessions", json={"profile_id": "luanti"},
            )
            assert r.status_code == 201
        r = client.post(
            "/api/game-stream/sessions", json={"profile_id": "luanti"},
        )
        assert r.status_code == 429

    def test_get_session_user_scoped(self, stream_client):
        client, store, _ = stream_client
        # Create session as usr_other.
        sid = _run(store.create_session(user_id="usr_other", profile_id="luanti"))
        r = client.get(f"/api/game-stream/sessions/{sid}")
        # usr_test must not see it.
        assert r.status_code == 404

    def test_readiness_reports_waiting_stream(self, stream_client):
        client, _, runtime = stream_client
        info = _run(runtime.start_session(user_id="usr_test", profile_id="luanti"))
        r = client.get(f"/api/game-stream/sessions/{info.session_id}/readiness")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is False
        assert body["stage"] == "waiting_stream"
        assert body["stream_port"] == 40000
        assert body["container_alive"] is True
        assert body["probe"]["target"].endswith(":40000/")

    def test_readiness_marks_ready_when_stream_accepts(self, stream_client, monkeypatch):
        from augmentum.proxy import game_stream_routes

        async def fake_probe(port):
            return {
                "ok": True,
                "status": 200,
                "target": f"http://probe-host:{port}/",
                "error": "",
            }

        monkeypatch.setattr(game_stream_routes, "_probe_stream_root", fake_probe)
        client, store, runtime = stream_client
        info = _run(runtime.start_session(user_id="usr_test", profile_id="luanti"))
        r = client.get(f"/api/game-stream/sessions/{info.session_id}/readiness")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is True
        assert body["stage"] == "ready"
        row = _run(store.get_session(info.session_id, user_id="usr_test"))
        assert row["status"] == "ready"

    def test_readiness_user_scoped(self, stream_client):
        client, store, _ = stream_client
        sid = _run(store.create_session(user_id="usr_other", profile_id="luanti"))
        r = client.get(f"/api/game-stream/sessions/{sid}/readiness")
        assert r.status_code == 404

    def test_stop_session_releases_and_marks_stopped(self, stream_client):
        client, store, runtime = stream_client
        # Start one
        info = _run(runtime.start_session(user_id="usr_test", profile_id="luanti"))
        # Stop via DELETE
        r = client.delete(f"/api/game-stream/sessions/{info.session_id}")
        assert r.status_code == 200
        row = _run(store.get_session(info.session_id, user_id="usr_test"))
        assert row["status"] == "stopped"
        assert row["exit_reason"] == "clean"

    def test_heartbeat_marks_ready_session_connected(self, stream_client):
        client, store, runtime = stream_client
        info = _run(runtime.start_session(user_id="usr_test", profile_id="luanti"))
        assert _run(runtime.mark_ready(info.session_id, user_id="usr_test"))
        r = client.post(f"/api/game-stream/sessions/{info.session_id}/heartbeat")
        assert r.status_code == 200
        assert r.json()["status"] == "connected"
        row = _run(store.get_session(info.session_id, user_id="usr_test"))
        assert row["status"] == "connected"

    def test_heartbeat_user_scoped(self, stream_client):
        client, store, _ = stream_client
        sid = _run(store.create_session(user_id="usr_other", profile_id="luanti"))
        r = client.post(f"/api/game-stream/sessions/{sid}/heartbeat")
        assert r.status_code == 404


# ── Telemetry ────────────────────────────────────────────────────────


class TestTelemetry:
    def test_post_telemetry_404_for_other_users_session(self, stream_client):
        client, store, _ = stream_client
        sid = _run(store.create_session(user_id="usr_other", profile_id="luanti"))
        r = client.post(
            f"/api/game-stream/sessions/{sid}/telemetry",
            json={"rtt_ms": 42.0},
        )
        assert r.status_code == 404

    def test_post_telemetry_writes_row(self, stream_client):
        client, store, runtime = stream_client
        info = _run(runtime.start_session(user_id="usr_test", profile_id="luanti"))
        r = client.post(
            f"/api/game-stream/sessions/{info.session_id}/telemetry",
            json={"rtt_ms": 42.0, "fps": 60.0, "bitrate_kbps": 4000},
        )
        assert r.status_code == 200
        rows = _run(store.recent_telemetry(info.session_id, user_id="usr_test"))
        assert len(rows) == 1
        assert rows[0]["rtt_ms"] == 42.0
        assert rows[0]["fps"] == 60.0
        assert rows[0]["bitrate_kbps"] == 4000
