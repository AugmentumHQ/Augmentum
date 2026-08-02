"""HTTP + WebSocket integration tests for game_agent_routes.

Uses starlette's TestClient (FastAPI's bundled test harness) against
a fresh FastAPI app per test so app.state is isolated. The LLM is
always a deterministic stub; the orchestrator is real.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.proxy.game_agent_routes import router


async def _stub_llm_noop(_prompt: str, _frame: bytes | None) -> str:
    """A deterministic LLM stub that emits an empty plan -- session ends
    promptly when the orchestrator gets stopped from the outside."""

    return json.dumps(
        {
            "observations": ["stubbed"],
            "state_update": "",
            "actions": [],
            "confidence": 0.5,
            "next_check_in_ms": 500,
        }
    )


def _make_app(tmp_path: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.game_agent_llm = _stub_llm_noop
    app.state.game_agent_log_dir = tmp_path
    return app


def _wait_for_log(path: Path, timeout_s: float = 2.0) -> list[dict]:
    """Poll until the log file has at least a session_end line."""

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            lines = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            if any(e["kind"] == "session_end" for e in lines):
                return lines
        time.sleep(0.05)
    raise AssertionError(f"log {path} did not finalize within {timeout_s}s")


def test_503_when_llm_unconfigured(tmp_path: Path) -> None:
    """@example: starting a session without an LLM configured returns 503."""

    app = FastAPI()
    app.include_router(router)
    app.state.game_agent_log_dir = tmp_path
    # NOTE: deliberately no app.state.game_agent_llm

    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={"surface": "mock", "objective": "x"},
        )
        assert r.status_code == 503


def test_mock_session_runs_end_to_end(tmp_path: Path) -> None:
    """@example: POST /sessions for mock starts immediately; stop finalizes the log."""

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={"surface": "mock", "objective": "demo"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "running"
        assert body["bridge_ws_url"] is None
        session_id = body["session_id"]

        # Give the orchestrator a moment to write the header + caps.
        time.sleep(0.2)

        status = client.get(f"/api/game-agent/sessions/{session_id}").json()
        assert status["session_id"] == session_id
        assert status["surface"] == "mock"
        assert status["status"] in {"running", "stopped"}

        stop = client.post(f"/api/game-agent/sessions/{session_id}/stop")
        assert stop.status_code == 200

        # The orchestrator should finish soon after stop().
        lines = _wait_for_log(tmp_path / f"{session_id}.ndjson")
        kinds = [e["kind"] for e in lines]
        assert kinds[0] == "session"
        assert "surface_caps" in kinds
        assert kinds[-1] == "session_end"


def test_bridged_session_returns_ws_url_and_status_pending(tmp_path: Path) -> None:
    """@example: js13k session returns a bridge URL and stays pending."""

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "js13k",
                "objective": "play tiny game",
                "semantic_inputs": ["left", "right", "jump"],
                "log_schema": "js13k.v1",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "pending_bridge"
        assert body["bridge_ws_url"] is not None
        assert f"/bridge/{body['session_id']}" in body["bridge_ws_url"]
        assert body["bridge_ws_url"].startswith("ws://")


def test_bridged_session_requires_semantic_inputs(tmp_path: Path) -> None:
    """@example: js13k without semantic_inputs is rejected at 400."""

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={"surface": "js13k", "objective": "x"},
        )
        assert r.status_code == 400
        assert "semantic_inputs" in r.json()["error"]


def test_unknown_session_404(tmp_path: Path) -> None:
    """@example: GET and stop on an unknown session_id 404."""

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/game-agent/sessions/s_nope").status_code == 404
        assert client.post("/api/game-agent/sessions/s_nope/stop").status_code == 404


def test_bridged_session_rejects_one_profile_without_the_other(tmp_path: Path) -> None:
    """@example: passing only controller_profile (no game_profile) returns 400."""

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "js13k",
                "objective": "x",
                "semantic_inputs": ["confirm"],
                "log_schema": "js13k.v1",
                "controller_profile": "gba",
                # game_profile deliberately missing
            },
        )
        assert r.status_code == 400
        assert "together" in r.json()["error"].lower()


def test_bridged_session_rejects_unknown_profile_id(tmp_path: Path) -> None:
    """@example: an unregistered profile id is a 400, not a 500."""

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "js13k",
                "objective": "x",
                "semantic_inputs": ["confirm"],
                "log_schema": "js13k.v1",
                "controller_profile": "does_not_exist",
                "game_profile": "pokemon_rs",
            },
        )
        assert r.status_code == 400
        # ProfileLoadError surfaces its message; we only care it's not 500.
        assert "does_not_exist" in r.json()["error"]


def test_bridged_session_accepts_known_profile_pair(tmp_path: Path) -> None:
    """@example: gba + pokemon_rs compose cleanly and the session stays pending."""

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "emulatorjs",
                "objective": "advance dialog",
                "semantic_inputs": ["a", "b"],
                "log_schema": "pokemon_rs.v1",
                "controller_profile": "gba",
                "game_profile": "pokemon_rs",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "pending_bridge"
        assert body["bridge_ws_url"] is not None


def test_emulator_session_returns_ws_url_and_status_pending(tmp_path: Path) -> None:
    """@example: streamed-emulator session returns a bridge URL and stays pending.

    Mirror of the ``emulatorjs`` case: the route accepts ``surface:
    "emulator"``, composes optional control profiles, and hands back a
    bridge_ws_url. The in-container agent-bridge.py daemon dials that
    URL from inside the streaming container.
    """

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "emulator",
                "objective": "win a melee match",
                "semantic_inputs": [
                    "button_a", "button_b", "button_x", "button_y",
                    "dpad_up", "dpad_down", "dpad_left", "dpad_right",
                    "start", "select",
                ],
                "log_schema": "emulator.v1",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "pending_bridge"
        assert body["bridge_ws_url"] is not None
        assert "/surfaces/emulator/bridge/" in body["bridge_ws_url"]
        # The token rides in the URL so a no-cookie dialler (the
        # in-container agent-bridge.py daemon) can authenticate.
        assert "?token=" in body["bridge_ws_url"]


def test_bridge_url_carries_session_scoped_token(tmp_path: Path) -> None:
    """@example: two sessions yield two different bridge_tokens.

    The token is per-session, not per-process, so an attacker who saw
    one session's URL cannot use it to dial another session even when
    both belong to the same user.
    """

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        def _start() -> str:
            r = client.post(
                "/api/game-agent/sessions",
                json={
                    "surface": "js13k",
                    "objective": "x",
                    "semantic_inputs": ["a"],
                    "log_schema": "js13k.v1",
                },
            )
            assert r.status_code == 200, r.text
            return r.json()["bridge_ws_url"]

        url_a = _start()
        url_b = _start()
        token_a = url_a.rsplit("token=", 1)[-1]
        token_b = url_b.rsplit("token=", 1)[-1]
        assert token_a != token_b
        assert len(token_a) >= 32  # secrets.token_urlsafe(32) is ~43 chars


def test_ws_bridge_accepts_token_only_dialler(tmp_path: Path) -> None:
    """@example: a client with only the bridge_token can open the WS.

    Models the in-container agent-bridge.py daemon: it has no user
    cookie, only the token augmentum embedded in the URL it was
    handed. Without token auth this dialler would 404 at the WS
    handshake.
    """

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "emulator",
                "objective": "smoke",
                "semantic_inputs": ["button_a"],
                "log_schema": "emulator.v1",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        token = body["bridge_ws_url"].rsplit("token=", 1)[-1]
        session_id = body["session_id"]

        bridge_path = (
            f"/api/game-agent/surfaces/emulator/bridge/{session_id}"
            f"?token={token}"
        )
        with client.websocket_connect(bridge_path) as ws:
            # The orchestrator starts as soon as the WS accepts; sending
            # ``bye`` lets the session finalise without us hanging on
            # frames or actions.
            ws.send_text(json.dumps({"kind": "bye"}))
            time.sleep(0.3)

        lines = _wait_for_log(tmp_path / f"{session_id}.ndjson")
        kinds = [e["kind"] for e in lines]
        assert kinds[-1] == "session_end"


def test_ws_bridge_rejects_anonymous_dialler_without_token(tmp_path: Path) -> None:
    """@example: knowing the session id is not enough on an anon session.

    ROOT CAUSE:
      An anonymously-created session has owner_user_id "", which equals
      the "" every unauthenticated dialler reports, so `_owned_session`
      returned the record and the owner check degenerated into "knows
      the session id" — re-opening the session-id oracle the bridge
      token exists to close. The route then accepted the socket and
      never sent an error, wedging any caller waiting on a reply.
    """

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "emulator",
                "objective": "smoke",
                "semantic_inputs": ["button_a"],
                "log_schema": "emulator.v1",
            },
        )
        session_id = r.json()["session_id"]
        # No ?token= at all — the pre-fix oracle path.
        path = f"/api/game-agent/surfaces/emulator/bridge/{session_id}"
        with client.websocket_connect(path) as ws:
            err = json.loads(ws.receive_text())
            assert err.get("error") == "no such session"


def test_ws_bridge_rejects_wrong_token(tmp_path: Path) -> None:
    """@example: wrong token + no user cookie -> 404-equivalent close.

    Owner check fails (no user) AND token mismatches, so the route
    closes the socket with the "no such session" error envelope
    instead of accepting the bridge.
    """

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "emulator",
                "objective": "smoke",
                "semantic_inputs": ["button_a"],
                "log_schema": "emulator.v1",
            },
        )
        session_id = r.json()["session_id"]
        bridge_path = (
            f"/api/game-agent/surfaces/emulator/bridge/{session_id}"
            f"?token=not-the-real-token"
        )
        with client.websocket_connect(bridge_path) as ws:
            err = json.loads(ws.receive_text())
            assert err.get("error") == "no such session"


def test_ws_bridge_runs_session_to_completion(tmp_path: Path) -> None:
    """@example: WS bridge handshake starts the orchestrator; bye ends it.

    This is the full vertical: POST creates a pending session, the
    client opens the WS, sends an event, then sends ``{"kind":"bye"}``,
    and the session's log finalizes with session_end.
    """

    app = _make_app(tmp_path)
    with TestClient(app) as client:
        r = client.post(
            "/api/game-agent/sessions",
            json={
                "surface": "js13k",
                "objective": "bridge smoke",
                "semantic_inputs": ["confirm"],
                "log_schema": "js13k.v1",
            },
        )
        assert r.status_code == 200
        session_id = r.json()["session_id"]
        # Dial exactly like the real client: the server-issued
        # bridge_ws_url carries ?token=. An anonymous session has no
        # owner to authenticate against, so the token is the only
        # credential that can open the bridge.
        token = r.json()["bridge_ws_url"].rsplit("token=", 1)[-1]
        bridge_path = (
            f"/api/game-agent/surfaces/js13k/bridge/{session_id}?token={token}"
        )

        with client.websocket_connect(bridge_path) as ws:
            # Push one event frame so the log captures a real event.
            ws.send_text(
                json.dumps(
                    {"kind": "event", "data": {"event": "spawn", "n": 1}}
                )
            )
            time.sleep(0.15)
            # Signal the bridge to bye out -- adapter forwards as a
            # bridge_bye event AND requests orchestrator stop.
            ws.send_text(json.dumps({"kind": "bye"}))
            # Give the server a moment to finalize before the WS
            # context exits and starlette closes the socket.
            time.sleep(0.3)

        lines = _wait_for_log(tmp_path / f"{session_id}.ndjson")
        kinds = [e["kind"] for e in lines]
        assert kinds[-1] == "session_end"
        # Our pushed event arrived through the bridge.
        events = [e for e in lines if e["kind"] == "event"]
        assert any(
            e["payload"]["data"].get("event") == "spawn" for e in events
        )
