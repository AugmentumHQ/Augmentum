"""Route-level tests for the pushed pi session mirror endpoints.

Complements tests/test_pi_run_store.py (store invariants) by exercising the
HTTP layer itself: auth gating (503 without a user), request validation
(400/404), idempotent event retry THROUGH the endpoint, incremental reads
via since_seq, and cross-user isolation at the route boundary.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.proxy.pi_run_routes import router
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager


def _run(coro):
    return asyncio.run(coro)


def _make_app(backend: SQLiteBackend, *, user_id: str | None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.state_manager = StateManager(backend)

    if user_id is not None:
        @app.middleware("http")
        async def _inject_user(request, call_next):
            class _U:
                def __init__(self, uid):
                    self.id = uid
            request.scope["user"] = _U(user_id)
            return await call_next(request)

    return app


@pytest.fixture
def backend(tmp_path):
    be = SQLiteBackend(str(tmp_path / "pi-run-routes.db"))
    _run(be.connect())
    try:
        yield be
    finally:
        _run(be.close())


def _create_run(client, run_id="r1", **overrides):
    body = {
        "run_id": run_id,
        "project": "augmentum",
        "session_file": "C:/sessions/x.jsonl",
        "title": "test session",
        "model": "d/deepseek-v4-flash",
    }
    body.update(overrides)
    return client.post("/api/coder/external/pi/runs", json=body)


def test_all_endpoints_require_user(backend):
    app = _make_app(backend, user_id=None)
    with TestClient(app) as client:
        assert _create_run(client).status_code == 503
        assert client.post("/api/coder/external/pi/runs/r1/events", json={"events": []}).status_code == 503
        assert client.post("/api/coder/external/pi/runs/r1/finish", json={}).status_code == 503
        assert client.get("/api/coder/external/pi/runs").status_code == 503
        assert client.get("/api/coder/external/pi/runs/r1").status_code == 503


def test_create_requires_run_id(backend):
    app = _make_app(backend, user_id="alice")
    with TestClient(app) as client:
        r = _create_run(client, run_id="  ")
        assert r.status_code == 400


def test_roundtrip_and_idempotent_retry_via_http(backend):
    app = _make_app(backend, user_id="alice")
    with TestClient(app) as client:
        assert _create_run(client).status_code == 200

        events = {"events": [
            {"seq": 0, "kind": "message", "text": "hello"},
            {"seq": 1, "kind": "tool_call", "tool": "bash", "text": "ls", "path": ""},
        ]}
        r = client.post("/api/coder/external/pi/runs/r1/events", json=events)
        assert r.status_code == 200
        assert r.json()["inserted"] == 2

        # Network-blip retry of the same batch inserts nothing (idempotent).
        r = client.post("/api/coder/external/pi/runs/r1/events", json=events)
        assert r.status_code == 200
        assert r.json()["inserted"] == 0

        # Empty batch is a cheap no-op, not an error.
        r = client.post("/api/coder/external/pi/runs/r1/events", json={"events": []})
        assert r.status_code == 200
        assert r.json()["inserted"] == 0

        # Incremental read — the SSE poller's contract.
        r = client.get("/api/coder/external/pi/runs/r1", params={"since_seq": 0})
        assert r.status_code == 200
        assert [e["seq"] for e in r.json()["events"]] == [1]

        # Finish → terminal status + files list visible in the listing.
        r = client.post("/api/coder/external/pi/runs/r1/finish", json={
            "status": "done", "outcome": "ok",
            "files_changed": ["a.py"], "num_turns": 3,
        })
        assert r.status_code == 200
        r = client.get("/api/coder/external/pi/runs")
        runs = r.json()["runs"]
        assert runs[0]["id"] == "r1"
        assert runs[0]["status"] == "done"
        assert runs[0]["engine"] == "pi"

        # Re-attach (host resumed the session) flips it back to running.
        assert _create_run(client).status_code == 200
        r = client.get("/api/coder/external/pi/runs/r1")
        assert r.json()["status"] == "running"


def test_get_unknown_run_404(backend):
    app = _make_app(backend, user_id="alice")
    with TestClient(app) as client:
        assert client.get("/api/coder/external/pi/runs/nope").status_code == 404


def test_cross_user_isolation_at_route_boundary(backend):
    alice = _make_app(backend, user_id="alice")
    bob = _make_app(backend, user_id="bob")
    with TestClient(alice) as a:
        assert _create_run(a).status_code == 200
        a.post("/api/coder/external/pi/runs/r1/events", json={
            "events": [{"seq": 0, "kind": "message", "text": "secret"}],
        })
    with TestClient(bob) as b:
        # Bob can't read Alice's run…
        assert b.get("/api/coder/external/pi/runs/r1").status_code == 404
        # …and doesn't see it in his listing.
        assert b.get("/api/coder/external/pi/runs").json()["runs"] == []
        # Bob pushing events at Alice's run_id is rejected (0 inserted):
        # the (run_id, seq) unique index is shared, so an unchecked insert
        # would let him squat seq slots on her run.
        r = b.post("/api/coder/external/pi/runs/r1/events", json={
            "events": [{"seq": 5, "kind": "message", "text": "bob"}],
        })
        assert r.json()["inserted"] == 0
    with TestClient(alice) as a:
        events = a.get("/api/coder/external/pi/runs/r1").json()["events"]
        assert [e["seq"] for e in events] == [0]


def test_project_filter_and_limit_clamp(backend):
    app = _make_app(backend, user_id="alice")
    with TestClient(app) as client:
        _create_run(client, run_id="r1", project="augmentum")
        _create_run(client, run_id="r2", project="other")
        r = client.get("/api/coder/external/pi/runs", params={"project": "other"})
        assert [x["id"] for x in r.json()["runs"]] == ["r2"]
        # limit is clamped server-side, not an error.
        r = client.get("/api/coder/external/pi/runs", params={"limit": 9999})
        assert r.status_code == 200
        assert len(r.json()["runs"]) == 2
