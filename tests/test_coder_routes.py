from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.classifier.router import Mode
from augmentum.coder.models import ContainerInfo
from augmentum.proxy.coder_routes import router
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager


def _run(coro):
    return asyncio.run(coro)


def _workspace(
    id: str = "ws123",
    name: str = "test",
    status: str = "running",
    container_id: str = "abc",
) -> ContainerInfo:
    return ContainerInfo(
        id=id,
        name=name,
        status=status,
        container_id=container_id,
        created_at=1.0,
    )


async def _seed_user(conn, user_id: str, username: str) -> None:
    await conn.execute(
        "INSERT INTO users (id, username, display_name, password_hash, role) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, username, username.title(), "pw", "user"),
    )
    await conn.commit()


async def _seed_workspace(
    conn,
    workspace_id: str,
    *,
    user_id: str | None,
    name: str = "test",
) -> None:
    await conn.execute(
        "INSERT INTO project_checkouts (id, name, status, created_at, user_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (workspace_id, name, "running", time.time(), user_id),
    )
    await conn.commit()


async def _fetch_one(conn, query: str, params: tuple):
    cursor = await conn.execute(query, params)
    return await cursor.fetchone()


def _make_manager(backend: SQLiteBackend):
    mgr = AsyncMock()
    mgr.list_workspaces = AsyncMock(return_value=[])

    async def _create_workspace(**kwargs):
        info = _workspace()
        info.tooling_profile = kwargs.get("tooling_profile") or "standard"
        await _seed_workspace(
            backend.conn,
            info.id,
            user_id=kwargs.get("user_id"),
            name=info.name,
        )
        return info

    mgr.create_workspace = AsyncMock(side_effect=_create_workspace)
    mgr.start = AsyncMock(return_value=_workspace(status="running"))
    mgr.stop = AsyncMock(return_value=_workspace(status="stopped"))
    mgr.delete = AsyncMock()
    mgr.enable_published_ports = AsyncMock(return_value=(_workspace(status="running"), True))
    mgr.list_ports = AsyncMock(return_value=[])
    mgr.file_list = AsyncMock(return_value=[])
    mgr.file_read = AsyncMock(return_value="file content")
    mgr.file_write = AsyncMock()
    return mgr


def _make_app(backend: SQLiteBackend, manager, *, user_id: str | None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.container_manager = manager
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
    be = SQLiteBackend(str(tmp_path / "coder-routes.db"))
    _run(be.connect())
    _run(_seed_user(be.conn, "alice", "alice"))
    _run(_seed_user(be.conn, "bob", "bob"))
    try:
        yield be
    finally:
        _run(be.close())


def test_coder_mode_exists():
    assert hasattr(Mode, "CODER")
    assert Mode.CODER.value == "coder"


def test_handler_factory_returns_coder():
    from augmentum.modes.coder.handler import CoderHandler
    from augmentum.proxy.handler_factory import get_handler_for_mode

    backend = AsyncMock()
    app_state = type("State", (), {})()
    app_state.tool_registry = None
    app_state.state_manager = None
    app_state.permission_registry = None
    app_state.review_registry = None
    app_state.container_manager = None
    handler = get_handler_for_mode(
        Mode.CODER,
        backend,
        "test-session",
        app_state,
    )
    assert isinstance(handler, CoderHandler)


def test_list_workspaces_requires_auth(backend):
    app = _make_app(backend, _make_manager(backend), user_id=None)
    with TestClient(app) as client:
        resp = client.get("/api/coder/workspaces")
    assert resp.status_code == 401


def test_list_workspaces_filters_to_owned_ids(backend):
    manager = _make_manager(backend)
    manager.list_workspaces = AsyncMock(return_value=[
        _workspace(id="ws-alice", name="alice"),
        _workspace(id="ws-bob", name="bob"),
    ])
    _run(_seed_workspace(backend.conn, "ws-alice", user_id="alice", name="alice"))
    _run(_seed_workspace(backend.conn, "ws-bob", user_id="bob", name="bob"))

    app = _make_app(backend, manager, user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/workspaces")
    assert resp.status_code == 200
    body = resp.json()
    assert [w["id"] for w in body["workspaces"]] == ["ws-alice"]


def test_create_workspace_stamps_owner(backend):
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.post("/api/coder/workspaces", json={"name": "test"})
    assert resp.status_code == 201
    row = _run(_fetch_one(
        backend.conn,
        "SELECT user_id FROM project_checkouts WHERE id = ?",
        ("ws123",),
    ))
    assert row[0] == "alice"


def test_create_workspace_forwards_tooling_profile(backend):
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/workspaces",
            json={"name": "test", "tooling_profile": "power"},
        )

    assert resp.status_code == 201
    assert manager.create_workspace.await_args.kwargs["base_image"] == "augmentum-workspace"
    assert manager.create_workspace.await_args.kwargs["tooling_profile"] == "power"


def test_create_workspace_rejects_unknown_tooling_profile(backend):
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/workspaces",
            json={"name": "test", "tooling_profile": "kitchen-sink"},
        )

    assert resp.status_code == 422
    manager.create_workspace.assert_not_awaited()


def test_read_file_requires_workspace_ownership(backend):
    app = _make_app(backend, _make_manager(backend), user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/files/ws-missing/read?path=/workspace/test.py")
    assert resp.status_code == 404


def test_read_file_for_owned_workspace(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.get("/api/coder/files/ws123/read?path=/workspace/test.py")
    assert resp.status_code == 200
    assert resp.json()["content"] == "file content"


def test_read_file_rejects_path_traversal(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.get("/api/coder/files/ws123/read?path=/workspace/../etc/passwd")
    assert resp.status_code == 400
    manager.file_read.assert_not_awaited()


def test_read_file_rejects_path_outside_workspace(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.get("/api/coder/files/ws123/read?path=/etc/passwd")
    assert resp.status_code == 400
    manager.file_read.assert_not_awaited()


def test_write_file_rejects_path_traversal(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.put(
            "/api/coder/files/ws123/write",
            json={"path": "/workspace/../etc/cron.d/x", "content": "*/1 * * * * root id"},
        )
    assert resp.status_code == 400
    manager.file_write.assert_not_awaited()


def test_write_file_rejects_path_outside_workspace(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.put(
            "/api/coder/files/ws123/write",
            json={"path": "/etc/passwd", "content": "x"},
        )
    assert resp.status_code == 400
    manager.file_write.assert_not_awaited()


def test_publish_workspace_ports_requires_workspace_ownership(backend):
    app = _make_app(backend, _make_manager(backend), user_id="alice")
    with TestClient(app) as client:
        resp = client.post("/api/coder/workspaces/ws-missing/ports/publish")
    assert resp.status_code == 404


def test_publish_workspace_ports_for_owned_workspace(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.post("/api/coder/workspaces/ws123/ports/publish")
    assert resp.status_code == 200
    body = resp.json()
    assert body["changed"] is True
    assert body["workspace"]["id"] == "ws123"
    manager.enable_published_ports.assert_awaited_once_with("ws123")


def test_workspace_ports_reports_unpublished_preview_state(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)
    manager.list_ports = AsyncMock(return_value=[
        {"container_port": 3000, "host_port": 0, "listening": False},
        {"container_port": 8080, "host_port": 0, "listening": False},
    ])
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.get("/api/coder/workspaces/ws123/ports")
    assert resp.status_code == 200
    body = resp.json()
    assert body["preview"]["state"] == "not_published"
    assert body["preview"]["published"] is False
    assert body["preview"]["ready"] is False
    assert body["preview"]["primary_url"] is None


def test_workspace_ports_reports_ready_preview_urls(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)
    manager.list_ports = AsyncMock(return_value=[
        {"container_port": 3000, "host_port": 45123, "listening": True},
        {"container_port": 5173, "host_port": 0, "listening": False},
    ])
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.get("/api/coder/workspaces/ws123/ports")
    assert resp.status_code == 200
    body = resp.json()
    assert body["preview"]["state"] == "ready"
    assert body["preview"]["published"] is True
    assert body["preview"]["ready"] is True
    assert body["preview"]["ready_count"] == 1
    # Preview URLs are now same-origin proxy paths so the iframe loads
    # under Augmentum's CSP and is reachable from any device that can
    # reach Augmentum (phone-on-LAN), not just the Docker host.
    assert body["preview"]["primary_url"] == "/api/coder/preview/ws123/3000/"
    assert body["preview"]["urls"] == ["/api/coder/preview/ws123/3000/"]


def test_save_conversation_upserts_even_after_prior_writes(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")

    with TestClient(app) as client:
        resp = client.put(
            "/api/coder/conversation/ws123",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200

        saved = client.get("/api/coder/conversation/ws123")
        assert saved.status_code == 200
        assert saved.json()["messages"] == [{"role": "user", "content": "hello"}]

    row = _run(_fetch_one(
        backend.conn,
        "SELECT conversation FROM coder_sessions WHERE session_id = ? AND user_id = ?",
        ("ws123", "alice"),
    ))
    assert row is not None


def test_compact_conversation_replaces_middle_and_saves(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")
    messages = [{"role": "user", "content": "start"}]
    for i in range(18):
        messages.append({
            "role": "assistant",
            "content": f"assistant {i} " + ("alpha beta gamma " * 80),
        })
        messages.append({
            "role": "user",
            "content": f"user {i} " + ("delta epsilon zeta " * 80),
        })

    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/conversation/ws123/compact",
            json={"messages": messages, "keep_recent": 6, "force": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["compacted"] is True
        assert body["tokens"]["tokens_after"] < body["tokens"]["tokens_before"]
        assert body["messages"][0]["content"] == "start"
        assert "<compacted" in body["messages"][1]["content"]

        saved = client.get("/api/coder/conversation/ws123")
        assert saved.status_code == 200
        saved_messages = saved.json()["messages"]
        assert saved_messages == body["messages"]


def test_git_token_routes_require_auth(backend):
    app = _make_app(backend, _make_manager(backend), user_id=None)
    with TestClient(app) as client:
        resp = client.post("/api/coder/git-tokens", json={
            "host": "github.com",
            "token": "tok",
            "username": "oauth2",
        })
    assert resp.status_code == 401


def test_git_tokens_are_user_scoped(backend):
    alice_app = _make_app(backend, _make_manager(backend), user_id="alice")
    bob_app = _make_app(backend, _make_manager(backend), user_id="bob")

    with TestClient(alice_app) as client:
        resp = client.post("/api/coder/git-tokens", json={
            "host": "github.com",
            "token": "alice-token",
            "username": "oauth2",
        })
        assert resp.status_code == 200

    with TestClient(bob_app) as client:
        resp = client.get("/api/coder/git-tokens")
        assert resp.status_code == 200
        assert resp.json()["tokens"] == []


def test_git_credential_proxy_resolves_workspace_owner_token(backend, monkeypatch):
    _run(_seed_workspace(backend.conn, "ws-alice", user_id="alice"))
    alice_app = _make_app(backend, _make_manager(backend), user_id="alice")

    with TestClient(alice_app) as client:
        resp = client.post("/api/coder/git-tokens", json={
            "host": "github.com",
            "token": "alice-token",
            "username": "oauth2",
        })
        assert resp.status_code == 200

    monkeypatch.setattr(
        "augmentum.proxy.coder_routes._is_docker_internal",
        lambda _ip: True,
    )
    proxy_app = _make_app(backend, _make_manager(backend), user_id=None)
    with TestClient(proxy_app) as client:
        resp = client.get(
            "/api/coder/git-credential",
            params={"host": "github.com", "workspace_id": "ws-alice"},
        )
    assert resp.status_code == 200
    assert resp.text == "alice-token"


def test_workspace_profile_routes_roundtrip(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")

    with TestClient(app) as client:
        put = client.put(
            "/api/coder/workspaces/ws123/profile",
            json={"entries": [{
                "category": "command",
                "key": "test_command",
                "value": "pytest",
                "confidence": 0.9,
            }]},
        )
        assert put.status_code == 200
        got = client.get("/api/coder/workspaces/ws123/profile")

    assert got.status_code == 200
    body = got.json()
    assert body["entries"][0]["category"] == "command"
    assert body["entries"][0]["key"] == "test_command"
    assert "command.test_command" in body["rendered"]


def test_workspace_service_routes_start_logs_stop(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)

    async def _run_command(_workspace_id, cmd, timeout=30.0, **_kwargs):
        joined = " ".join(cmd)
        if "nohup" in joined:
            return "4321\n"
        if "tail -n" in joined:
            return "server ready\n"
        if "kill 4321" in joined:
            return "stopped\n"
        if "kill -0 4321" in joined:
            return "running\n"
        return ""

    manager.run_command = AsyncMock(side_effect=_run_command)
    manager.file_read = AsyncMock(return_value="[]")
    manager.file_write = AsyncMock()
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        started = client.post(
            "/api/coder/workspaces/ws123/services",
            json={"name": "web", "command": "npm run dev", "ports": [5173]},
        )
        assert started.status_code == 201
        service_id = started.json()["service"]["id"]

        listed = client.get("/api/coder/workspaces/ws123/services")
        logs = client.get(f"/api/coder/workspaces/ws123/services/{service_id}/logs")
        stopped = client.delete(f"/api/coder/workspaces/ws123/services/{service_id}")

    assert listed.status_code == 200
    assert listed.json()["services"][0]["status"] == "running"
    assert logs.status_code == 200
    assert "server ready" in logs.json()["logs"]
    assert stopped.status_code == 200
    assert stopped.json()["service"]["status"] == "stopped"


def test_workspace_service_user_toggle_stop_keeps_row(backend):
    # Soft-stop route is the one bound to the user-controlled toggle:
    # it kills the process but leaves the row + config intact so the
    # /start companion route can revive it without remembering the
    # command. Sibling DELETE route deletes the row entirely.
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)

    async def _run_command(_workspace_id, cmd, timeout=30.0, **_kwargs):
        joined = " ".join(cmd)
        if "nohup" in joined:
            return "9001\n"
        if "kill 9001" in joined:
            return "stopped\n"
        if "kill -0 9001" in joined:
            return "running\n"
        return ""

    manager.run_command = AsyncMock(side_effect=_run_command)
    manager.file_read = AsyncMock(return_value="[]")
    manager.file_write = AsyncMock()
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        started = client.post(
            "/api/coder/workspaces/ws123/services",
            json={"name": "api", "command": "uvicorn app:app", "ports": [8000]},
        )
        sid = started.json()["service"]["id"]
        soft = client.post(f"/api/coder/workspaces/ws123/services/{sid}/stop")
        listed = client.get("/api/coder/workspaces/ws123/services")

    assert soft.status_code == 200
    assert soft.json()["service"]["status"] == "stopped"
    # Row is still present after a soft stop — distinguishes it from
    # DELETE which removes the row.
    assert any(s["id"] == sid for s in listed.json()["services"])


def test_workspace_service_user_toggle_restart_reuses_id(backend):
    # The restart route must hit the same service_id, not mint a new
    # one. This is the load-bearing contract for the user-toggle UX:
    # without it, toggling off/on would produce phantom rows.
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    manager = _make_manager(backend)

    pids = iter(["4321\n", "5555\n"])

    async def _run_command(_workspace_id, cmd, timeout=30.0, **_kwargs):
        joined = " ".join(cmd)
        if "nohup" in joined:
            return next(pids)
        if "kill 4321" in joined:
            return "stopped\n"
        if "kill -0 4321" in joined:
            return "stopped\n"  # confirms first pid is gone
        if "kill -0 5555" in joined:
            return "running\n"
        return ""

    manager.run_command = AsyncMock(side_effect=_run_command)
    manager.file_read = AsyncMock(return_value="[]")
    manager.file_write = AsyncMock()
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        started = client.post(
            "/api/coder/workspaces/ws123/services",
            json={"name": "web", "command": "npm run dev", "ports": [3000]},
        )
        original_id = started.json()["service"]["id"]
        client.post(f"/api/coder/workspaces/ws123/services/{original_id}/stop")
        restarted = client.post(f"/api/coder/workspaces/ws123/services/{original_id}/start")

    assert restarted.status_code == 200
    svc = restarted.json()["service"]
    assert svc["id"] == original_id  # same handle
    assert svc["status"] == "running"
    assert svc["pid"] == 5555  # new process




def test_coder_run_routes_are_user_scoped(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))

    async def _seed_run_and_event():
        await backend.conn.execute(
            """
            INSERT INTO coder_turn_runs
                (id, user_id, project_id, session_id, strategy, model,
                 status, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("ctr_test", "alice", "ws123", "sess", "hybrid", "model",
             "completed", time.time(), time.time()),
        )
        await backend.conn.execute(
            """
            INSERT INTO coder_turn_events
                (run_id, seq, timestamp, type, phase, status, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ctr_test", 1, time.time(), "complete", "executing", "complete", "{}"),
        )
        await backend.conn.commit()

    _run(_seed_run_and_event())

    alice_app = _make_app(backend, _make_manager(backend), user_id="alice")
    bob_app = _make_app(backend, _make_manager(backend), user_id="bob")

    with TestClient(alice_app) as client:
        run = client.get("/api/coder/runs/ctr_test")
        events = client.get("/api/coder/runs/ctr_test/events")
    with TestClient(bob_app) as client:
        bob_run = client.get("/api/coder/runs/ctr_test")

    assert run.status_code == 200
    assert run.json()["run"]["id"] == "ctr_test"
    assert events.status_code == 200
    assert events.json()["events"][0]["status"] == "complete"
    assert bob_run.status_code == 404


# ---------------------------------------------------------------------------
# Background-run reattach routes — /active-run, /cancel, /stream
# ---------------------------------------------------------------------------


async def _seed_run(conn, run_id: str, *, user_id: str, workspace_id: str,
                    status: str = "running") -> None:
    await conn.execute(
        """
        INSERT INTO coder_turn_runs
            (id, user_id, project_id, session_id, strategy, model,
             status, started_at, updated_at)
        VALUES (?, ?, ?, ?, '', '', ?, ?, ?)
        """,
        (run_id, user_id, workspace_id, workspace_id, status, time.time(), time.time()),
    )
    await conn.commit()


async def _seed_finished_run(conn, run_id: str, *, user_id: str,
                             model: str, oracle: dict | None) -> None:
    import json as _json

    metrics = {"oracle": oracle} if oracle is not None else {}
    await conn.execute(
        """
        INSERT INTO coder_turn_runs
            (id, user_id, project_id, session_id, strategy, model,
             status, started_at, updated_at, metrics_json)
        VALUES (?, ?, 'ws123', 'ws123', '', ?, 'completed', ?, ?, ?)
        """,
        (run_id, user_id, model, time.time(), time.time(), _json.dumps(metrics)),
    )
    await conn.commit()


def test_oracle_stats_route(backend):
    _run(_seed_finished_run(
        backend.conn, "ctr_v1", user_id="alice", model="m1",
        oracle={"wrote": True, "oracle_calls": 1, "kinds": ["test"],
                "verified_after_last_write": True, "last_outcome": "green",
                "no_oracle_done": False},
    ))
    _run(_seed_finished_run(
        backend.conn, "ctr_v2", user_id="alice", model="m1",
        oracle={"wrote": True, "oracle_calls": 0, "kinds": [],
                "verified_after_last_write": False, "last_outcome": "",
                "no_oracle_done": True},
    ))
    _run(_seed_finished_run(
        backend.conn, "ctr_old", user_id="alice", model="m1", oracle=None,
    ))
    app = _make_app(backend, _make_manager(backend), user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/oracle-stats")
    assert resp.status_code == 200
    stats = resp.json()["stats"]
    assert stats["runs"] == 2
    assert stats["runs_without_telemetry"] == 1
    assert stats["write_runs"] == 2
    assert stats["no_oracle_done"] == 1
    assert stats["no_oracle_done_rate"] == 0.5
    assert stats["per_model"]["m1"]["runs"] == 2


def test_oracle_stats_requires_auth(backend):
    app = _make_app(backend, _make_manager(backend), user_id=None)
    with TestClient(app) as client:
        resp = client.get("/api/coder/oracle-stats")
    assert resp.status_code == 401


def test_oracle_stats_is_user_scoped(backend):
    _run(_seed_finished_run(
        backend.conn, "ctr_a", user_id="alice", model="m1",
        oracle={"wrote": True, "oracle_calls": 0, "kinds": [],
                "verified_after_last_write": False, "last_outcome": "",
                "no_oracle_done": True},
    ))
    bob_app = _make_app(backend, _make_manager(backend), user_id="bob")
    with TestClient(bob_app) as client:
        resp = client.get("/api/coder/oracle-stats")
    assert resp.status_code == 200
    stats = resp.json()["stats"]
    assert stats["runs"] == 0 and stats["write_runs"] == 0


def test_active_run_returns_null_when_idle(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/workspaces/ws123/active-run")
    assert resp.status_code == 200
    assert resp.json()["run_id"] is None


def test_active_run_falls_back_to_ledger_when_broker_missing(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    _run(_seed_run(backend.conn, "ctr_active", user_id="alice", workspace_id="ws123"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/workspaces/ws123/active-run")
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == "ctr_active"
    assert body["source"] == "ledger"


def test_active_run_is_user_scoped(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    _run(_seed_run(backend.conn, "ctr_alice", user_id="alice", workspace_id="ws123"))
    bob_app = _make_app(backend, _make_manager(backend), user_id="bob")
    with TestClient(bob_app) as client:
        # Bob doesn't own ws123 — ownership gate returns 404 before we
        # ever hit the ledger.
        resp = client.get("/api/coder/workspaces/ws123/active-run")
    assert resp.status_code == 404


def test_cancel_route_invokes_broker(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    _run(_seed_run(backend.conn, "ctr_cancel", user_id="alice", workspace_id="ws123"))

    cancelled: list[tuple[str, str]] = []

    class _FakeBroker:
        def cancel(self, run_id: str, *, reason: str = "user_cancel") -> bool:
            cancelled.append((run_id, reason))
            return True

    app = _make_app(backend, _make_manager(backend), user_id="alice")
    app.state.coder_run_broker = _FakeBroker()
    with TestClient(app) as client:
        # No body → default reason "user_cancel".
        resp = client.post("/api/coder/runs/ctr_cancel/cancel")
    assert resp.status_code == 200
    assert resp.json() == {
        "cancelled": True, "run_id": "ctr_cancel", "reason": "user_cancel",
    }
    assert cancelled == [("ctr_cancel", "user_cancel")]


def test_cancel_route_propagates_reason_from_body(backend):
    """UI sends {"reason": "slash_clear"} so the next turn's prior_turns
    block can show why the run ended — not just that it did."""
    _run(_seed_workspace(backend.conn, "ws-reason", user_id="alice"))
    _run(_seed_run(backend.conn, "ctr_reason", user_id="alice", workspace_id="ws-reason"))

    cancelled: list[tuple[str, str]] = []

    class _FakeBroker:
        def cancel(self, run_id: str, *, reason: str = "user_cancel") -> bool:
            cancelled.append((run_id, reason))
            return True

    app = _make_app(backend, _make_manager(backend), user_id="alice")
    app.state.coder_run_broker = _FakeBroker()
    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/runs/ctr_reason/cancel",
            json={"reason": "slash_clear"},
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["reason"] == "slash_clear"
    assert cancelled == [("ctr_reason", "slash_clear")]


def test_cancel_route_404_for_other_users_run(backend):
    _run(_seed_workspace(backend.conn, "ws-bob", user_id="bob"))
    _run(_seed_run(backend.conn, "ctr_bob", user_id="bob", workspace_id="ws-bob"))
    alice_app = _make_app(backend, _make_manager(backend), user_id="alice")
    with TestClient(alice_app) as client:
        resp = client.post("/api/coder/runs/ctr_bob/cancel")
    assert resp.status_code == 404


def test_stream_route_replays_ledger_when_broker_gone(backend):
    """The dead-broker case: run finished while the client was away.
    The /stream endpoint replays ledger events + emits a final_state
    chunk carrying the saved assistant message."""
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    _run(_seed_run(backend.conn, "ctr_done", user_id="alice", workspace_id="ws123",
                   status="completed"))

    async def _seed_replay():
        await backend.conn.execute(
            """
            INSERT INTO coder_turn_events
                (run_id, seq, timestamp, type, phase, status, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ctr_done", 1, time.time(), "strategy", "planning", "strategy",
             '{"strategy":"hybrid"}'),
        )
        await backend.conn.execute(
            """
            INSERT INTO coder_turn_events
                (run_id, seq, timestamp, type, phase, status, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ctr_done", 2, time.time(), "tool_call", "executing", "tool_call",
             '{"tool_call":{"id":"t1","tool":"file_read","input":{"path":"x"}}}'),
        )
        await backend.conn.execute(
            """
            INSERT INTO coder_sessions
                (session_id, workspace_id, phase, conversation, user_id,
                 created_at, updated_at)
            VALUES (?, ?, 'waiting', ?, ?, ?, ?)
            """,
            ("ws123", "ws123",
             '[{"role":"user","content":"hi"},'
             ' {"role":"assistant","content":"done!"}]',
             "alice", time.time(), time.time()),
        )
        await backend.conn.commit()

    _run(_seed_replay())

    app = _make_app(backend, _make_manager(backend), user_id="alice")
    # No broker on app.state → forces replay path.
    with TestClient(app) as client:
        resp = client.get("/api/coder/runs/ctr_done/stream")
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.strip().split("\n") if ln]
    payloads = [__import__("json").loads(ln) for ln in lines]

    # Replay chunks carry replay=True; the final synthetic chunk has
    # done=True + the assistant message.
    assert any(p["augmentum"].get("status") == "strategy" for p in payloads)
    assert any(p["augmentum"].get("status") == "tool_call" for p in payloads)
    final = payloads[-1]
    assert final["done"] is True
    assert final["augmentum"].get("final_state") is True
    assert final["message"]["content"] == "done!"


def test_stream_route_falls_back_to_replay_when_entry_evicted_mid_stream(backend):
    """Race: the route snapshots broker.get(run_id) before streaming;
    the sweeper can evict the entry before the first subscribe
    iteration. The live path must then fall through to ledger replay
    instead of returning an empty 200."""
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    _run(_seed_run(backend.conn, "ctr_race", user_id="alice", workspace_id="ws123",
                   status="completed"))

    async def _seed_events():
        await backend.conn.execute(
            """
            INSERT INTO coder_turn_events
                (run_id, seq, timestamp, type, phase, status, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("ctr_race", 1, time.time(), "strategy", "planning", "strategy",
             '{"strategy":"hybrid"}'),
        )
        await backend.conn.commit()

    _run(_seed_events())

    class _EvictingBroker:
        """get() returns a live-looking entry for the route's snapshot,
        then None — simulating the sweeper evicting between the check
        and the first stream iteration. subscribe() yields nothing,
        matching the real broker's behavior for an unknown run_id."""

        def __init__(self):
            self._gets = 0

        def get(self, run_id):
            self._gets += 1
            return object() if self._gets == 1 else None

        async def subscribe(self, run_id, *, since_seq=0):
            return
            yield  # pragma: no cover — makes this an async generator

    app = _make_app(backend, _make_manager(backend), user_id="alice")
    app.state.coder_run_broker = _EvictingBroker()
    with TestClient(app) as client:
        resp = client.get("/api/coder/runs/ctr_race/stream")
    assert resp.status_code == 200
    lines = [ln for ln in resp.text.strip().split("\n") if ln]
    assert lines, "eviction race must not produce an empty stream"
    payloads = [__import__("json").loads(ln) for ln in lines]
    assert any(p["augmentum"].get("status") == "strategy" for p in payloads)
    assert payloads[-1]["done"] is True
    assert payloads[-1]["augmentum"].get("final_state") is True


def test_stream_route_skips_replayed_events_below_since(backend):
    _run(_seed_workspace(backend.conn, "ws123", user_id="alice"))
    _run(_seed_run(backend.conn, "ctr_since", user_id="alice", workspace_id="ws123",
                   status="completed"))

    async def _seed_events():
        for seq in (1, 2, 3):
            await backend.conn.execute(
                """
                INSERT INTO coder_turn_events
                    (run_id, seq, timestamp, type, phase, status, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("ctr_since", seq, time.time(), "x", "p", f"s{seq}", "{}"),
            )
        await backend.conn.commit()

    _run(_seed_events())

    app = _make_app(backend, _make_manager(backend), user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/runs/ctr_since/stream?since=2")
    lines = [ln for ln in resp.text.strip().split("\n") if ln]
    statuses = [
        __import__("json").loads(ln)["augmentum"].get("status") for ln in lines
    ]
    # seq 1 and 2 skipped, seq 3 replayed, then final_state.
    assert "s1" not in statuses
    assert "s2" not in statuses
    assert "s3" in statuses


# ----------------------------------------------------------------------
# Rename + Pause routes (manager surface additions)
# ----------------------------------------------------------------------

def test_rename_workspace_updates_db(backend):
    _run(_seed_workspace(backend.conn, "ws-rn", user_id="alice", name="old-name"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")

    with TestClient(app) as client:
        resp = client.put(
            "/api/coder/workspaces/ws-rn/name",
            json={"name": "new-name"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"workspace_id": "ws-rn", "name": "new-name"}

    row = _run(_fetch_one(
        backend.conn,
        "SELECT name FROM project_checkouts WHERE id = ?",
        ("ws-rn",),
    ))
    assert row[0] == "new-name"


def test_rename_workspace_strips_whitespace(backend):
    _run(_seed_workspace(backend.conn, "ws-rn", user_id="alice", name="old"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")

    with TestClient(app) as client:
        resp = client.put(
            "/api/coder/workspaces/ws-rn/name",
            json={"name": "  padded  "},
        )
    assert resp.status_code == 200
    assert resp.json()["name"] == "padded"


def test_rename_workspace_rejects_empty(backend):
    _run(_seed_workspace(backend.conn, "ws-rn", user_id="alice"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")

    with TestClient(app) as client:
        # Pure-whitespace input → 422 (validator strips then rejects).
        resp = client.put(
            "/api/coder/workspaces/ws-rn/name",
            json={"name": "   "},
        )
    assert resp.status_code == 422


def test_rename_workspace_rejects_too_long(backend):
    _run(_seed_workspace(backend.conn, "ws-rn", user_id="alice"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")

    with TestClient(app) as client:
        resp = client.put(
            "/api/coder/workspaces/ws-rn/name",
            json={"name": "x" * 200},
        )
    assert resp.status_code == 422


def test_rename_workspace_requires_ownership(backend):
    _run(_seed_workspace(backend.conn, "ws-bob", user_id="bob", name="bob's"))
    app = _make_app(backend, _make_manager(backend), user_id="alice")

    with TestClient(app) as client:
        resp = client.put(
            "/api/coder/workspaces/ws-bob/name",
            json={"name": "stolen"},
        )
    assert resp.status_code == 404
    # Bob's row unchanged.
    row = _run(_fetch_one(
        backend.conn,
        "SELECT name FROM project_checkouts WHERE id = ?",
        ("ws-bob",),
    ))
    assert row[0] == "bob's"


def test_pause_workspace_route_invokes_manager(backend):
    _run(_seed_workspace(backend.conn, "ws-pp", user_id="alice"))
    manager = _make_manager(backend)
    manager.pause = AsyncMock(return_value=_workspace(id="ws-pp", status="paused"))
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.post("/api/coder/workspaces/ws-pp/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"
    manager.pause.assert_awaited_once_with("ws-pp")


def test_pause_workspace_route_requires_ownership(backend):
    _run(_seed_workspace(backend.conn, "ws-bob", user_id="bob"))
    manager = _make_manager(backend)
    manager.pause = AsyncMock()
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.post("/api/coder/workspaces/ws-bob/pause")
    assert resp.status_code == 404
    manager.pause.assert_not_awaited()


# ----------------------------------------------------------------------
# Import workspace (counterpart to /export)
# ----------------------------------------------------------------------

def _gzipped_workspace_tar(files: dict[str, bytes]) -> bytes:
    import gzip
    import io
    import tarfile
    import time as _time

    raw = io.BytesIO()
    now = int(_time.time())
    with tarfile.open(fileobj=raw, mode="w") as tar:
        d = tarfile.TarInfo(name="workspace")
        d.type = tarfile.DIRTYPE
        d.mode = 0o755
        d.mtime = now
        tar.addfile(d)
        for rel, data in files.items():
            ti = tarfile.TarInfo(name=f"workspace/{rel}")
            ti.size = len(data)
            ti.mtime = now
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(data))
    return gzip.compress(raw.getvalue())


def test_import_workspace_creates_and_extracts(backend):
    manager = _make_manager(backend)
    manager.import_archive_into = AsyncMock()
    app = _make_app(backend, manager, user_id="alice")

    archive = _gzipped_workspace_tar({"hello.txt": b"world"})
    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/workspaces/import",
            data={"name": "imported-ws", "tooling_profile": "browser"},
            files={"archive": ("ws.tar.gz", archive, "application/gzip")},
        )
    assert resp.status_code == 201
    # Create was called with the user-supplied name + profile + uid.
    create_call = manager.create_workspace.await_args
    assert create_call.kwargs["name"] == "imported-ws"
    assert create_call.kwargs["tooling_profile"] == "browser"
    assert create_call.kwargs["user_id"] == "alice"
    # Then extraction was invoked with the raw archive bytes.
    extract_call = manager.import_archive_into.await_args
    assert extract_call.args[1] == archive


def test_import_workspace_rejects_empty_archive(backend):
    manager = _make_manager(backend)
    manager.import_archive_into = AsyncMock()
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/workspaces/import",
            data={"name": "empty"},
            files={"archive": ("empty.tar.gz", b"", "application/gzip")},
        )
    assert resp.status_code == 400
    manager.create_workspace.assert_not_awaited()
    manager.import_archive_into.assert_not_awaited()


def test_import_workspace_cleans_up_on_extract_failure(backend):
    # Bad archive bytes — create_workspace succeeds but import_archive_into
    # raises ValueError. The route must delete the half-created workspace
    # so the user doesn't end up with an empty skeleton row.
    manager = _make_manager(backend)
    manager.import_archive_into = AsyncMock(
        side_effect=ValueError("Invalid gzip archive"),
    )
    app = _make_app(backend, manager, user_id="alice")

    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/workspaces/import",
            data={"name": "bad"},
            files={"archive": ("bad.tar.gz", b"not gzip at all", "application/gzip")},
        )
    assert resp.status_code == 400
    assert "Invalid gzip" in resp.json()["error"]
    # Cleanup ran with keep_volume=False (purge everything).
    delete_call = manager.delete.await_args
    assert delete_call.kwargs.get("keep_volume") is False


def test_import_workspace_requires_auth(backend):
    app = _make_app(backend, _make_manager(backend), user_id=None)
    with TestClient(app) as client:
        resp = client.post(
            "/api/coder/workspaces/import",
            data={"name": "x"},
            files={"archive": ("x.tar.gz", b"data", "application/gzip")},
        )
    assert resp.status_code == 401
