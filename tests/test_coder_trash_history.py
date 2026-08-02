"""Sprint 2 backend tests: soft-delete/trash, restore, and commit-history.

Two layers:
  * ContainerManager.file_trash / file_restore — the reversible-delete
    logic (script building + manifest parsing + conflict guards), tested
    with a mocked _run_command so no container is needed.
  * Route layer — the delete/restore/trash/checkpoint-show endpoints,
    tested with a mocked manager (same harness style as
    test_coder_routes / test_coder_text_search).
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from augmentum.coder.containers import ContainerManager
from augmentum.proxy.coder_routes import router
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager


def _run(coro):
    return asyncio.run(coro)


def _bare_manager(run_output="") -> ContainerManager:
    """A ContainerManager with __init__ bypassed and _run_command mocked."""
    mgr = ContainerManager.__new__(ContainerManager)
    mgr._run_command = AsyncMock(return_value=run_output)
    return mgr


# ---------------------------------------------------------------------------
# Container-level: file_trash
# ---------------------------------------------------------------------------


def test_file_trash_returns_id_and_builds_move_script():
    mgr = _bare_manager("")
    trash_id = _run(mgr.file_trash("ws1", "/workspace/junk.py"))
    assert trash_id and len(trash_id) == 16
    script = mgr._run_command.call_args[0][1][2]  # bash -c <script>
    # Moves the source into the trash dir and excludes trash from git.
    assert "/workspace/.augmentum/trash/" in script
    # shlex.quote leaves a plain path unquoted; assert the mv still targets it.
    assert "mv -- /workspace/junk.py" in script
    assert ".git/info/exclude" in script


# ---------------------------------------------------------------------------
# Container-level: file_restore
# ---------------------------------------------------------------------------


def test_file_restore_rejects_bad_id():
    mgr = _bare_manager("")
    res = _run(mgr.file_restore("ws1", "../etc"))
    assert res["restored"] is False
    assert "invalid" in res["reason"]


def test_file_restore_missing_manifest():
    mgr = _bare_manager("")   # cat returns empty
    res = _run(mgr.file_restore("ws1", "abc123def456"))
    assert res["restored"] is False
    assert "not found" in res["reason"]


def test_file_restore_ok():
    manifest = json.dumps({
        "trash_id": "abc123def456", "original": "/workspace/a.py",
        "name": "a.py", "deleted_at": 1,
    })
    mgr = ContainerManager.__new__(ContainerManager)
    # First call (cat manifest) returns the JSON; second (the mv script)
    # returns the __OK__ sentinel.
    mgr._run_command = AsyncMock(side_effect=[manifest, "__OK__\n"])
    res = _run(mgr.file_restore("ws1", "abc123def456"))
    assert res == {"restored": True, "path": "/workspace/a.py"}


def test_file_restore_conflict_when_occupied():
    manifest = json.dumps({
        "trash_id": "abc123def456", "original": "/workspace/a.py",
        "name": "a.py", "deleted_at": 1,
    })
    mgr = ContainerManager.__new__(ContainerManager)
    mgr._run_command = AsyncMock(side_effect=[manifest, "__OCCUPIED__\n"])
    res = _run(mgr.file_restore("ws1", "abc123def456"))
    assert res["restored"] is False
    assert "already exists" in res["reason"]


# ---------------------------------------------------------------------------
# Container-level: file_list_trash
# ---------------------------------------------------------------------------


def test_file_list_trash_parses_and_sorts_newest_first():
    m1 = json.dumps({"trash_id": "a" * 12, "original": "/workspace/x", "name": "x", "deleted_at": 10})
    m2 = json.dumps({"trash_id": "b" * 12, "original": "/workspace/y", "name": "y", "deleted_at": 20})
    mgr = _bare_manager(f"{m1}\n{m2}\n")
    items = _run(mgr.file_list_trash("ws1"))
    assert [i["trash_id"] for i in items] == ["b" * 12, "a" * 12]  # newest first


# ---------------------------------------------------------------------------
# Route-level
# ---------------------------------------------------------------------------


async def _seed_user(conn, user_id, username):
    await conn.execute(
        "INSERT INTO users (id, username, display_name, password_hash, role) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, username, username.title(), "pw", "user"),
    )
    await conn.commit()


async def _seed_workspace(conn, workspace_id, *, user_id):
    await conn.execute(
        "INSERT INTO project_checkouts (id, name, status, created_at, user_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (workspace_id, "test", "running", time.time(), user_id),
    )
    await conn.commit()


def _make_app(backend, manager, *, user_id):
    app = FastAPI()
    app.include_router(router)
    app.state.container_manager = manager
    app.state.state_manager = StateManager(backend)

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
    be = SQLiteBackend(str(tmp_path / "trash-routes.db"))
    _run(be.connect())
    _run(_seed_user(be.conn, "alice", "alice"))
    _run(_seed_workspace(be.conn, "ws1", user_id="alice"))
    try:
        yield be
    finally:
        _run(be.close())


def test_delete_route_soft_deletes_by_default(backend):
    mgr = AsyncMock()
    mgr.file_trash = AsyncMock(return_value="deadbeefcafe0001")
    mgr.file_delete = AsyncMock()
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.request(
            "DELETE", "/api/coder/files/ws1",
            json={"path": "/workspace/a.py"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["trashed"] is True
    assert data["trash_id"] == "deadbeefcafe0001"
    mgr.file_trash.assert_called_once()
    mgr.file_delete.assert_not_called()


def test_delete_route_permanent_hard_deletes(backend):
    mgr = AsyncMock()
    mgr.file_trash = AsyncMock()
    mgr.file_delete = AsyncMock()
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.request(
            "DELETE", "/api/coder/files/ws1",
            json={"path": "/workspace/a.py", "permanent": True},
        )
    assert resp.status_code == 200
    assert resp.json()["trashed"] is False
    mgr.file_delete.assert_called_once()
    mgr.file_trash.assert_not_called()


def test_restore_route_conflict_returns_409(backend):
    mgr = AsyncMock()
    mgr.file_restore = AsyncMock(return_value={"restored": False, "reason": "a file already exists at the original path"})
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.post("/api/coder/files/ws1/restore", json={"trash_id": "abc123def456"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["reason"]


def test_restore_route_ok(backend):
    mgr = AsyncMock()
    mgr.file_restore = AsyncMock(return_value={"restored": True, "path": "/workspace/a.py"})
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.post("/api/coder/files/ws1/restore", json={"trash_id": "abc123def456"})
    assert resp.status_code == 200
    assert resp.json()["path"] == "/workspace/a.py"


def test_checkpoint_show_rejects_bad_hash(backend):
    mgr = AsyncMock()
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/checkpoints/ws1/show", params={"hash": "not-a-hash!"})
    assert resp.status_code == 400


def test_checkpoint_show_splits_meta_and_diff(backend):
    show_out = (
        "abc1234567890000000000000000000000000000\n"
        "Alice\n"
        "1700000000\n"
        "Fix the thing\n"
        "\n"
        "diff --git a/x.py b/x.py\n"
        "+added line\n"
    )
    mgr = AsyncMock()
    mgr.git_show = AsyncMock(return_value=show_out)
    app = _make_app(backend, mgr, user_id="alice")
    with TestClient(app) as client:
        resp = client.get("/api/coder/checkpoints/ws1/show", params={"hash": "abc1234"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["meta"]["author"] == "Alice"
    assert data["meta"]["subject"] == "Fix the thing"
    assert data["meta"]["timestamp"] == 1700000000
    assert "diff --git a/x.py b/x.py" in data["diff"]
    assert "+added line" in data["diff"]
    # Header lines must NOT leak into the diff body.
    assert "Fix the thing" not in data["diff"]
