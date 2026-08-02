"""Save-to-Library REST route tests.

Exercises ``augmentum/proxy/library_save_routes.py`` end-to-end via
TestClient. The conftest ``app`` fixture installs a mock SessionManager
that authenticates "Bearer test-token" as ``usr_test``; this file wires
a real SQLite backend on top so the routes can actually persist.

The save / preflight paths normally call into the coder subsystem
(``gather_preview_state`` + ``snapshot_container_path``). Those depend
on a running Docker workspace, so we monkey-patch them at the
library_save_routes module to inject deterministic preview state and
extracted-content paths — same approach the coder route tests take
for container interactions.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TEST_USER_ID = "usr_test"  # conftest.test_user
WORKSPACE_ID = "ws-test-abc"


@pytest.fixture
def library_client(app, tmp_path: Path):
    """Wire a real PublicationStore + LibraryStorage on the app and
    register the workspace + user rows the routes look up.

    Yields ``(client, store, storage_root)`` so tests can both make
    HTTP calls and reach the underlying store for setup / assertions.
    """
    from augmentum.library.publications import LibraryStorage, PublicationStore
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    # SQLiteBackend.connect() runs the migration suite, which creates
    # project_checkouts, library_publications, users, etc. with their
    # production schemas. We just need to insert the workspace rows the
    # routes look up. The auth user comes from the mocked SessionManager
    # in the conftest ``app`` fixture, so no users-table row is needed.
    _run(backend.connect())
    conn = backend._conn

    import time as _time
    now = _time.time()
    _run(conn.execute(
        "INSERT INTO project_checkouts (id, user_id, name, created_at) "
        "VALUES (?, ?, ?, ?)",
        (WORKSPACE_ID, TEST_USER_ID, "Tower Defense WS", now),
    ))
    _run(conn.execute(
        "INSERT INTO project_checkouts (id, user_id, name, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("ws-other", "usr_other", "Other WS", now),
    ))
    _run(conn.commit())

    storage_root = tmp_path / "library_published"
    storage = LibraryStorage(storage_root)
    store = PublicationStore(conn, storage)

    app.state.state_manager = StateManager(backend)
    app.state.publication_store = store

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})

    yield tc, store, storage_root
    _run(backend.close())


def _mk_extracted(tmp_path: Path, *, name: str = "snap") -> Path:
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    (src / "index.html").write_text("<html><body>hi</body></html>")
    (src / "game.js").write_text("console.log('hi')")
    return src


# ── List / get / patch / delete (no coder dependency) ─────────────────


class TestListPublications:
    def test_empty_for_fresh_user(self, library_client):
        client, _, _ = library_client
        r = client.get("/api/library/publications")
        assert r.status_code == 200
        assert r.json() == {"publications": []}

    def test_returns_only_own_publications(self, library_client, tmp_path: Path):
        """Cross-tenant isolation: another user's saves are invisible."""
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        _run(store.create_or_overwrite(
            user_id="usr_other", workspace_id="ws-other",
            title="Theirs", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        mine = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="Mine", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))

        r = client.get("/api/library/publications")
        pubs = r.json()["publications"]
        assert len(pubs) == 1
        assert pubs[0]["id"] == mine["id"]
        assert pubs[0]["title"] == "Mine"


class TestGetPublication:
    def test_get_own(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="A", description="d", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.get(f"/api/library/publications/{row['id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == row["id"]
        assert body["title"] == "A"
        assert body["launch_url"] == f"/api/library/play/{row['id']}"
        # storage_path is server-only; never leaks to client.
        assert "storage_path" not in body

    def test_get_cross_tenant_returns_404(self, library_client, tmp_path: Path):
        """Critical isolation: must never leak existence across users."""
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id="usr_other", workspace_id="ws-other",
            title="Hidden", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.get(f"/api/library/publications/{row['id']}")
        assert r.status_code == 404

    def test_get_missing_returns_404(self, library_client):
        client, _, _ = library_client
        r = client.get("/api/library/publications/pub_does_not_exist")
        assert r.status_code == 404


class TestPatchPublication:
    def test_rename(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="Old", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.patch(f"/api/library/publications/{row['id']}",
                         json={"title": "New", "description": "updated"})
        assert r.status_code == 200
        body = r.json()
        assert body["title"] == "New"
        assert body["description"] == "updated"

    def test_rename_to_collision_409(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        a = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="A", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="B", description="", kind="game",
            source_path=_mk_extracted(tmp_path, name="snap2"),
            entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.patch(f"/api/library/publications/{a['id']}",
                         json={"title": "B"})
        assert r.status_code == 409
        assert r.json()["error"] == "title_collision"


class TestDeletePublication:
    def test_delete_own(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="Doomed", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.delete(f"/api/library/publications/{row['id']}")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

        # Subsequent GET is 404.
        r2 = client.get(f"/api/library/publications/{row['id']}")
        assert r2.status_code == 404

    def test_delete_cross_tenant_404(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id="usr_other", workspace_id="ws-other",
            title="Theirs", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.delete(f"/api/library/publications/{row['id']}")
        assert r.status_code == 404


# ── Assets / play / launch / download ─────────────────────────────────


class TestAssets:
    def test_serves_content_file(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="A", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.get(f"/api/library/publications/{row['id']}/assets/index.html")
        assert r.status_code == 200
        assert "hi" in r.text

    def test_blocks_traversal(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="A", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.get(f"/api/library/publications/{row['id']}/assets/../../etc/passwd")
        # FastAPI's path param normalization may reject this before our
        # handler runs (404 from the router) or after (404 from us).
        # Both are acceptable — what matters is "never 200, never serve".
        assert r.status_code in (404, 422)

    def test_cross_tenant_404(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id="usr_other", workspace_id="ws-other",
            title="Theirs", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.get(f"/api/library/publications/{row['id']}/assets/index.html")
        assert r.status_code == 404


class TestPlay:
    def test_renders_sandboxed_iframe(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="Play Test", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.get(f"/api/library/play/{row['id']}")
        assert r.status_code == 200
        body = r.text
        # Critical security invariant: iframe sandbox MUST NOT include
        # allow-same-origin or the artifact can read top-level Augmentum.
        assert 'sandbox="allow-scripts' in body
        assert "allow-same-origin" not in body
        assert f"/api/library/publications/{row['id']}/assets/index.html" in body


class TestLaunch:
    def test_bumps_counter(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="Counter", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.post(f"/api/library/publications/{row['id']}/launch")
        assert r.status_code == 200
        r2 = client.post(f"/api/library/publications/{row['id']}/launch")
        assert r2.status_code == 200

        refreshed = _run(store.get(row["id"], user_id=TEST_USER_ID))
        assert refreshed["launch_count"] == 2
        assert refreshed["last_launched_at"] is not None


class TestDownload:
    def test_serves_zip(self, library_client, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        row = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="DL", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))
        r = client.get(f"/api/library/publications/{row['id']}/download")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"
        # Real zip files start with the PK\x03\x04 local file header signature.
        assert r.content[:4] == b"PK\x03\x04"


# ── Preflight (with mocked preview state) ─────────────────────────────


class TestPreflight:
    def test_no_preview_when_workspace_idle(self, library_client, monkeypatch):
        """gather_preview_state returns kind='none' when no service runs."""
        client, _, _ = library_client

        from augmentum.library.coder_bridge import PreviewSnapshot

        async def fake_gather(*, request, workspace_id, user_id):
            return PreviewSnapshot(
                preview_kind="none", primary_url=None, served_dir=None,
                container_port=None, host_port=None, service_name=None,
            )

        monkeypatch.setattr(
            "augmentum.proxy.library_save_routes.gather_preview_state",
            fake_gather,
        )
        r = client.post(
            "/api/library/save/preflight",
            json={"workspace_id": WORKSPACE_ID},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["preview_kind"] == "none"
        assert body["preview_ready"] is False

    def test_workspace_not_found_404(self, library_client):
        client, _, _ = library_client
        r = client.post(
            "/api/library/save/preflight",
            json={"workspace_id": "ws-not-mine"},
        )
        assert r.status_code == 404

    def test_title_collision_reported(self, library_client, monkeypatch, tmp_path: Path):
        client, store, _ = library_client
        src = _mk_extracted(tmp_path)
        existing = _run(store.create_or_overwrite(
            user_id=TEST_USER_ID, workspace_id=WORKSPACE_ID,
            title="Dup", description="", kind="game",
            source_path=src, entry_point="index.html",
            max_bytes=10_000_000, user_budget_bytes=10_000_000,
        ))

        from augmentum.library.coder_bridge import PreviewSnapshot

        async def fake_gather(*, request, workspace_id, user_id):
            return PreviewSnapshot(
                preview_kind="static",
                primary_url=f"/api/coder/preview/{workspace_id}/8000/",
                served_dir="/workspace/dist",
                container_port=8000, host_port=54321, service_name="serve",
                file_count=3, estimated_size_bytes=2048,
            )

        monkeypatch.setattr(
            "augmentum.proxy.library_save_routes.gather_preview_state",
            fake_gather,
        )
        r = client.post(
            "/api/library/save/preflight",
            json={"workspace_id": WORKSPACE_ID, "proposed_title": "Dup"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["preview_kind"] == "static"
        assert body["preview_ready"] is True
        assert body["title_collision"] is True
        assert body["existing_publication_id"] == existing["id"]


# ── Save (with mocked container extraction) ───────────────────────────


class TestSave:
    def test_save_static_success(self, library_client, monkeypatch, tmp_path: Path):
        """Happy path: static preview, extracted dir contains index.html,
        save creates a row, response includes launch_url."""
        client, _, _ = library_client

        from augmentum.library.coder_bridge import PreviewSnapshot

        async def fake_gather(*, request, workspace_id, user_id):
            return PreviewSnapshot(
                preview_kind="static",
                primary_url=f"/api/coder/preview/{workspace_id}/8000/",
                served_dir="/workspace/dist",
                container_port=8000, host_port=54321, service_name="serve",
            )

        async def fake_snapshot(*, request, workspace_id, container_path, host_dest_dir):
            return _mk_extracted(host_dest_dir, name="dist")

        monkeypatch.setattr(
            "augmentum.proxy.library_save_routes.gather_preview_state",
            fake_gather,
        )
        monkeypatch.setattr(
            "augmentum.proxy.library_save_routes.snapshot_container_path",
            fake_snapshot,
        )

        r = client.post(
            "/api/library/save",
            json={"workspace_id": WORKSPACE_ID, "title": "Tower MVP"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["title"] == "Tower MVP"
        assert body["action"] == "created"
        assert body["version"] == 1
        assert body["launch_url"].startswith("/api/library/play/")

    def test_save_rejects_dynamic_preview(self, library_client, monkeypatch):
        client, _, _ = library_client

        from augmentum.library.coder_bridge import PreviewSnapshot

        async def fake_gather(*, request, workspace_id, user_id):
            return PreviewSnapshot(
                preview_kind="dynamic", primary_url="/x",
                served_dir="/workspace", container_port=8000,
                host_port=54321, service_name="uvicorn",
            )

        monkeypatch.setattr(
            "augmentum.proxy.library_save_routes.gather_preview_state",
            fake_gather,
        )
        r = client.post(
            "/api/library/save",
            json={"workspace_id": WORKSPACE_ID, "title": "Would Break"},
        )
        assert r.status_code == 409
        assert r.json()["error"] == "dynamic_preview"

    def test_save_overwrite_bumps_version(self, library_client, monkeypatch, tmp_path: Path):
        client, _, _ = library_client

        from augmentum.library.coder_bridge import PreviewSnapshot

        async def fake_gather(*, request, workspace_id, user_id):
            return PreviewSnapshot(
                preview_kind="static",
                primary_url=f"/api/coder/preview/{workspace_id}/8000/",
                served_dir="/workspace/dist",
                container_port=8000, host_port=54321, service_name="serve",
            )

        async def fake_snapshot(*, request, workspace_id, container_path, host_dest_dir):
            return _mk_extracted(host_dest_dir, name="dist")

        monkeypatch.setattr(
            "augmentum.proxy.library_save_routes.gather_preview_state",
            fake_gather,
        )
        monkeypatch.setattr(
            "augmentum.proxy.library_save_routes.snapshot_container_path",
            fake_snapshot,
        )

        # First save.
        r1 = client.post("/api/library/save",
                         json={"workspace_id": WORKSPACE_ID, "title": "Game"})
        assert r1.status_code == 200
        first_id = r1.json()["publication_id"]

        # Same title without on_collision → 409.
        r2 = client.post("/api/library/save",
                         json={"workspace_id": WORKSPACE_ID, "title": "Game"})
        assert r2.status_code == 409
        assert r2.json()["error"] == "title_collision"

        # With on_collision=overwrite → 200, same id, version=2.
        r3 = client.post("/api/library/save",
                         json={"workspace_id": WORKSPACE_ID, "title": "Game",
                               "on_collision": "overwrite"})
        assert r3.status_code == 200, r3.text
        body = r3.json()
        assert body["publication_id"] == first_id
        assert body["version"] == 2
        assert body["action"] == "overwritten"

    def test_save_workspace_ownership_check(self, library_client, monkeypatch):
        client, _, _ = library_client
        # Save against another user's workspace must 404 — not 403,
        # not 200. Don't leak existence.
        r = client.post(
            "/api/library/save",
            json={"workspace_id": "ws-other", "title": "Sneaky"},
        )
        assert r.status_code == 404
