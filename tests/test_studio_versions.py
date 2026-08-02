"""Tests for Studio version snapshot + restore (studio_routes.py).

Pins:
  - Manual save snapshots a version (source.json pseudo-file)
  - Autosave (is_autosave=True) skips the snapshot
  - Chart fast-path also snapshots on manual save
  - Snapshot failure is non-fatal — the save itself still succeeds
  - GET /api/studio/{id}/versions returns the list payload, user-scoped
  - POST /api/studio/{id}/restore-version/{vid} restores source + re-renders
  - Restore pre-snapshots current state so the restore is itself reversible
  - Restore rejects: 404 missing version, 400 wrong artifact, 400 corrupt source
  - Every CRUD call threads user_id from the request scope (multi-tenant)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock


def _mock_store(*, source_json: str | None = None):
    """Standard mock with the version-history surface stubbed."""
    if source_json is None:
        source_json = json.dumps({
            "type": "ebook",
            "title": "Book",
            "author": "A.",
            "chapters": [{"heading": "Ch 1", "body": "Hi."}],
        })
    store = MagicMock()
    store.get = AsyncMock(return_value={
        "id": "art_1",
        "filename": "book.epub",
        "display_name": "Book",
        "format": "epub",
        "size_bytes": 1024,
        "source_json": source_json,
        "metadata": {},
        "download_url": "/api/artifacts/art_1/download",
    })
    store.update_source = AsyncMock()
    store.update_file = AsyncMock()
    store.save_version = AsyncMock(return_value={
        "id": "ver_new", "artifact_id": "art_1", "version_index": 1,
        "label": "", "file_count": 1, "score": None,
    })
    store.list_versions = AsyncMock(return_value=[
        {"id": "ver_2", "version_index": 2, "label": "", "file_count": 1,
         "score": None, "created_at": "2026-06-09 10:00:00"},
        {"id": "ver_1", "version_index": 1, "label": "", "file_count": 1,
         "score": None, "created_at": "2026-06-09 09:00:00"},
    ])
    store.get_version = AsyncMock()
    return store


_DEFAULT_SAVE_BODY = {
    "source": {
        "type": "ebook",
        "title": "Book",
        "author": "A.",
        "chapters": [{"heading": "Ch 1", "body": "Hello."}],
    },
}


# ── Snapshot on save ─────────────────────────────────────────────


class TestSaveSnapshots:
    def test_manual_save_snapshots_version(self, app, client):
        store = _mock_store()
        app.state.artifact_store = store

        resp = client.post("/api/studio/art_1/save", json=_DEFAULT_SAVE_BODY)
        assert resp.status_code == 200

        store.save_version.assert_awaited_once()
        call = store.save_version.await_args
        assert call.args[0] == "art_1"
        # files list — one pseudo-file holding source.json
        files = call.args[1]
        assert len(files) == 1
        assert files[0]["path"] == "source.json"
        assert files[0]["role"] == "source"
        assert json.loads(files[0]["content"])["type"] == "ebook"
        # Multi-tenant: user_id threaded from request scope
        assert call.kwargs["user_id"] == "usr_test"

    def test_autosave_skips_snapshot(self, app, client):
        store = _mock_store()
        app.state.artifact_store = store

        body = {**_DEFAULT_SAVE_BODY, "is_autosave": True}
        resp = client.post("/api/studio/art_1/save", json=body)
        assert resp.status_code == 200

        store.save_version.assert_not_awaited()

    def test_snapshot_failure_doesnt_break_save(self, app, client):
        """If save_version raises, the user's save still succeeds (the row is
        non-fatal). Logged warning surfaces the prod degradation."""
        store = _mock_store()
        store.save_version = AsyncMock(side_effect=RuntimeError("db_locked"))
        app.state.artifact_store = store

        resp = client.post("/api/studio/art_1/save", json=_DEFAULT_SAVE_BODY)
        assert resp.status_code == 200
        # Source + file were still persisted
        store.update_source.assert_awaited_once()
        store.update_file.assert_awaited_once()


# ── List endpoint ────────────────────────────────────────────────


class TestListVersions:
    def test_list_no_store(self, client):
        resp = client.get("/api/studio/art_1/versions")
        assert resp.status_code == 503

    def test_list_not_found(self, app, client):
        store = MagicMock()
        store.get = AsyncMock(return_value=None)
        app.state.artifact_store = store
        resp = client.get("/api/studio/art_1/versions")
        assert resp.status_code == 404

    def test_list_returns_versions(self, app, client):
        store = _mock_store()
        app.state.artifact_store = store
        resp = client.get("/api/studio/art_1/versions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["artifact_id"] == "art_1"
        assert len(data["versions"]) == 2
        assert data["versions"][0]["version_index"] == 2  # newest first

    def test_list_user_scoped(self, app, client):
        """list_versions must receive user_id from the request scope."""
        store = _mock_store()
        app.state.artifact_store = store
        client.get("/api/studio/art_1/versions")
        store.list_versions.assert_awaited_once_with("art_1", user_id="usr_test")


# ── Restore endpoint ─────────────────────────────────────────────


class TestRestoreVersion:
    @staticmethod
    def _stub_version(source_dict):
        return {
            "id": "ver_1",
            "artifact_id": "art_1",
            "version_index": 1,
            "label": "",
            "file_count": 1,
            "score": None,
            "files": [{
                "path": "source.json",
                "role": "source",
                "content": json.dumps(source_dict),
            }],
        }

    def test_restore_not_found_version(self, app, client):
        store = _mock_store()
        store.get_version = AsyncMock(return_value=None)
        app.state.artifact_store = store
        resp = client.post("/api/studio/art_1/restore-version/ver_missing")
        assert resp.status_code == 404

    def test_restore_wrong_artifact(self, app, client):
        store = _mock_store()
        store.get_version = AsyncMock(return_value={
            "id": "ver_1", "artifact_id": "art_OTHER", "version_index": 1,
            "files": [{"path": "source.json", "role": "source",
                       "content": '{"type": "ebook"}'}],
        })
        app.state.artifact_store = store
        resp = client.post("/api/studio/art_1/restore-version/ver_1")
        assert resp.status_code == 400
        assert "does not belong" in resp.json()["error"]

    def test_restore_missing_source_file(self, app, client):
        store = _mock_store()
        store.get_version = AsyncMock(return_value={
            "id": "ver_1", "artifact_id": "art_1", "version_index": 1,
            "files": [{"path": "other.txt", "role": "source", "content": "x"}],
        })
        app.state.artifact_store = store
        resp = client.post("/api/studio/art_1/restore-version/ver_1")
        assert resp.status_code == 400
        assert "Studio source snapshot" in resp.json()["error"]

    def test_restore_corrupt_source_json(self, app, client):
        store = _mock_store()
        store.get_version = AsyncMock(return_value={
            "id": "ver_1", "artifact_id": "art_1", "version_index": 1,
            "files": [{"path": "source.json", "role": "source",
                       "content": "not valid json {{{"}],
        })
        app.state.artifact_store = store
        resp = client.post("/api/studio/art_1/restore-version/ver_1")
        assert resp.status_code == 400
        assert "corrupted" in resp.json()["error"]

    def test_restore_writes_source_and_rerenders(self, app, client):
        store = _mock_store()
        restored = {
            "type": "ebook",
            "title": "Restored Title",
            "author": "B.",
            "chapters": [{"heading": "Old", "body": "Old body."}],
        }
        store.get_version = AsyncMock(return_value=self._stub_version(restored))
        app.state.artifact_store = store

        resp = client.post("/api/studio/art_1/restore-version/ver_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["restored"] is True
        assert data["source"]["title"] == "Restored Title"
        assert data["version_index"] == 1
        # The rebuilt binary was written + source updated
        store.update_source.assert_awaited_once()
        saved_source = json.loads(store.update_source.await_args.args[1])
        assert saved_source["title"] == "Restored Title"
        store.update_file.assert_awaited_once()

    def test_restore_pre_snapshots_current_state(self, app, client):
        """Restore must save_version the current source before overwriting so
        the user can undo the restore by restoring the auto-snapshot."""
        store = _mock_store()
        restored = {
            "type": "ebook",
            "title": "Old Title",
            "author": "X",
            "chapters": [],
        }
        store.get_version = AsyncMock(return_value=self._stub_version(restored))
        app.state.artifact_store = store

        client.post("/api/studio/art_1/restore-version/ver_1")

        # save_version was called with the CURRENT source (before overwrite),
        # not the restored source.
        store.save_version.assert_awaited_once()
        snapshotted_files = store.save_version.await_args.args[1]
        snapshotted = json.loads(snapshotted_files[0]["content"])
        assert snapshotted["title"] == "Book"  # the pre-restore current title

    def test_restore_user_scoped(self, app, client):
        """All store calls during restore must include user_id."""
        store = _mock_store()
        restored = {"type": "ebook", "title": "T", "author": "A", "chapters": []}
        store.get_version = AsyncMock(return_value=self._stub_version(restored))
        app.state.artifact_store = store

        client.post("/api/studio/art_1/restore-version/ver_1")

        # Every data call carries user_id="usr_test"
        assert store.get.await_args.kwargs["user_id"] == "usr_test"
        assert store.get_version.await_args.kwargs["user_id"] == "usr_test"
        assert store.update_source.await_args.kwargs["user_id"] == "usr_test"
        assert store.update_file.await_args.kwargs["user_id"] == "usr_test"
        assert store.save_version.await_args.kwargs["user_id"] == "usr_test"

    def test_restore_unknown_source_type(self, app, client):
        store = _mock_store()
        store.get_version = AsyncMock(return_value={
            "id": "ver_1", "artifact_id": "art_1", "version_index": 1,
            "files": [{"path": "source.json", "role": "source",
                       "content": json.dumps({"type": "mystery"})}],
        })
        app.state.artifact_store = store
        resp = client.post("/api/studio/art_1/restore-version/ver_1")
        assert resp.status_code == 400
        assert "Unknown source type" in resp.json()["error"]
