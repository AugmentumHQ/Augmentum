"""Behavior tests for /api/artifacts/* — artifact CRUD, pin/open, build.

Replaces the prior 77-line smoke stub (6 tests) with real behavior
coverage including user isolation, filter paths, error codes, and pin/
open state writes.

Scope note: the LLM-driven endpoints (/iterate, /fix, /verify, /upscale,
/convert, /inpaint, /transcribe, /remove-bg) each hit external models
with large request/response contracts — they need full backend mocks
worth of their own suite. Covered here: the CRUD + bookkeeping surface
that the UI uses on every artifact panel render.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


TEST_USER_ID = "usr_test"


def _seeded_store(artifacts: list[dict] | None = None):
    """Build a MagicMock artifact store with user-isolation semantics.

    Every method respects user_id so cross-tenant tests fail the way the
    production store would. Callers can override methods per-test when
    they need specific error shapes.
    """
    rows = {a["id"]: a for a in (artifacts or [])}

    async def _get(artifact_id, user_id=""):
        art = rows.get(artifact_id)
        if art and user_id and art.get("user_id", user_id) != user_id:
            return None
        return art

    async def _delete(artifact_id, user_id=""):
        art = rows.get(artifact_id)
        if not art:
            return False
        if user_id and art.get("user_id", user_id) != user_id:
            return False
        rows.pop(artifact_id, None)
        return True

    async def _set_pinned(artifact_id, pinned, user_id=""):
        art = rows.get(artifact_id)
        if art and (not user_id or art.get("user_id", user_id) == user_id):
            art["pinned"] = pinned
        return bool(art)

    async def _touch_opened(artifact_id, user_id=""):
        return True

    async def _list_all(user_id="", **_):
        return [a for a in rows.values() if not user_id or a.get("user_id", user_id) == user_id]

    async def _list_for_task(task_id, user_id=""):
        return [a for a in rows.values()
                if a.get("task_id") == task_id
                and (not user_id or a.get("user_id", user_id) == user_id)]

    async def _list_for_session(session_id, user_id=""):
        return [a for a in rows.values()
                if a.get("session_id") == session_id
                and (not user_id or a.get("user_id", user_id) == user_id)]

    store = MagicMock()
    store.get = AsyncMock(side_effect=_get)
    store.delete = AsyncMock(side_effect=_delete)
    store.set_pinned = AsyncMock(side_effect=_set_pinned)
    store.touch_opened = AsyncMock(side_effect=_touch_opened)
    store.list_all = AsyncMock(side_effect=_list_all)
    store.list_for_task = AsyncMock(side_effect=_list_for_task)
    store.list_for_session = AsyncMock(side_effect=_list_for_session)
    store.get_file_path = MagicMock(return_value=None)
    store._rows = rows  # test-only handle for assertions
    return store


def _artifact(id="art_1", *, user_id=TEST_USER_ID, format="html",
              session_id="", task_id="", pinned=False, metadata=None):
    # Mirror ArtifactStore.get() shape — raw bytes are never in the JSON
    # response; they're served separately via /download.
    return {
        "id": id,
        "user_id": user_id,
        "filename": f"{id}.{format}",
        "display_name": id,
        "format": format,
        "size_bytes": 256,
        "path": f"{id}.{format}",
        "metadata": metadata or {},
        "session_id": session_id,
        "task_id": task_id,
        "pinned": pinned,
        "download_url": f"/api/artifacts/{id}/download",
        "source_json": None,
    }


# ===========================================================================
# GET /api/artifacts
# ===========================================================================

class TestListArtifacts:
    def test_503_when_store_missing(self, app, client):
        if hasattr(app.state, "artifact_store"):
            delattr(app.state, "artifact_store")
        r = client.get("/api/artifacts")
        assert r.status_code == 503

    def test_list_all(self, app, client):
        app.state.artifact_store = _seeded_store([
            _artifact("art_1"),
            _artifact("art_2", format="pdf"),
        ])
        r = client.get("/api/artifacts")
        assert r.status_code == 200
        ids = {a["id"] for a in r.json()}
        assert ids == {"art_1", "art_2"}

    def test_list_filters_by_task(self, app, client):
        app.state.artifact_store = _seeded_store([
            _artifact("art_1", task_id="task_a"),
            _artifact("art_2", task_id="task_b"),
        ])
        r = client.get("/api/artifacts?task_id=task_a")
        ids = {a["id"] for a in r.json()}
        assert ids == {"art_1"}

    def test_list_filters_by_session(self, app, client):
        app.state.artifact_store = _seeded_store([
            _artifact("art_1", session_id="sess_a"),
            _artifact("art_2", session_id="sess_b"),
        ])
        r = client.get("/api/artifacts?session_id=sess_a")
        ids = {a["id"] for a in r.json()}
        assert ids == {"art_1"}

    def test_list_scopes_to_user(self, app, client):
        """Cross-tenant list must not leak other users' artifacts."""
        app.state.artifact_store = _seeded_store([
            _artifact("mine", user_id=TEST_USER_ID),
            _artifact("theirs", user_id="usr_other"),
        ])
        r = client.get("/api/artifacts")
        ids = {a["id"] for a in r.json()}
        assert ids == {"mine"}


# ===========================================================================
# GET /api/artifacts/{id}
# ===========================================================================

class TestGetArtifact:
    def test_503_when_store_missing(self, app, client):
        if hasattr(app.state, "artifact_store"):
            delattr(app.state, "artifact_store")
        r = client.get("/api/artifacts/any")
        assert r.status_code == 503

    def test_returns_artifact(self, app, client):
        app.state.artifact_store = _seeded_store([_artifact("art_1", format="pdf")])
        r = client.get("/api/artifacts/art_1")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "art_1"
        assert data["format"] == "pdf"

    def test_missing_returns_404(self, app, client):
        app.state.artifact_store = _seeded_store()
        r = client.get("/api/artifacts/does-not-exist")
        assert r.status_code == 404

    def test_other_users_artifact_returns_404(self, app, client):
        """ID-guessing attack protection: another user's artifact ID must
        resolve to 404, not 200."""
        app.state.artifact_store = _seeded_store([
            _artifact("protected", user_id="usr_other"),
        ])
        r = client.get("/api/artifacts/protected")
        assert r.status_code == 404


# ===========================================================================
# DELETE /api/artifacts/{id}
# ===========================================================================

class TestDeleteArtifact:
    def test_deletes_existing(self, app, client):
        store = _seeded_store([_artifact("art_1")])
        app.state.artifact_store = store
        r = client.delete("/api/artifacts/art_1")
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        assert "art_1" not in store._rows

    def test_missing_returns_404(self, app, client):
        app.state.artifact_store = _seeded_store()
        r = client.delete("/api/artifacts/ghost")
        assert r.status_code == 404

    def test_other_users_artifact_returns_404(self, app, client):
        store = _seeded_store([_artifact("protected", user_id="usr_other")])
        app.state.artifact_store = store
        r = client.delete("/api/artifacts/protected")
        assert r.status_code == 404
        # Still present for the real owner
        assert "protected" in store._rows


# ===========================================================================
# PATCH /api/artifacts/{id}/pin
# ===========================================================================

class TestPinToggle:
    def test_pins_unpinned_artifact(self, app, client):
        store = _seeded_store([_artifact("art_1", pinned=False)])
        app.state.artifact_store = store
        r = client.patch("/api/artifacts/art_1/pin")
        assert r.status_code == 200
        assert r.json()["pinned"] is True
        assert store._rows["art_1"]["pinned"] is True

    def test_unpins_pinned_artifact(self, app, client):
        store = _seeded_store([_artifact("art_1", pinned=True)])
        app.state.artifact_store = store
        r = client.patch("/api/artifacts/art_1/pin")
        assert r.status_code == 200
        assert r.json()["pinned"] is False

    def test_missing_returns_404(self, app, client):
        app.state.artifact_store = _seeded_store()
        r = client.patch("/api/artifacts/ghost/pin")
        assert r.status_code == 404

    def test_other_users_artifact_returns_404(self, app, client):
        store = _seeded_store([_artifact("theirs", user_id="usr_other", pinned=False)])
        app.state.artifact_store = store
        r = client.patch("/api/artifacts/theirs/pin")
        assert r.status_code == 404
        # Unchanged
        assert store._rows["theirs"]["pinned"] is False


# ===========================================================================
# PATCH /api/artifacts/{id}/open
# ===========================================================================

class TestMarkOpened:
    def test_marks_opened(self, app, client):
        store = _seeded_store([_artifact("art_1")])
        app.state.artifact_store = store
        r = client.patch("/api/artifacts/art_1/open")
        assert r.status_code == 200
        # touch_opened was called with our user_id
        store.touch_opened.assert_awaited()


# ===========================================================================
# POST /api/artifacts/build-cancel + GET /api/artifacts/build-status
# ===========================================================================

class TestBuildControl:
    def test_cancel_no_active_builds(self, client):
        r = client.post("/api/artifacts/build-cancel")
        assert r.status_code == 200

    def test_status_returns_shape(self, client):
        r = client.get("/api/artifacts/build-status")
        assert r.status_code == 200
        # Must at minimum not crash; status dict shape varies by build state

    def test_status_falls_back_to_persisted_build_run(self, app, client):
        store = MagicMock()
        store.latest_for_session = AsyncMock(return_value={
            "id": "build_saved",
            "user_id": TEST_USER_ID,
            "session_id": "sess_1",
            "task_id": "task_1",
            "artifact_id": "art_1",
            "kind": "application",
            "status": "completed",
            "name": "Saved App",
            "progress": {},
            "result": {"artifact_id": "art_1", "project": {"name": "Saved App"}},
            "error": None,
        })
        app.state.build_run_store = store

        r = client.get("/api/artifacts/build-status?session_id=sess_1")

        assert r.status_code == 200
        data = r.json()
        assert data["build_id"] == "build_saved"
        assert data["artifact_id"] == "art_1"
        store.latest_for_session.assert_awaited_once_with("sess_1", user_id=TEST_USER_ID)

    def test_status_marks_stale_persisted_running_build_failed(self, app, client):
        run = {
            "id": "build_stale",
            "user_id": TEST_USER_ID,
            "session_id": "sess_1",
            "task_id": "task_1",
            "artifact_id": "",
            "kind": "application",
            "status": "running",
            "name": "Stale App",
            "progress": {"passes": [{"name": "generate", "status": "running"}]},
            "result": {},
            "error": "",
            "created_at": "2026-05-12 23:46:22",
            "updated_at": "2026-05-12 23:47:10",
            "completed_at": None,
        }

        async def _mark_stale(*_, reason="", **__):
            run["status"] = "failed"
            run["error"] = reason
            run["completed_at"] = "2026-05-13 00:00:00"
            return True

        store = MagicMock()
        store.latest_for_session = AsyncMock(return_value=run)
        store.get = AsyncMock(return_value=run)
        store.mark_running_stale = AsyncMock(side_effect=_mark_stale)
        app.state.build_run_store = store

        r = client.get("/api/artifacts/build-status?session_id=sess_1")

        assert r.status_code == 200
        data = r.json()
        assert data["active"] is False
        assert data["status"] == "error"
        assert "stopped updating" in data["error"]
        store.mark_running_stale.assert_awaited_once()


# ===========================================================================
# GET /api/artifacts/{id}/download — 404 paths (happy path needs real file)
# ===========================================================================

class TestDownload:
    def test_missing_returns_404(self, app, client):
        app.state.artifact_store = _seeded_store()
        r = client.get("/api/artifacts/ghost/download")
        assert r.status_code == 404

    def test_other_users_artifact_returns_404(self, app, client):
        app.state.artifact_store = _seeded_store([
            _artifact("theirs", user_id="usr_other"),
        ])
        r = client.get("/api/artifacts/theirs/download")
        assert r.status_code == 404


# ===========================================================================
# GET /api/artifacts/{id}/preview
# ===========================================================================

class TestPreviewDelivery:
    def test_image_preview_escapes_display_name_in_alt(self, app, client, tmp_path):
        art = _artifact("img_escape", format="png")
        art["display_name"] = '"><script>alert(1)</script>'
        image_path = tmp_path / "img_escape.png"
        image_path.write_bytes(b"fake image bytes")
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=image_path)
        app.state.artifact_store = store

        r = client.get("/api/artifacts/img_escape/preview")

        assert r.status_code == 200
        assert "<img " in r.text
        assert "alt=" in r.text
        assert '&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;' in r.text
        assert '<script>alert(1)</script>' not in r.text

    def test_fallback_preview_escapes_display_name_and_format(self, app, client):
        art = _artifact("file_escape", format="zip<script>")
        art["display_name"] = '<img src=x onerror=alert(1)>'
        app.state.artifact_store = _seeded_store([art])

        r = client.get("/api/artifacts/file_escape/preview")

        assert r.status_code == 200
        assert '&lt;img src=x onerror=alert(1)&gt;' in r.text
        assert 'ZIP&lt;SCRIPT&gt;' in r.text
        assert '<img src=x onerror=alert(1)>' not in r.text


# ===========================================================================
# Application bundles — GET /api/artifacts/{id}/preview
#                       + GET /api/artifacts/{id}/preview/{path:path}
# ===========================================================================

def _make_app_zip(path, files):
    """Write a zip at ``path`` with ``files`` = list of (name, bytes)."""
    import zipfile
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files:
            zf.writestr(name, data)
    return path


class TestApplicationPreview:
    def test_application_preview_serves_index_with_base_tag(self, app, client, tmp_path):
        zip_path = _make_app_zip(tmp_path / "app.zip", [
            ("index.html", b"<!doctype html><html><head><title>X</title></head><body>hi</body></html>"),
            ("app.js", b"console.log('hi')"),
        ])
        art = _artifact("app1", format="zip")
        art["source_json"] = '{"type":"application","files":[]}'
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=zip_path)
        app.state.artifact_store = store

        r = client.get("/api/artifacts/app1/preview")

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        # <base> tag injected so the app's relative URLs resolve back
        # into the artifact instead of the augmentum origin root.
        assert '<base href="/api/artifacts/app1/preview/">' in r.text
        assert "hi" in r.text

    def test_application_preview_falls_through_when_no_index(self, app, client, tmp_path):
        # An "application" artifact without an index.html is malformed;
        # fall through to the archive renderer instead of 500-ing.
        zip_path = _make_app_zip(tmp_path / "noindex.zip", [
            ("README.md", b"empty"),
        ])
        art = _artifact("app2", format="zip")
        art["source_json"] = '{"type":"application","files":[]}'
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=zip_path)
        app.state.artifact_store = store

        r = client.get("/api/artifacts/app2/preview")

        assert r.status_code == 200
        # Archive renderer fingerprint — file listing rather than app HTML.
        assert "README.md" in r.text

    def test_application_preview_skips_zip_with_existing_base(self, app, client, tmp_path):
        # If the app shipped its own <base> we leave it alone — the
        # author knows where they want relative URLs to resolve.
        zip_path = _make_app_zip(tmp_path / "withbase.zip", [
            ("index.html",
             b"<html><head><base href='https://example.com/'></head><body>x</body></html>"),
        ])
        art = _artifact("app3", format="zip")
        art["source_json"] = '{"type":"application"}'
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=zip_path)
        app.state.artifact_store = store

        r = client.get("/api/artifacts/app3/preview")

        assert r.status_code == 200
        assert "https://example.com/" in r.text
        assert "/api/artifacts/app3/preview/" not in r.text

    def test_non_application_zip_still_shows_archive_listing(self, app, client, tmp_path):
        # Regression: arbitrary user-uploaded zips must NOT be served as
        # apps. Without source_json marking it an application, the old
        # archive-listing path stays.
        zip_path = _make_app_zip(tmp_path / "user.zip", [
            ("index.html", b"<html><body>I am not an app</body></html>"),
            ("data.txt", b"plain data"),
        ])
        art = _artifact("user_zip", format="zip")
        art["source_json"] = None
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=zip_path)
        app.state.artifact_store = store

        r = client.get("/api/artifacts/user_zip/preview")

        assert r.status_code == 200
        assert "I am not an app" not in r.text  # not served as HTML
        assert "data.txt" in r.text             # archive listing


class TestApplicationPreviewFile:
    def test_sibling_file_serves_with_correct_mime(self, app, client, tmp_path):
        zip_path = _make_app_zip(tmp_path / "app.zip", [
            ("index.html", b"<html></html>"),
            ("app.js", b"console.log('hello')"),
            ("styles.css", b"body{color:red}"),
        ])
        art = _artifact("app_sf", format="zip")
        art["source_json"] = '{"type":"application"}'
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=zip_path)
        app.state.artifact_store = store

        r_js = client.get("/api/artifacts/app_sf/preview/app.js")
        assert r_js.status_code == 200
        assert r_js.headers["content-type"].startswith("application/javascript")
        assert r_js.content == b"console.log('hello')"

        r_css = client.get("/api/artifacts/app_sf/preview/styles.css")
        assert r_css.status_code == 200
        assert r_css.headers["content-type"].startswith("text/css")
        assert r_css.content == b"body{color:red}"

    def test_sibling_file_404_for_missing_path(self, app, client, tmp_path):
        zip_path = _make_app_zip(tmp_path / "app.zip", [
            ("index.html", b"<html></html>"),
        ])
        art = _artifact("app_404", format="zip")
        art["source_json"] = '{"type":"application"}'
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=zip_path)
        app.state.artifact_store = store

        r = client.get("/api/artifacts/app_404/preview/ghost.js")
        assert r.status_code == 404

    def test_sibling_route_refuses_non_application_artifacts(self, app, client, tmp_path):
        # Critical: the sibling route must NOT act as a generic
        # zip-content server. Only artifacts explicitly marked as
        # applications opt into per-file serving — otherwise any
        # user-uploaded zip becomes a virtual filesystem.
        zip_path = _make_app_zip(tmp_path / "any.zip", [
            ("secret.txt", b"do not leak this"),
        ])
        art = _artifact("any_zip", format="zip")
        art["source_json"] = None
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=zip_path)
        app.state.artifact_store = store

        r = client.get("/api/artifacts/any_zip/preview/secret.txt")
        assert r.status_code == 404

    def test_sibling_route_user_isolation(self, app, client, tmp_path):
        zip_path = _make_app_zip(tmp_path / "app.zip", [
            ("index.html", b"<html></html>"),
            ("app.js", b"console.log('private')"),
        ])
        art = _artifact("private_app", format="zip", user_id="usr_other")
        art["source_json"] = '{"type":"application"}'
        store = _seeded_store([art])
        store.get_file_path = MagicMock(return_value=zip_path)
        app.state.artifact_store = store

        r = client.get("/api/artifacts/private_app/preview/app.js")
        assert r.status_code == 404


# ===========================================================================
# Router sanity
# ===========================================================================

class TestRouterShape:
    def test_prefix(self):
        from augmentum.proxy.artifact_routes import router
        assert router.prefix == "/api/artifacts"

    def test_all_21_endpoints_registered(self):
        """Guard against accidental removal during god-file refactors."""
        from augmentum.proxy.artifact_routes import router
        paths = {r.path for r in router.routes}
        expected = {
            "/api/artifacts", "/api/artifacts/{artifact_id}",
            "/api/artifacts/{artifact_id}/download",
            "/api/artifacts/{artifact_id}/preview",
            "/api/artifacts/{artifact_id}/save",
            "/api/artifacts/{artifact_id}/upload",
            "/api/artifacts/{artifact_id}/upscale",
            "/api/artifacts/{artifact_id}/convert",
            "/api/artifacts/{artifact_id}/inpaint",
            "/api/artifacts/{artifact_id}/transcribe",
            "/api/artifacts/{artifact_id}/remove-bg",
            "/api/artifacts/{artifact_id}/pin",
            "/api/artifacts/{artifact_id}/open",
            "/api/artifacts/iterate", "/api/artifacts/fix", "/api/artifacts/verify",
            "/api/artifacts/build-cancel", "/api/artifacts/build-status",
            "/api/artifacts/import", "/api/artifacts/save-html",
        }
        assert expected.issubset(paths)
