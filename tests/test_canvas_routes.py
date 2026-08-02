"""Tests for the Session Canvas binding routes (canvas_routes.py).

Pins:
  - GET resolves an explicit pin first, falls back to the session's latest
    artifact, and returns artifact_id=None when the session has none
  - A stale pin (artifact deleted) self-heals: the binding is cleared and
    GET falls back to the latest artifact
  - PUT validates the artifact exists + belongs to the user before pinning
  - PUT requires an artifact_id (400 otherwise)
  - Every store call threads user_id from the request scope (multi-tenant)
  - No artifact_store wired -> 503
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock


def _artifact(artifact_id: str = "art_1") -> dict:
    return {
        "id": artifact_id,
        "filename": "app.html",
        "display_name": "My App",
        "format": "html",
    }


def _app_artifact(artifact_id: str = "art_app") -> dict:
    """An editable application bundle (source_json type:application + files)."""
    return {
        "id": artifact_id,
        "filename": "index.html",
        "display_name": "My App",
        "format": "zip",
        "source_json": json.dumps({
            "type": "application",
            "files": [
                {"path": "index.html", "role": "entry",
                 "content": "<html><body>hi</body></html>"},
            ],
        }),
    }


def _store(*, pin=None, get_result="__artifact__", session_list=None, versions=None):
    """Mock ArtifactStore with the canvas-binding surface stubbed."""
    store = MagicMock()
    store.get_canvas_binding = AsyncMock(return_value=pin)
    store.set_canvas_binding = AsyncMock()
    store.clear_canvas_binding = AsyncMock()
    resolved_get = _artifact() if get_result == "__artifact__" else get_result
    store.get = AsyncMock(return_value=resolved_get)
    store.list_for_session = AsyncMock(return_value=session_list or [])
    store.list_versions = AsyncMock(return_value=versions if versions is not None else [{}])
    store.save_version = AsyncMock(return_value={"version_index": 1})
    store.update_file = AsyncMock(return_value=True)
    store.update_source = AsyncMock(return_value=True)
    store.get_version = AsyncMock(return_value=None)
    return store


def _builder(*, patches: int = 2, response: str = "patch"):
    """Mock application builder exposing the quick-edit surface."""
    b = MagicMock()
    b._max_tokens = 8192
    b._call_llm = AsyncMock(return_value=response)
    b._apply_file_patches = MagicMock(return_value=patches)
    return b


def _wire_builder(app, builder):
    registry = MagicMock()
    registry.get = MagicMock(return_value=builder)
    app.state.tool_registry = registry


# ── GET resolve ──────────────────────────────────────────────────


class TestGetCanvas:
    def test_no_store(self, client):
        resp = client.get("/api/canvas/sess_1")
        assert resp.status_code == 503

    def test_empty_session(self, app, client):
        app.state.artifact_store = _store(pin=None, session_list=[])
        resp = client.get("/api/canvas/sess_1")
        assert resp.status_code == 200
        assert resp.json() == {"artifact_id": None}

    def test_falls_back_to_latest(self, app, client):
        store = _store(pin=None, session_list=[_artifact("art_latest")], versions=[{}, {}])
        app.state.artifact_store = store
        resp = client.get("/api/canvas/sess_1")
        data = resp.json()
        assert data["artifact_id"] == "art_latest"
        assert data["pinned"] is False
        assert data["version_count"] == 2
        assert data["preview_url"] == "/api/artifacts/art_latest/preview"

    def test_uses_explicit_pin(self, app, client):
        store = _store(pin="art_pinned", get_result=_artifact("art_pinned"))
        app.state.artifact_store = store
        resp = client.get("/api/canvas/sess_1")
        data = resp.json()
        assert data["artifact_id"] == "art_pinned"
        assert data["pinned"] is True
        store.list_for_session.assert_not_awaited()  # pin short-circuits the fallback

    def test_stale_pin_self_heals(self, app, client):
        # Pin points at a deleted artifact -> get() returns None -> clear + fall back.
        store = _store(pin="gone", get_result=None, session_list=[_artifact("art_latest")])
        app.state.artifact_store = store
        resp = client.get("/api/canvas/sess_1")
        assert resp.json()["artifact_id"] == "art_latest"
        store.clear_canvas_binding.assert_awaited_once()

    def test_user_scoped(self, app, client):
        store = _store(pin=None, session_list=[])
        app.state.artifact_store = store
        client.get("/api/canvas/sess_1")
        store.get_canvas_binding.assert_awaited_once_with("sess_1", user_id="usr_test")


# ── PUT pin ──────────────────────────────────────────────────────


class TestSetCanvas:
    def test_pins_artifact(self, app, client):
        store = _store(get_result=_artifact("art_1"))
        app.state.artifact_store = store
        resp = client.put("/api/canvas/sess_1", json={"artifact_id": "art_1"})
        assert resp.status_code == 200
        assert resp.json()["pinned"] is True
        store.set_canvas_binding.assert_awaited_once_with("sess_1", "art_1", user_id="usr_test")

    def test_requires_artifact_id(self, app, client):
        app.state.artifact_store = _store()
        resp = client.put("/api/canvas/sess_1", json={})
        assert resp.status_code == 400

    def test_unknown_artifact_404(self, app, client):
        store = _store(get_result=None)
        app.state.artifact_store = store
        resp = client.put("/api/canvas/sess_1", json={"artifact_id": "nope"})
        assert resp.status_code == 404
        store.set_canvas_binding.assert_not_awaited()


# ── POST /{id}/edit — in-place quick edit ────────────────────────


class TestEditCanvas:
    def test_requires_description(self, app, client):
        app.state.artifact_store = _store()
        resp = client.post("/api/canvas/sess_1/edit", json={})
        assert resp.status_code == 400

    def test_no_artifact_404(self, app, client):
        app.state.artifact_store = _store(pin=None, session_list=[])
        _wire_builder(app, _builder())
        resp = client.post("/api/canvas/sess_1/edit", json={"description": "make it blue"})
        assert resp.status_code == 404

    def test_non_app_artifact_400(self, app, client):
        # Plain HTML artifact has no app source_json -> not editable.
        app.state.artifact_store = _store(pin="art_1", get_result=_artifact("art_1"))
        _wire_builder(app, _builder())
        resp = client.post("/api/canvas/sess_1/edit", json={"description": "x"})
        assert resp.status_code == 400

    def test_builder_unavailable_503(self, app, client):
        app.state.artifact_store = _store(pin="art_app", get_result=_app_artifact())
        app.state.tool_registry = None
        resp = client.post("/api/canvas/sess_1/edit", json={"description": "x"})
        assert resp.status_code == 503

    def test_no_patches_422(self, app, client):
        app.state.artifact_store = _store(pin="art_app", get_result=_app_artifact())
        _wire_builder(app, _builder(patches=0))
        resp = client.post("/api/canvas/sess_1/edit", json={"description": "x"})
        assert resp.status_code == 422

    def test_applies_and_persists(self, app, client):
        store = _store(pin="art_app", get_result=_app_artifact(), versions=[{}])
        app.state.artifact_store = store
        _wire_builder(app, _builder(patches=3))
        resp = client.post("/api/canvas/sess_1/edit", json={"description": "make it blue"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["artifact_id"] == "art_app"
        assert data["patches_applied"] == 3
        assert data["editable"] is True
        assert "changed_files" in data
        # The bundle + source are rewritten so the live preview reflects it.
        store.update_file.assert_awaited_once()
        store.update_source.assert_awaited_once()
        store.save_version.assert_awaited()  # new version snapshot

    def test_seeds_original_version(self, app, client):
        # No prior versions -> seed the pre-edit state as "Original" first.
        store = _store(pin="art_app", get_result=_app_artifact(), versions=[])
        app.state.artifact_store = store
        _wire_builder(app, _builder())
        resp = client.post("/api/canvas/sess_1/edit", json={"description": "tweak"})
        assert resp.status_code == 200
        assert store.save_version.await_count == 2
        assert store.save_version.await_args_list[0].kwargs.get("label") == "Original"

    def test_user_scoped(self, app, client):
        store = _store(pin="art_app", get_result=_app_artifact(), versions=[{}])
        app.state.artifact_store = store
        _wire_builder(app, _builder())
        client.post("/api/canvas/sess_1/edit", json={"description": "x"})
        # The write threads user_id from the request scope.
        expected_files = json.loads(_app_artifact()["source_json"])["files"]
        store.save_version.assert_awaited_with(
            "art_app", expected_files, user_id="usr_test", label="x",
        )


# ── Version stepper surface ──────────────────────────────────────


class TestVersions:
    def test_summary_includes_versions(self, app, client):
        store = _store(
            pin="art_app", get_result=_app_artifact(),
            versions=[
                {"id": "v2", "version_index": 2, "label": "blue"},
                {"id": "v1", "version_index": 1, "label": "Original"},
            ],
        )
        app.state.artifact_store = store
        data = client.get("/api/canvas/sess_1").json()
        assert data["version_count"] == 2
        assert [v["id"] for v in data["versions"]] == ["v2", "v1"]

    def test_version_preview_renders(self, app, client):
        store = _store(pin="art_app", get_result=_app_artifact())
        store.get_version = AsyncMock(return_value={
            "artifact_id": "art_app",
            "files": [{"path": "index.html", "role": "entry",
                       "content": "<html><body>old</body></html>"}],
        })
        app.state.artifact_store = store
        resp = client.get("/api/canvas/sess_1/version/ver_1/preview")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"]

    def test_version_preview_wrong_artifact_404(self, app, client):
        store = _store(pin="art_app", get_result=_app_artifact())
        store.get_version = AsyncMock(return_value={
            "artifact_id": "someone_elses", "files": [{"path": "i.html", "content": "x"}],
        })
        app.state.artifact_store = store
        resp = client.get("/api/canvas/sess_1/version/ver_1/preview")
        assert resp.status_code == 404

    def test_version_preview_no_canvas_artifact_404(self, app, client):
        app.state.artifact_store = _store(pin=None, session_list=[])
        resp = client.get("/api/canvas/sess_1/version/ver_1/preview")
        assert resp.status_code == 404
