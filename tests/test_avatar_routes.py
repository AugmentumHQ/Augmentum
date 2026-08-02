"""Tests for proxy/avatar_routes.py — avatar API endpoints."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager


@pytest.fixture
def sqlite_client(app):
    """Client with real SQLite for avatar routes."""
    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)
    tc = TestClient(app)
    # Conftest's `app` fixture installs a mock session manager that accepts
    # any bearer token. Add the header so routes see the admin test_user.
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc
    asyncio.get_event_loop().run_until_complete(backend.close())


class TestAvatarList:
    def test_list_avatars(self, sqlite_client):
        resp = sqlite_client.get("/api/avatar/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "avatars" in data
        assert isinstance(data["avatars"], list)

    def test_list_bundled(self, sqlite_client):
        resp = sqlite_client.get("/api/avatar/bundled")
        assert resp.status_code == 200
        data = resp.json()
        assert "avatars" in data


class TestRemovedEndpoints:
    def test_generation_status_removed(self, sqlite_client):
        resp = sqlite_client.get("/api/avatar/generation-status")
        assert resp.status_code in (404, 405)

    def test_generate_removed(self, sqlite_client):
        resp = sqlite_client.post("/api/avatar/generate", json={"image_id": "x"})
        assert resp.status_code in (404, 405)

    def test_lipsync_capabilities_removed(self, sqlite_client):
        resp = sqlite_client.get("/api/avatar/lipsync-capabilities")
        assert resp.status_code in (404, 405)

    def test_segment_removed(self, sqlite_client):
        resp = sqlite_client.post("/api/avatar/avt_test/segment")
        assert resp.status_code in (404, 405)


class TestAvatarHelpers:
    def test_safe_avatar_id_valid(self):
        from augmentum.proxy.avatar_routes import _safe_avatar_id
        assert _safe_avatar_id("avt_123") == "avt_123"

    def test_safe_avatar_id_path_traversal(self):
        from augmentum.proxy.avatar_routes import _safe_avatar_id
        assert _safe_avatar_id("../etc/passwd") is None
        assert _safe_avatar_id("foo/bar") is None
        assert _safe_avatar_id("foo\\bar") is None

    def test_avatar_to_response(self):
        from augmentum.proxy.avatar_routes import _avatar_to_response
        resp = _avatar_to_response({
            "id": "avt_test",
            "type": "vrm",
            "character_id": "ch_1",
            "mannerisms": "{}",
        })
        assert resp["id"] == "avt_test"
        assert resp["vrm_url"] == "/api/avatar/avt_test.vrm"
        assert resp["thumbnail_url"] == "/api/avatar/avt_test/thumbnail"
        assert isinstance(resp["mannerisms"], dict)
        assert "name" in resp

    def test_avatar_to_response_bundled_has_name(self):
        from augmentum.proxy.avatar_routes import _avatar_to_response
        resp = _avatar_to_response({
            "id": "bundled_f_gentle",
            "type": "vrm",
            "is_bundled": 1,
            "mannerisms": "{}",
        })
        assert resp["name"] == "Gentle"
        assert resp["is_bundled"] is True

    def test_avatar_to_response_portrait_type(self):
        from augmentum.proxy.avatar_routes import _avatar_to_response
        resp = _avatar_to_response({
            "id": "avt_portrait",
            "type": "portrait",
        })
        assert resp["vrm_url"] is None
        assert resp["portrait_url"] == "/api/avatar/avt_portrait/portrait"

    def test_avatar_to_response_bad_mannerisms(self):
        from augmentum.proxy.avatar_routes import _avatar_to_response
        resp = _avatar_to_response({
            "id": "avt_bad",
            "type": "vrm",
            "mannerisms": "not json",
        })
        assert resp["mannerisms"] == {}


class TestAvatarCRUD:
    def test_get_nonexistent(self, sqlite_client):
        resp = sqlite_client.get("/api/avatar/avt_nonexistent/meta")
        assert resp.status_code in (404, 500)

    def test_delete_nonexistent(self, sqlite_client):
        resp = sqlite_client.delete("/api/avatar/avt_nonexistent")
        assert resp.status_code in (404, 500)
