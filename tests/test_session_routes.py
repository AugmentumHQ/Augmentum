"""Tests for session_routes.py — session management API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# GET /api/sessions/
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_no_sqlite_returns_empty(self, client):
        """With MemoryBackend (no SQLite), returns empty list."""
        resp = client.get("/api/sessions/")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert data["sessions"] == []


# ---------------------------------------------------------------------------
# GET /api/sessions/{id}/export
# ---------------------------------------------------------------------------


class TestExportSession:
    def test_export_no_state_manager_returns_503(self, client):
        original = client.app.state.state_manager
        client.app.state.state_manager = None
        try:
            resp = client.get("/api/sessions/test-session-id/export")
            assert resp.status_code == 503
        finally:
            client.app.state.state_manager = original

    def test_export_missing_session_returns_404(self, sqlite_client):
        resp = sqlite_client.get("/api/sessions/nonexistent-id/export")
        assert resp.status_code == 404

    def test_export_response_shape(self, sqlite_client):
        """Even for missing sessions, the error response is well-formed JSON."""
        resp = sqlite_client.get("/api/sessions/some-id/export")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data


# ---------------------------------------------------------------------------
# Chat session routes (/api/chats/) - defined in chat_routes.py
# ---------------------------------------------------------------------------


class TestChatSessionList:
    def test_list_chats_no_sqlite_returns_empty(self, client):
        resp = client.get("/api/chats/")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data

    def test_list_chats_meta_only(self, client):
        resp = client.get("/api/chats/?meta=1")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data

    def test_get_chat_no_sqlite_returns_503(self, client):
        resp = client.get("/api/chats/some-session-id")
        assert resp.status_code == 503
