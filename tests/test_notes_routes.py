"""Tests for notes_routes.py — browse notes CRUD."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def _mock_notes_store():
    store = MagicMock()
    store.list_stubs = AsyncMock(return_value=[
        {"id": "n1", "title": "Test Note", "created_at": "2024-01-01T00:00:00Z"},
    ])
    store.get = AsyncMock(return_value={
        "id": "n1", "title": "Test Note", "content": "# Hello",
        "tags": ["test"], "source_url": "", "source_title": "",
        "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z",
    })
    store.create = AsyncMock()
    store.update = AsyncMock(return_value={
        "id": "n1", "title": "Updated Note", "content": "# Updated",
    })
    store.delete = AsyncMock(return_value=True)
    return store


class TestListNotes:
    def test_list_no_store(self, client):
        resp = client.get("/api/browse/notes")
        assert resp.status_code == 503

    def test_list_success(self, app, client):
        app.state.notes_store = _mock_notes_store()
        resp = client.get("/api/browse/notes")
        assert resp.status_code == 200
        assert len(resp.json()["notes"]) == 1


class TestGetNote:
    def test_get_no_store(self, client):
        resp = client.get("/api/browse/notes/n1")
        assert resp.status_code == 503

    def test_get_not_found(self, app, client):
        store = _mock_notes_store()
        store.get = AsyncMock(return_value=None)
        app.state.notes_store = store
        resp = client.get("/api/browse/notes/nonexistent")
        assert resp.status_code == 404

    def test_get_success(self, app, client):
        app.state.notes_store = _mock_notes_store()
        resp = client.get("/api/browse/notes/n1")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Test Note"


class TestCreateNote:
    def test_create_no_store(self, client):
        resp = client.post(
            "/api/browse/notes",
            json={"title": "New Note", "content": "Hello"},
        )
        assert resp.status_code == 503

    def test_create_success(self, app, client, monkeypatch):
        app.state.notes_store = _mock_notes_store()
        # Mock reputation update to avoid side effects
        monkeypatch.setattr(
            "augmentum.proxy.notes_routes._update_reputation",
            AsyncMock(),
        )
        resp = client.post(
            "/api/browse/notes",
            json={"title": "New Note", "content": "Hello world"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["title"] == "New Note"

    def test_create_invalid_json(self, app, client):
        app.state.notes_store = _mock_notes_store()
        resp = client.post(
            "/api/browse/notes",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400


class TestUpdateNote:
    def test_update_no_store(self, client):
        resp = client.put("/api/browse/notes/n1", json={"title": "Updated"})
        assert resp.status_code == 503

    def test_update_not_found(self, app, client):
        store = _mock_notes_store()
        store.update = AsyncMock(return_value=None)
        app.state.notes_store = store
        resp = client.put(
            "/api/browse/notes/nonexistent",
            json={"title": "Updated"},
        )
        assert resp.status_code == 404

    def test_update_success(self, app, client):
        app.state.notes_store = _mock_notes_store()
        resp = client.put(
            "/api/browse/notes/n1",
            json={"title": "Updated Note"},
        )
        assert resp.status_code == 200


class TestDeleteNote:
    def test_delete_no_store(self, client):
        resp = client.delete("/api/browse/notes/n1")
        assert resp.status_code == 503

    def test_delete_not_found(self, app, client):
        store = _mock_notes_store()
        store.delete = AsyncMock(return_value=False)
        app.state.notes_store = store
        resp = client.delete("/api/browse/notes/nonexistent")
        assert resp.status_code == 404

    def test_delete_success(self, app, client):
        app.state.notes_store = _mock_notes_store()
        resp = client.delete("/api/browse/notes/n1")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
