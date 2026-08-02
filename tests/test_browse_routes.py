"""Tests for browse notes CRUD API routes (/api/browse/notes)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def notes_client(app):
    """TestClient with a SettingsStore so browse note routes can function.

    Notes are persisted via the SettingsStore (key-value in SQLite),
    not the state_manager backend directly.
    """
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.settings_store import SettingsStore

    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.settings_store = SettingsStore(backend.conn)

    yield TestClient(app)

    asyncio.get_event_loop().run_until_complete(backend.close())


# ── List (empty) ──────────────────────────────────────────────────────────

def test_list_notes_empty(notes_client):
    """GET /api/browse/notes returns empty list initially."""
    resp = notes_client.get("/api/browse/notes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["notes"] == []


# ── POST (create) ────────────────────────────────────────────────────────

def test_create_note(notes_client):
    """POST /api/browse/notes creates a note and returns 201."""
    body = {
        "title": "Test Note",
        "content": "Some content here",
        "tags": ["test", "example"],
        "source_url": "https://example.com",
    }
    resp = notes_client.post("/api/browse/notes", json=body)
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Test Note"
    assert data["content"] == "Some content here"
    assert data["tags"] == ["test", "example"]
    assert "id" in data
    assert "created_at" in data


def test_create_note_defaults(notes_client):
    """POST /api/browse/notes uses defaults for missing fields."""
    resp = notes_client.post("/api/browse/notes", json={})
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "Untitled"
    assert data["content"] == ""
    assert data["tags"] == []


# ── GET single ────────────────────────────────────────────────────────────

def test_get_note(notes_client):
    """GET /api/browse/notes/{id} returns the full note."""
    create_resp = notes_client.post("/api/browse/notes", json={
        "title": "Retrieve Me",
        "content": "Full content body",
    })
    note_id = create_resp.json()["id"]

    resp = notes_client.get(f"/api/browse/notes/{note_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == note_id
    assert data["title"] == "Retrieve Me"
    assert data["content"] == "Full content body"


def test_get_note_not_found(notes_client):
    """GET /api/browse/notes/{id} returns 404 for missing note."""
    resp = notes_client.get("/api/browse/notes/nonexistent")
    assert resp.status_code == 404
    assert "error" in resp.json()


# ── PUT (update) ─────────────────────────────────────────────────────────

def test_update_note(notes_client):
    """PUT /api/browse/notes/{id} updates an existing note."""
    create_resp = notes_client.post("/api/browse/notes", json={
        "title": "Original",
        "content": "Original content",
    })
    note_id = create_resp.json()["id"]

    resp = notes_client.put(f"/api/browse/notes/{note_id}", json={
        "title": "Updated Title",
        "content": "Updated content",
        "tags": ["updated"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Updated Title"
    assert data["content"] == "Updated content"
    assert data["tags"] == ["updated"]


def test_update_note_partial(notes_client):
    """PUT /api/browse/notes/{id} updates only provided fields."""
    create_resp = notes_client.post("/api/browse/notes", json={
        "title": "Keep This",
        "content": "Original content",
    })
    note_id = create_resp.json()["id"]

    resp = notes_client.put(f"/api/browse/notes/{note_id}", json={
        "content": "New content only",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Keep This"  # unchanged
    assert data["content"] == "New content only"


def test_update_note_not_found(notes_client):
    """PUT /api/browse/notes/{id} returns 404 for missing note."""
    resp = notes_client.put("/api/browse/notes/nonexistent", json={"title": "X"})
    assert resp.status_code == 404


# ── DELETE ────────────────────────────────────────────────────────────────

def test_delete_note(notes_client):
    """DELETE /api/browse/notes/{id} removes the note."""
    create_resp = notes_client.post("/api/browse/notes", json={"title": "Doomed"})
    note_id = create_resp.json()["id"]

    resp = notes_client.delete(f"/api/browse/notes/{note_id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Confirm gone
    assert notes_client.get(f"/api/browse/notes/{note_id}").status_code == 404


def test_delete_note_not_found(notes_client):
    """DELETE /api/browse/notes/{id} returns 404 for missing note."""
    resp = notes_client.delete("/api/browse/notes/nonexistent")
    assert resp.status_code == 404


# ── List (populated) ─────────────────────────────────────────────────────

def test_list_notes_returns_stubs(notes_client):
    """GET /api/browse/notes returns metadata stubs with preview."""
    notes_client.post("/api/browse/notes", json={
        "title": "Note A",
        "content": "Content for note A which is quite long " * 5,
    })
    notes_client.post("/api/browse/notes", json={
        "title": "Note B",
        "content": "Short content",
    })

    resp = notes_client.get("/api/browse/notes")
    assert resp.status_code == 200
    notes = resp.json()["notes"]
    assert len(notes) == 2
    # Stubs should have preview but limited length
    for n in notes:
        assert "id" in n
        assert "title" in n
        assert "preview" in n
        assert len(n["preview"]) <= 120


# ── No settings store fallback ────────────────────────────────────────────

def test_list_notes_no_store(client):
    """Without settings_store, notes endpoint returns 503."""
    # The default conftest client has no settings_store
    resp = client.get("/api/browse/notes")
    assert resp.status_code == 503
