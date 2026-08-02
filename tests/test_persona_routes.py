"""Tests for persona CRUD API routes (/api/personas/)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sqlite_client(app):
    """TestClient with a real SQLite backend so persona routes can function."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)

    tc = TestClient(app)
    # Auth header — app fixture installs a mock session_manager that accepts
    # any bearer token for the admin test_user.
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc

    asyncio.get_event_loop().run_until_complete(backend.close())


# ── List (empty) ──────────────────────────────────────────────────────────

def test_list_personas_empty(sqlite_client):
    """GET /api/personas/ returns empty list initially."""
    resp = sqlite_client.get("/api/personas/")
    assert resp.status_code == 200
    assert resp.json()["personas"] == []


# ── POST (create) ────────────────────────────────────────────────────────

def test_create_persona(sqlite_client):
    """POST /api/personas/ creates a new persona and returns 201."""
    body = {
        "name": "Test User",
        "appearance": "Tall, dark hair",
        "description": "A test persona",
    }
    resp = sqlite_client.post("/api/personas/", json=body)
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Test User"
    assert data["appearance"] == "Tall, dark hair"
    assert data["description"] == "A test persona"
    assert "id" in data


def test_create_persona_name_required(sqlite_client):
    """POST /api/personas/ returns 400 when name is missing."""
    resp = sqlite_client.post("/api/personas/", json={"description": "No name"})
    assert resp.status_code == 400
    assert "Name" in resp.json()["error"] or "name" in resp.json()["error"].lower()


def test_create_persona_empty_name(sqlite_client):
    """POST /api/personas/ returns 400 when name is empty string."""
    resp = sqlite_client.post("/api/personas/", json={"name": "  "})
    assert resp.status_code == 400


# ── GET single ────────────────────────────────────────────────────────────

def test_get_persona(sqlite_client):
    """GET /api/personas/{id} returns the persona."""
    create_resp = sqlite_client.post("/api/personas/", json={
        "name": "Retrieve Me",
        "appearance": "Blue eyes",
        "description": "Retrievable persona",
    })
    persona_id = create_resp.json()["id"]

    resp = sqlite_client.get(f"/api/personas/{persona_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == persona_id
    assert data["name"] == "Retrieve Me"
    assert data["appearance"] == "Blue eyes"


def test_get_persona_not_found(sqlite_client):
    """GET /api/personas/{id} returns 404 for missing persona."""
    resp = sqlite_client.get("/api/personas/nonexistent")
    assert resp.status_code == 404
    assert "error" in resp.json()


# ── PUT (update) ─────────────────────────────────────────────────────────

def test_update_persona(sqlite_client):
    """PUT /api/personas/{id} updates an existing persona."""
    create_resp = sqlite_client.post("/api/personas/", json={
        "name": "Original Name",
        "description": "Original desc",
    })
    persona_id = create_resp.json()["id"]

    resp = sqlite_client.put(f"/api/personas/{persona_id}", json={
        "name": "Updated Name",
        "description": "Updated desc",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Updated Name"
    assert data["description"] == "Updated desc"


def test_update_persona_partial(sqlite_client):
    """PUT /api/personas/{id} updates only provided fields."""
    create_resp = sqlite_client.post("/api/personas/", json={
        "name": "Keep This",
        "description": "Original",
        "appearance": "Original look",
    })
    persona_id = create_resp.json()["id"]

    resp = sqlite_client.put(f"/api/personas/{persona_id}", json={
        "description": "New description",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Keep This"  # unchanged
    assert data["description"] == "New description"


def test_update_persona_not_found(sqlite_client):
    """PUT /api/personas/{id} returns 404 for missing persona."""
    resp = sqlite_client.put("/api/personas/nonexistent", json={"name": "X"})
    assert resp.status_code == 404


def test_update_persona_empty_name(sqlite_client):
    """PUT /api/personas/{id} returns 400 when setting name to empty."""
    create_resp = sqlite_client.post("/api/personas/", json={"name": "Valid"})
    persona_id = create_resp.json()["id"]

    resp = sqlite_client.put(f"/api/personas/{persona_id}", json={"name": "  "})
    assert resp.status_code == 400


# ── DELETE ────────────────────────────────────────────────────────────────

def test_delete_persona(sqlite_client):
    """DELETE /api/personas/{id} removes the persona."""
    create_resp = sqlite_client.post("/api/personas/", json={"name": "Doomed"})
    persona_id = create_resp.json()["id"]

    resp = sqlite_client.delete(f"/api/personas/{persona_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == persona_id

    # Confirm gone
    assert sqlite_client.get(f"/api/personas/{persona_id}").status_code == 404


def test_delete_persona_not_found(sqlite_client):
    """DELETE /api/personas/{id} returns 404 for missing persona."""
    resp = sqlite_client.delete("/api/personas/nonexistent")
    assert resp.status_code == 404


# ── List (populated) ─────────────────────────────────────────────────────

def test_list_personas_returns_all(sqlite_client):
    """GET /api/personas/ returns all personas after creation."""
    sqlite_client.post("/api/personas/", json={"name": "Persona A"})
    sqlite_client.post("/api/personas/", json={"name": "Persona B"})

    resp = sqlite_client.get("/api/personas/")
    assert resp.status_code == 200
    personas = resp.json()["personas"]
    assert len(personas) == 2
    names = {p["name"] for p in personas}
    assert names == {"Persona A", "Persona B"}


# ── Default persona ──────────────────────────────────────────────────────

def test_create_default_persona(sqlite_client):
    """Creating a persona with is_default=True sets it as default."""
    resp = sqlite_client.post("/api/personas/", json={
        "name": "Default One",
        "is_default": True,
    })
    assert resp.status_code == 201
    assert resp.json()["is_default"] is True


# ── MemoryBackend fallback ────────────────────────────────────────────────

def test_list_personas_no_sqlite(client):
    """With MemoryBackend, list returns empty personas."""
    resp = client.get("/api/personas/")
    assert resp.status_code == 200
    assert resp.json()["personas"] == []


def test_get_persona_no_sqlite(client):
    """With MemoryBackend, GET single persona returns 503."""
    resp = client.get("/api/personas/any-id")
    assert resp.status_code == 503
