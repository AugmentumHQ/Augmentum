"""Tests for character card CRUD API routes (/api/characters/)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sqlite_client(app):
    """TestClient with a real SQLite backend so character routes can function."""
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)

    yield TestClient(app)

    asyncio.get_event_loop().run_until_complete(backend.close())


# ── List (empty) ──────────────────────────────────────────────────────────

def test_list_characters_empty(sqlite_client):
    """GET /api/characters/ returns empty list initially."""
    resp = sqlite_client.get("/api/characters/")
    assert resp.status_code == 200
    assert resp.json()["characters"] == []


# ── PUT (upsert) ─────────────────────────────────────────────────────────

def test_put_creates_character(sqlite_client):
    """PUT /api/characters/{id} saves a new character."""
    body = {
        "name": "Alice",
        "description": "A curious adventurer",
        "personality": "Brave and kind",
        "greeting": "Hello there!",
    }
    resp = sqlite_client.put("/api/characters/ch-001", json=body)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["id"] == "ch-001"


def test_put_upserts_existing(sqlite_client):
    """PUT /api/characters/{id} updates an existing character."""
    body1 = {"name": "Alice", "description": "Version 1"}
    sqlite_client.put("/api/characters/ch-001", json=body1)

    body2 = {"name": "Alice Updated", "description": "Version 2"}
    resp = sqlite_client.put("/api/characters/ch-001", json=body2)
    assert resp.status_code == 200

    get_resp = sqlite_client.get("/api/characters/ch-001")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Alice Updated"
    assert get_resp.json()["description"] == "Version 2"


# ── GET single ────────────────────────────────────────────────────────────

def test_get_character_returns_data(sqlite_client):
    """GET /api/characters/{id} returns the character."""
    body = {
        "name": "Bob",
        "description": "A friendly NPC",
        "personality": "Cheerful",
        "scenario": "A tavern",
        "greeting": "Welcome, traveler!",
    }
    sqlite_client.put("/api/characters/ch-002", json=body)

    resp = sqlite_client.get("/api/characters/ch-002")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "ch-002"
    assert data["name"] == "Bob"
    assert data["description"] == "A friendly NPC"
    assert data["greeting"] == "Welcome, traveler!"


def test_get_character_not_found(sqlite_client):
    """GET /api/characters/{id} returns 404 for missing character."""
    resp = sqlite_client.get("/api/characters/nonexistent")
    assert resp.status_code == 404
    assert "error" in resp.json()


# ── DELETE ────────────────────────────────────────────────────────────────

def test_delete_character(sqlite_client):
    """DELETE /api/characters/{id} removes the character."""
    sqlite_client.put("/api/characters/ch-003", json={"name": "Doomed"})

    resp = sqlite_client.delete("/api/characters/ch-003")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Confirm gone
    assert sqlite_client.get("/api/characters/ch-003").status_code == 404


def test_delete_character_not_found(sqlite_client):
    """DELETE /api/characters/{id} returns 404 for missing character."""
    resp = sqlite_client.delete("/api/characters/nonexistent")
    assert resp.status_code == 404


# ── POST /import (bulk) ──────────────────────────────────────────────────

def test_import_characters_bulk(sqlite_client):
    """POST /api/characters/import upserts an array of characters."""
    body = {
        "characters": [
            {"id": "ch-a", "name": "Alpha", "description": "First"},
            {"id": "ch-b", "name": "Beta", "description": "Second"},
            {"id": "ch-c", "name": "Gamma", "description": "Third"},
        ]
    }
    resp = sqlite_client.post("/api/characters/import", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["imported"] == 3

    # Verify each character exists
    for cid in ["ch-a", "ch-b", "ch-c"]:
        assert sqlite_client.get(f"/api/characters/{cid}").status_code == 200


def test_import_characters_skips_no_id(sqlite_client):
    """POST /api/characters/import skips entries without an id."""
    body = {
        "characters": [
            {"name": "No ID Character"},
            {"id": "ch-valid", "name": "Has ID"},
        ]
    }
    resp = sqlite_client.post("/api/characters/import", json=body)
    assert resp.status_code == 200
    assert resp.json()["imported"] == 1


def test_import_characters_bad_format(sqlite_client):
    """POST /api/characters/import returns 400 for non-list characters."""
    resp = sqlite_client.post("/api/characters/import", json={"characters": "not-a-list"})
    assert resp.status_code == 400


# ── List (populated) ─────────────────────────────────────────────────────

def test_list_characters_returns_all(sqlite_client):
    """GET /api/characters/ returns all characters after creation."""
    sqlite_client.put("/api/characters/ch-x", json={"name": "X"})
    sqlite_client.put("/api/characters/ch-y", json={"name": "Y"})

    resp = sqlite_client.get("/api/characters/")
    assert resp.status_code == 200
    chars = resp.json()["characters"]
    assert len(chars) == 2
    names = {c["name"] for c in chars}
    assert names == {"X", "Y"}


# ── MemoryBackend fallback ────────────────────────────────────────────────

def test_list_characters_no_sqlite(client):
    """With MemoryBackend, list returns empty characters."""
    resp = client.get("/api/characters/")
    assert resp.status_code == 200
    assert resp.json()["characters"] == []


def test_get_character_no_sqlite(client):
    """With MemoryBackend, GET single character returns 503."""
    resp = client.get("/api/characters/any-id")
    assert resp.status_code == 503
