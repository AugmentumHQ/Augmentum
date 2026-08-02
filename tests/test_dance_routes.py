"""Smoke tests for /api/dance/* routes."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager


@pytest.fixture
def sqlite_client(app):
    backend = SQLiteBackend(":memory:")
    asyncio.get_event_loop().run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc
    asyncio.get_event_loop().run_until_complete(backend.close())


class TestDanceHistoryRoutes:
    def test_history_starts_empty(self, sqlite_client):
        resp = sqlite_client.get("/api/dance/history")
        assert resp.status_code == 200
        assert resp.json() == {"entries": []}

    def test_history_append_then_list(self, sqlite_client):
        post = sqlite_client.post("/api/dance/history", json={
            "anim_id": "kebab-dance",
            "label": "kebab dance",
            "played_at": 1_000_000,
            "duration_sec": 12.0,
            "mode": "chat-call",
        })
        assert post.status_code == 200
        get = sqlite_client.get("/api/dance/history")
        entries = get.json()["entries"]
        assert len(entries) == 1
        assert entries[0]["anim_id"] == "kebab-dance"

    def test_history_rejects_missing_anim_id(self, sqlite_client):
        resp = sqlite_client.post("/api/dance/history", json={"label": "x"})
        assert resp.status_code == 400

    def test_history_clear(self, sqlite_client):
        sqlite_client.post("/api/dance/history", json={
            "anim_id": "x", "label": "x",
            "played_at": 1, "duration_sec": 1.0,
        })
        resp = sqlite_client.delete("/api/dance/history")
        assert resp.status_code == 200
        assert resp.json()["cleared"] == 1


class TestDanceRatingsRoutes:
    def test_ratings_start_empty(self, sqlite_client):
        resp = sqlite_client.get("/api/dance/ratings")
        assert resp.status_code == 200
        assert resp.json() == {"ratings": {}}

    def test_put_like(self, sqlite_client):
        resp = sqlite_client.put(
            "/api/dance/ratings/kebab-dance", json={"kind": "like"},
        )
        assert resp.status_code == 200
        assert resp.json()["rating"]["kind"] == "like"

    def test_put_longer_accumulates(self, sqlite_client):
        sqlite_client.put(
            "/api/dance/ratings/kebab-dance", json={"kind": "longer"},
        )
        resp = sqlite_client.put(
            "/api/dance/ratings/kebab-dance", json={"kind": "longer"},
        )
        assert resp.json()["rating"]["slotBonusSec"] == 16

    def test_clear_via_put(self, sqlite_client):
        sqlite_client.put(
            "/api/dance/ratings/kebab-dance", json={"kind": "like"},
        )
        resp = sqlite_client.put(
            "/api/dance/ratings/kebab-dance", json={"kind": "clear"},
        )
        assert resp.status_code == 200
        assert resp.json()["rating"] is None
        get = sqlite_client.get("/api/dance/ratings")
        assert "kebab-dance" not in get.json()["ratings"]

    def test_delete_clears(self, sqlite_client):
        sqlite_client.put(
            "/api/dance/ratings/kebab-dance", json={"kind": "broken"},
        )
        resp = sqlite_client.delete("/api/dance/ratings/kebab-dance")
        assert resp.status_code == 200
        get = sqlite_client.get("/api/dance/ratings")
        assert "kebab-dance" not in get.json()["ratings"]

    def test_rejects_invalid_kind(self, sqlite_client):
        resp = sqlite_client.put(
            "/api/dance/ratings/kebab-dance", json={"kind": "love"},
        )
        assert resp.status_code == 400
