"""Smoke tests for /api/dance/loops/* routes."""
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


class TestLoopsList:
    def test_list_empty(self, sqlite_client):
        resp = sqlite_client.get("/api/dance/loops")
        assert resp.status_code == 200
        assert resp.json() == {"loops": [], "active_id": None}


class TestLoopsCreate:
    def test_create_minimal(self, sqlite_client):
        resp = sqlite_client.post("/api/dance/loops", json={
            "name": "chill",
            "animation_ids": ["kebab-dance", "user:abc"],
        })
        assert resp.status_code == 200
        loop = resp.json()["loop"]
        assert loop["name"] == "chill"
        assert loop["animation_ids"] == ["kebab-dance", "user:abc"]
        assert loop["is_active"] is False

    def test_rejects_missing_name(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/dance/loops", json={"animation_ids": []},
        )
        assert resp.status_code == 400

    def test_rejects_non_list_ids(self, sqlite_client):
        resp = sqlite_client.post("/api/dance/loops", json={
            "name": "x", "animation_ids": "not-a-list",
        })
        assert resp.status_code == 400


class TestLoopsUpdate:
    def test_update_name(self, sqlite_client):
        created = sqlite_client.post(
            "/api/dance/loops", json={"name": "old"},
        ).json()["loop"]
        resp = sqlite_client.put(
            f"/api/dance/loops/{created['id']}", json={"name": "renamed"},
        )
        assert resp.status_code == 200
        assert resp.json()["loop"]["name"] == "renamed"

    def test_update_404_missing(self, sqlite_client):
        resp = sqlite_client.put(
            "/api/dance/loops/loop_nope_aaa", json={"name": "x"},
        )
        assert resp.status_code == 404

    def test_update_rejects_path_traversal(self, sqlite_client):
        resp = sqlite_client.put(
            "/api/dance/loops/..%2Fhostile", json={"name": "x"},
        )
        assert resp.status_code in (400, 404)


class TestLoopsDelete:
    def test_delete(self, sqlite_client):
        created = sqlite_client.post(
            "/api/dance/loops", json={"name": "x"},
        ).json()["loop"]
        resp = sqlite_client.delete(f"/api/dance/loops/{created['id']}")
        assert resp.status_code == 200
        assert sqlite_client.get("/api/dance/loops").json()["loops"] == []

    def test_delete_404_missing(self, sqlite_client):
        resp = sqlite_client.delete("/api/dance/loops/loop_nope_aaa")
        assert resp.status_code == 404


class TestActiveLoop:
    def test_activate(self, sqlite_client):
        loop = sqlite_client.post(
            "/api/dance/loops", json={"name": "x"},
        ).json()["loop"]
        resp = sqlite_client.put(
            "/api/dance/loops/active", json={"loop_id": loop["id"]},
        )
        assert resp.status_code == 200
        assert resp.json()["active"]["id"] == loop["id"]
        # List reflects the active state.
        listed = sqlite_client.get("/api/dance/loops").json()
        assert listed["active_id"] == loop["id"]

    def test_activate_switches(self, sqlite_client):
        a = sqlite_client.post(
            "/api/dance/loops", json={"name": "a"},
        ).json()["loop"]
        b = sqlite_client.post(
            "/api/dance/loops", json={"name": "b"},
        ).json()["loop"]
        sqlite_client.put(
            "/api/dance/loops/active", json={"loop_id": a["id"]},
        )
        sqlite_client.put(
            "/api/dance/loops/active", json={"loop_id": b["id"]},
        )
        listed = sqlite_client.get("/api/dance/loops").json()
        assert listed["active_id"] == b["id"]
        # And only one row reports is_active.
        active = [l for l in listed["loops"] if l["is_active"]]
        assert len(active) == 1
        assert active[0]["id"] == b["id"]

    def test_clear(self, sqlite_client):
        loop = sqlite_client.post(
            "/api/dance/loops", json={"name": "x"},
        ).json()["loop"]
        sqlite_client.put(
            "/api/dance/loops/active", json={"loop_id": loop["id"]},
        )
        resp = sqlite_client.put(
            "/api/dance/loops/active", json={"loop_id": None},
        )
        assert resp.status_code == 200
        assert resp.json()["active"] is None
        assert sqlite_client.get(
            "/api/dance/loops",
        ).json()["active_id"] is None

    def test_delete_active_clears_active(self, sqlite_client):
        loop = sqlite_client.post(
            "/api/dance/loops", json={"name": "x"},
        ).json()["loop"]
        sqlite_client.put(
            "/api/dance/loops/active", json={"loop_id": loop["id"]},
        )
        sqlite_client.delete(f"/api/dance/loops/{loop['id']}")
        listed = sqlite_client.get("/api/dance/loops").json()
        assert listed["active_id"] is None
