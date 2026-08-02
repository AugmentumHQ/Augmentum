"""Tests for memory_routes.py — memory CRUD, search, config, diagnostics."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from fastapi.testclient import TestClient


def _mock_memory_store():
    """Create a mock MemoryStore with standard methods."""
    store = MagicMock()
    mock_mem = MagicMock(
        id="mem_1", content="Test fact", memory_type="fact",
        importance=0.8, confidence=0.9, source_type="user_manual",
        tier="active", access_count=1,
        valid_from="2024-01-01T00:00:00Z", valid_until=None,
        created_at="2024-01-01T00:00:00Z", updated_at="2024-01-01T00:00:00Z",
        superseded_by=None,
    )
    store.list_all = AsyncMock(return_value=[mock_mem])
    store.recall = AsyncMock(return_value=[mock_mem])
    store.store = AsyncMock(return_value="mem_new")
    store.edit = AsyncMock(return_value=True)
    store.forget = AsyncMock(return_value=True)
    store.update_tier = AsyncMock(return_value=True)
    store.count = AsyncMock(return_value={"fact": 5, "preference": 2})
    store.get = AsyncMock(return_value=mock_mem)
    store.get_history = AsyncMock(return_value=[mock_mem])
    store._vec_enabled = False
    store._backend = MagicMock()
    store._backend.conn = MagicMock()
    store._backend.conn.execute = AsyncMock()
    store._backend.conn.commit = AsyncMock()
    return store


class TestListFacts:
    def test_list_facts_no_store(self, client):
        resp = client.get("/v1/memory/facts")
        assert resp.status_code == 503

    def test_list_facts_success(self, app, client):
        app.state.memory_store = _mock_memory_store()
        resp = client.get("/v1/memory/facts?user_id=default")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["memories"][0]["id"] == "mem_1"


class TestSearchMemories:
    def test_search_no_store(self, client):
        resp = client.get("/v1/memory/search?q=test")
        assert resp.status_code == 503

    def test_search_success(self, app, client):
        app.state.memory_store = _mock_memory_store()
        resp = client.get("/v1/memory/search?q=test")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1
        assert resp.json()["query"] == "test"


class TestStoreMemory:
    def test_store_no_store(self, client):
        resp = client.post("/v1/memory/store", json={"content": "hello"})
        assert resp.status_code == 503

    def test_store_success(self, app, client):
        app.state.memory_store = _mock_memory_store()
        resp = client.post(
            "/v1/memory/store",
            json={"content": "User prefers dark mode", "memory_type": "preference"},
        )
        assert resp.status_code == 201
        assert resp.json()["id"] == "mem_new"
        assert resp.json()["status"] == "stored"


class TestEditMemory:
    def test_edit_not_found(self, app, client):
        store = _mock_memory_store()
        store.edit = AsyncMock(return_value=False)
        app.state.memory_store = store
        resp = client.put(
            "/v1/memory/facts/nonexistent",
            json={"content": "updated"},
        )
        assert resp.status_code == 404

    def test_edit_success(self, app, client):
        app.state.memory_store = _mock_memory_store()
        resp = client.put(
            "/v1/memory/facts/mem_1",
            json={"content": "updated fact", "importance": 0.9},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"


class TestDeleteMemory:
    def test_delete_not_found(self, app, client):
        store = _mock_memory_store()
        store.forget = AsyncMock(return_value=False)
        app.state.memory_store = store
        resp = client.delete("/v1/memory/facts/nonexistent")
        assert resp.status_code == 404

    def test_delete_success(self, app, client):
        app.state.memory_store = _mock_memory_store()
        resp = client.delete("/v1/memory/facts/mem_1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestMemoryTier:
    def test_update_tier_invalid(self, app, client):
        app.state.memory_store = _mock_memory_store()
        resp = client.put(
            "/v1/memory/facts/mem_1/tier",
            json={"tier": "bogus"},
        )
        assert resp.status_code == 400

    def test_update_tier_not_found(self, app, client):
        store = _mock_memory_store()
        store.update_tier = AsyncMock(return_value=False)
        app.state.memory_store = store
        resp = client.put(
            "/v1/memory/facts/mem_1/tier",
            json={"tier": "active"},
        )
        assert resp.status_code == 404


class TestMemoryStats:
    def test_stats_no_store(self, client):
        resp = client.get("/v1/memory/stats")
        assert resp.status_code == 503

    def test_stats_success(self, app, client):
        app.state.memory_store = _mock_memory_store()
        resp = client.get("/v1/memory/stats?user_id=default")
        assert resp.status_code == 200
        assert "counts" in resp.json()


class TestMemoryDiagnostics:
    def test_diagnostics_no_store(self, client):
        resp = client.get("/v1/memory/diagnostics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["memory_store_initialized"] is False


class TestMemoryConfig:
    def test_get_config(self, client):
        resp = client.get("/v1/memory/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "memory_enabled" in data

    def test_update_config_unknown_key(self, client):
        resp = client.put("/v1/memory/config", json={"bogus_key": True})
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" in data
        assert any("Unknown" in e for e in data["errors"])

    def test_update_config_valid(self, app, client):
        app.state.settings_store = MagicMock()
        app.state.settings_store.set = AsyncMock()
        resp = client.put(
            "/v1/memory/config",
            json={"memory_enabled": True, "memory_recall_limit": 5},
        )
        assert resp.status_code == 200
        assert "updated" in resp.json()


class TestMemoryHistory:
    def test_history_not_found(self, app, client):
        store = _mock_memory_store()
        store.get_history = AsyncMock(return_value=[])
        app.state.memory_store = store
        resp = client.get("/v1/memory/facts/nonexistent/history")
        assert resp.status_code == 404


class TestMemoryNotifications:
    def test_get_notifications(self, client, monkeypatch):
        monkeypatch.setattr(
            "augmentum.memory.integration.get_pending_notifications",
            lambda user_id: [{"id": "n1", "content": "new fact"}],
        )
        resp = client.get("/v1/memory/notifications")
        assert resp.status_code == 200
        assert len(resp.json()["notifications"]) == 1


class TestMemoryProfile:
    def test_get_profile_not_initialized(self, client):
        resp = client.get("/v1/memory/profile")
        assert resp.status_code == 200
        assert resp.json()["profile"] == ""

    def test_context_preview_no_store(self, client):
        resp = client.get("/v1/memory/context-preview")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_memories" in data
