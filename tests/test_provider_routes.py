"""Tests for provider_routes.py — provider management API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# GET /api/providers/
# ---------------------------------------------------------------------------


class TestListProviders:
    def test_list_returns_builtin_providers(self, client):
        resp = client.get("/api/providers/")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        providers = data["providers"]
        assert isinstance(providers, list)
        # The mock registry has "ollama" in backends
        names = [p["id"] for p in providers]
        assert "ollama" in names

    def test_builtin_providers_are_type_builtin(self, client):
        resp = client.get("/api/providers/")
        data = resp.json()
        for p in data["providers"]:
            if p["id"] == "ollama":
                assert p["type"] == "builtin"
                assert p["is_enabled"] is True


# ---------------------------------------------------------------------------
# POST /api/providers/ — create
# ---------------------------------------------------------------------------


class TestCreateProvider:
    def test_create_reserved_id_returns_400(self, client):
        resp = client.post("/api/providers/", json={
            "id": "ollama",
            "name": "My Ollama",
            "base_url": "http://localhost:11434",
        })
        assert resp.status_code == 400
        assert "reserved" in resp.json()["error"]

    def test_create_duplicate_returns_409(self, client):
        # First, register a backend with this ID in the mock registry
        client.app.state.provider_registry.backends["my-provider"] = MagicMock()
        resp = client.post("/api/providers/", json={
            "id": "my-provider",
            "name": "My Provider",
            "base_url": "http://localhost:8000",
        })
        assert resp.status_code == 409
        # Cleanup
        del client.app.state.provider_registry.backends["my-provider"]

    def test_create_no_sqlite_returns_503(self, client):
        # The base client uses MemoryBackend, not SQLite — store returns None
        resp = client.post("/api/providers/", json={
            "id": "new-provider",
            "name": "New Provider",
            "base_url": "http://localhost:8000",
        })
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# PUT /api/providers/{id} — update
# ---------------------------------------------------------------------------


class TestUpdateProvider:
    def test_update_reserved_returns_400(self, client):
        resp = client.put("/api/providers/ollama", json={"name": "New Name"})
        assert resp.status_code == 400

    def test_update_no_fields_returns_400(self, client):
        resp = client.put("/api/providers/some-id", json={})
        # No SQLite → 503 before field check, or 400 after
        assert resp.status_code in (400, 503)

    def test_update_no_sqlite_returns_503(self, client):
        resp = client.put("/api/providers/some-id", json={"name": "Updated"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# DELETE /api/providers/{id}
# ---------------------------------------------------------------------------


class TestDeleteProvider:
    def test_delete_reserved_returns_400(self, client):
        resp = client.delete("/api/providers/ollama")
        assert resp.status_code == 400

    def test_delete_no_sqlite_returns_503(self, client):
        resp = client.delete("/api/providers/some-id")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# POST /api/providers/{id}/test
# ---------------------------------------------------------------------------


class TestTestProvider:
    def test_test_nonexistent_returns_404(self, client):
        client.app.state.provider_registry.get_backend = MagicMock(return_value=None)
        resp = client.post("/api/providers/nonexistent/test")
        assert resp.status_code == 404

    def test_test_existing_returns_models(self, client):
        from augmentum.models.base import ModelInfo

        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(return_value=[
            ModelInfo(name="test-model", model="test-model", size=1000, digest="xyz",
                      modified_at="2024-01-01"),
        ])
        client.app.state.provider_registry.get_backend = MagicMock(return_value=mock_backend)

        resp = client.post("/api/providers/my-provider/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["models"]) == 1

    def test_test_backend_error_returns_502(self, client):
        mock_backend = MagicMock()
        mock_backend.list_models = AsyncMock(side_effect=Exception("Connection refused"))
        client.app.state.provider_registry.get_backend = MagicMock(return_value=mock_backend)

        resp = client.post("/api/providers/my-provider/test")
        assert resp.status_code == 502
        assert resp.json()["status"] == "error"


# ---------------------------------------------------------------------------
# GET /api/providers/profiles
# ---------------------------------------------------------------------------


class TestProviderProfiles:
    def test_profiles_returns_list(self, client):
        resp = client.get("/api/providers/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # Each profile should have id and name
        if data:
            assert "id" in data[0]
            assert "name" in data[0]
