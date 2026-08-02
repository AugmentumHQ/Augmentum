"""Tests for cloud_image_routes.py — cloud image provider CRUD and generation."""

from __future__ import annotations


class TestCloudImageProviderCRUD:
    def test_list_providers_no_db(self, client):
        """Without SQLite backend, returns empty list."""
        resp = client.get("/api/image/cloud/providers")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_providers_empty(self, sqlite_client):
        resp = sqlite_client.get("/api/image/cloud/providers")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 0

    def test_create_provider(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/image/cloud/providers",
            json={
                "id": "openai",
                "name": "OpenAI DALL-E",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "openai"

    def test_delete_provider_not_found(self, sqlite_client):
        resp = sqlite_client.delete("/api/image/cloud/providers/nonexistent")
        assert resp.status_code == 404


class TestCloudImageGeneration:
    def test_generate_no_providers(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/image/cloud/generate",
            json={"prompt": "a cat"},
        )
        # No provider available
        assert resp.status_code in (404, 503)
