"""Tests for marketplace_routes.py — catalog, service enable/disable, hardware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _mock_service_manager():
    """Create a mock ServiceManager."""
    mgr = MagicMock()
    entry = MagicMock(
        id="svc_ollama", name="Ollama", description="LLM backend",
        category=MagicMock(value="llm"), image="ollama/ollama:latest",
        host_port=11434,
        gpu=MagicMock(required=False, vram_mb=0),
        api_type="ollama", features=["chat"],
    )
    mgr.catalog.list_all.return_value = [entry]
    mgr.catalog.list_by_category.return_value = [entry]
    mgr.catalog._entries = [entry]
    mgr.catalog._by_id = {entry.id: entry}
    mgr.list_managed = AsyncMock(return_value=[])

    managed_svc = MagicMock(
        id="svc_ollama", name="Ollama", status="running",
        host_port=11434, container_id="abc123def456",
        category="llm", image="ollama/ollama:latest",
        enabled=True, error=None,
    )
    mgr.enable_service = AsyncMock(return_value=managed_svc)
    mgr.disable_service = AsyncMock()
    mgr.get_status = AsyncMock(return_value=MagicMock(value="running"))
    mgr._find_container = AsyncMock(return_value=None)
    return mgr


class TestCatalog:
    def test_list_catalog_no_manager(self, app, client):
        """Without service_manager, _mgr() raises RuntimeError which propagates."""
        import pytest
        with pytest.raises(RuntimeError, match="Service manager not available"):
            client.get("/api/marketplace/catalog")

    def test_list_catalog_success(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.get("/api/marketplace/catalog")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == "svc_ollama"
        assert "gpu" in data[0]

    def test_list_catalog_invalid_category(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.get("/api/marketplace/catalog?category=bogus")
        assert resp.status_code == 400
        assert "Invalid category" in resp.json()["error"]


class TestServices:
    def test_list_services(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.get("/api/marketplace/services")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_enable_service_success(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.post("/api/marketplace/services/svc_ollama/enable")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "svc_ollama"
        assert data["status"] == "running"

    def test_enable_service_error(self, app, client):
        mgr = _mock_service_manager()
        mgr.enable_service = AsyncMock(side_effect=ValueError("Already enabled"))
        app.state.service_manager = mgr
        resp = client.post("/api/marketplace/services/svc_ollama/enable")
        assert resp.status_code == 400

    def test_disable_service_success(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.post("/api/marketplace/services/svc_ollama/disable")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_disable_service_error(self, app, client):
        mgr = _mock_service_manager()
        mgr.disable_service = AsyncMock(side_effect=Exception("Docker error"))
        app.state.service_manager = mgr
        resp = client.post("/api/marketplace/services/svc_ollama/disable")
        assert resp.status_code == 500

    def test_service_status(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.get("/api/marketplace/services/svc_ollama/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_service_logs_no_container(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.post("/api/marketplace/services/svc_ollama/logs")
        assert resp.status_code == 404


class TestCustomService:
    def test_create_custom_missing_fields(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.post(
            "/api/marketplace/services/custom",
            json={"description": "no name or image"},
        )
        assert resp.status_code == 400

    def test_create_custom_success(self, app, client):
        mgr = _mock_service_manager()
        app.state.service_manager = mgr
        resp = client.post(
            "/api/marketplace/services/custom",
            json={
                "name": "Custom LLM",
                "category": "llm",
                "image": "custom/llm:latest",
                "internal_port": 8080,
                "host_port": 9090,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "status" in data


class TestHardware:
    def test_detect_hardware_no_manager(self, app, client):
        # service_manager not set => still works (docker_available=False)
        resp = client.get("/api/marketplace/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpu_available" in data
        assert data["docker_available"] is False

    def test_detect_hardware_with_manager(self, app, client):
        app.state.service_manager = _mock_service_manager()
        resp = client.get("/api/marketplace/hardware")
        assert resp.status_code == 200
        assert resp.json()["docker_available"] is True
