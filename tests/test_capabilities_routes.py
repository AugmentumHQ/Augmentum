"""Tests for /api/capabilities read-only host inventory."""

from __future__ import annotations

from augmentum.tools.base import ToolCategory


class DemoTool:
    name = "demo_fetch"
    description = "Demo fetch tool"
    category = ToolCategory.FETCH
    input_schema = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Target URL"}},
        "required": ["url"],
    }
    requires_services = ["demo"]
    produces = ["text"]
    consumes = []
    long_running = False


def test_lists_capability_hosts(client):
    response = client.get("/api/capabilities")

    assert response.status_code == 200
    data = response.json()
    assert data["version"] == 1
    assert data["frontdesk"]["schema"] == "augmentum.capabilities.inventory"
    assert "image_enabled" in data
    assert "endpoints" in data
    assert set(data["hosts"]) == {
        "models",
        "modes",
        "tools",
        "sessions",
        "devices",
        "surfaces",
        "powers",
        "jobs",
        "events",
    }
    assert "/v1/chat/completions" in data["routes"]["compatibility"]
    assert data["hosts"]["models"]["available"] is True
    assert data["hosts"]["models"]["backend_count"] == 1


def test_capabilities_root_is_registered_once(app):
    matches = [
        route
        for route in app.routes
        if getattr(route, "path", "") == "/api/capabilities"
        and "GET" in getattr(route, "methods", set())
    ]

    assert len(matches) == 1
    assert matches[0].endpoint.__name__ == "capabilities"


def test_model_host_reflects_provider_registry(app, client):
    app.state.provider_registry._default = "ollama"
    app.state.provider_registry._model_map = {"llama3.1:8b": "ollama"}
    app.state.provider_registry._discovered = [{"key": "ollama"}]

    response = client.get("/api/capabilities/models")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "models"
    assert data["default_backend"] == "ollama"
    assert data["available_backends"] == ["ollama"]
    assert data["backend_count"] == 1
    assert data["model_map_count"] == 1
    assert data["discovered_count"] == 1
    assert data["items"][0]["id"] == "ollama"
    assert data["items"][0]["kind"] == "backend"


def test_missing_provider_registry_degrades(app, client):
    delattr(app.state, "provider_registry")

    response = client.get("/api/capabilities/models")

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False
    assert data["status"] == "unavailable"
    assert data["default_backend"] == ""
    assert data["backend_count"] == 0
    assert data["items"] == []


def test_tools_host_reflects_runtime_registry(app, client):
    app.state.tool_registry.register(DemoTool())

    response = client.get("/api/capabilities/tools")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "tools"
    assert data["count"] == 1
    assert data["categories"] == [category.value for category in ToolCategory]
    assert data["phase_categories"]["gather"]
    assert data["items"][0]["id"] == "demo_fetch"
    assert data["items"][0]["category"] == "fetch"
    assert data["items"][0]["requires_services"] == ["demo"]
    assert data["links"]["primary"] == "/api/tools"


def test_tools_endpoint_uses_tool_host_ui_shape(app, client):
    app.state.tool_registry.register(DemoTool())

    response = client.get("/api/tools")

    assert response.status_code == 200
    data = response.json()
    assert data["categories"] == [category.value for category in ToolCategory]
    assert data["tools"][0]["name"] == "demo_fetch"
    assert data["tools"][0]["category"] == "fetch"
    assert data["tools"][0]["params"] == [
        {
            "name": "url",
            "type": "str",
            "required": True,
            "description": "Target URL",
        },
    ]


def test_tools_metrics_uses_tool_host(app, client):
    app.state.tool_registry.metrics.record("demo_fetch", success=True, elapsed_ms=12.0)

    response = client.get("/api/tools/metrics")

    assert response.status_code == 200
    assert response.json()["demo_fetch"]["calls"] == 1


def test_missing_tool_registry_degrades(app, client):
    delattr(app.state, "tool_registry")

    tools_response = client.get("/api/tools")
    host_response = client.get("/api/capabilities/tools")

    assert tools_response.status_code == 200
    assert tools_response.json() == []
    assert host_response.status_code == 200
    assert host_response.json()["available"] is False
    assert host_response.json()["status"] == "unavailable"


def test_frontdesk_isolates_host_descriptor_failure(app):
    from augmentum.capabilities.frontdesk import CapabilityContext, CapabilityFrontdesk

    class BrokenHost:
        id = "broken"
        title = "BrokenHost"
        order = 1
        endpoint_prefixes = ["/api/broken"]

        def describe(self, _context):
            raise RuntimeError("boom")

    frontdesk = CapabilityFrontdesk([BrokenHost()])
    data = frontdesk.inventory(CapabilityContext(app.state))

    assert data["hosts"]["broken"]["available"] is False
    assert data["hosts"]["broken"]["status"] == "error"
    assert data["hosts"]["broken"]["unavailable_reason"] == "descriptor_failed"


def test_unknown_capability_host_returns_404(client):
    response = client.get("/api/capabilities/unknown")

    assert response.status_code == 404
    assert response.json()["error"] == "Capability host not found"


def test_router_shape():
    from augmentum.proxy.capabilities_routes import router

    assert router.prefix == "/api/capabilities"
