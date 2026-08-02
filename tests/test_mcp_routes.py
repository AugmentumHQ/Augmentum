"""Tests for mcp_routes.py — MCP server management endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _mock_mcp_client():
    client = MagicMock()
    client.connected_servers = ["test-server"]
    mock_tool = MagicMock(name="test_tool", description="A test tool")
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.schema = {}
    mock_tool.source = "test-server"
    client.get_server_tools = MagicMock(return_value=[mock_tool])
    client.list_all_tools = MagicMock(return_value=[mock_tool])
    client.connect_http = AsyncMock(return_value=[mock_tool])
    client.disconnect = AsyncMock()
    client.ping_server = AsyncMock(return_value=(True, ""))
    return client


class TestListServers:
    def test_list_no_client(self, client):
        resp = client.get("/v1/mcp/servers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["servers"] == []

    def test_list_success(self, app, client):
        app.state.mcp_client = _mock_mcp_client()
        resp = client.get("/v1/mcp/servers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert len(data["servers"]) == 1
        assert data["servers"][0]["name"] == "test-server"


class TestConnectServer:
    def test_connect_no_client(self, client):
        resp = client.post("/v1/mcp/connect", json={"name": "test"})
        assert resp.status_code == 400

    def test_connect_missing_name(self, app, client):
        app.state.mcp_client = _mock_mcp_client()
        resp = client.post("/v1/mcp/connect", json={})
        assert resp.status_code == 400

    def test_connect_stdio_blocked(self, app, client):
        app.state.mcp_client = _mock_mcp_client()
        resp = client.post(
            "/v1/mcp/connect",
            json={"name": "bad", "command": "python -m evil"},
        )
        assert resp.status_code == 403

    def test_connect_missing_url(self, app, client):
        app.state.mcp_client = _mock_mcp_client()
        resp = client.post("/v1/mcp/connect", json={"name": "test"})
        assert resp.status_code == 400
        assert "url" in resp.json()["error"].lower()


class TestDisconnectServer:
    def test_disconnect_no_client(self, client):
        resp = client.delete("/v1/mcp/servers/test-server")
        assert resp.status_code == 400

    def test_disconnect_not_found(self, app, client):
        mcp = _mock_mcp_client()
        mcp.disconnect = AsyncMock(side_effect=ValueError("Not found"))
        app.state.mcp_client = mcp
        resp = client.delete("/v1/mcp/servers/nonexistent")
        assert resp.status_code == 404


class TestListTools:
    def test_list_tools_no_client(self, client):
        resp = client.get("/v1/mcp/tools")
        assert resp.status_code == 200
        assert resp.json()["tools"] == []

    def test_list_tools_success(self, app, client):
        app.state.mcp_client = _mock_mcp_client()
        resp = client.get("/v1/mcp/tools")
        assert resp.status_code == 200
        tools = resp.json()["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"


class TestPersistence:
    """Server configs added via /connect must survive restart."""

    def test_connect_persists_to_settings_store(self, app, client, monkeypatch):
        from augmentum.config import settings as global_settings
        monkeypatch.setattr(global_settings, "mcp_servers", "", raising=False)
        # Stub the SSRF validator so unit tests don't hit DNS.
        from augmentum.utils import safe_http
        monkeypatch.setattr(safe_http.SafeHttpClient, "_validate_url", lambda self, url: "example.com")
        monkeypatch.setattr(safe_http.SafeHttpClient, "_check_resolved_ips", AsyncMock())
        app.state.mcp_client = _mock_mcp_client()

        captured: dict[str, str] = {}
        async def _set(key, value):
            captured[key] = value
        app.state.settings_store = MagicMock(set=_set)

        resp = client.post(
            "/v1/mcp/connect",
            json={"name": "remote-mcp", "url": "https://example.com/mcp"},
        )
        assert resp.status_code == 200, resp.text
        assert "mcp_servers" in captured
        import json as _json
        persisted = _json.loads(captured["mcp_servers"])
        assert any(s["name"] == "remote-mcp" and s["url"] == "https://example.com/mcp" for s in persisted)

    def test_connect_replaces_same_name(self, app, client, monkeypatch):
        from augmentum.config import settings as global_settings
        # Pre-populate with an entry that should be replaced
        monkeypatch.setattr(
            global_settings, "mcp_servers",
            '[{"name":"remote-mcp","url":"https://old.example.com/mcp"}]',
            raising=False,
        )
        from augmentum.utils import safe_http
        monkeypatch.setattr(safe_http.SafeHttpClient, "_validate_url", lambda self, url: "example.com")
        monkeypatch.setattr(safe_http.SafeHttpClient, "_check_resolved_ips", AsyncMock())
        app.state.mcp_client = _mock_mcp_client()

        captured: dict[str, str] = {}
        async def _set(key, value):
            captured[key] = value
        app.state.settings_store = MagicMock(set=_set)

        resp = client.post(
            "/v1/mcp/connect",
            json={"name": "remote-mcp", "url": "https://new.example.com/mcp"},
        )
        assert resp.status_code == 200, resp.text
        import json as _json
        persisted = _json.loads(captured["mcp_servers"])
        urls = [s["url"] for s in persisted if s["name"] == "remote-mcp"]
        assert urls == ["https://new.example.com/mcp"]  # no duplicate, replaced

    def test_disconnect_removes_from_settings_store(self, app, client, monkeypatch):
        from augmentum.config import settings as global_settings
        monkeypatch.setattr(
            global_settings, "mcp_servers",
            '[{"name":"keepme","url":"https://a"},{"name":"test-server","url":"https://b"}]',
            raising=False,
        )
        app.state.mcp_client = _mock_mcp_client()

        captured: dict[str, str] = {}
        async def _set(key, value):
            captured[key] = value
        app.state.settings_store = MagicMock(set=_set)

        resp = client.delete("/v1/mcp/servers/test-server")
        assert resp.status_code == 200, resp.text
        import json as _json
        persisted = _json.loads(captured["mcp_servers"])
        assert [s["name"] for s in persisted] == ["keepme"]
