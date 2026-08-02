"""Tests for MCP client integration — client, bridge, tool registry wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.mcp.bridge import (
    MCPToolWrapper,
    register_mcp_tools,
    unregister_mcp_tools,
)
from augmentum.mcp.client import MCPClientManager, MCPServerConnection, ToolInfo
from augmentum.tools.base import ToolCategory, ToolResult
from augmentum.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fake MCP types (mirror mcp.types just enough for testing without subprocess)
# ---------------------------------------------------------------------------


@dataclass
class FakeMCPTool:
    name: str = "echo"
    title: str | None = None
    description: str | None = "Echoes input"
    inputSchema: dict = field(default_factory=lambda: {
        "type": "object",
        "properties": {"text": {"type": "string"}},
    })
    outputSchema: dict | None = None
    icons: Any = None
    annotations: Any = None
    meta: Any = None
    execution: Any = None


@dataclass
class FakeTextContent:
    type: str = "text"
    text: str = ""
    annotations: Any = None
    meta: Any = None


@dataclass
class FakeCallToolResult:
    content: list = field(default_factory=list)
    structuredContent: Any = None
    isError: bool = False
    meta: Any = None


@dataclass
class FakeListToolsResult:
    tools: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper to create a mock MCPServerConnection
# ---------------------------------------------------------------------------


def _make_mock_connection(
    name: str = "test-server",
    tools: list[FakeMCPTool] | None = None,
) -> MCPServerConnection:
    """Create a mock server connection with a fake session."""
    if tools is None:
        tools = [FakeMCPTool()]

    session = AsyncMock()
    session.call_tool = AsyncMock(
        return_value=FakeCallToolResult(
            content=[FakeTextContent(text="hello world")]
        )
    )
    session.list_tools = AsyncMock(
        return_value=FakeListToolsResult(tools=tools)
    )

    return MCPServerConnection(
        name=name,
        session=session,
        tools=tools,
    )


# ===========================================================================
# TestMCPClientManager
# ===========================================================================


class TestMCPClientManager:
    """Tests for MCPClientManager connection and tool management."""

    def test_init_empty(self):
        mgr = MCPClientManager()
        assert mgr.server_count == 0
        assert mgr.connected_servers == []

    def test_list_all_tools_empty(self):
        mgr = MCPClientManager()
        assert mgr.list_all_tools() == []

    def test_list_all_tools_with_connection(self):
        mgr = MCPClientManager()
        tools = [
            FakeMCPTool(name="echo", description="Echo"),
            FakeMCPTool(name="add", description="Add numbers"),
        ]
        mgr._servers["calc"] = _make_mock_connection("calc", tools)

        result = mgr.list_all_tools()
        assert len(result) == 2
        assert result[0].name == "calc/echo"
        assert result[1].name == "calc/add"
        assert result[0].source == "calc"

    def test_connected_servers(self):
        mgr = MCPClientManager()
        mgr._servers["a"] = _make_mock_connection("a")
        mgr._servers["b"] = _make_mock_connection("b")
        assert set(mgr.connected_servers) == {"a", "b"}
        assert mgr.server_count == 2

    def test_get_server_tools(self):
        mgr = MCPClientManager()
        tools = [FakeMCPTool(name="t1"), FakeMCPTool(name="t2")]
        mgr._servers["srv"] = _make_mock_connection("srv", tools)
        result = mgr.get_server_tools("srv")
        assert len(result) == 2
        assert result[0].name == "t1"

    def test_get_server_tools_not_connected(self):
        mgr = MCPClientManager()
        with pytest.raises(ValueError, match="not connected"):
            mgr.get_server_tools("missing")

    async def test_call_tool_success(self):
        mgr = MCPClientManager()
        conn = _make_mock_connection("srv")
        mgr._servers["srv"] = conn

        result = await mgr.call_tool("srv", "echo", {"text": "hi"})
        assert result == "hello world"
        conn.session.call_tool.assert_awaited_once_with("echo", {"text": "hi"})

    async def test_call_tool_error(self):
        mgr = MCPClientManager()
        conn = _make_mock_connection("srv")
        conn.session.call_tool.return_value = FakeCallToolResult(
            content=[FakeTextContent(text="something went wrong")],
            isError=True,
        )
        mgr._servers["srv"] = conn

        with pytest.raises(RuntimeError, match="something went wrong"):
            await mgr.call_tool("srv", "echo", {})

    async def test_call_tool_server_not_connected(self):
        mgr = MCPClientManager()
        with pytest.raises(ValueError, match="not connected"):
            await mgr.call_tool("missing", "echo", {})

    async def test_call_tool_multiple_content_blocks(self):
        mgr = MCPClientManager()
        conn = _make_mock_connection("srv")
        conn.session.call_tool.return_value = FakeCallToolResult(
            content=[
                FakeTextContent(text="line1"),
                FakeTextContent(text="line2"),
            ]
        )
        mgr._servers["srv"] = conn
        result = await mgr.call_tool("srv", "echo", {})
        assert result == "line1\nline2"

    async def test_disconnect(self):
        mgr = MCPClientManager()
        conn = _make_mock_connection("srv")
        conn._session_exit = AsyncMock()
        conn._session_exit.__aexit__ = AsyncMock()
        conn._cm_exit = AsyncMock()
        conn._cm_exit.__aexit__ = AsyncMock()
        mgr._servers["srv"] = conn

        await mgr.disconnect("srv")
        assert "srv" not in mgr._servers
        assert mgr.server_count == 0

    async def test_disconnect_not_connected(self):
        mgr = MCPClientManager()
        with pytest.raises(ValueError, match="not connected"):
            await mgr.disconnect("missing")

    async def test_disconnect_all(self):
        mgr = MCPClientManager()
        for name in ("a", "b", "c"):
            conn = _make_mock_connection(name)
            conn._session_exit = None
            conn._cm_exit = None
            mgr._servers[name] = conn

        await mgr.disconnect_all()
        assert mgr.server_count == 0

    async def test_connect_stdio_duplicate_name(self):
        mgr = MCPClientManager()
        mgr._servers["dup"] = _make_mock_connection("dup")
        with pytest.raises(ValueError, match="already connected"):
            await mgr.connect_stdio("dup", "echo")


# ===========================================================================
# TestToolInfo
# ===========================================================================


class TestToolInfo:
    def test_tool_info_fields(self):
        info = ToolInfo(
            name="srv/echo",
            description="Echoes input",
            schema={"type": "object"},
            source="srv",
        )
        assert info.name == "srv/echo"
        assert info.source == "srv"


# ===========================================================================
# TestMCPToolWrapper
# ===========================================================================


class TestMCPToolWrapper:
    """Tests for the MCP-to-Augmentum tool bridge."""

    def _make_wrapper(self, tool_name="echo") -> MCPToolWrapper:
        mcp_tool = FakeMCPTool(name=tool_name, description="Test tool")
        client = MagicMock(spec=MCPClientManager)
        client.call_tool = AsyncMock(return_value="result text")
        return MCPToolWrapper(
            mcp_tool=mcp_tool,
            server_name="test-srv",
            client=client,
        )

    def test_name_is_namespaced(self):
        wrapper = self._make_wrapper("my_tool")
        assert wrapper.name == "test-srv/my_tool"

    def test_description(self):
        wrapper = self._make_wrapper()
        assert wrapper.description == "Test tool"

    def test_category_default(self):
        wrapper = self._make_wrapper()
        assert wrapper.category == ToolCategory.EXECUTE

    def test_custom_category(self):
        mcp_tool = FakeMCPTool()
        client = MagicMock()
        wrapper = MCPToolWrapper(mcp_tool, "srv", client, category=ToolCategory.SEARCH)
        assert wrapper.category == ToolCategory.SEARCH

    def test_input_schema(self):
        wrapper = self._make_wrapper()
        schema = wrapper.input_schema
        assert "properties" in schema
        assert "text" in schema["properties"]

    def test_server_name(self):
        wrapper = self._make_wrapper()
        assert wrapper.server_name == "test-srv"

    def test_remote_tool_name(self):
        wrapper = self._make_wrapper("calculate")
        assert wrapper.remote_tool_name == "calculate"

    async def test_execute_success(self):
        wrapper = self._make_wrapper()
        result = await wrapper.execute(text="hello")
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "result text"
        assert result.metadata["source"] == "mcp"
        assert result.metadata["server"] == "test-srv"
        wrapper._client.call_tool.assert_awaited_once_with("test-srv", "echo", {"text": "hello"})

    async def test_execute_runtime_error(self):
        wrapper = self._make_wrapper()
        wrapper._client.call_tool = AsyncMock(side_effect=RuntimeError("tool failed"))
        result = await wrapper.execute(text="hello")
        assert result.success is False
        assert "tool failed" in result.error

    async def test_execute_unexpected_error(self):
        wrapper = self._make_wrapper()
        wrapper._client.call_tool = AsyncMock(side_effect=ConnectionError("lost"))
        result = await wrapper.execute(text="hello")
        assert result.success is False
        assert "MCP tool error" in result.error

    def test_validate_input_always_true(self):
        wrapper = self._make_wrapper()
        assert wrapper.validate_input(text="x") is True


# ===========================================================================
# TestBridgeRegistration
# ===========================================================================


class TestBridgeRegistration:
    """Tests for register/unregister MCP tools in the Augmentum tool registry."""

    def _setup(self):
        tools = [
            FakeMCPTool(name="echo", description="Echo"),
            FakeMCPTool(name="add", description="Add"),
        ]
        client = MCPClientManager()
        client._servers["test-srv"] = _make_mock_connection("test-srv", tools)
        registry = ToolRegistry()
        return client, registry

    def test_register_mcp_tools(self):
        client, registry = self._setup()
        names = register_mcp_tools(client, "test-srv", registry)
        assert len(names) == 2
        assert "test-srv/echo" in names
        assert "test-srv/add" in names

        # Tools are now in the registry
        t = registry.get("test-srv/echo")
        assert t is not None
        assert isinstance(t, MCPToolWrapper)
        assert t.description == "Echo"

    def test_register_mcp_tools_appear_in_list(self):
        client, registry = self._setup()
        register_mcp_tools(client, "test-srv", registry)
        all_tools = registry.list_tools()
        mcp_names = [t.name for t in all_tools if isinstance(t, MCPToolWrapper)]
        assert set(mcp_names) == {"test-srv/echo", "test-srv/add"}

    def test_register_with_custom_category(self):
        client, registry = self._setup()
        register_mcp_tools(client, "test-srv", registry, category=ToolCategory.SEARCH)
        t = registry.get("test-srv/echo")
        assert t.category == ToolCategory.SEARCH

    def test_unregister_mcp_tools(self):
        client, registry = self._setup()
        register_mcp_tools(client, "test-srv", registry)
        assert registry.get("test-srv/echo") is not None

        removed = unregister_mcp_tools("test-srv", registry)
        assert len(removed) == 2
        assert registry.get("test-srv/echo") is None
        assert registry.get("test-srv/add") is None

    def test_unregister_only_removes_target_server(self):
        client, registry = self._setup()
        # Add tools from two servers
        tools2 = [FakeMCPTool(name="mul", description="Multiply")]
        client._servers["other-srv"] = _make_mock_connection("other-srv", tools2)
        register_mcp_tools(client, "test-srv", registry)
        register_mcp_tools(client, "other-srv", registry)

        removed = unregister_mcp_tools("test-srv", registry)
        assert len(removed) == 2
        # Other server's tools remain
        assert registry.get("other-srv/mul") is not None

    def test_unregister_empty_server(self):
        registry = ToolRegistry()
        removed = unregister_mcp_tools("nonexistent", registry)
        assert removed == []

    def test_mcp_tools_available_in_apply_phase(self):
        """MCP tools with EXECUTE category appear in the APPLY phase."""
        client, registry = self._setup()
        register_mcp_tools(client, "test-srv", registry)
        apply_tools = registry.get_for_phase("apply")
        mcp_apply = [t for t in apply_tools if isinstance(t, MCPToolWrapper)]
        assert len(mcp_apply) == 2

    def test_mcp_tools_not_in_assess_phase(self):
        """MCP tools should not appear in phases with no matching category."""
        client, registry = self._setup()
        register_mcp_tools(client, "test-srv", registry)
        assess_tools = registry.get_for_phase("assess")
        assert len(assess_tools) == 0

    async def test_registered_tool_executes(self):
        """End-to-end: register MCP tool, look it up, call execute()."""
        client, registry = self._setup()
        register_mcp_tools(client, "test-srv", registry)

        tool = registry.get("test-srv/echo")
        assert tool is not None

        # The mock session on the connection will return "hello world"
        result = await tool.execute(text="test input")
        assert result.success is True
        assert result.output == "hello world"


# ===========================================================================
# TestMCPClientConnectStdio (mocked subprocess)
# ===========================================================================


class TestMCPClientConnectStdio:
    """Test connect_stdio with mocked MCP SDK internals."""

    async def test_connect_stdio_mocked(self):
        """Verify connect_stdio wires up transport → session → tool discovery."""
        mgr = MCPClientManager()

        fake_tools = [
            FakeMCPTool(name="greet", description="Greet someone"),
        ]

        # Mock the stdio_client context manager
        fake_read = MagicMock()
        fake_write = MagicMock()
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__ = AsyncMock(return_value=(fake_read, fake_write))
        mock_transport_cm.__aexit__ = AsyncMock(return_value=False)

        # Mock ClientSession
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=FakeListToolsResult(tools=fake_tools)
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("augmentum.mcp.client.stdio_client", return_value=mock_transport_cm),
            patch("augmentum.mcp.client.ClientSession", return_value=mock_session),
        ):
            tools = await mgr.connect_stdio("greet-srv", "node", args=["server.js"])

        assert len(tools) == 1
        assert tools[0].name == "greet"
        assert mgr.server_count == 1
        assert "greet-srv" in mgr.connected_servers

        # Verify session was initialized
        mock_session.initialize.assert_awaited_once()

    async def test_connect_http_mocked(self):
        """Verify connect_http wires up transport → session → tool discovery."""
        mgr = MCPClientManager()

        fake_tools = [FakeMCPTool(name="lookup", description="Look up data")]

        fake_read = MagicMock()
        fake_write = MagicMock()
        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__ = AsyncMock(
            return_value=(fake_read, fake_write, lambda: None)
        )
        mock_transport_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=FakeListToolsResult(tools=fake_tools)
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # Patch the module-level import inside connect_http
        mock_http_module = MagicMock()
        mock_http_module.streamablehttp_client = MagicMock(return_value=mock_transport_cm)

        with (
            patch.dict("sys.modules", {"mcp.client.streamable_http": mock_http_module}),
            patch("augmentum.mcp.client.ClientSession", return_value=mock_session),
        ):
            tools = await mgr.connect_http(
                "remote-srv", "https://mcp.example.com/v1"
            )

        assert len(tools) == 1
        assert tools[0].name == "lookup"
        assert "remote-srv" in mgr.connected_servers

    async def test_connect_and_disconnect_full_cycle(self):
        """Connect → discover tools → register → call → disconnect → gone."""
        mgr = MCPClientManager()
        registry = ToolRegistry()

        fake_tools = [FakeMCPTool(name="echo", description="Echo")]

        mock_transport_cm = AsyncMock()
        mock_transport_cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_transport_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=FakeListToolsResult(tools=fake_tools)
        )
        mock_session.call_tool = AsyncMock(
            return_value=FakeCallToolResult(
                content=[FakeTextContent(text="echoed: hi")]
            )
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("augmentum.mcp.client.stdio_client", return_value=mock_transport_cm),
            patch("augmentum.mcp.client.ClientSession", return_value=mock_session),
        ):
            await mgr.connect_stdio("echo-srv", "echo-server")

        # Register tools into Augmentum registry
        register_mcp_tools(mgr, "echo-srv", registry)
        tool = registry.get("echo-srv/echo")
        assert tool is not None

        # Call the tool
        result = await tool.execute(text="hi")
        assert result.success is True
        assert result.output == "echoed: hi"

        # Disconnect
        await mgr.disconnect("echo-srv")
        assert mgr.server_count == 0

        # Unregister
        unregister_mcp_tools("echo-srv", registry)
        assert registry.get("echo-srv/echo") is None


# ===========================================================================
# TestMCPServerConfig
# ===========================================================================


class TestMCPServerConfig:
    """Tests for MCP configuration settings."""

    def test_mcp_enabled_default(self):
        # mcp_enabled defaults False — privacy-first opt-in. Flipping it on
        # exposes /mcp to authenticated MCP clients.
        from augmentum.config import Settings
        s = Settings()
        assert s.mcp_enabled is False
        assert s.mcp_servers == ""

    def test_mcp_servers_json_parsing(self):
        """Config value is a JSON string that can be parsed."""
        import json
        config_val = '[{"name":"test","command":"echo","args":["hello"]}]'
        servers = json.loads(config_val)
        assert len(servers) == 1
        assert servers[0]["name"] == "test"
        assert servers[0]["command"] == "echo"


# ===========================================================================
# Fake tool for MCP server tests
# ===========================================================================


class FakeAugmentumTool:
    """Minimal Augmentum Tool for testing the MCP server wrapper."""

    def __init__(self, name: str, desc: str, schema: dict | None = None):
        self._name = name
        self._desc = desc
        self._schema = schema or {"type": "object", "properties": {"input": {"type": "string"}}}
        self._last_kwargs: dict = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._desc

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return self._schema

    async def execute(self, **kwargs) -> ToolResult:
        self._last_kwargs = kwargs
        return ToolResult(success=True, output=f"executed {self._name}: {kwargs}")

    def validate_input(self, **kwargs) -> bool:
        return True


class FakeFailingTool(FakeAugmentumTool):
    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=False, output="", error="tool failed")


# ===========================================================================
# TestMCPServer
# ===========================================================================


class TestMCPServer:
    """Tests for the MCP server that exposes Augmentum tools."""

    def _make_registry(self, tools: list | None = None) -> ToolRegistry:
        registry = ToolRegistry()
        if tools is None:
            tools = [
                FakeAugmentumTool("echo", "Echo input back"),
                FakeAugmentumTool("calculate", "Perform calculations"),
            ]
        for t in tools:
            registry.register(t)
        return registry

    def test_create_mcp_server_basic(self):
        from augmentum.mcp.server import create_mcp_server
        registry = self._make_registry()
        mcp = create_mcp_server(registry)
        assert mcp is not None
        assert mcp.name == "augmentum"

    async def test_all_tools_exposed(self):
        from augmentum.mcp.server import create_mcp_server
        registry = self._make_registry()
        mcp = create_mcp_server(registry)
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "echo" in tool_names
        assert "calculate" in tool_names

    async def test_tool_descriptions_preserved(self):
        from augmentum.mcp.server import create_mcp_server
        registry = self._make_registry()
        mcp = create_mcp_server(registry)
        tools = await mcp.list_tools()
        echo_tool = next(t for t in tools if t.name == "echo")
        assert echo_tool.description == "Echo input back"

    async def test_memory_tools_registered_when_store_provided(self):
        from augmentum.mcp.server import create_mcp_server
        registry = self._make_registry()
        mock_store = MagicMock()
        mcp = create_mcp_server(registry, memory_store=mock_store)
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "memory_recall" in tool_names
        assert "memory_store" in tool_names
        assert "memory_count" in tool_names

    async def test_no_memory_tools_without_store(self):
        from augmentum.mcp.server import create_mcp_server
        registry = self._make_registry()
        mcp = create_mcp_server(registry, memory_store=None)
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        assert "memory_recall" not in tool_names

    def test_streamable_http_app_created(self):
        from augmentum.mcp.server import create_mcp_server
        registry = self._make_registry()
        mcp = create_mcp_server(registry)
        starlette_app = mcp.streamable_http_app()
        assert starlette_app is not None

    async def test_call_tool_success(self):
        from augmentum.mcp.server import create_mcp_server
        tool = FakeAugmentumTool("echo", "Echo")
        registry = self._make_registry([tool])
        mcp = create_mcp_server(registry)
        # call_tool returns (content_list, structured_content)
        content_list, _ = await mcp.call_tool("echo", {"input": "hello"})
        assert len(content_list) > 0
        text_parts = [c.text for c in content_list if hasattr(c, "text")]
        full_text = "\n".join(text_parts)
        assert "executed echo" in full_text

    async def test_call_tool_error_returns_error_text(self):
        from augmentum.mcp.server import create_mcp_server
        tool = FakeFailingTool("bad_tool", "Always fails")
        registry = self._make_registry([tool])
        mcp = create_mcp_server(registry)
        content_list, _ = await mcp.call_tool("bad_tool", {"input": "test"})
        text_parts = [c.text for c in content_list if hasattr(c, "text")]
        full_text = "\n".join(text_parts)
        assert "Error: tool failed" in full_text

    def test_mount_mcp_server(self):
        from fastapi import FastAPI

        from augmentum.mcp.server import create_mcp_server, mount_mcp_server
        app = FastAPI()
        registry = self._make_registry()
        mcp = create_mcp_server(registry)
        mount_mcp_server(app, mcp)
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert any("/mcp" in p for p in route_paths)

    async def test_multiple_tools_all_exposed(self):
        from augmentum.mcp.server import create_mcp_server
        tools = [FakeAugmentumTool(f"tool_{i}", f"Tool {i}") for i in range(5)]
        registry = self._make_registry(tools)
        mcp = create_mcp_server(registry)
        exposed = await mcp.list_tools()
        assert len(exposed) == 5


# ===========================================================================
# TestMCPRoutes (REST API)
# ===========================================================================


class TestMCPRoutes:
    """Tests for the MCP management REST API endpoints."""

    def _make_app(self, mcp_client=None):
        """Create a minimal FastAPI app with MCP routes.

        The mcp_routes admin gate (added in commit c79c5dd) reads
        ``request.scope["user"]`` — without the full auth middleware in
        place, we inject a tiny ASGI shim that stamps an admin user so
        the mutation endpoints reach their handlers instead of 401-ing
        on the guard.
        """
        from fastapi import FastAPI
        from starlette.types import ASGIApp, Receive, Scope, Send

        from augmentum.proxy.mcp_routes import router

        class _AdminUser:
            id = "test-admin"
            is_admin = True

        class _AdminScopeMiddleware:
            def __init__(self, app: ASGIApp) -> None:
                self.app = app

            async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
                if scope["type"] == "http":
                    scope["user"] = _AdminUser()
                await self.app(scope, receive, send)

        app = FastAPI()
        app.add_middleware(_AdminScopeMiddleware)
        app.include_router(router)
        app.state.mcp_client = mcp_client
        app.state.tool_registry = ToolRegistry()
        return app

    async def test_list_servers_disabled(self):
        from httpx import ASGITransport, AsyncClient
        app = self._make_app(mcp_client=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/mcp/servers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["servers"] == []

    async def test_list_servers_enabled_empty(self):
        from httpx import ASGITransport, AsyncClient
        mgr = MCPClientManager()
        app = self._make_app(mcp_client=mgr)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/mcp/servers")
        data = resp.json()
        assert data["enabled"] is True
        assert data["servers"] == []

    async def test_list_servers_with_connections(self):
        from httpx import ASGITransport, AsyncClient
        mgr = MCPClientManager()
        tools = [FakeMCPTool(name="t1", description="Tool 1")]
        mgr._servers["srv1"] = _make_mock_connection("srv1", tools)
        app = self._make_app(mcp_client=mgr)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/mcp/servers")
        data = resp.json()
        assert len(data["servers"]) == 1
        assert data["servers"][0]["name"] == "srv1"
        assert data["servers"][0]["tool_count"] == 1

    async def test_list_all_tools_endpoint(self):
        from httpx import ASGITransport, AsyncClient
        mgr = MCPClientManager()
        tools = [FakeMCPTool(name="echo"), FakeMCPTool(name="add")]
        mgr._servers["srv"] = _make_mock_connection("srv", tools)
        app = self._make_app(mcp_client=mgr)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/mcp/tools")
        data = resp.json()
        assert len(data["tools"]) == 2
        names = [t["name"] for t in data["tools"]]
        assert "srv/echo" in names
        assert "srv/add" in names

    async def test_list_tools_no_client(self):
        from httpx import ASGITransport, AsyncClient
        app = self._make_app(mcp_client=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/v1/mcp/tools")
        assert resp.json()["tools"] == []

    async def test_connect_missing_name(self):
        from httpx import ASGITransport, AsyncClient
        mgr = MCPClientManager()
        app = self._make_app(mcp_client=mgr)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/v1/mcp/connect", json={"command": "echo"})
        assert resp.status_code == 400
        assert "name" in resp.json()["error"]

    async def test_connect_missing_transport(self):
        from httpx import ASGITransport, AsyncClient
        mgr = MCPClientManager()
        app = self._make_app(mcp_client=mgr)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/v1/mcp/connect", json={"name": "test"})
        assert resp.status_code == 400
        assert "command" in resp.json()["error"] or "url" in resp.json()["error"]

    async def test_connect_disabled(self):
        from httpx import ASGITransport, AsyncClient
        app = self._make_app(mcp_client=None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/v1/mcp/connect", json={"name": "x", "command": "y"})
        assert resp.status_code == 400

    async def test_disconnect_not_connected(self):
        from httpx import ASGITransport, AsyncClient
        mgr = MCPClientManager()
        app = self._make_app(mcp_client=mgr)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/v1/mcp/servers/nonexistent")
        assert resp.status_code == 404

    async def test_disconnect_success(self):
        from httpx import ASGITransport, AsyncClient
        mgr = MCPClientManager()
        conn = _make_mock_connection("srv")
        conn._session_exit = None
        conn._cm_exit = None
        mgr._servers["srv"] = conn
        app = self._make_app(mcp_client=mgr)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/v1/mcp/servers/srv")
        assert resp.status_code == 200
        assert resp.json()["status"] == "disconnected"
        assert mgr.server_count == 0
