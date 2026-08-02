"""Tests for augmentum/mcp/ — MCP server, client, and bridge modules."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from augmentum.mcp.bridge import MCPToolWrapper, _DEFAULT_CATEGORY
from augmentum.mcp.client import MCPClientManager, MCPServerConnection, ToolInfo
from augmentum.tools.base import ToolCategory, ToolResult


class TestMCPClientManager:
    """Verify MCPClientManager construction and tool listing."""

    def test_constructs_empty(self):
        mgr = MCPClientManager()
        assert mgr.server_count == 0
        assert mgr.connected_servers == []

    def test_list_all_tools_empty(self):
        mgr = MCPClientManager()
        assert mgr.list_all_tools() == []

    def test_duplicate_connect_raises(self):
        mgr = MCPClientManager()
        # Inject a fake server
        mgr._servers["test"] = MagicMock(spec=MCPServerConnection)
        import pytest
        with pytest.raises(ValueError, match="already connected"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                mgr.connect_stdio("test", "echo")
            )

    def test_disconnect_nonexistent_raises(self):
        mgr = MCPClientManager()
        import pytest
        with pytest.raises(ValueError, match="not connected"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(mgr.disconnect("nope"))

    def test_list_tools_with_injected_server(self):
        mgr = MCPClientManager()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}
        conn = MCPServerConnection(
            name="myserver",
            session=MagicMock(),
            tools=[mock_tool],
        )
        mgr._servers["myserver"] = conn
        tools = mgr.list_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "myserver/test_tool"
        assert tools[0].source == "myserver"

    def test_connected_servers_property(self):
        mgr = MCPClientManager()
        mgr._servers["a"] = MagicMock()
        mgr._servers["b"] = MagicMock()
        assert sorted(mgr.connected_servers) == ["a", "b"]
        assert mgr.server_count == 2


class TestToolInfo:
    """Verify ToolInfo dataclass."""

    def test_fields(self):
        info = ToolInfo(name="srv/tool", description="desc", schema={}, source="srv")
        assert info.name == "srv/tool"
        assert info.source == "srv"


class TestMCPToolWrapper:
    """Verify bridge wraps MCP tools as Augmentum tools."""

    def _make_wrapper(self):
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search the web"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        mock_client = MagicMock(spec=MCPClientManager)
        return MCPToolWrapper(mock_tool, "external", mock_client)

    def test_name_is_namespaced(self):
        wrapper = self._make_wrapper()
        assert wrapper.name == "external/search"

    def test_description(self):
        wrapper = self._make_wrapper()
        assert wrapper.description == "Search the web"

    def test_category_default(self):
        wrapper = self._make_wrapper()
        assert wrapper.category == _DEFAULT_CATEGORY

    def test_input_schema(self):
        wrapper = self._make_wrapper()
        schema = wrapper.input_schema
        assert "properties" in schema
        assert "query" in schema["properties"]

    def test_server_name(self):
        wrapper = self._make_wrapper()
        assert wrapper.server_name == "external"

    def test_remote_tool_name(self):
        wrapper = self._make_wrapper()
        assert wrapper.remote_tool_name == "search"

    async def test_execute_success(self):
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search"
        mock_tool.inputSchema = {}
        mock_client = MagicMock(spec=MCPClientManager)
        mock_client.call_tool = AsyncMock(return_value="result text")
        wrapper = MCPToolWrapper(mock_tool, "srv", mock_client)
        result = await wrapper.execute(query="test")
        assert result.success is True
        assert result.output == "result text"

    async def test_execute_runtime_error(self):
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search"
        mock_tool.inputSchema = {}
        mock_client = MagicMock(spec=MCPClientManager)
        mock_client.call_tool = AsyncMock(side_effect=RuntimeError("MCP error"))
        wrapper = MCPToolWrapper(mock_tool, "srv", mock_client)
        result = await wrapper.execute(query="test")
        assert result.success is False
        assert "MCP error" in result.error


class TestMCPServerUserScoping:
    """Verify _user_id_from_ctx extracts user identity from the MCP request scope.

    The MCP server-side memory tools must refuse to operate without an
    authenticated user — protects against the 'shared global memory_store
    leaks across tenants' regression.
    """

    def _make_ctx(self, scope_user):
        ctx = MagicMock()
        request = MagicMock()
        request.scope = {"user": scope_user}
        ctx.request_context.request = request
        return ctx

    def test_none_ctx_returns_empty(self):
        from augmentum.mcp.server import _user_id_from_ctx
        assert _user_id_from_ctx(None) == ""

    def test_no_user_in_scope_returns_empty(self):
        from augmentum.mcp.server import _user_id_from_ctx
        ctx = self._make_ctx(None)
        assert _user_id_from_ctx(ctx) == ""

    def test_user_with_id_is_extracted(self):
        from augmentum.mcp.server import _user_id_from_ctx
        fake_user = MagicMock()
        fake_user.id = "usr_abc123"
        ctx = self._make_ctx(fake_user)
        assert _user_id_from_ctx(ctx) == "usr_abc123"

    def test_missing_request_attribute(self):
        from augmentum.mcp.server import _user_id_from_ctx
        ctx = MagicMock()
        # Simulate request_context being unavailable
        type(ctx).request_context = property(lambda self: (_ for _ in ()).throw(ValueError("no ctx")))
        assert _user_id_from_ctx(ctx) == ""

    def test_request_is_none(self):
        from augmentum.mcp.server import _user_id_from_ctx
        ctx = MagicMock()
        ctx.request_context.request = None
        assert _user_id_from_ctx(ctx) == ""


class TestMCPServerResources:
    """Verify _register_resources registers character + knowledge pack templates."""

    def _make_server(self, backend=None, pack_manager=None):
        """Build a real FastMCP server wired to the resource registration path."""
        from augmentum.mcp.server import create_mcp_server
        from augmentum.tools.registry import ToolRegistry

        app = MagicMock()
        app.state.backend = backend
        app.state.pack_manager = pack_manager
        return create_mcp_server(
            tool_registry=ToolRegistry(),
            memory_store=None,
            app=app,
        )

    def test_resource_templates_registered(self):
        srv = self._make_server()
        # FastMCP stores templates by URI pattern
        keys = list(srv._resource_manager._templates.keys())
        assert "augmentum://characters/{character_id}" in keys
        assert "augmentum://knowledge/{pack_id}" in keys

    def test_listing_tools_registered(self):
        srv = self._make_server()
        tool_names = [t.name for t in srv._tool_manager._tools.values()]
        assert "list_character_cards" in tool_names
        assert "list_knowledge_packs" in tool_names

    def test_omitting_app_skips_resources(self):
        """Without ``app=...`` the server still works but exposes no resources."""
        from augmentum.mcp.server import create_mcp_server
        from augmentum.tools.registry import ToolRegistry

        srv = create_mcp_server(tool_registry=ToolRegistry(), memory_store=None)
        # No character/knowledge templates registered
        keys = list(srv._resource_manager._templates.keys())
        assert all("augmentum://" not in k for k in keys)
        # No listing tools either
        tool_names = [t.name for t in srv._tool_manager._tools.values()]
        assert "list_character_cards" not in tool_names


class TestMCPServerPrompts:
    """Verify _register_prompts registers preset + modular composer prompts."""

    def _make_server(self):
        from augmentum.mcp.server import create_mcp_server
        from augmentum.tools.registry import ToolRegistry

        app = MagicMock()
        app.state.backend = None
        app.state.pack_manager = None
        return create_mcp_server(
            tool_registry=ToolRegistry(),
            memory_store=None,
            app=app,
        )

    def test_prompts_registered(self):
        srv = self._make_server()
        prompts = list(srv._prompt_manager._prompts.keys())
        assert "apply_prompt_preset" in prompts
        assert "compose_modular_prompt" in prompts

    def test_list_presets_tool_registered(self):
        srv = self._make_server()
        tool_names = [t.name for t in srv._tool_manager._tools.values()]
        assert "list_prompt_presets" in tool_names

    def test_omitting_app_skips_prompts(self):
        from augmentum.mcp.server import create_mcp_server
        from augmentum.tools.registry import ToolRegistry

        srv = create_mcp_server(tool_registry=ToolRegistry(), memory_store=None)
        prompts = list(srv._prompt_manager._prompts.keys())
        assert "apply_prompt_preset" not in prompts
        assert "compose_modular_prompt" not in prompts


class TestMCPDefaultDisabled:
    """The mcp_enabled setting defaults to False — privacy-first opt-in."""

    def test_default_is_false(self):
        from augmentum.config import Settings
        # Construct a fresh Settings instance to see the declared default,
        # bypassing any env-var overrides or settings_store restoration that
        # may have mutated the module-level singleton.
        fresh = Settings()
        assert fresh.mcp_enabled is False, (
            "mcp_enabled MUST default to False — flipping it on exposes "
            "/mcp and connects to external servers. Explicit admin opt-in only."
        )
