"""Bridge between MCP tools and the Augmentum tool registry.

MCPToolWrapper makes each MCP tool look like a native Augmentum Tool,
so the UARF engine can select and call MCP tools exactly like built-in tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

    from augmentum.mcp.client import MCPClientManager

log = get_logger(__name__)


# MCP tools get registered in the APPLY category by default, making them
# available during UARF's APPLY phase (where general-purpose tool use happens).
_DEFAULT_CATEGORY = ToolCategory.EXECUTE


class MCPToolWrapper(Tool):
    """Wraps a single MCP server tool as an Augmentum Tool instance."""

    def __init__(
        self,
        mcp_tool: MCPTool,
        server_name: str,
        client: MCPClientManager,
        category: ToolCategory = _DEFAULT_CATEGORY,
    ) -> None:
        self._mcp_tool = mcp_tool
        self._server_name = server_name
        self._client = client
        self._category = category

    @property
    def name(self) -> str:
        """Namespaced tool name: ``server_name/tool_name``."""
        return f"{self._server_name}/{self._mcp_tool.name}"

    @property
    def description(self) -> str:
        return self._mcp_tool.description or f"MCP tool from {self._server_name}"

    @property
    def category(self) -> ToolCategory:
        return self._category

    @property
    def input_schema(self) -> dict:
        """Return the MCP tool's JSON Schema for LLM tool-use."""
        return self._mcp_tool.inputSchema if self._mcp_tool.inputSchema else {}

    @property
    def server_name(self) -> str:
        return self._server_name

    @property
    def remote_tool_name(self) -> str:
        """The original tool name on the MCP server."""
        return self._mcp_tool.name

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the MCP tool via the client manager."""
        try:
            result_text = await self._client.call_tool(
                self._server_name, self._mcp_tool.name, kwargs
            )
            return ToolResult(
                success=True,
                output=result_text,
                metadata={
                    "source": "mcp",
                    "server": self._server_name,
                    "tool": self._mcp_tool.name,
                },
            )
        except RuntimeError as exc:
            return ToolResult(
                success=False,
                output="",
                error=str(exc),
                metadata={
                    "source": "mcp",
                    "server": self._server_name,
                    "tool": self._mcp_tool.name,
                },
            )
        except Exception as exc:
            log.error(
                "mcp_tool_execution_failed",
                server=self._server_name,
                tool=self._mcp_tool.name,
                error=str(exc),
                exc_info=True,
            )
            return ToolResult(
                success=False,
                output="",
                error=f"MCP tool error: {exc}",
                metadata={
                    "source": "mcp",
                    "server": self._server_name,
                    "tool": self._mcp_tool.name,
                },
            )


def register_mcp_tools(
    client: MCPClientManager,
    server_name: str,
    tool_registry: Any,
    category: ToolCategory = _DEFAULT_CATEGORY,
) -> list[str]:
    """Register all tools from an MCP server into the Augmentum tool registry.

    Returns the list of registered tool names.
    """
    from augmentum.tools.registry import ToolRegistry

    assert isinstance(tool_registry, ToolRegistry)

    mcp_tools = client.get_server_tools(server_name)
    registered: list[str] = []

    for mcp_tool in mcp_tools:
        wrapper = MCPToolWrapper(mcp_tool, server_name, client, category)
        tool_registry.register(wrapper)
        registered.append(wrapper.name)

    log.info(
        "mcp_tools_registered",
        server=server_name,
        count=len(registered),
        tools=registered,
    )
    return registered


def unregister_mcp_tools(server_name: str, tool_registry: Any) -> list[str]:
    """Remove all MCP tools from a specific server from the registry.

    Returns the list of removed tool names.
    """
    from augmentum.tools.registry import ToolRegistry

    assert isinstance(tool_registry, ToolRegistry)

    to_remove = [
        name
        for name, tool in tool_registry._tools.items()
        if isinstance(tool, MCPToolWrapper) and tool.server_name == server_name
    ]

    for name in to_remove:
        del tool_registry._tools[name]

    log.info(
        "mcp_tools_unregistered",
        server=server_name,
        count=len(to_remove),
        tools=to_remove,
    )
    return to_remove
