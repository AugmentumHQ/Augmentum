"""MCP client manager — connects to external MCP servers and discovers tools."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Implementation
from mcp.types import Tool as MCPTool

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Timeout constants for MCP operations
_MCP_INIT_TIMEOUT: float = 30.0  # session.initialize() and list_tools()
_MCP_CALL_TIMEOUT: float = 60.0  # call_tool()


@dataclass
class MCPServerConnection:
    """Tracks a single connected MCP server."""

    name: str
    session: ClientSession
    tools: list[MCPTool] = field(default_factory=list)
    # Keep references to the transport streams for cleanup
    _read_stream: Any = field(default=None, repr=False)
    _write_stream: Any = field(default=None, repr=False)
    _cm_exit: Any = field(default=None, repr=False)
    _session_exit: Any = field(default=None, repr=False)


@dataclass
class ToolInfo:
    """Flattened view of an MCP tool for listing/display."""

    name: str
    description: str
    schema: dict
    source: str  # server name


class MCPClientManager:
    """Manages connections to external MCP servers.

    Supports stdio transport (subprocess-based MCP servers) and
    Streamable HTTP transport (remote MCP servers).
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConnection] = {}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect_stdio(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> list[MCPTool]:
        """Connect to an MCP server via stdio transport.

        Args:
            name: Unique identifier for this server connection.
            command: The executable to run (e.g. "npx", "python").
            args: Arguments to pass to the command.
            env: Optional environment variables for the subprocess.

        Returns:
            List of tools discovered on the server.
        """
        if name in self._servers:
            raise ValueError(f"Server '{name}' is already connected")

        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )

        # Enter the stdio_client context manager manually so we can
        # keep the connection open until disconnect() is called.
        cm = stdio_client(params, errlog=sys.stderr)
        read_stream, write_stream = await cm.__aenter__()

        # Create and initialize the session
        session = ClientSession(
            read_stream,
            write_stream,
            client_info=Implementation(name="augmentum", version="0.1.0"),
        )
        await session.__aenter__()

        try:
            await asyncio.wait_for(session.initialize(), timeout=_MCP_INIT_TIMEOUT)
            tools_result = await asyncio.wait_for(
                session.list_tools(), timeout=_MCP_INIT_TIMEOUT,
            )
        except Exception:
            # Rollback: clean up session and transport on init failure
            try:
                await session.__aexit__(None, None, None)
            except Exception as cleanup_exc:
                log.debug(
                    "mcp_stdio_session_rollback_failed",
                    server=name,
                    error=str(cleanup_exc),
                )
            try:
                await cm.__aexit__(None, None, None)
            except Exception as cleanup_exc:
                log.debug(
                    "mcp_stdio_transport_rollback_failed",
                    server=name,
                    error=str(cleanup_exc),
                )
            raise

        tools = tools_result.tools

        conn = MCPServerConnection(
            name=name,
            session=session,
            tools=tools,
            _read_stream=read_stream,
            _write_stream=write_stream,
            _cm_exit=cm,
            _session_exit=session,
        )
        self._servers[name] = conn

        log.info(
            "mcp_server_connected",
            server=name,
            transport="stdio",
            tool_count=len(tools),
            tools=[t.name for t in tools],
        )
        return tools

    async def connect_http(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> list[MCPTool]:
        """Connect to an MCP server via Streamable HTTP transport.

        Args:
            name: Unique identifier for this server connection.
            url: The HTTP(S) URL of the MCP server.
            headers: Optional HTTP headers (e.g. auth tokens).

        Returns:
            List of tools discovered on the server.
        """
        if name in self._servers:
            raise ValueError(f"Server '{name}' is already connected")

        from mcp.client.streamable_http import streamablehttp_client

        cm = streamablehttp_client(url=url, headers=headers)
        read_stream, write_stream, _get_session_id = await cm.__aenter__()

        session = ClientSession(
            read_stream,
            write_stream,
            client_info=Implementation(name="augmentum", version="0.1.0"),
        )
        await session.__aenter__()

        try:
            await asyncio.wait_for(session.initialize(), timeout=_MCP_INIT_TIMEOUT)
            tools_result = await asyncio.wait_for(
                session.list_tools(), timeout=_MCP_INIT_TIMEOUT,
            )
        except Exception:
            # Rollback: clean up session and transport on init failure
            try:
                await session.__aexit__(None, None, None)
            except Exception as cleanup_exc:
                log.debug(
                    "mcp_http_session_rollback_failed",
                    server=name,
                    error=str(cleanup_exc),
                )
            try:
                await cm.__aexit__(None, None, None)
            except Exception as cleanup_exc:
                log.debug(
                    "mcp_http_transport_rollback_failed",
                    server=name,
                    error=str(cleanup_exc),
                )
            raise

        tools = tools_result.tools

        conn = MCPServerConnection(
            name=name,
            session=session,
            tools=tools,
            _read_stream=read_stream,
            _write_stream=write_stream,
            _cm_exit=cm,
            _session_exit=session,
        )
        self._servers[name] = conn

        log.info(
            "mcp_server_connected",
            server=name,
            transport="http",
            tool_count=len(tools),
            tools=[t.name for t in tools],
        )
        return tools

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server and clean up resources."""
        conn = self._servers.pop(name, None)
        if conn is None:
            raise ValueError(f"Server '{name}' is not connected")

        # Exit session context, then transport context
        try:
            if conn._session_exit:
                await conn._session_exit.__aexit__(None, None, None)
        except Exception:
            log.debug("mcp_session_close_error", server=name, exc_info=True)

        try:
            if conn._cm_exit:
                await conn._cm_exit.__aexit__(None, None, None)
        except Exception:
            log.debug("mcp_transport_close_error", server=name, exc_info=True)

        log.info("mcp_server_disconnected", server=name)

    async def disconnect_all(self) -> None:
        """Disconnect from all connected servers."""
        names = list(self._servers.keys())
        for name in names:
            try:
                await self.disconnect(name)
            except Exception:
                log.warning("mcp_disconnect_failed", server=name, exc_info=True)

    # ------------------------------------------------------------------
    # Tool operations
    # ------------------------------------------------------------------

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """Call a tool on a connected MCP server.

        Returns the text content of the result.
        """
        conn = self._servers.get(server_name)
        if conn is None:
            raise ValueError(f"Server '{server_name}' is not connected")

        result = await asyncio.wait_for(
            conn.session.call_tool(tool_name, arguments or {}),
            timeout=_MCP_CALL_TIMEOUT,
        )

        if result.isError:
            # Concatenate error content
            parts = []
            for block in result.content:
                text = getattr(block, "text", None)
                if text:
                    parts.append(text)
            error_text = "\n".join(parts) if parts else "Unknown MCP tool error"
            raise RuntimeError(error_text)

        # Extract text content from result blocks
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)

    def list_all_tools(self) -> list[ToolInfo]:
        """List tools from all connected servers.

        Tool names are namespaced as ``server_name/tool_name``.
        """
        all_tools: list[ToolInfo] = []
        for name, conn in self._servers.items():
            for tool in conn.tools:
                all_tools.append(
                    ToolInfo(
                        name=f"{name}/{tool.name}",
                        description=tool.description or "",
                        schema=tool.inputSchema if tool.inputSchema else {},
                        source=name,
                    )
                )
        return all_tools

    def get_server_tools(self, server_name: str) -> list[MCPTool]:
        """Get tools for a specific server."""
        conn = self._servers.get(server_name)
        if conn is None:
            raise ValueError(f"Server '{server_name}' is not connected")
        return conn.tools

    @property
    def connected_servers(self) -> list[str]:
        """Names of all connected servers."""
        return list(self._servers.keys())

    @property
    def server_count(self) -> int:
        return len(self._servers)

    async def ping_server(self, name: str, timeout: float = 3.0) -> tuple[bool, str]:
        """Probe a connected server with a lightweight list_tools call.

        Returns ``(healthy, error_detail)``. A healthy server returns
        ``(True, "")``. Unreachable / errored servers return
        ``(False, "<truncated error>")``. Never raises.
        """
        conn = self._servers.get(name)
        if conn is None:
            return False, "not connected"
        try:
            await asyncio.wait_for(conn.session.list_tools(), timeout=timeout)
            return True, ""
        except TimeoutError:
            return False, f"timeout after {timeout}s"
        except Exception as exc:
            return False, str(exc)[:200]
