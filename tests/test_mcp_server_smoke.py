"""Smoke test for the Augmentum MCP server surface.

Proves ``create_mcp_server()`` exposes the built-in tools + the user-scoped
memory tools that MCP clients (Claude Desktop/Code, Cursor) consume — the
surface the integration recipe (docs/connect-to-claude-mcp.md) and the
``/api/capabilities`` ``mcp`` endpoint depend on. Lightweight: a fake tool
registry + a non-None memory store, no live HTTP handshake.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from augmentum.mcp.server import create_mcp_server, mount_mcp_server


class _FakeResult:
    success = True
    output = "ok"
    error = ""


class _FakeTool:
    name = "web_search"
    description = "Search the web."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    async def execute(self, **_kwargs):
        return _FakeResult()


class _FakeRegistry:
    def list_tools(self):
        return [_FakeTool()]


async def _tool_names(mcp) -> set[str]:
    tools = await mcp.list_tools()
    return {t.name for t in tools}


@pytest.mark.asyncio
async def test_mcp_server_exposes_builtin_and_memory_tools():
    store = MagicMock()  # non-None → memory tools get registered
    mcp = create_mcp_server(_FakeRegistry(), store)
    names = await _tool_names(mcp)
    # built-in tool from the registry
    assert "web_search" in names
    # user-scoped memory tools — the differentiated surface for MCP clients
    assert "memory_recall" in names
    assert "memory_store" in names


@pytest.mark.asyncio
async def test_mcp_server_skips_memory_tools_without_store():
    mcp = create_mcp_server(_FakeRegistry(), memory_store=None)
    names = await _tool_names(mcp)
    assert "web_search" in names
    assert "memory_recall" not in names  # no store → no memory tools


def test_mount_serves_clean_mcp_path():
    from fastapi import FastAPI

    mcp = create_mcp_server(_FakeRegistry(), memory_store=None)
    # The inner Streamable-HTTP app must serve at ROOT so the /mcp mount yields
    # the clean /mcp/ URL — not the doubled /mcp/mcp/ that 404s the intuitive
    # path. Pins the streamable_http_path="/" fix in create_mcp_server.
    inner_paths = [getattr(r, "path", "") for r in mcp.streamable_http_app().routes]
    assert "/" in inner_paths, f"expected root-served MCP app, got {inner_paths}"

    app = FastAPI()
    mount_mcp_server(app, mcp)
    # The recipe + /api/capabilities advertise /mcp/ — assert it's mounted there.
    assert any(getattr(r, "path", "") == "/mcp" for r in app.routes)
