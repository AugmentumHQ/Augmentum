"""The /mcp/ mount must actually serve requests (task-group init bug).

FastAPI's ``app.mount`` does not run a mounted sub-app's lifespan, and
FastMCP's Streamable-HTTP transport only initializes its task group
inside ``session_manager.run()``. Until 2026-07-18 the mount had never
served a single request — every JSON-RPC call 500'd with "Task group is
not initialized". These tests exercise the exact wiring pattern
server.py uses now: enter the session-manager context in the parent
lifespan, exit it on shutdown.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from augmentum.mcp.server import create_mcp_server, mount_mcp_server
from augmentum.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mcp_initialize_roundtrip_through_mount():
    # httpx's ASGITransport does not run FastAPI lifespans, so enter the
    # session-manager context directly — the same enter/exit pair
    # server.py's lifespan performs around the mount.
    mcp_srv = create_mcp_server(ToolRegistry())
    app = FastAPI()
    mount_mcp_server(app, mcp_srv)

    ctx = mcp_srv.session_manager.run()
    await ctx.__aenter__()
    try:
        await _roundtrip(app)
    finally:
        await ctx.__aexit__(None, None, None)


async def _roundtrip(app: FastAPI) -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t",
    ) as client, client.stream(
            "POST", "/mcp/",
            json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "1.0"},
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
        ) as resp:
            assert resp.status_code == 200, await resp.aread()
            body = (await resp.aread()).decode("utf-8", "replace")
    assert "serverInfo" in body and "augmentum" in body


@pytest.mark.asyncio
async def test_mcp_500s_without_session_manager_run():
    """Documents the failure mode the lifespan wiring exists to prevent —
    if this ever starts passing, the transport no longer needs the
    parent-lifespan context and the server.py wiring can be simplified."""
    mcp_srv = create_mcp_server(ToolRegistry())
    app = FastAPI()
    mount_mcp_server(app, mcp_srv)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t",
    ) as client:
        with pytest.raises((RuntimeError, ExceptionGroup)):
            resp = await client.post(
                "/mcp/",
                json={
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "probe", "version": "1.0"},
                    },
                },
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
            )
            # Some transport versions surface it as a 500 instead of raising.
            assert resp.status_code == 500
            raise RuntimeError("500")
