"""MCP management API — connect/disconnect MCP servers, list tools."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.auth.guards import require_admin
from augmentum.config import settings
from augmentum.utils.logging import get_logger
from augmentum.utils.secrets import sanitize_error_detail

log = get_logger(__name__)

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])


def _get_mcp_client(request: Request):
    """Get the MCP client manager from app state, or None."""
    return getattr(request.app.state, "mcp_client", None)


def _parse_persisted_servers() -> list[dict]:
    """Read the current persisted mcp_servers JSON list. Returns [] on any error.

    The setting may contain a mix of stdio (env-var-seeded) and HTTP entries.
    On first save via API, env-var stdio entries get captured here too.
    """
    raw = settings.mcp_servers or ""
    if not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [s for s in parsed if isinstance(s, dict) and s.get("name")]
    except json.JSONDecodeError:
        log.warning("mcp_servers_persisted_invalid_json", raw=raw[:200])
    return []


async def _persist_servers(request: Request, servers: list[dict]) -> None:
    """Write the server list to settings_store + update the live Settings attr.

    Mirrors the persistence path used by /api/config/tools so a restart
    restores the same configuration.
    """
    store = getattr(request.app.state, "settings_store", None)
    serialized = json.dumps(servers)
    if store is not None:
        try:
            await store.set("mcp_servers", serialized)
        except Exception:
            log.warning("mcp_servers_persist_failed", exc_info=True)
    object.__setattr__(settings, "mcp_servers", serialized)


@router.get("/servers")
async def list_servers(request: Request) -> JSONResponse:
    """List all connected MCP servers, their tool counts, and health.

    Health is probed in parallel via ``client.ping_server`` — a lightweight
    ``list_tools`` round-trip. Query param ``?health=false`` skips the
    probe (useful for tight polling in UIs that don't care about health
    every tick).
    """
    import asyncio as _asyncio

    client = _get_mcp_client(request)
    if client is None:
        return JSONResponse({"servers": [], "enabled": False})

    names = list(client.connected_servers)
    if not names:
        return JSONResponse({"servers": [], "enabled": True})

    skip_health = request.query_params.get("health", "").lower() == "false"

    # Static data: names + tool count + cached tool list
    basic = []
    for name in names:
        tools = client.get_server_tools(name)
        basic.append({
            "name": name,
            "tool_count": len(tools),
            "tools": [{"name": t.name, "description": t.description or ""} for t in tools],
        })

    if skip_health:
        for b in basic:
            b["healthy"] = None  # unknown — health probe was skipped
            b["last_error"] = ""
        return JSONResponse({"servers": basic, "enabled": True})

    # Probe health in parallel. Each ping has its own 3s timeout inside
    # ping_server; gather itself doesn't need one.
    probes = await _asyncio.gather(
        *[client.ping_server(name) for name in names],
        return_exceptions=True,
    )
    for server, probe in zip(basic, probes, strict=True):
        if isinstance(probe, BaseException):
            server["healthy"] = False
            server["last_error"] = str(probe)[:200]
        else:
            healthy, err = probe
            server["healthy"] = healthy
            server["last_error"] = err

    return JSONResponse({"servers": basic, "enabled": True})


@router.post("/connect")
async def connect_server(request: Request) -> JSONResponse:
    """Connect to a new MCP server.

    Body: {"name": "...", "url": "...", "headers": {...}}

    Note: stdio (subprocess) connections are restricted to the
    ``AUGMENTUM_MCP_SERVERS`` environment variable for security.
    Only HTTP-based MCP servers can be connected via this API.

    Admin only — MCP servers expose tools to every tenant, so adding one
    is an install-wide decision.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    client = _get_mcp_client(request)
    if client is None:
        return JSONResponse({"error": "MCP is not enabled"}, status_code=400)

    body = await request.json()
    name = body.get("name")
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    try:
        if "command" in body:
            return JSONResponse(
                {"error": "stdio (subprocess) MCP servers cannot be added via the API. "
                 "Configure them in the AUGMENTUM_MCP_SERVERS environment variable."},
                status_code=403,
            )

        if "url" not in body:
            return JSONResponse(
                {"error": "'url' is required for HTTP MCP server connections"},
                status_code=400,
            )

        # Validate URL against SSRF before connecting
        from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError

        try:
            safe_client = SafeHttpClient()
            hostname = safe_client._validate_url(body["url"])
            await safe_client._check_resolved_ips(hostname)
        except SafeHttpError as exc:
            return JSONResponse(
                {"error": f"Blocked: {exc}"},
                status_code=403,
            )

        tools = await client.connect_http(
            name, body["url"], headers=body.get("headers"),
        )

        # Register tools into the tool registry
        tool_registry = getattr(request.app.state, "tool_registry", None)
        if tool_registry:
            from augmentum.mcp.bridge import register_mcp_tools
            register_mcp_tools(client, name, tool_registry)

        # Persist the server config so it survives restart. Replace any
        # existing entry with the same name so the user can re-Connect
        # with updated headers/URL without leaking duplicates.
        persisted = [s for s in _parse_persisted_servers() if s.get("name") != name]
        entry: dict = {"name": name, "url": body["url"]}
        if body.get("headers"):
            entry["headers"] = body["headers"]
        persisted.append(entry)
        await _persist_servers(request, persisted)

        return JSONResponse({
            "name": name,
            "tool_count": len(tools),
            "tools": [{"name": t.name, "description": t.description or ""} for t in tools],
        })
    except ValueError as exc:
        return JSONResponse(
            {"error": sanitize_error_detail(str(exc))}, status_code=409,
        )
    except Exception as exc:
        log.error("mcp_connect_failed", server=name, error=str(exc), exc_info=True)
        return JSONResponse(
            {"error": f"Failed to connect: {sanitize_error_detail(str(exc))}"},
            status_code=503,
        )


@router.delete("/servers/{name}")
async def disconnect_server(name: str, request: Request) -> JSONResponse:
    """Disconnect from an MCP server and remove its tools. Admin only."""
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    client = _get_mcp_client(request)
    if client is None:
        return JSONResponse({"error": "MCP is not enabled"}, status_code=400)

    try:
        # Unregister tools first
        tool_registry = getattr(request.app.state, "tool_registry", None)
        if tool_registry:
            from augmentum.mcp.bridge import unregister_mcp_tools
            unregister_mcp_tools(name, tool_registry)

        await client.disconnect(name)

        # Drop from persisted list so a restart doesn't reconnect.
        # Stdio entries (env-var-seeded) are dropped here too; admin can
        # restore them by un-setting AUGMENTUM_MCP_SERVERS and restarting,
        # or by re-PUTting the mcp_servers setting.
        persisted = [s for s in _parse_persisted_servers() if s.get("name") != name]
        await _persist_servers(request, persisted)

        return JSONResponse({"status": "disconnected", "name": name})
    except ValueError as exc:
        return JSONResponse(
            {"error": sanitize_error_detail(str(exc))}, status_code=404,
        )
    except Exception as exc:
        log.error("mcp_disconnect_failed", server=name, error=str(exc), exc_info=True)
        return JSONResponse(
            {"error": f"Failed to disconnect: {sanitize_error_detail(str(exc))}"},
            status_code=503,
        )


@router.get("/tools")
async def list_all_tools(request: Request) -> JSONResponse:
    """List all MCP tools across all connected servers."""
    client = _get_mcp_client(request)
    if client is None:
        return JSONResponse({"tools": []})

    tools = client.list_all_tools()
    return JSONResponse({
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "schema": t.schema,
                "source": t.source,
            }
            for t in tools
        ]
    })
