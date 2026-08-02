"""Auto-route registration for Tools that declare ``surfaces.http_route``.

Phase 1 of the Unified Primitive Layer (see
``docs/superpowers/specs/2026-06-01-unified-primitive-layer-design.md``).

After the ToolRegistry is populated in server.py lifespan, call
:func:`register_tool_routes` to bind a ``POST {route}`` for every Tool
that opted into HTTP exposure. The generic dispatcher reads the JSON
body, runs the Tool's ``execute(**body, _context={"user_id": ...})``,
and returns ``ToolResult.metadata`` (or ``ToolResult.error``).

Tools that need custom request/response shaping should keep their
hand-written route in the appropriate ``*_routes.py`` file and leave
``http_route=None``. The auto-dispatcher is the cheap path; the
hand-written route is the escape hatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from augmentum.tools.base import invoke_tool
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

    from augmentum.tools.base import Tool
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


def _make_dispatcher(tool: Tool):
    """Build a FastAPI handler that runs the given Tool from a JSON body."""

    async def _dispatch(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001 — surface to client
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")

        # Inject context so user-scoped tools can resolve the caller.
        # Strip any caller-supplied identity fields and overwrite with
        # the authenticated user — Tool.extract_user_id reads `_user_id`
        # from the top level first AND `_context["user_id"]` second, so
        # both surfaces have to be force-set to the authed user or a
        # crafted body lets one authed user impersonate any other.
        body.pop("_user_id", None)
        ctx = body.get("_context")
        if not isinstance(ctx, dict):
            ctx = {}
        ctx["user_id"] = _user_id(request)
        body["_context"] = ctx

        try:
            result = await invoke_tool(tool, body)
        except TypeError as exc:
            # Most likely a kwargs mismatch — bad input from caller.
            raise HTTPException(status_code=400, detail=f"Invalid arguments: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            log.warning("auto_tool_dispatch_failed", tool=tool.name, error=str(exc))
            raise HTTPException(status_code=500, detail=f"Tool execution failed: {exc}") from exc

        if not result.success:
            status = 400 if result.validation_error else 500
            return JSONResponse(
                status_code=status,
                content={"ok": False, "error": result.error, "warnings": result.warnings},
            )

        payload: dict = {
            "ok": True,
            "output": result.output,
            "metadata": result.metadata,
        }
        if result.warnings:
            payload["warnings"] = result.warnings
        if result.card:
            payload["card"] = result.card
        return JSONResponse(payload)

    _dispatch.__name__ = f"_tool_dispatch_{tool.name}"
    return _dispatch


def register_tool_routes(app: FastAPI, registry: ToolRegistry) -> list[str]:
    """Bind ``POST {tool.surfaces.http_route}`` for every Tool that
    opted in. Returns the list of routes registered. Idempotent — a
    second call against the same app is a no-op for already-bound paths.
    """
    bound: list[str] = []
    existing_paths = {getattr(r, "path", None) for r in app.routes}

    for tool in registry.list_tools():
        route = tool.surfaces.http_route
        if not route:
            continue
        if route in existing_paths:
            log.debug("auto_route_skip_existing", tool=tool.name, route=route)
            continue
        app.add_api_route(
            route,
            _make_dispatcher(tool),
            methods=["POST"],
            name=f"tool_dispatch_{tool.name}",
            tags=["tools"],
        )
        bound.append(route)
        existing_paths.add(route)
        log.info("auto_route_bound", tool=tool.name, route=route)

    return bound
