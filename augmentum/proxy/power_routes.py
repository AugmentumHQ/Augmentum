"""REST API routes for Augmentum Powers."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.powers import PowerStateStore
from augmentum.utils.logging import get_logger

router = APIRouter(prefix="/api/powers", tags=["powers"])
log = get_logger(__name__)


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _registry(request: Request):
    return getattr(request.app.state, "power_registry", None)


def _state_store(request: Request) -> PowerStateStore | None:
    settings_store = getattr(request.app.state, "settings_store", None)
    if settings_store is None:
        return None
    return PowerStateStore(settings_store)


def _runtime_deps(request: Request) -> dict:
    return {
        "mcp_client": getattr(request.app.state, "mcp_client", None),
        "tool_registry": getattr(request.app.state, "tool_registry", None),
    }


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return body if isinstance(body, dict) else {}


@router.get("")
async def list_powers(request: Request, workspace_id: str = "") -> JSONResponse:
    registry = _registry(request)
    if registry is None:
        return JSONResponse({"powers": [], "active": None})
    try:
        registry.rescan()
    except Exception as exc:
        log.debug("powers_rescan_failed", error=str(exc))
    state = _state_store(request)
    user_id = _user_id(request)
    enabled_map = await state.get_enabled_map(user_id) if state else {}
    active = await state.get_active_power(user_id, workspace_id=workspace_id) if state else None
    deps = _runtime_deps(request)

    powers = []
    for manifest in registry.list_powers():
        enabled = enabled_map.get(manifest.id, True)
        health = registry.evaluate_health(manifest, **deps)
        powers.append(
            manifest.to_summary_dict(
                enabled=enabled,
                health=health,
                active=active if active and active.power_id == manifest.id else None,
            ),
        )
    return JSONResponse({"powers": powers, "active": active.to_dict() if active else None})


@router.get("/active")
async def get_active_power(request: Request, workspace_id: str = "") -> JSONResponse:
    registry = _registry(request)
    state = _state_store(request)
    if registry is None or state is None:
        return JSONResponse({"active": None})
    user_id = _user_id(request)
    active = await state.get_active_power(user_id, workspace_id=workspace_id)
    if active is None:
        return JSONResponse({"active": None})
    manifest = registry.get_power(active.power_id)
    if manifest is None:
        return JSONResponse({"active": active.to_dict()})
    enabled = await state.is_enabled(user_id, manifest.id)
    health = registry.evaluate_health(manifest, **_runtime_deps(request))
    return JSONResponse(
        {
            "active": active.to_dict(),
            "power": manifest.to_summary_dict(enabled=enabled, health=health, active=active),
        },
    )


@router.post("/rescan")
async def rescan_powers(request: Request) -> JSONResponse:
    registry = _registry(request)
    if registry is None:
        return JSONResponse({"error": "Power registry not available"}, status_code=503)
    registry.rescan()
    return JSONResponse({"rescanned": True, "count": len(registry.list_powers())})


@router.post("/clear-activation")
async def clear_active_power(request: Request) -> JSONResponse:
    state = _state_store(request)
    if state is None:
        return JSONResponse({"error": "Settings store not available"}, status_code=503)
    body = await _json_body(request)
    await state.clear_active_power(_user_id(request), workspace_id=str(body.get("workspace_id", "")))
    return JSONResponse({"cleared": True})


@router.get("/{power_id}")
async def get_power(request: Request, power_id: str, workspace_id: str = "") -> JSONResponse:
    registry = _registry(request)
    if registry is None:
        return JSONResponse({"error": "Power registry not available"}, status_code=503)
    manifest = registry.get_power(power_id)
    if manifest is None:
        return JSONResponse({"error": "Power not found"}, status_code=404)
    state = _state_store(request)
    user_id = _user_id(request)
    active = await state.get_active_power(user_id, workspace_id=workspace_id) if state else None
    enabled = await state.is_enabled(user_id, power_id) if state else True
    health = registry.evaluate_health(manifest, **_runtime_deps(request))
    return JSONResponse(
        manifest.to_detail_dict(
            enabled=enabled,
            health=health,
            active=active if active and active.power_id == power_id else None,
        ),
    )


@router.post("/{power_id}/activate")
async def activate_power(request: Request, power_id: str) -> JSONResponse:
    registry = _registry(request)
    state = _state_store(request)
    if registry is None or state is None:
        return JSONResponse({"error": "Powers not available"}, status_code=503)
    manifest = registry.get_power(power_id)
    if manifest is None:
        return JSONResponse({"error": "Power not found"}, status_code=404)
    user_id = _user_id(request)
    if not await state.is_enabled(user_id, power_id):
        return JSONResponse({"error": "Power is disabled"}, status_code=409)
    body = await _json_body(request)
    workspace_id = str(body.get("workspace_id", ""))
    active = await state.activate_power(
        user_id,
        workspace_id=workspace_id,
        power_id=power_id,
        source=str(body.get("source", "manual") or "manual"),
        scope=str(body.get("scope", "workspace") or "workspace"),
        reason=str(body.get("reason", "") or ""),
    )
    health = registry.evaluate_health(manifest, **_runtime_deps(request))
    return JSONResponse(
        {
            "active": active.to_dict(),
            "power": manifest.to_summary_dict(enabled=True, health=health, active=active),
        },
    )


@router.post("/{power_id}/enable")
async def enable_power(request: Request, power_id: str) -> JSONResponse:
    registry = _registry(request)
    state = _state_store(request)
    if registry is None or state is None:
        return JSONResponse({"error": "Powers not available"}, status_code=503)
    if registry.get_power(power_id) is None:
        return JSONResponse({"error": "Power not found"}, status_code=404)
    await state.set_enabled(_user_id(request), power_id, True)
    return JSONResponse({"enabled": True})


@router.post("/{power_id}/disable")
async def disable_power(request: Request, power_id: str) -> JSONResponse:
    registry = _registry(request)
    state = _state_store(request)
    if registry is None or state is None:
        return JSONResponse({"error": "Powers not available"}, status_code=503)
    if registry.get_power(power_id) is None:
        return JSONResponse({"error": "Power not found"}, status_code=404)
    user_id = _user_id(request)
    await state.set_enabled(user_id, power_id, False)
    body = await _json_body(request)
    workspace_id = str(body.get("workspace_id", ""))
    active = await state.get_active_power(user_id, workspace_id=workspace_id)
    if active and active.power_id == power_id:
        await state.clear_active_power(user_id, workspace_id=workspace_id)
    return JSONResponse({"enabled": False})
