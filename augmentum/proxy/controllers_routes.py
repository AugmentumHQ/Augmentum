"""Controller framework routes -- system catalog + per-user remaps.

Read paths:
  GET /api/controllers/profiles                 -> all known systems + defaults
  GET /api/controllers/{system_id}              -> resolved layout (default + override)
  GET /api/controllers/{system_id}/remap        -> raw user override only

Write paths:
  PUT    /api/controllers/{system_id}/remap     -> save partial override
  DELETE /api/controllers/{system_id}/remap     -> reset to defaults

All write paths require auth + user_id-scope. The system catalog read
is auth-required but doesn't filter (the catalog is server-level
data, same shape for every user). Master toggle is
``controller_remap_enabled`` -- when off, GETs still serve the
catalog (so the UI can render hints) but PUT/DELETE return 503 so
nobody silently writes data the system will then ignore.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.config import settings
from augmentum.controllers import ControllerService, get_system_profile
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/controllers", tags=["controllers"])


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _service(request: Request) -> ControllerService | None:
    return getattr(request.app.state, "controller_service", None)


def _gate_read(request: Request) -> JSONResponse | None:
    """Reads always work when the service is up -- the catalog is
    static and useful to surface even when remaps are disabled.
    """
    if _service(request) is None:
        return JSONResponse(
            {"error": "Controllers framework unavailable"},
            status_code=503,
        )
    return None


def _gate_write(request: Request) -> JSONResponse | None:
    """Writes additionally require the master toggle."""
    if (gate := _gate_read(request)) is not None:
        return gate
    if not getattr(settings, "controller_remap_enabled", True):
        return JSONResponse(
            {"error": "Controller remapping is disabled"},
            status_code=503,
        )
    return None


# ── Catalog ──────────────────────────────────────────────────────────


@router.get("/profiles")
async def list_profiles(request: Request) -> JSONResponse:
    """List all systems + their canonical default bindings."""
    if (gate := _gate_read(request)) is not None:
        return gate
    if not _user_id(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return JSONResponse({
        "profiles": [p.to_dict() for p in _service(request).list_systems()],
    })


# ── Resolved layout ──────────────────────────────────────────────────


@router.get("/{system_id}")
async def get_resolved(
    request: Request, system_id: str,
) -> JSONResponse:
    """Resolved (default + user override) layout for one system.

    Engine adapters consume this shape -- the launch handle's metadata
    embeds it for emulator-* runtimes.
    """
    if (gate := _gate_read(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    layout = await _service(request).resolve(user_id=uid, system_id=system_id)
    if layout is None:
        return JSONResponse({"error": "Unknown system"}, status_code=404)
    return JSONResponse({"layout": layout.to_dict()})


# ── User remap CRUD ──────────────────────────────────────────────────


@router.get("/{system_id}/remap")
async def get_remap(
    request: Request, system_id: str,
) -> JSONResponse:
    """Raw user override only (no defaults merged in). Returns 200
    with an empty body when the user has no override yet -- the UI
    can render that as "all defaults" without polling for 404s.
    """
    if (gate := _gate_read(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if get_system_profile(system_id) is None:
        return JSONResponse({"error": "Unknown system"}, status_code=404)
    remap = await _service(request).get_user_remap(
        user_id=uid, system_id=system_id,
    )
    if remap is None:
        return JSONResponse({
            "system_id": system_id,
            "bindings": {},
            "pad_routing": "index",
            "created_at": None,
            "updated_at": None,
        })
    return JSONResponse({"remap": remap.to_dict()})


@router.put("/{system_id}/remap")
async def put_remap(
    request: Request, system_id: str,
) -> JSONResponse:
    """Save a partial override.

    Body shape:
        {
          "bindings": { "<action_id>": { keyboard, gamepad_button, ... } },
          "pad_routing": "index" | "firstpress"
        }

    Bindings is a partial dict -- omitted actions inherit from
    defaults at resolve time. To explicitly clear an override and
    restore the default for one action, send that action with all
    fields null.
    """
    if (gate := _gate_write(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    body: dict[str, Any] = {}
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "JSON body required"}, status_code=400)

    bindings = body.get("bindings")
    pad_routing = body.get("pad_routing")
    if bindings is not None and not isinstance(bindings, dict):
        return JSONResponse(
            {"error": "bindings must be an object"}, status_code=400,
        )
    try:
        remap = await _service(request).update_remap(
            user_id=uid,
            system_id=system_id,
            bindings=bindings,
            pad_routing=pad_routing,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"remap": remap.to_dict()})


@router.delete("/{system_id}/remap")
async def delete_remap(
    request: Request, system_id: str,
) -> JSONResponse:
    """Reset to defaults (drops the user's override row)."""
    if (gate := _gate_write(request)) is not None:
        return gate
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if get_system_profile(system_id) is None:
        return JSONResponse({"error": "Unknown system"}, status_code=404)
    await _service(request).reset_remap(user_id=uid, system_id=system_id)
    return JSONResponse({"ok": True})
