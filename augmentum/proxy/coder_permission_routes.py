"""HTTP endpoints for the coder permission-approval flow.

See ``augmentum/coder/permissions.py`` for the registry that backs these
endpoints. Flow:

- Coder agent calls a tool that needs approval. The handler's
  permission_callback posts a ``PermissionRequest`` into the registry
  and awaits its future.
- The UI polls ``GET /v1/coder/permissions/pending`` every ~2s. For
  each pending request it shows an approval modal.
- When the user clicks Allow or Deny the UI hits
  ``POST /v1/coder/permissions/{id}/approve`` or ``…/deny``. The
  registry resolves the future; the callback returns.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1/coder/permissions", tags=["coder-permissions"])


def _get_registry(request: Request):
    return getattr(request.app.state, "permission_registry", None)


def _current_user_id(request: Request) -> str:
    """Scope requests to the caller's user. Empty string in no-auth
    setups (registry returns all pending for an empty user id so dev
    setups without auth still work)."""
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


@router.get("/pending")
async def list_pending(request: Request) -> JSONResponse:
    """Return the current user's pending approval requests."""
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"enabled": False, "pending": []})

    user_id = _current_user_id(request)
    pending = [r.to_dict() for r in registry.pending_for(user_id)]
    return JSONResponse({"enabled": True, "pending": pending})


@router.get("/audit")
async def list_audit(
    request: Request,
    workspace_id: str = "",
    limit: int = 100,
) -> JSONResponse:
    """Durable decision history (migration 260): who allowed/denied
    which tool, when, and via what channel (user modal / policy /
    timeout / disconnect). Newest first."""
    from augmentum.coder.permission_audit import resolve_store

    store = resolve_store(request.app.state)
    if store is None:
        return JSONResponse({"enabled": False, "events": []})

    user_id = _current_user_id(request)
    events = await store.list_events(
        user_id=user_id, workspace_id=workspace_id, limit=limit,
    )
    return JSONResponse({"enabled": True, "events": events})


@router.post("/{request_id}/approve")
async def approve(request_id: str, request: Request) -> JSONResponse:
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "permissions disabled"}, status_code=400)

    user_id = _current_user_id(request)
    req = registry.get(request_id)
    if req is None:
        return JSONResponse({"error": "unknown request"}, status_code=404)
    if user_id and req.user_id and req.user_id != user_id:
        return JSONResponse({"error": "not owner"}, status_code=403)

    ok = registry.resolve(request_id, approved=True)
    if not ok:
        return JSONResponse(
            {"error": "already resolved or expired"}, status_code=409,
        )
    return JSONResponse({"status": "approved", "id": request_id})


@router.post("/{request_id}/deny")
async def deny(request_id: str, request: Request) -> JSONResponse:
    registry = _get_registry(request)
    if registry is None:
        return JSONResponse({"error": "permissions disabled"}, status_code=400)

    user_id = _current_user_id(request)
    req = registry.get(request_id)
    if req is None:
        return JSONResponse({"error": "unknown request"}, status_code=404)
    if user_id and req.user_id and req.user_id != user_id:
        return JSONResponse({"error": "not owner"}, status_code=403)

    ok = registry.resolve(request_id, approved=False)
    if not ok:
        return JSONResponse(
            {"error": "already resolved or expired"}, status_code=409,
        )
    return JSONResponse({"status": "denied", "id": request_id})
