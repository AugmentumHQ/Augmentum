"""Route-layer authorization guards.

These helpers sit inside the mutation handlers that edit shared-infrastructure
rows (providers, load balancers, MCP servers, managed services, etc.) so
that only users with ``role == "admin"`` can create, update, or delete them.
Reads stay open to every authenticated caller because the provider list is
used across many user-facing flows (voice playback, chat, image generation)
and gating it admin-only would break non-admin UX.

Usage:

    from augmentum.auth.guards import require_admin

    @router.post("/some-shared-resource")
    async def create(request: Request) -> JSONResponse:
        if (forbidden := require_admin(request)) is not None:
            return forbidden
        ...
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


def _current_user(request: Request):
    """Return the authenticated User from the ASGI scope, or None."""
    return request.scope.get("user")


def is_admin(request: Request) -> bool:
    """Return True when the authenticated caller has the admin role."""
    user = _current_user(request)
    return bool(user and getattr(user, "is_admin", False))


def require_admin(request: Request) -> JSONResponse | None:
    """Return a 401/403 JSONResponse when the caller is not admin, else None.

    Call sites follow the walrus-assignment pattern so the guard fits in one
    line at the top of a handler:

        if (forbidden := require_admin(request)) is not None:
            return forbidden
    """
    user = _current_user(request)
    if user is None:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not getattr(user, "is_admin", False):
        return JSONResponse(
            {"error": "Admin privilege required for this operation"},
            status_code=403,
        )
    return None
