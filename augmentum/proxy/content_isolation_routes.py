"""Content iframe origin isolation — mint + auth handoff.

Extends the coder-preview isolation mechanism to other "untrusted
content" iframes (knowledge packs first, game bundles and emulator
artifacts to follow). Same shape as the coder path:

  1. Parent UI POSTs ``/api/content/preview-token`` with ``{kind, id}``
     from the main origin (carries the user's session cookie).
  2. Server validates the user's permission to view that resource,
     mints a one-time ``pvt_*`` token bound to ``(user_id, kind, id)``,
     returns it alongside the isolated-origin URL.
  3. Parent UI sets the iframe ``src=`` to
     ``<isolated_origin><resource_path>?_pvt=<token>``.
  4. The isolated listener (Caddy → FastAPI with the
     X-Augmentum-Preview-Listener header) routes the request to the
     resource handler. The handler calls :func:`check_content_isolated_auth`
     which redeems the token and sets a preview-session cookie scoped
     to the isolated origin. Subsequent in-iframe requests use that
     cookie.
  5. Cookies for Augmentum's main origin never reach the isolated
     origin, so a malicious script inside the iframe can't call
     ``/api/auth/keys`` with the user's credentials.

Off by default: requires :setting:`content_iframe_isolation_enabled`
(and the Caddy/compose listener wired). When disabled the mint
endpoint returns 501 and the parent UI gracefully falls back to
same-origin same as today's behaviour.

Spec: see the coder-preview isolation design doc for the underlying
mechanism; this module is the generalisation pass.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["content-isolation"])


# Kinds the generic mint accepts. Adding a new kind requires:
#   1. Add it here.
#   2. Add the resource path prefix to ``_ISOLATED_PATH_PREFIXES`` in
#      ``proxy/server.py`` so the listener gate routes the path.
#   3. Implement ownership validation in ``_validate_kind_access`` below.
#   4. Call :func:`check_content_isolated_auth` from the resource's
#      route handler when the request scope has
#      ``augmentum_preview_isolated``.
_KIND_KNOWLEDGE_PACK = "knowledge_pack"
# A saved app-builder / coder artifact (zip or html) previewed inline in
# the library hero / detail pane. The resource id is the artifact_id; the
# served paths are /api/artifacts/{id}/preview[/{sub}].
_KIND_ARTIFACT_APP = "artifact_app"
# A library publication played via /api/library/play/{id}; the served
# paths are /api/library/publications/{id}/assets/{sub}.
_KIND_PUBLICATION = "publication"
_ALLOWED_KINDS: frozenset[str] = frozenset(
    {_KIND_KNOWLEDGE_PACK, _KIND_ARTIFACT_APP, _KIND_PUBLICATION}
)


class _MintTokenRequest(BaseModel):
    kind: Literal["knowledge_pack", "artifact_app", "publication"] = Field(
        ...,
        description="Resource kind. Workspace previews use the legacy "
                    "/api/coder/preview-token/{workspace_id} route.",
    )
    id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Resource identifier — pack_id for knowledge_pack, etc.",
    )


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


async def _validate_kind_access(
    request: Request, kind: str, resource_id: str,
) -> tuple[bool, str]:
    """Verify the caller may preview ``(kind, resource_id)``.

    Returns ``(allowed, reason)``. ``reason`` is surfaced as the JSON
    error message on a 403/404; never leak resource existence to the
    caller (use the same shape for "not found" and "not yours").
    """
    if kind == _KIND_KNOWLEDGE_PACK:
        # Knowledge packs are server-level (no user_id column) so the
        # check is "does this pack exist + is it active". Anything an
        # authenticated user can already see in the Browse panel they
        # can mint a token for.
        pack_mgr = getattr(request.app.state, "pack_manager", None)
        if pack_mgr is None:
            return False, "Pack manager not initialised"
        zim = getattr(pack_mgr, "_zim_packs", {}).get(resource_id)
        if zim is None:
            return False, "Knowledge pack not found"
        if not getattr(zim, "active", False):
            return False, "Knowledge pack inactive"
        return True, ""
    if kind == _KIND_ARTIFACT_APP:
        # Artifacts are user-scoped. The token is only minted for an
        # artifact the caller owns; cross-tenant returns the same
        # "not found" shape so existence never leaks.
        uid = _user_id(request)
        if not uid:
            return False, "Artifact not found"
        store = getattr(request.app.state, "artifact_store", None)
        if store is None:
            return False, "Artifact storage not available"
        info = await store.get(resource_id, user_id=uid)
        if not info:
            return False, "Artifact not found"
        return True, ""
    if kind == _KIND_PUBLICATION:
        # Publications are user-scoped (own row OR installed). The
        # publication store's get() already enforces the tenant check.
        uid = _user_id(request)
        if not uid:
            return False, "Publication not found"
        store = getattr(request.app.state, "publication_store", None)
        if store is None:
            return False, "Library not initialised"
        row = await store.get(resource_id, user_id=uid)
        if not row:
            return False, "Publication not found"
        return True, ""
    return False, f"Unknown kind: {kind}"


def _isolated_origin(request: Request) -> str:
    """Derive the isolated origin URL. Mirrors coder_routes._isolated_preview_origin."""
    import re

    from augmentum.config import settings
    explicit = (getattr(settings, "coder_preview_isolated_origin", "") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    host_header = request.headers.get("host") or ""
    host_only = host_header.split(":", 1)[0].strip()
    if not host_only:
        return ""
    if not re.match(r"^[A-Za-z0-9._\-]+$", host_only):
        return ""
    scheme = "https" if request.url.scheme == "https" else "http"
    port = int(getattr(settings, "coder_preview_isolated_port", 6444))
    return f"{scheme}://{host_only}:{port}"


@router.post("/api/content/preview-token")
async def mint_content_preview_token(
    body: _MintTokenRequest, request: Request,
) -> JSONResponse:
    """Generic mint for content iframe isolation tokens.

    Authenticated by the main session cookie. Returns a single-use
    token bound to ``(user_id, kind, id)`` plus the isolated-origin
    URL the iframe should load from.

    Status codes:
      201 — token minted (body: ``{token, expires_in, isolated_origin}``)
      401 — caller has no session
      403 — caller doesn't own the resource (or kind not allowed)
      404 — resource doesn't exist (returned identically to 403 above
            for kinds where existence is a privacy concern; current
            kinds use the same shape to keep messages clean)
      501 — content isolation disabled in settings
      503 — preview token store not initialised

    The endpoint sits on the MAIN origin (parent UI fetches it). The
    returned token is then redeemed by the isolated listener on first
    iframe request — see :func:`check_content_isolated_auth`.
    """
    from augmentum.config import settings

    if not getattr(settings, "content_iframe_isolation_enabled", False):
        return JSONResponse(
            {"error": "Content iframe isolation is disabled"}, status_code=501,
        )

    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if body.kind not in _ALLOWED_KINDS:
        return JSONResponse(
            {"error": f"Unsupported kind: {body.kind}"}, status_code=400,
        )

    allowed, reason = await _validate_kind_access(request, body.kind, body.id)
    if not allowed:
        log.info(
            "content_preview_token_denied",
            kind=body.kind, resource_id=body.id, user_id=uid, reason=reason,
        )
        return JSONResponse({"error": reason}, status_code=404)

    store = getattr(request.app.state, "preview_token_store", None)
    if store is None:
        return JSONResponse(
            {"error": "Preview token store unavailable"}, status_code=503,
        )

    try:
        token, expires_at = store.mint(
            user_id=uid,
            workspace_id=body.id,  # field name is legacy; carries the resource id for non-workspace kinds
            ttl_s=float(getattr(settings, "coder_preview_token_ttl_seconds", 60)),
            kind=body.kind,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    import time as _time
    expires_in = max(1, int(round(expires_at - _time.time())))
    isolated = _isolated_origin(request)
    if not isolated:
        return JSONResponse(
            {"error": "Could not derive isolated origin"}, status_code=503,
        )
    return JSONResponse({
        "token": token,
        "expires_in": expires_in,
        "isolated_origin": isolated,
    }, status_code=201)


async def check_content_isolated_auth(
    request: Request, kind: str, resource_id: str,
) -> Response | None:
    """Validate a request to ``/api/<content>/...`` on the isolated origin.

    Returns ``None`` when the request may proceed; otherwise returns a
    Response (302 redirect after token redemption, 401 when the
    preview session is expired, 403 on kind/resource mismatch) that
    the caller must return.

    Two paths:
      - ``?_pvt=<token>`` present → consume one-time token, mint a
        preview-session cookie scoped to the isolated origin, 302 to
        the same URL with ``_pvt`` stripped.
      - ``preview_session`` cookie present → validate, extend sliding
        TTL, proceed.
      - Neither → 401 with ``X-Augmentum-Preview-Expired: true``.

    Kind mismatch on token or session is the load-bearing check: a
    workspace token can't open a knowledge_pack path even though both
    travel through the same store. The listener gate path-prefix
    allowlist is the outer defence; this check is the inner.
    """
    from augmentum.config import settings

    if not request.scope.get("augmentum_preview_isolated"):
        # Main origin — the route's existing logic still applies
        # (which for knowledge packs is "no auth required, packs are
        # server-level"). Caller continues unchanged.
        return None

    token_store = getattr(request.app.state, "preview_token_store", None)
    session_store = getattr(request.app.state, "preview_session_store", None)
    if token_store is None or session_store is None:
        return JSONResponse(
            {"error": "Preview auth unavailable"}, status_code=503,
        )

    token = request.query_params.get("_pvt", "")
    if token:
        record = token_store.consume(token)
        if record is None:
            return JSONResponse(
                {"error": "Invalid or expired preview token"},
                status_code=401,
            )
        if record.kind != kind:
            log.warning(
                "content_preview_token_kind_mismatch",
                token_kind=record.kind, requested_kind=kind,
                user_id=record.user_id,
            )
            return JSONResponse(
                {"error": "Token kind mismatch"}, status_code=403,
            )
        if record.workspace_id != resource_id:
            log.warning(
                "content_preview_token_resource_mismatch",
                token_resource=record.workspace_id,
                requested_resource=resource_id,
                user_id=record.user_id, kind=kind,
            )
            return JSONResponse(
                {"error": "Token resource mismatch"}, status_code=403,
            )
        cookie_value = session_store.mint(
            user_id=record.user_id,
            workspace_id=resource_id,
            kind=kind,
        )
        # Redirect to the same path with ``_pvt`` stripped so the
        # cookie does the work on subsequent requests.
        remaining_qs = "&".join(
            f"{k}={v}" for k, v in request.query_params.items() if k != "_pvt"
        )
        redirect_path = request.url.path
        if remaining_qs:
            redirect_path += "?" + remaining_qs
        response = Response(status_code=302)
        response.headers["Location"] = redirect_path
        response.set_cookie(
            key="preview_session",
            value=cookie_value,
            httponly=True,
            secure=(request.url.scheme == "https"),
            samesite="lax",
            max_age=int(getattr(settings, "coder_preview_session_ttl_seconds", 1800)),
            path="/",
        )
        return response

    cookie = request.cookies.get("preview_session", "")
    if not cookie:
        return JSONResponse(
            {"error": "Preview session required"},
            status_code=401,
            headers={"X-Augmentum-Preview-Expired": "true"},
        )
    record = session_store.get(cookie)
    if record is None:
        return JSONResponse(
            {"error": "Preview session expired"},
            status_code=401,
            headers={"X-Augmentum-Preview-Expired": "true"},
        )
    if record.kind != kind:
        log.warning(
            "content_preview_session_kind_mismatch",
            session_kind=record.kind, requested_kind=kind,
            user_id=record.user_id,
        )
        return JSONResponse(
            {"error": "Session kind mismatch"}, status_code=403,
        )
    if record.workspace_id != resource_id:
        log.warning(
            "content_preview_session_resource_mismatch",
            session_resource=record.workspace_id,
            requested_resource=resource_id,
            user_id=record.user_id, kind=kind,
        )
        return JSONResponse(
            {"error": "Session resource mismatch"}, status_code=403,
        )
    # Surface the redeemed user for user-scoped resource handlers
    # (artifact / publication previews look up by user_id). Mirrors the
    # coder preview proxy's contract — NOT scope["user"], which the
    # AuthMiddleware owns and which is absent on the isolated origin.
    request.scope["augmentum_preview_user_id"] = record.user_id
    return None
