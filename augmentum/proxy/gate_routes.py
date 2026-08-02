"""Front-gate auth endpoint — the trust check behind the identity-aware proxy.

Caddy's ``forward_auth`` calls ``GET /api/gate/verify?svc=<id>`` on every hit
to a ``<service>.<gate_domain>`` host, forwarding the original request's
``Cookie``. We validate the Augmentum session, authorize the user against the
service, and — for Basic-auth (managed_auth) upstreams — return the credential
to inject server-side via the ``X-Aug-Gate-Authz`` response header (Caddy copies
it onto the upstream request; the snippet then strips the cookie). The browser
never sees the upstream credential, and the upstream never sees the Augmentum
session.

Verdicts:
  - 200 + ``X-Aug-Gate-Authz: Basic <b64>``  → trusted; dissolve login.
  - 302 → /?next=…  (SPA root, login view)     → no/invalid session.
  - 403                                         → authed but no access to svc.

This route is in ``_PUBLIC_PATHS`` (it validates the cookie itself, ahead of
the upstream), so it must do its own auth — never assume ``request.scope.user``.

See docs/superpowers/specs/2026-06-19-front-gate-identity-aware-proxy-design.md
"""

from __future__ import annotations

import base64
from http.cookies import SimpleCookie
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from augmentum.media.store import MediaServerStore
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/gate", tags=["gate"])


def _session_token(request: Request) -> str:
    """Extract the Augmentum session token from the (forwarded) request."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    cookie_header = request.headers.get("cookie", "")
    if cookie_header:
        try:
            jar = SimpleCookie()
            jar.load(cookie_header)
            morsel = jar.get("augmentum_session")
            if morsel and morsel.value:
                return morsel.value.strip()
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _store(request: Request) -> MediaServerStore | None:
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return MediaServerStore(backend.conn) if isinstance(backend, SQLiteBackend) else None


def _db_conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm else None
    return backend.conn if isinstance(backend, SQLiteBackend) else None


def _login_redirect(request: Request) -> Response:
    """302 the browser to the Augmentum login on the gate apex, with ?next.

    Targets the SPA at ``/ui/``, NOT ``/login`` or ``/``. Augmentum's login
    is a client-side view inside the single-page app, which is served at the
    public ``/ui/`` mount (StaticFiles html=True). There is no server
    ``/login`` route — sending the browser there 401s at the auth middleware
    before any HTML loads (the "Unauthorized" the user sees). The bare ``/``
    is the Ollama-compat stub ("Ollama is running"), not the app shell. The
    SPA reads ``?next`` after a successful sign-in and forwards the browser
    back to the service subdomain (auth.js::_safeGateNext).
    """
    from augmentum.config import settings
    fwd_host = request.headers.get("x-forwarded-host", "")
    fwd_uri = request.headers.get("x-forwarded-uri", "/")
    nxt = f"https://{fwd_host}{fwd_uri}" if fwd_host else fwd_uri
    base = f"https://{settings.gate_domain}:6443" if settings.gate_domain else ""
    return RedirectResponse(f"{base}/ui/?next={quote(nxt, safe='')}", status_code=302)


@router.get("/info")
async def gate_info(request: Request) -> JSONResponse:
    """Front-gate config for the UI (gate domain → build ``<svc>.<domain>`` URLs).

    Authed (not public): goes through the normal middleware. Returns the
    domain only — no secrets.
    """
    from augmentum.config import settings
    return JSONResponse({"gate_domain": settings.gate_domain or ""})


@router.get("/verify")
async def gate_verify(request: Request) -> Response:
    """forward_auth verdict for ``<svc>.<gate_domain>``."""
    svc = (request.query_params.get("svc") or "").strip()
    sm = getattr(request.app.state, "session_manager", None)
    if sm is None or not svc:
        # No auth subsystem / malformed gate config → fail closed.
        return JSONResponse({"error": "unavailable"}, status_code=503)

    token = _session_token(request)
    user = await sm.validate_token(token) if token else None
    if not user or not getattr(user, "is_active", False):
        return _login_redirect(request)

    # Workspace services (ws-* prefix) are owned by the user who created
    # the workspace — verify ownership instead of provider visibility.
    if svc.startswith("ws-"):
        # svc is "ws-<slug>" where slug == _workspace_slug(name, workspace_id)
        # = "<name-slug>-<workspace_id[:6]>". Authorize by recomputing that slug
        # for each of THIS user's LAN-exposed workspaces and exact-matching.
        # The previous `id LIKE '%<fragment>'` was wrong twice over: the
        # fragment is the id PREFIX (not a suffix), and a bare LIKE could
        # authorize a different workspace that merely shares the fragment.
        slug = svc[len("ws-"):]
        conn = getattr(request.app.state, "conn", None)
        ws_authorized = False
        if conn and slug:
            try:
                from augmentum.coder.containers import _workspace_slug
                rows = await conn.execute_fetchall(
                    "SELECT id, name FROM project_checkouts "
                    "WHERE user_id = ? AND lan_accessible = 1",
                    (user.id,),
                )
                ws_authorized = any(_workspace_slug(r[1], r[0]) == slug for r in rows)
            except Exception:
                log.warning("gate_workspace_lookup_failed", svc=svc, exc_info=True)
        if not ws_authorized:
            log.info("gate_denied_workspace", svc=svc, user=user.id)
            return JSONResponse(
                {"error": "no access to this workspace"}, status_code=403,
            )
        return Response(status_code=200)

    mgr = getattr(request.app.state, "service_manager", None)
    sd = mgr.get_definition(svc) if mgr else None
    from augmentum.providers.service_auth import (
        needs_managed_auth,
        resolve_managed_credentials,
    )

    # BASIC mode (managed-auth media servers): the user must already have this
    # provider connected (their own row or an admin-shared one) — same
    # visibility the Access panel grants. On success, hand Caddy the credential
    # to inject; the snippet maps it to Authorization + strips the cookie.
    if sd is not None and needs_managed_auth(sd):
        store = _store(request)
        if store is None:
            return JSONResponse({"error": "unavailable"}, status_code=503)
        visible = await store.list_visible(user_id=user.id)
        if not any(s.provider == svc for s in visible):
            log.info("gate_denied_no_access", svc=svc, user=user.id)
            return JSONResponse(
                {"error": "no access to this service"}, status_code=403,
            )
        username, password = await resolve_managed_credentials(svc, _db_conn(request))
        token_b64 = base64.b64encode(f"{username}:{password}".encode()).decode("ascii")
        resp = Response(status_code=200)
        resp.headers["X-Aug-Gate-Authz"] = f"Basic {token_b64}"
        return resp

    # ACCESS mode (generic service apps — n8n etc.): install-wide. Any active
    # Augmentum user may reach an INSTALLED service; the app keeps its own
    # login for identity. Authorize by confirming a live managed row exists
    # (installed_image is non-empty only when the service is provisioned) —
    # never authorize a bare catalog id that isn't actually running.
    if sd is not None and mgr is not None and await mgr.installed_image(svc):
        return Response(status_code=200)

    # Unknown / not installed → fail closed.
    log.info("gate_denied_unknown_service", svc=svc, user=user.id)
    return JSONResponse({"error": "no access to this service"}, status_code=403)
