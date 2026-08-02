"""Origin proxy HTTP routes.

Two endpoints:

  POST /api/cast/games/{title_id}/proxy/start
      Mint a ProxySession + return the surface_url the receiver should
      open. The classifier route (``POST /classify``) calls this
      internally when it picks the proxy strategy; library2 can also
      call it directly for "force proxy" testing flows.

  GET  /api/cast/game-proxy/{token}/{path:path}
      Serve a proxied asset. Token is validated against the session
      store; the path is appended to the session's source_origin to
      reach the upstream URL. HTML + CSS responses get URL-rewritten
      + the adapter loader injected.

Auth on the proxy endpoint is via the path-embedded token. We don't
require a session cookie because cross-origin iframed games don't
carry one reliably — and the token itself is the credential (bound
to user, receiver, title, origin, TTL).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from augmentum.cast.games.proxy.fetcher import (
    ProxyFetcher,
    is_url_safe,
)
from augmentum.cast.games.proxy.rewriter import (
    DEFAULT_CDN_ALLOWLIST,
    inject_adapter_loader,
    rewrite_csp,
    rewrite_css,
    rewrite_html,
)
from augmentum.cast.games.proxy.session_store import ProxySessionStore
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Mounted at root (not under /api/cast/games) so the path is short +
# friendly in CSP allowances. The auth middleware exempts this prefix
# because the token IS the credential.
proxy_router = APIRouter(prefix="/api/cast/game-proxy", tags=["cast"])

# Profile-management adjacency — start endpoint lives on the same
# router as /classify so they share the prefix.
start_router = APIRouter(prefix="/api/cast/games", tags=["cast"])


# ── Pydantic shapes ──────────────────────────────────────────────


class ProxyStartRequest(BaseModel):
    receiver_id: str = ""
    source_url: str
    ttl_s: float | None = None


# ── Helpers ──────────────────────────────────────────────────────


def _require_user(request: Request) -> str:
    user = request.scope.get("user")
    if user is None:
        raise HTTPException(401, "auth required")
    user_id = getattr(user, "id", "")
    if not user_id:
        raise HTTPException(401, "auth required")
    return user_id


def _store_or_503(request: Request) -> ProxySessionStore:
    s = getattr(request.app.state, "cast_proxy_session_store", None)
    if s is None:
        raise HTTPException(503, "cast proxy session store unavailable")
    return s


def _fetcher_or_503(request: Request) -> ProxyFetcher:
    f = getattr(request.app.state, "cast_proxy_fetcher", None)
    if f is None:
        raise HTTPException(503, "cast proxy fetcher unavailable")
    return f


def _proxy_base(token: str) -> str:
    return f"/api/cast/game-proxy/{token}"


def _content_type_kind(ct: str) -> str:
    """Classify a Content-Type into one of (html | css | other)."""
    base = (ct or "").split(";", 1)[0].strip().lower()
    if base in ("text/html", "application/xhtml+xml"):
        return "html"
    if base in ("text/css",):
        return "css"
    return "other"


# ── /api/cast/games/{title_id}/proxy/start ───────────────────────


@start_router.post("/{title_id}/proxy/start")
async def start_proxy_session(
    title_id: str,
    payload: ProxyStartRequest,
    request: Request,
) -> dict[str, Any]:
    user_id = _require_user(request)
    store = _store_or_503(request)

    if not is_url_safe(payload.source_url):
        raise HTTPException(400, "source_url is not a safe public URL")

    try:
        session = store.mint(
            user_id=user_id,
            receiver_id=payload.receiver_id,
            title_id=title_id,
            source_url=payload.source_url,
            ttl_s=payload.ttl_s,
        )
    except ValueError as err:
        raise HTTPException(400, str(err))

    # Compute the entry path under our proxy. Empty path lands on the
    # source's root (which usually 302s to its index).
    parsed = urlparse(payload.source_url)
    entry_path = (parsed.path or "/")
    if parsed.query:
        entry_path = f"{entry_path}?{parsed.query}"
    surface_url = f"{_proxy_base(session.token)}{entry_path}"

    return {
        "token": session.token,
        "expires_at": session.expires_at,
        "source_origin": session.source_origin,
        "surface_url": surface_url,
    }


# ── /api/cast/game-proxy/{token}/{path} ─────────────────────────


@proxy_router.get("/{token}/{path:path}")
@proxy_router.head("/{token}/{path:path}")
async def proxy_asset(
    token: str,
    path: str,
    request: Request,
) -> Response:
    store = _store_or_503(request)
    fetcher = _fetcher_or_503(request)

    session = store.get(token)
    if session is None:
        raise HTTPException(404, "proxy session not found or expired")

    # Build upstream URL: source_origin + path[?query].
    qs = request.url.query
    upstream_path = "/" + path.lstrip("/")
    upstream = session.source_origin + upstream_path
    if qs:
        upstream = f"{upstream}?{qs}"

    if not is_url_safe(upstream):
        raise HTTPException(400, "unsafe upstream URL")

    try:
        result = await fetcher.fetch(
            upstream,
            source_origin=session.source_origin,
            user_id=session.user_id,
            request_headers={
                "accept": request.headers.get("accept", "*/*"),
                "accept-encoding": "identity",  # we re-encode if needed
                "accept-language": request.headers.get("accept-language", ""),
                "user-agent": request.headers.get("user-agent", ""),
                "range": request.headers.get("range", ""),
            },
        )
    except PermissionError as err:
        raise HTTPException(400, str(err))
    except httpx.HTTPError as err:
        log.warning(
            "cast_proxy_upstream_error",
            token_prefix=token[:8], url=upstream, error=str(err),
        )
        raise HTTPException(502, f"upstream fetch failed: {err}")

    body = result.body
    headers = dict(result.headers)
    kind = _content_type_kind(result.content_type)
    proxy_base = _proxy_base(token)

    # HTML: rewrite URLs + inject the adapter loader.
    if kind == "html":
        try:
            html_str = body.decode("utf-8", errors="replace")
        except (LookupError, UnicodeDecodeError):
            html_str = body.decode("latin-1", errors="replace")
        rewritten = rewrite_html(
            html_str,
            proxy_base=proxy_base,
            source_origin=session.source_origin,
            page_url=upstream,
            cdn_allowlist=DEFAULT_CDN_ALLOWLIST,
        )
        injected = inject_adapter_loader(rewritten)
        body = injected.encode("utf-8")
        headers["content-length"] = str(len(body))
    elif kind == "css":
        try:
            css_str = body.decode("utf-8", errors="replace")
        except (LookupError, UnicodeDecodeError):
            css_str = body.decode("latin-1", errors="replace")
        rewritten = rewrite_css(
            css_str,
            proxy_base=proxy_base,
            source_origin=session.source_origin,
            page_url=upstream,
        )
        body = rewritten.encode("utf-8")
        headers["content-length"] = str(len(body))

    # Rewrite CSP if present so our injection + CDN pass-through work.
    if "content-security-policy" in {h.lower() for h in headers}:
        for k in list(headers.keys()):
            if k.lower() == "content-security-policy":
                headers[k] = rewrite_csp(
                    headers[k], proxy_base=proxy_base,
                )

    return Response(
        content=body,
        status_code=result.status,
        headers=headers,
        media_type=result.content_type or None,
    )
