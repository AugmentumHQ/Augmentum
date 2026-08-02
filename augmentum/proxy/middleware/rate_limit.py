"""Per-user/IP sliding window rate limiter middleware."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

import structlog

log = structlog.get_logger(__name__)


class RateLimitMiddleware:
    """Sliding window rate limiter.

    Raw ASGI (not BaseHTTPMiddleware) so StreamingResponse bodies flush
    chunk-by-chunk. BaseHTTPMiddleware wraps the response through an anyio
    memory stream that holds chunks until the upstream generator finishes,
    which surfaces to the user as "chat is not streamed."

    Groups endpoints into buckets with different limits:
    - chat: /api/chat, /v1/chat/completions (default 30 req/min)
    - image: /api/image/generate, /api/image/img2img, /api/image/inpaint (default 10 req/min)
    - voice: /ws/voice (default 5 req/min)
    - tools: /api/artifacts, /api/browse (default 20 req/min)
    - upload: /api/files/upload (default 30 req/min)
    - default: everything else (default 60 req/min)

    Health/static endpoints are exempt.
    """

    # Endpoint group definitions: prefix -> group name
    _GROUPS = {
        "/api/chat": "chat",
        "/v1/chat/completions": "chat",
        "/api/image/generate": "image",
        "/api/image/img2img": "image",
        "/api/image/inpaint": "image",
        "/ws/voice": "voice",
        "/api/artifacts/iterate": "image",
        "/api/artifacts/build-status": "default",
        "/api/browse/fetch": "tools",
        "/api/files/upload": "upload",
    }

    # Exempt paths (no rate limiting).
    #
    # /api/fabric/ exemption: peer-to-peer protocol traffic is signed
    # (authenticated by ed25519, not user tokens) and includes the
    # load-coordination poll loop that fires ~4 req/s during cold
    # model loads. The "default" 60/min would 429 mid-cold-load and
    # the sender treats 429 as PeerProtocolError — load gate breaks.
    # Operator-driven fabric admin endpoints (/pair-with-remote,
    # /discover, /unpair) sit on the same prefix; those are
    # admin-gated + infrequent so exemption is fine for them too.
    _EXEMPT = {
        "/", "/api/health", "/api/config", "/ui", "/api/models",
        "/metrics", "/api/fabric/",
    }

    def __init__(self, app: ASGIApp, limits: dict[str, int] | None = None, enabled: bool = True):
        self._app = app
        self._enabled = enabled
        self._limits = {
            "chat": 30,
            "image": 10,
            "voice": 5,
            "tools": 20,
            "upload": 30,
            "default": 60,
            **(limits or {}),
        }
        # {(client_key, group): deque of timestamps}
        self._windows: dict[tuple[str, str], deque] = defaultdict(deque)
        self._window_seconds = 60
        self._cleanup_counter = 0

    def _get_client_key(self, scope: Scope) -> str:
        """Get rate limit key — user_id for authenticated, IP for anonymous."""
        user = scope.get("user")
        if user:
            return f"user:{user.id}"
        # Fall back to IP for unauthenticated paths
        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for", b"").decode("latin-1")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        client = scope.get("client")
        return f"ip:{client[0]}" if client else "ip:unknown"

    def _get_group(self, path: str) -> str | None:
        """Return the rate limit group for a path, or None if exempt."""
        for exempt in self._EXEMPT:
            if path.startswith(exempt):
                return None
        for prefix, group in self._GROUPS.items():
            if path.startswith(prefix):
                return group
        return "default"

    def _cleanup_old_entries(self) -> None:
        """Periodically remove expired window entries."""
        self._cleanup_counter += 1
        if self._cleanup_counter < 100:
            return
        self._cleanup_counter = 0
        now = time.monotonic()
        cutoff = now - self._window_seconds
        empty_keys = []
        for key, timestamps in self._windows.items():
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            if not timestamps:
                empty_keys.append(key)
        for key in empty_keys:
            del self._windows[key]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Only rate-limit HTTP. WebSockets and lifespan pass through untouched
        # (matches prior BaseHTTPMiddleware behavior, which skipped non-http).
        if scope["type"] != "http" or not self._enabled:
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        group = self._get_group(path)
        if group is None:
            await self._app(scope, receive, send)
            return

        client_key = self._get_client_key(scope)
        key = (client_key, group)
        limit = self._limits.get(group, 60)
        now = time.monotonic()
        cutoff = now - self._window_seconds

        window = self._windows[key]
        # Remove expired entries
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= limit:
            retry_after = int(window[0] - cutoff) + 1
            log.warning("rate_limited", client=client_key, group=group, limit=limit, path=path)
            response = JSONResponse(
                status_code=429,
                content={
                    "error": f"Rate limit exceeded ({limit} requests per minute for {group} endpoints)",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
            await response(scope, receive, send)
            return

        window.append(now)
        self._cleanup_old_entries()

        await self._app(scope, receive, send)
