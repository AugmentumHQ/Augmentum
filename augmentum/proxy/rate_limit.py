"""Per-IP token bucket rate limiter middleware."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Paths exempt from rate limiting (health checks, static assets, etc.)
_EXEMPT_PREFIXES = ("/ui", "/api/health", "/mcp", "/ws")


class _Bucket:
    __slots__ = ("tokens", "last_refill")

    def __init__(self, capacity: float) -> None:
        self.tokens = capacity
        self.last_refill = time.monotonic()


class RateLimitMiddleware:
    """Token-bucket rate limiter keyed by client IP.

    Each IP gets ``capacity`` tokens, refilled at ``rate`` tokens/second.
    A request consumes 1 token. When empty, returns 429.
    Stale buckets are pruned periodically to prevent memory growth.

    Only mutating requests (POST/PUT/PATCH/DELETE) are rate-limited.
    GET/HEAD/OPTIONS are exempt because the bundled UI fires many
    parallel reads on page load and during polling.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        rate: float = 2.0,
        capacity: int = 120,
    ) -> None:
        self._app = app
        self._rate = rate          # tokens per second
        self._capacity = capacity  # max burst
        self._buckets: dict[str, _Bucket] = defaultdict(lambda: _Bucket(self._capacity))
        self._last_prune = time.monotonic()
        self._prune_interval = 300.0  # seconds

    def _get_client_ip(self, scope: Scope) -> str:
        # X-Forwarded-For is only trusted behind a known reverse proxy.
        # Default to direct peer address for safety.
        client = scope.get("client")
        return client[0] if client else "unknown"

    def _try_consume(self, ip: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets[ip]

        # Refill tokens
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self._capacity, bucket.tokens + elapsed * self._rate)
        bucket.last_refill = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True
        return False

    def _maybe_prune(self) -> None:
        now = time.monotonic()
        if now - self._last_prune < self._prune_interval:
            return
        self._last_prune = now
        # Remove buckets that have been full (idle) for > prune_interval
        stale = [
            ip for ip, b in self._buckets.items()
            if (now - b.last_refill) > self._prune_interval
        ]
        for ip in stale:
            del self._buckets[ip]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        if any(path.startswith(p) for p in _EXEMPT_PREFIXES) or path == "/":
            await self._app(scope, receive, send)
            return

        # Only rate-limit mutating methods — the UI fires many parallel
        # GET requests on startup and during polling that would exhaust
        # any reasonable bucket.  Reads are cheap and safe to allow freely.
        method = scope.get("method", "GET")
        if method in ("GET", "HEAD", "OPTIONS"):
            await self._app(scope, receive, send)
            return

        ip = self._get_client_ip(scope)
        self._maybe_prune()

        if not self._try_consume(ip):
            log.warning("rate_limited", client_ip=ip, path=path, method=method)
            response = JSONResponse(
                {"error": "Too many requests. Please slow down."},
                status_code=429,
                headers={"Retry-After": str(int(self._capacity / self._rate))},
            )
            await response(scope, receive, send)
            return

        await self._app(scope, receive, send)
