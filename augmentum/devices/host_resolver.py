"""LAN-reachable host resolution for TV-bound URLs.

The augmentum container often serves multiple "names":

  - `localhost` / `127.0.0.1` — when the operator opens the UI on the
    same machine augmentum runs on
  - `host.docker.internal` — Docker Desktop's name for the host
  - The host's LAN IP (e.g. `192.168.1.10`) — what other devices on
    the network use

Most of those don't matter. But when augmentum hands a URL to a TV
("here's the stream, fetch it"), the URL must be something the **TV**
can resolve. `localhost` means the TV's own loopback; `host.docker.
internal` means nothing outside Docker; only the LAN IP works.

This module solves that without making the user configure anything.
Every incoming HTTP request brushes past `observe_request()`, which
remembers the host whenever it's non-loopback. When the cast
machinery later asks for `public_host()`, the resolver returns:

  1. An explicit operator override if set (`AUGMENTUM_PUBLIC_HOST`)
  2. The current request's Host header if it's non-loopback
  3. The most recently learned non-loopback Host
  4. Empty string — caller decides whether to error or fall through

The auto-learn step means: as soon as ANY device on the LAN opens
augmentum (your phone, another laptop), the server learns its
LAN-visible name. From then on, casts initiated from a localhost
session still produce TV-reachable URLs.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

log = get_logger(__name__)


_LOOPBACK_HOSTNAMES: frozenset[str] = frozenset({
    "",
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "host.docker.internal",  # Docker-only, not LAN-reachable
})


def _strip_port(host: str) -> str:
    """Return the bare hostname half of a `host:port` value."""
    h = (host or "").strip()
    if not h:
        return ""
    # IPv6 in brackets: [::1]:6443
    if h.startswith("["):
        end = h.find("]")
        if end > 0:
            return h[1:end].lower()
        return h.lower()
    # IPv4 / hostname
    return h.split(":", 1)[0].strip().lower()


def _port(host: str) -> str:
    """Return the port half of a host value, if it has one."""
    h = (host or "").strip()
    if not h:
        return ""
    if h.startswith("["):
        end = h.find("]")
        if end > 0 and h[end + 1:].startswith(":"):
            return h[end + 2:].strip()
        return ""
    if h.count(":") == 1:
        return h.rsplit(":", 1)[1].strip()
    return ""


def _is_loopback_host(host: str) -> bool:
    bare = _strip_port(host)
    if bare in _LOOPBACK_HOSTNAMES:
        return True
    if bare.endswith(".localhost"):
        return True
    return False


def _visible_scheme_from_request(request: "Request") -> str:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    if forwarded in {"http", "https"}:
        return forwarded
    return request.url.scheme or ""


def _visible_scheme_from_scope(scope: dict) -> str:
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-proto" and value:
            proto = value.decode("latin-1", errors="ignore").split(",", 1)[0].strip()
            if proto in {"http", "https"}:
                return proto
            break
    return str(scope.get("scheme") or "")


def _host_preference(host: str, scheme: str = "") -> int:
    """Prefer the browser-facing HTTPS proxy over direct backend HTTP."""
    if (scheme or "").strip().lower() == "https":
        return 30
    if _port(host) == "6443":
        return 20
    if _port(host) == "6100":
        return 5
    return 10


class PublicHostResolver:
    """Tracks the LAN-reachable host:port of augmentum for TV-bound URLs.

    Thread-safe (the observe path can fire from any worker; the read
    path runs on the route layer). Reads are wait-free; writes lock
    only briefly to update the most-recent value.
    """

    def __init__(self, *, configured: str = "") -> None:
        self._configured: str = (configured or "").strip()
        self._learned: str = ""
        self._learned_preference: int = 0
        self._learned_at: float = 0.0
        self._lock = threading.Lock()

    # ---- configuration -------------------------------------------------------

    def configure(self, host: str) -> None:
        """Operator-supplied override, e.g. from `AUGMENTUM_PUBLIC_HOST`."""
        with self._lock:
            self._configured = (host or "").strip()

    # ---- learning ------------------------------------------------------------

    def observe_request(self, request: "Request") -> None:
        """Remember the host of any non-loopback request.

        Called from middleware on every request. Cheap — early-exits on
        loopback (the common case for same-host development) without
        touching the lock.
        """
        host = self._extract_host(request)
        self._learn(host, scheme=_visible_scheme_from_request(request))

    def observe_scope(self, scope: dict) -> None:
        """ASGI-native version of observe_request.

        Used by raw ASGI middleware that doesn't construct a Request.
        Identical semantics; just reads from the scope's header bytes.
        """
        if scope.get("type") != "http":
            return
        host = ""
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-host" and value:
                host = value.decode("latin-1", errors="ignore").split(",")[0].strip()
                break
        if not host:
            for name, value in scope.get("headers", []):
                if name == b"host" and value:
                    host = value.decode("latin-1", errors="ignore").strip()
                    break
        if not host:
            scope_host = scope.get("server")
            if isinstance(scope_host, tuple) and scope_host:
                host_part = scope_host[0]
                port = scope_host[1] if len(scope_host) > 1 else None
                if host_part:
                    host = f"{host_part}:{port}" if port else str(host_part)
        self._learn(host, scheme=_visible_scheme_from_scope(scope))

    def _learn(self, host: str, *, scheme: str = "") -> None:
        if not host or _is_loopback_host(host):
            return
        preference = _host_preference(host, scheme)
        with self._lock:
            if self._learned and host != self._learned and preference < self._learned_preference:
                return
            if host != self._learned:
                # When two clients hit the server on different reachable
                # addresses (e.g. LAN 192.168.x + Tailscale 100.x), every
                # request flips this and would log a line. The log is only
                # useful the first time we learn a host — drop to debug
                # for subsequent flips so the steady-state log is quiet.
                first_learn = not self._learned
                self._learned = host
                self._learned_preference = preference
                self._learned_at = time.time()
                if first_learn:
                    log.info("public_host_learned", host=host)
                else:
                    log.debug("public_host_changed", host=host)

    # ---- query ---------------------------------------------------------------

    def public_host(self, request: "Request | None" = None) -> str:
        """Best-known LAN-reachable host:port. Empty if nothing known yet."""
        with self._lock:
            configured = self._configured
            learned = self._learned

        if configured:
            return configured

        if request is not None:
            current = self._extract_host(request)
            if current and not _is_loopback_host(current):
                return current

        return learned

    def public_url(
        self,
        path: str,
        *,
        request: "Request | None" = None,
        scheme: str = "",
    ) -> str:
        """Build a fully-qualified URL using the best-known public host.

        Returns empty string if no public host is known and the caller
        provided no request to fall back on. Caller decides how to
        handle that — typical move is to use the request URL as-is and
        log a warning that the TV may not be able to reach it.
        """
        host = self.public_host(request=request)
        if not host:
            return ""
        chosen_scheme = (scheme or "").strip()
        if not chosen_scheme and request is not None:
            chosen_scheme = _visible_scheme_from_request(request) or "https"
        if not chosen_scheme:
            chosen_scheme = "https"
        path_part = path if path.startswith("/") else f"/{path}"
        return f"{chosen_scheme}://{host}{path_part}"

    def is_loopback(self, host: str) -> bool:
        return _is_loopback_host(host)

    # ---- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_host(request: "Request") -> str:
        """Pull a Host value from the request, preferring proxy-set headers."""
        for header in ("x-forwarded-host", "host"):
            value = request.headers.get(header, "")
            if value:
                # X-Forwarded-Host may carry a chain — first is the user-facing one
                return value.split(",")[0].strip()
        # Fall back to scope's authority if no header (rare).
        scope_host = request.scope.get("server")
        if isinstance(scope_host, tuple) and scope_host:
            host_part = scope_host[0]
            port = scope_host[1] if len(scope_host) > 1 else None
            if host_part:
                return f"{host_part}:{port}" if port else str(host_part)
        return ""

    def state(self) -> dict[str, str]:
        """Diagnostic snapshot — useful for debug routes."""
        with self._lock:
            return {
                "configured": self._configured,
                "learned": self._learned,
                "learned_preference": str(self._learned_preference),
                "learned_at": str(self._learned_at) if self._learned_at else "",
            }
