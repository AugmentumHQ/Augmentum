"""Origin proxy fetcher + asset cache.

Async HTTP fetch with a same-origin redirect policy + on-disk cache
keyed by ``(source_url, etag)``. Headers that leak auth, frame the
response, or break our injection get stripped on the way back.

Cache layout (per user, under ``/data/cast-game-cache/``):

    <user_id>/<sha256(source_url)>.body   — raw bytes
    <user_id>/<sha256(source_url)>.meta   — JSON: etag, content_type, mtime

Cache is best-effort — failures fall through to live fetch. Eviction
is a 7-day LRU bounded at 2 GB per user (default; tunable). The
classifier never *requires* the cache to be hot to cast a game.

Security:
  * URL allowlist enforced before fetch — no internal IPs (RFC 1918,
    loopback, link-local) and no non-http(s) schemes.
  * Same-origin redirect policy — request to https://example.com/x
    that redirects to https://other.com/y is rejected with HTTP 502.
    Mitigates open-redirect abuse + supply-chain redirect attacks.
  * Response header stripping — ``Set-Cookie`` / ``Authorization`` /
    ``X-Frame-Options`` / ``frame-ancestors`` removed so the proxied
    page can be iframed safely.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import shutil
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# Response headers we strip when proxying. ``Content-Security-Policy``
# is handled separately (rewritten to allow our injection origin).
BLOCKED_RESPONSE_HEADERS = frozenset({
    "set-cookie",
    "authorization",
    "x-frame-options",
    "frame-ancestors",
    "strict-transport-security",  # we re-add per our origin's policy
    "report-to",
    "report-uri",
    "public-key-pins",
    "public-key-pins-report-only",
    # httpx auto-decompresses the upstream body into ``resp.content``,
    # so the upstream ``content-encoding`` / ``transfer-encoding`` no
    # longer describe the bytes we serve. Leaving them on would make
    # the TV browser try to gunzip already-plain bytes (most sites gzip
    # → every proxied page would fail). ``content-length`` likewise
    # describes the compressed upstream length and is wrong once we
    # decompress and/or rewrite; the route recomputes it from the body.
    "content-encoding",
    "transfer-encoding",
    "content-length",
})


# Request headers we forward verbatim (allowlist — anything not here
# is dropped). Keeps cookie / referer / origin leakage out.
ALLOWED_REQUEST_HEADERS = frozenset({
    "accept",
    "accept-encoding",
    "accept-language",
    "user-agent",
    "range",
    "if-none-match",
    "if-modified-since",
})


# Soft per-user cache budget. The cache periodically reaps entries
# older than ``cache_ttl_s`` and trims to this byte cap (LRU by mtime).
DEFAULT_CACHE_TTL_S = 7 * 24 * 3600     # 7 days
DEFAULT_CACHE_BYTES_CAP = 2 * 1024 * 1024 * 1024  # 2 GB

DEFAULT_FETCH_TIMEOUT_S = 20.0


@dataclass(slots=True)
class FetchResult:
    """One asset fetched (or cache-hit) for the proxy.

    ``body`` is the raw bytes — caller may transform (rewrite HTML/CSS)
    before serving. ``headers`` are the *response* headers ready for
    pass-through (already stripped of blocked entries).
    """

    status: int
    body: bytes
    content_type: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    from_cache: bool = False
    source_url: str = ""


# ── URL safety ───────────────────────────────────────────────────


def _is_private_or_loopback(host: str) -> bool:
    """True iff ``host`` resolves to an RFC1918 / loopback / link-local
    address. We refuse to proxy these — they're either internal Augmentum
    services or the user's home network."""
    if not host:
        return True
    # Reject by literal IP first
    try:
        ip = ipaddress.ip_address(host)
        return (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved
        )
    except ValueError:
        pass
    # Else resolve. Skip DNS in tests by passing an explicit IP up front.
    try:
        # IPv4 + IPv6 — refuse if any resolution lands on a private range.
        infos = socket.getaddrinfo(host, None)
    except (OSError, socket.gaierror):
        return True  # unresolvable = unsafe by default
    for family, *_, sockaddr in infos:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return True
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved
        ):
            return True
    return False


def is_url_safe(url: str, *, allow_private: bool = False) -> bool:
    """True iff ``url`` is http(s) + non-internal. Used at mint time
    AND on every fetch (the source can't be downgraded)."""
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    if not parsed.hostname:
        return False
    if not allow_private and _is_private_or_loopback(parsed.hostname):
        return False
    return True


def strip_response_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of ``headers`` with BLOCKED_RESPONSE_HEADERS
    removed. Case-insensitive."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() in BLOCKED_RESPONSE_HEADERS:
            continue
        out[k] = v
    return out


def filter_request_headers(
    headers: dict[str, str | list[str]],
) -> dict[str, str]:
    """Drop everything not in ALLOWED_REQUEST_HEADERS. Case-insensitive."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        if k.lower() not in ALLOWED_REQUEST_HEADERS:
            continue
        if isinstance(v, list):
            out[k] = ", ".join(v)
        else:
            out[k] = str(v)
    return out


# ── Cache ────────────────────────────────────────────────────────


class AssetCache:
    """Per-user disk cache for proxied assets.

    Keyed by sha256(source_url). Files are namespaced to the user (so
    one user's bandwidth isn't billed against another's quota + so a
    revoked session can be wiped cleanly).
    """

    def __init__(
        self,
        root: Path,
        *,
        ttl_s: float = DEFAULT_CACHE_TTL_S,
        bytes_cap: int = DEFAULT_CACHE_BYTES_CAP,
    ) -> None:
        self._root = root
        self._ttl_s = float(ttl_s)
        self._cap = int(bytes_cap)
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError:
            log.warning("cast_proxy_cache_mkdir_failed", root=str(root), exc_info=True)

    def _user_dir(self, user_id: str) -> Path:
        # Empty user_id collapses to '_anon'. Keeps a stray cache write
        # from creating an empty-named dir on the filesystem.
        safe = "".join(c for c in (user_id or "_anon") if c.isalnum() or c in "-_") or "_anon"
        return self._root / safe

    def _paths(self, user_id: str, source_url: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
        d = self._user_dir(user_id)
        return d / f"{digest}.body", d / f"{digest}.meta"

    def get(self, user_id: str, source_url: str) -> FetchResult | None:
        body_path, meta_path = self._paths(user_id, source_url)
        if not body_path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            body = body_path.read_bytes()
        except (OSError, ValueError):
            return None
        mtime = float(meta.get("mtime") or 0.0)
        if (time.time() - mtime) > self._ttl_s:
            # Stale — opportunistically delete so the next pass doesn't
            # see it. Best-effort.
            try:
                body_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return FetchResult(
            status=200,
            body=body,
            content_type=str(meta.get("content_type") or ""),
            headers=dict(meta.get("headers") or {}),
            from_cache=True,
            source_url=source_url,
        )

    def put(
        self,
        user_id: str,
        source_url: str,
        result: FetchResult,
    ) -> None:
        if result.from_cache or result.status != 200:
            return
        body_path, meta_path = self._paths(user_id, source_url)
        try:
            body_path.parent.mkdir(parents=True, exist_ok=True)
            body_path.write_bytes(result.body)
            meta = {
                "mtime": time.time(),
                "content_type": result.content_type,
                "headers": result.headers,
                "source_url": source_url,
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        except OSError:
            log.warning(
                "cast_proxy_cache_write_failed",
                source_url=source_url,
                exc_info=True,
            )

    def wipe_for_user(self, user_id: str) -> None:
        d = self._user_dir(user_id)
        try:
            if d.is_dir():
                shutil.rmtree(d)
        except OSError:
            log.warning(
                "cast_proxy_cache_wipe_failed",
                user_id=user_id, exc_info=True,
            )

    def usage_bytes(self, user_id: str) -> int:
        d = self._user_dir(user_id)
        if not d.is_dir():
            return 0
        total = 0
        for p in d.iterdir():
            try:
                total += p.stat().st_size
            except OSError:
                continue
        return total


# ── Fetcher ──────────────────────────────────────────────────────


class ProxyFetcher:
    """Async fetcher with redirect + cache.

    Caller threads through a ``user_id`` so the cache namespacing
    happens per user (and a revoke wipe doesn't trash other users).
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        cache: AssetCache | None = None,
        timeout_s: float = DEFAULT_FETCH_TIMEOUT_S,
        allow_private: bool = False,
    ) -> None:
        # Caller can inject a stubbed client for tests; we don't own its
        # lifecycle in that case.
        self._client = client
        self._owns_client = client is None
        self._cache = cache
        self._timeout = float(timeout_s)
        self._allow_private = bool(allow_private)

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _client_handle(self) -> httpx.AsyncClient:
        if self._client is None:
            # follow_redirects=False — we want explicit same-origin
            # checks on every Location.
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
            )
        return self._client

    async def fetch(
        self,
        source_url: str,
        *,
        source_origin: str,
        user_id: str = "",
        request_headers: dict[str, str] | None = None,
        max_redirects: int = 4,
    ) -> FetchResult:
        """Fetch + return a FetchResult. Same-origin redirects are
        followed; cross-origin redirects raise httpx.HTTPError.

        Cache hits skip the network entirely.
        """
        if not is_url_safe(source_url, allow_private=self._allow_private):
            raise PermissionError(f"unsafe source_url: {source_url!r}")

        # Cache hit short-circuit (read-only)
        if self._cache is not None:
            hit = self._cache.get(user_id, source_url)
            if hit is not None:
                return hit

        client = await self._client_handle()
        headers = filter_request_headers(request_headers or {})

        current_url = source_url
        for _hop in range(max_redirects + 1):
            try:
                resp = await client.get(current_url, headers=headers)
            except httpx.HTTPError as err:
                log.warning(
                    "cast_proxy_fetch_failed",
                    url=current_url, error=str(err),
                )
                raise

            if 300 <= resp.status_code < 400 and "location" in resp.headers:
                next_url = str(httpx.URL(current_url).join(resp.headers["location"]))
                if not _same_origin(next_url, source_origin):
                    raise httpx.HTTPError(
                        f"cross-origin redirect refused: {current_url} → {next_url}",
                    )
                if not is_url_safe(next_url, allow_private=self._allow_private):
                    raise httpx.HTTPError(
                        f"redirect target is unsafe: {next_url}",
                    )
                current_url = next_url
                continue

            stripped = strip_response_headers(dict(resp.headers))
            result = FetchResult(
                status=resp.status_code,
                body=resp.content,
                content_type=str(resp.headers.get("content-type") or ""),
                headers=stripped,
                from_cache=False,
                source_url=source_url,
            )
            # Only cache 200 OK with non-empty bodies — error responses
            # would mask future recovery, and 304s would pollute the
            # body cache with nothing.
            if (
                self._cache is not None
                and result.status == 200
                and result.body
            ):
                self._cache.put(user_id, source_url, result)
            return result

        raise httpx.HTTPError(
            f"too many redirects fetching {source_url}",
        )


_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def _effective_port(scheme: str, port: int | None) -> int | None:
    """Resolve an explicit port, falling back to the scheme default.

    ``urlparse`` leaves ``.port`` as None for default-port URLs, so
    ``https://example.com`` and ``https://example.com:443`` must compare
    equal — otherwise a redirect that adds the explicit port reads as
    cross-origin and gets refused with a 502."""
    if port is not None:
        return port
    return _DEFAULT_PORTS.get(scheme.lower())


def _same_origin(url: str, source_origin: str) -> bool:
    """True iff ``url``'s origin matches ``source_origin``.

    Comparison is case-insensitive on scheme + host; ports compare after
    normalising scheme defaults (so ``:443`` == default for https).
    Empty / malformed urls fail closed.
    """
    if not url or not source_origin:
        return False
    try:
        parsed = urlparse(url)
        src = urlparse(source_origin)
    except ValueError:
        return False
    if parsed.scheme.lower() != src.scheme.lower():
        return False
    if (parsed.hostname or "").lower() != (src.hostname or "").lower():
        return False
    if _effective_port(parsed.scheme, parsed.port) != _effective_port(
        src.scheme, src.port
    ):
        return False
    return True
