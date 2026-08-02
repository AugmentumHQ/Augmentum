"""Fetch the Docker host's RAM/CPU from an optional host-side agent.

When Augmentum runs inside a container, ``psutil`` only sees the
container's view of the system — on Docker Desktop that's the Linux/WSL2
VM, whose RAM total and CPU usage diverge from what the host OS's Task
Manager / Activity Monitor reports. The host OS is otherwise opaque to
code inside the container.

If the operator runs ``scripts/host_stats_agent.py`` on the host, this
module fetches its readings so the resource panel can show "host" and
"container" side by side. When the agent isn't running, every probe
returns ``None`` cheaply (fast-fail + 60s backoff) and the UI falls back
to the container-only view.

Configuration (all optional):
  - ``AUGMENTUM_HOST_STATS_URL``   — full URL to the agent's /stats
    endpoint. Default when running in a container:
    ``http://host.docker.internal:6109/stats``. Unset and not in a
    container → host probing is disabled.
  - ``AUGMENTUM_HOST_STATS_TOKEN`` — shared secret appended as
    ``?token=…`` (matches the agent's ``--token`` flag).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_DEFAULT_CONTAINER_URL = "http://host.docker.internal:6109/stats"

# Serve a cached reading for this long before re-fetching. The agent
# samples CPU over ~100ms per request, so re-fetching every few seconds
# is fine; this just keeps a burst of concurrent UI polls to one GET.
_HIT_TTL_S = 5.0
# When the agent isn't reachable, don't retry on every poll — the common
# case is "no agent installed" and a refused connection every 3s is just
# noise. Re-probe at most this often while it's down.
_MISS_RETRY_S = 60.0
# Per-request timeout — the agent is local (host.docker.internal or
# loopback), so a missing one should fail fast rather than stall the
# resource panel. Connection-refused returns in <10ms; the previous
# connect=1.0/read=2.0 budget added up to 3s of latency to the first
# call after each _MISS_RETRY_S window when no agent is installed.
_TIMEOUT = httpx.Timeout(connect=0.2, read=0.5, write=0.5, pool=0.5)


@dataclass
class HostStats:
    ram_total_mb: int = 0
    ram_used_mb: int = 0
    ram_free_mb: int = 0
    cpu_pct: float = 0.0
    cpu_count: int = 0
    os_name: str = ""
    hostname: str = ""


# Module-level cache. ``_cache`` holds the last successful reading;
# ``_cache_at`` is when it was fetched; ``_last_attempt_at`` is when we
# last *tried* (success or failure) so misses can back off.
_cache: HostStats | None = None
_cache_at: float = 0.0
_last_attempt_at: float = 0.0
_logged_discovery = False
_lock: asyncio.Lock | None = None


def _resolve_url() -> str:
    override = os.environ.get("AUGMENTUM_HOST_STATS_URL", "").strip()
    if override:
        return override
    # Only auto-probe the default when we're actually containerised —
    # on bare metal psutil already reads the real host.
    if os.path.exists("/.dockerenv"):
        return _DEFAULT_CONTAINER_URL
    return ""


def _with_token(url: str) -> str:
    token = os.environ.get("AUGMENTUM_HOST_STATS_TOKEN", "").strip()
    if not token:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}token={token}"


def _parse(payload: dict) -> HostStats:
    ram = payload.get("ram") or {}
    return HostStats(
        ram_total_mb=int(ram.get("total_mb") or 0),
        ram_used_mb=int(ram.get("used_mb") or 0),
        ram_free_mb=int(ram.get("free_mb") or 0),
        cpu_pct=round(float(payload.get("cpu_pct") or 0.0), 1),
        cpu_count=int(payload.get("cpu_count") or 0),
        os_name=str(payload.get("os") or ""),
        hostname=str(payload.get("hostname") or ""),
    )


async def probe_host_stats(
    http_client: httpx.AsyncClient, *, cache_only: bool = False,
) -> HostStats | None:
    """Return the host machine's RAM/CPU, or ``None`` if no agent is up.

    Cached for ``_HIT_TTL_S`` on success; backs off ``_MISS_RETRY_S``
    between attempts while the agent is unreachable. Never raises.

    ``cache_only=True`` is the read-path contract (``GET /status``): never make
    the HTTP call inline — return the last-known reading (even if stale) or
    ``None``. The background sampler owns the refresh.
    """
    global _cache, _cache_at, _last_attempt_at, _logged_discovery, _lock

    url = _resolve_url()
    if not url:
        return None

    now = time.monotonic()
    if _cache is not None and (now - _cache_at) < _HIT_TTL_S:
        return _cache
    if cache_only:
        # Read path never probes — serve last-known (possibly stale) or None.
        return _cache
    if _cache is None and (now - _last_attempt_at) < _MISS_RETRY_S:
        return None

    if _lock is None:
        _lock = asyncio.Lock()

    async with _lock:
        # Re-check under the lock — a concurrent caller may have just
        # refreshed (or just failed and armed the backoff).
        now = time.monotonic()
        if _cache is not None and (now - _cache_at) < _HIT_TTL_S:
            return _cache
        if _cache is None and (now - _last_attempt_at) < _MISS_RETRY_S:
            return None

        _last_attempt_at = now
        try:
            resp = await http_client.get(_with_token(url), timeout=_TIMEOUT)
            resp.raise_for_status()
            stats = _parse(resp.json())
        except Exception:
            # Down / not installed / unparseable — the overwhelmingly
            # common case. Drop to debug so it isn't log spam.
            log.debug("host_stats_probe_failed", url=url, exc_info=True)
            _cache = None
            return None

        if not _logged_discovery:
            log.info("host_stats_agent_detected", url=url,
                     host=stats.hostname, os=stats.os_name)
            _logged_discovery = True
        _cache = stats
        _cache_at = time.monotonic()
        return stats
