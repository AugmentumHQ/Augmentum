"""Shared substrate for direct keyless data providers.

Every provider module builds its fetchers on :func:`fetch_json`, which
gives the whole layer the three behaviors the keyless API ecosystem
expects of a good citizen:

  * **Descriptive User-Agent** — MusicBrainz and Open Library REQUIRE
    an identifying UA (stock library UAs get throttled or 503'd);
    Open-Meteo and the gov services appreciate it.
  * **TTL response cache** — upstreams cache server-side anyway
    (USGS 60s, Open-Meteo model-update cadence); re-asking faster
    returns nothing new and burns goodwill quota.
  * **Per-provider min-interval throttle** — the 1 req/s class of
    policies (MusicBrainz, Open Library, Nominatim) enforced HERE so
    no caller has to remember.

Never raises: any failure logs a warning and returns ``None`` — the
calling verb degrades to an honest "couldn't reach that" speak line.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urlencode

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "Augmentum/1.0 (self-hosted personal AI hub)"

_CACHE_MAX_ENTRIES = 512

# key → (stored_at_monotonic, payload)
_cache: dict[str, tuple[float, Any]] = {}
# provider id → monotonic timestamp of last upstream call
_throttle_last: dict[str, float] = {}
# provider id → lock serializing that provider's upstream calls
_locks: dict[str, asyncio.Lock] = {}


def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    if not params:
        return url
    return f"{url}?{urlencode(sorted((k, str(v)) for k, v in params.items()))}"


def _cache_get(key: str, ttl_s: float) -> Any | None:
    hit = _cache.get(key)
    if hit is None:
        return None
    stored_at, payload = hit
    if time.monotonic() - stored_at >= ttl_s:
        _cache.pop(key, None)
        return None
    return payload


def _cache_put(key: str, payload: Any) -> None:
    if len(_cache) >= _CACHE_MAX_ENTRIES:
        # Evict the oldest entry — simple and sufficient at this size.
        oldest = min(_cache.items(), key=lambda kv: kv[1][0])[0]
        _cache.pop(oldest, None)
    _cache[key] = (time.monotonic(), payload)


def clear_cache() -> None:
    """Test hook."""
    _cache.clear()
    _throttle_last.clear()


async def fetch_json(
    provider: str,
    url: str,
    params: dict[str, Any] | None = None,
    *,
    ttl_s: float = 900.0,
    min_interval_s: float = 0.0,
    timeout_s: float = 8.0,
) -> Any | None:
    """GET ``url`` and decode JSON, with cache + throttle. None on failure."""
    key = _cache_key(url, params)
    cached = _cache_get(key, ttl_s)
    if cached is not None:
        return cached

    lock = _locks.setdefault(provider, asyncio.Lock())
    async with lock:
        # A concurrent caller may have populated the cache while we
        # waited on the lock.
        cached = _cache_get(key, ttl_s)
        if cached is not None:
            return cached

        if min_interval_s > 0:
            wait = (
                _throttle_last.get(provider, 0.0) + min_interval_s
                - time.monotonic()
            )
            if wait > 0:
                await asyncio.sleep(min(wait, 5.0))

        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=timeout_s,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
            ) as client:
                resp = await client.get(url, params=params)
            _throttle_last[provider] = time.monotonic()
            if resp.status_code != 200:
                log.warning(
                    "source_fetch_http_error",
                    provider=provider, status=resp.status_code, url=url,
                )
                return None
            data = resp.json()
        except Exception:  # noqa: BLE001 — sources soft-fail by contract
            log.warning("source_fetch_failed", provider=provider, url=url,
                        exc_info=True)
            return None

        _cache_put(key, data)
        return data
