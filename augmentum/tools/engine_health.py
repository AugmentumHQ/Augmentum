"""Client-side SearXNG engine health tracking.

SearXNG suspends engines internally when upstream providers rate-limit,
CAPTCHA, or block it, and reports them per-response in
``unresponsive_engines`` as ``[engine, reason]`` pairs — but it never
exposes the remaining suspension time over the JSON API. This module
keeps a process-wide client-side estimate so callers can:

- pick a fallback engine set from engines that are actually answering,
  instead of a static list that may itself be suspended
- tell the user "engines are rate-limited, retry in ~N min" instead of
  returning a silent empty result

The tracker is fed by every SearXNG response that passes through
:class:`~augmentum.tools.web_search.WebSearchTool` or
:func:`searxng_search_resilient`, so user searches, briefings, and
recurring searches all share one view of engine health.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)

# Reason-substring → estimated suspension TTL (seconds). SearXNG's own
# suspension windows vary per engine and error class; these mirror its
# defaults loosely. CAPTCHAs and access-denied blocks persist longest;
# plain rate limits clear faster. First match wins.
_REASON_TTLS: tuple[tuple[str, float], ...] = (
    ("captcha", 600.0),
    ("access denied", 600.0),
    ("too many requests", 300.0),
)
_DEFAULT_TTL = 180.0

# Preference-ordered candidates for a constrained fallback query.
# Ordering from 2026-06-10 live probes against this deployment: bing and
# duckduckgo answered every query; wikipedia never rate-limits but only
# covers knowledge queries; brave recovers quickly but suspends after
# short bursts; mojeek mostly denies access.
_FALLBACK_CANDIDATES: tuple[str, ...] = (
    "bing", "duckduckgo", "wikipedia", "brave", "mojeek",
)


class EngineHealthTracker:
    """Estimated per-engine suspension state, learned from responses.

    Time source is injectable for tests; defaults to ``time.monotonic``
    so wall-clock jumps can't fake an expiry.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._suspended_until: dict[str, float] = {}
        self._reasons: dict[str, str] = {}

    def record_response(self, data: dict[str, Any]) -> None:
        """Learn from one SearXNG JSON response.

        Engines in ``unresponsive_engines`` get a reason-aware TTL;
        engines that contributed results get their suspension cleared
        (they're demonstrably answering again).
        """
        if not isinstance(data, dict):
            return
        now = self._clock()
        for entry in data.get("unresponsive_engines") or []:
            engine, reason = _parse_unresponsive_entry(entry)
            if not engine:
                continue
            until = now + _ttl_for_reason(reason)
            # Never shorten an existing estimate from a fresher, longer one.
            if until > self._suspended_until.get(engine, 0.0):
                self._suspended_until[engine] = until
                self._reasons[engine] = reason
        for result in data.get("results") or []:
            if not isinstance(result, dict):
                continue
            names = set()
            single = str(result.get("engine") or "").strip().lower()
            if single:
                names.add(single)
            for name in result.get("engines") or []:
                name = str(name).strip().lower()
                if name:
                    names.add(name)
            for name in names:
                self._suspended_until.pop(name, None)
                self._reasons.pop(name, None)

    def is_suspended(self, engine: str) -> bool:
        until = self._suspended_until.get(engine.strip().lower())
        return until is not None and until > self._clock()

    def healthy_fallback_engines(self) -> str:
        """Comma-joined fallback candidates not currently suspended.

        Empty string means every candidate is suspended — callers should
        surface a rate-limited error rather than reissue a doomed query.
        """
        healthy = [e for e in _FALLBACK_CANDIDATES if not self.is_suspended(e)]
        return ",".join(healthy)

    def suspended_summary(self) -> list[dict[str, Any]]:
        """Currently-suspended engines with estimated seconds remaining."""
        now = self._clock()
        out: list[dict[str, Any]] = []
        for engine, until in sorted(self._suspended_until.items()):
            remaining = until - now
            if remaining <= 0:
                continue
            out.append({
                "engine": engine,
                "reason": self._reasons.get(engine, ""),
                "retry_in_seconds": int(remaining),
            })
        return out

    def earliest_retry_seconds(self) -> int | None:
        """Seconds until the first suspended fallback candidate recovers.

        None when no fallback candidate is suspended (nothing to wait
        for) — distinct from 0, which would mean "recovering right now".
        """
        now = self._clock()
        remaining = [
            self._suspended_until[e] - now
            for e in _FALLBACK_CANDIDATES
            if self._suspended_until.get(e, 0.0) > now
        ]
        if not remaining:
            return None
        return max(1, int(min(remaining)))


def _parse_unresponsive_entry(entry: Any) -> tuple[str, str]:
    """``unresponsive_engines`` entries are ``[engine, reason]`` pairs."""
    try:
        if isinstance(entry, list | tuple) and entry:
            engine = str(entry[0]).strip().lower()
            reason = str(entry[1]) if len(entry) > 1 else ""
            return engine, reason
    except Exception:
        pass
    return "", ""


def _ttl_for_reason(reason: str) -> float:
    lowered = reason.lower()
    for needle, ttl in _REASON_TTLS:
        if needle in lowered:
            return ttl
    return _DEFAULT_TTL


# Process-wide singleton. Suspension state describes the SearXNG
# instance, not a user, so sharing across tenants is correct.
TRACKER = EngineHealthTracker()


async def searxng_search_resilient(
    http_client: httpx.AsyncClient,
    base_url: str,
    query: str,
    *,
    categories: str | None = None,
    timeout: float = 15.0,
    headers: dict[str, str] | None = None,
    tracker: EngineHealthTracker | None = None,
) -> dict[str, Any]:
    """One SearXNG search with health recording + healthy-engine fallback.

    Issues the full fan-out, records engine health, and — when the
    response is empty *because* engines were unresponsive (infra
    failure, not a genuine no-match) — reissues once constrained to
    currently-healthy fallback engines. The fallback reissue is
    best-effort: its network errors are swallowed and the primary
    response is returned instead.

    Returns the final SearXNG JSON dict. When the fallback reissue
    produced the results, the dict carries ``augmentum_fallback_used:
    True``. Raises on network/HTTP failure of the primary call.
    """
    tracker = tracker or TRACKER
    params: dict[str, str] = {"q": query, "format": "json"}
    if categories:
        params["categories"] = categories
    resp = await http_client.get(
        f"{base_url.rstrip('/')}/search",
        params=params,
        timeout=timeout,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        data = {}
    tracker.record_response(data)

    if data.get("results") or not data.get("unresponsive_engines"):
        return data

    engines = tracker.healthy_fallback_engines()
    if not engines:
        log.warning(
            "searxng_all_fallback_engines_suspended",
            query=query[:80],
            suspended=tracker.suspended_summary(),
        )
        return data

    log.info(
        "searxng_fallback_reissue",
        query=query[:80],
        engines=engines,
    )
    try:
        fb_resp = await http_client.get(
            f"{base_url.rstrip('/')}/search",
            params={**params, "engines": engines},
            timeout=timeout,
            headers=headers,
        )
        fb_resp.raise_for_status()
        fb_data = fb_resp.json()
    except Exception as exc:
        log.warning(
            "searxng_fallback_reissue_failed",
            query=query[:80],
            error=str(exc)[:200],
        )
        return data
    if not isinstance(fb_data, dict):
        return data
    tracker.record_response(fb_data)
    if fb_data.get("results"):
        fb_data["augmentum_fallback_used"] = True
        return fb_data
    return data
