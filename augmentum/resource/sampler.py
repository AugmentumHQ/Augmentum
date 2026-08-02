"""Background resource sampler — keeps the heavy probe caches warm OFF the
request path so ``GET /api/resources/status`` only ever assembles from cache.

Spec §4.5/§4.6: the read path must never ``await`` a live docker/HTTP/nvidia-smi
probe. The ledger snapshot, sidecar-container probe, and host-stats probe each
own a short-TTL cache; this loop refreshes them on an interval shorter than
their TTLs, so a request handler's ``collect()`` / ``probe_*()`` calls always
hit a fresh cache and return in dict-read time regardless of consumer health.

Adaptive cadence: sample fast while the panel is actively polled, slow when
idle, so an unwatched panel costs almost nothing. The loop never raises — a
wedged daemon degrades to a stale cache, not a crash — and self-accounts its
own wall-clock so a slow pass backs the cadence off.
"""

from __future__ import annotations

import asyncio
import time

from augmentum.resource.container_probe import probe_sidecar_containers
from augmentum.resource.host_probe import probe_host_stats
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Sample every ACTIVE_INTERVAL_S while the panel was polled within
# ACTIVE_WINDOW_S; otherwise back off to IDLE_INTERVAL_S. ACTIVE is below the
# caches' TTLs (ledger 8s, container 6s, host 5s) so warm reads never go cold.
_ACTIVE_INTERVAL_S = 3.0
_IDLE_INTERVAL_S = 20.0
_ACTIVE_WINDOW_S = 30.0


def _interval(app_state) -> float:
    last = getattr(app_state, "resource_panel_last_access", 0.0)
    try:
        active = (time.monotonic() - float(last)) < _ACTIVE_WINDOW_S
    except (TypeError, ValueError):
        active = False
    return _ACTIVE_INTERVAL_S if active else _IDLE_INTERVAL_S


async def _sample_once(app_state) -> float:
    """Refresh every cache once; return wall-clock seconds spent. Never raises."""
    started = time.monotonic()
    ledger = getattr(app_state, "resource_ledger", None)
    if ledger is not None:
        try:
            await ledger.collect()  # warms the snapshot cache (TTL-gated)
        except Exception:
            log.warning("resource_sampler_collect_failed", exc_info=True)
    try:
        await probe_sidecar_containers(app_state)  # warms the container cache
    except Exception:
        log.warning("resource_sampler_container_probe_failed", exc_info=True)
    http = getattr(app_state, "http_client", None)
    if http is not None:
        try:
            await probe_host_stats(http)  # warms the host-stats cache
        except Exception:
            log.warning("resource_sampler_host_probe_failed", exc_info=True)
    return time.monotonic() - started


async def resource_sampler_loop(app_state) -> None:
    """Run forever, refreshing caches on an adaptive cadence."""
    log.info("resource_sampler_started")
    # Prime the caches once up front so the read path (cache_only) has
    # something to serve before the first interval elapses — otherwise the
    # first /status after boot would cold-probe inline (the very latency this
    # loop exists to remove). Never raises.
    try:
        await _sample_once(app_state)
    except Exception:  # pragma: no cover — defensive; loop must not die
        log.warning("resource_sampler_prime_failed", exc_info=True)
    while True:
        await asyncio.sleep(_interval(app_state))
        try:
            spent = await _sample_once(app_state)
            # Self-accounting: stash the sampler's own cost so it can surface as
            # a consumer row and so an over-budget pass is visible in logs.
            app_state.resource_sampler_last_cost_s = round(spent, 3)
            if spent > 2.0:
                log.warning("resource_sampler_slow_pass", seconds=round(spent, 2))
        except Exception:  # pragma: no cover — defensive; loop must not die
            log.warning("resource_sampler_pass_failed", exc_info=True)
