"""SearXNG search-side API routes.

Currently exposes the outbound-proxy manager state and a force-refresh
endpoint. Admin-only: SearXNG is a single shared container, so its
outbound routing is a per-install setting, not per-user.

The proxy configuration ITSELF (URL list, rotation toggle, interval,
fallback flag) is persisted via the standard
``/api/config/tools`` settings surface and the four-layer pattern — see
``search_proxies`` and friends in ``augmentum/config.py``. These routes
just give the UI a read view of live runtime state and a way to trigger
an immediate healthcheck without waiting for the next loop tick.
"""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from augmentum.auth.guards import require_admin
from augmentum.search import SearxngProxyManager
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/search/proxies", tags=["search"])


def _get_manager(request: Request) -> SearxngProxyManager:
    manager = getattr(request.app.state, "searxng_proxy_manager", None)
    if manager is None:
        raise HTTPException(
            503,
            "SearXNG proxy manager not initialised "
            "(SearXNG may be disabled or settings.yml unreachable from the Augmentum container)",
        )
    return manager


def _status_to_json(status) -> dict:
    """Dataclass → JSON-safe dict the UI can render directly."""
    return {
        "configured_count": status.configured_count,
        "healthy_count": status.healthy_count,
        "active_proxy": status.active_proxy,
        "direct_fallback_active": status.direct_fallback_active,
        "last_healthcheck": status.last_healthcheck,
        "proxies": [dataclasses.asdict(p) for p in status.proxies],
    }


@router.get("/status")
async def get_status(request: Request) -> JSONResponse:
    """Return the current proxy state for the Settings → Search panel."""
    require_admin(request)
    manager = _get_manager(request)
    return JSONResponse(_status_to_json(manager.status()))


@router.post("/test")
async def force_test(request: Request) -> JSONResponse:
    """Re-parse the proxy list, probe every entry, reconcile, return new status.

    Used by the "Test now" button. Cheap to call (parallel probes), but
    will write settings.yml and restart SearXNG if the active choice
    flips — same as a normal loop tick.
    """
    require_admin(request)
    manager = _get_manager(request)

    from augmentum.config import settings as live_settings

    await manager.update_proxy_list(getattr(live_settings, "search_proxies", "") or "")
    await manager.healthcheck_all()
    fallback = bool(getattr(live_settings, "search_proxy_fallback_direct_enabled", True))
    await manager.reconcile(fallback_to_direct=fallback)
    return JSONResponse(_status_to_json(manager.status()))
