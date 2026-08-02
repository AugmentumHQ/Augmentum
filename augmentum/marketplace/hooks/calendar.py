"""calendar hook — wire a CalDAV service into Augmentum's calendar system.

When a manifest declares ``integration.calendar``, the provisioned
service's events are synced into the ``calendar_events`` cache, where
the companion reads them for briefings and the ``calendar.today`` verb.

The hook config may carry:
  * ``protocol`` — always "caldav" (Phase 2 only supports CalDAV)
  * ``path`` — the calendar path on the server (e.g. ``/radicale/user/calendar.ics/``)
"""

from __future__ import annotations

from typing import Any

from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS, HookMeta
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _install(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Sync events from the connected CalDAV server into the cache."""
    cfg = manifest.integration.get("calendar", {}) or {}
    calendar_path = str(cfg.get("path") or "")
    protocol = cfg.get("protocol", "caldav")

    if protocol != "caldav":
        log.info(
            "calendar_hook_unknown_protocol",
            service_id=manifest.service_id, protocol=protocol,
        )
        return

    # Resolve the container's base URL — same pattern as provider_bridge.
    base_url = f"http://augmentum-{manifest.service_id}:{sd.internal_port}"
    if calendar_path and not calendar_path.startswith("/"):
        calendar_path = f"/{calendar_path}"

    # Managed credentials for auth.
    username = ""
    password = ""
    try:
        from augmentum.providers.service_auth import managed_service_credentials
        username, password = managed_service_credentials(manifest.service_id)
    except Exception:
        log.warning(
            "calendar_hook_no_credentials",
            service_id=manifest.service_id,
        )

    # Resolve DB connection.
    conn = _resolve_conn(request)
    if conn is None:
        log.warning(
            "calendar_hook_no_db",
            service_id=manifest.service_id,
        )
        return

    # Sync events into the cache.
    try:
        from augmentum.calendar.sync import sync_calendar_events
        n = await sync_calendar_events(
            conn,
            user_id=user_id,
            service_id=manifest.service_id,
            base_url=base_url,
            username=username,
            password=password,
            calendar_path=calendar_path,
        )
        # Record last_synced_at in the managed_services config_json so the
        # prompt-compose re-sync gate knows the cache is fresh.
        try:
            import time
            mgr = getattr(request.app.state, "service_manager", None)
            if mgr is not None and hasattr(mgr, "read_config_json"):
                cfg = await mgr.read_config_json(manifest.service_id) or {}
            else:
                cfg = {}
            cfg["last_synced_at"] = int(time.time())
            cfg["calendar_path"] = calendar_path
            if mgr is not None and hasattr(mgr, "update_config_json"):
                await mgr.update_config_json(manifest.service_id, cfg)
        except Exception:
            log.debug("calendar_hook_config_update_failed", exc_info=True)
        log.info(
            "calendar_hook_synced",
            service_id=manifest.service_id,
            events=n,
        )
    except Exception:
        log.warning(
            "calendar_hook_sync_failed",
            service_id=manifest.service_id, exc_info=True,
        )


async def _uninstall(request: Any, manifest: Any, sd: Any, user_id: str) -> None:
    """Remove cached calendar events for this service."""
    conn = _resolve_conn(request)
    if conn is None:
        return
    try:
        from augmentum.calendar.store import delete_stale_events
        deleted = await delete_stale_events(
            conn, user_id=user_id, service_id=manifest.service_id,
            seen_uids=set(),  # empty → delete ALL for this service
        )
        log.info(
            "calendar_hook_uninstalled",
            service_id=manifest.service_id, events_deleted=deleted,
        )
    except Exception:
        log.warning(
            "calendar_hook_uninstall_failed",
            service_id=manifest.service_id, exc_info=True,
        )


def _resolve_conn(request: Any) -> Any:
    """Best-effort aiosqlite connection from app.state."""
    sm = getattr(request.app.state, "state_manager", None)
    backend = getattr(sm, "backend", None) if sm is not None else None
    return getattr(backend, "conn", None)


KNOWN_INTEGRATION_HOOKS["calendar"] = (
    _install,
    _uninstall,
    HookMeta(
        label="Calendar",
        icon="📅",
        companion_hint="Companion knows your schedule",
        status_provider="calendar",
    ),
)
