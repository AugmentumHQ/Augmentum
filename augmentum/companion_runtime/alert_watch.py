"""Alert watch — NWS severe weather + USGS quakes near the user's home.

The first push-shaped consumer of the direct-sources layer: where
``weather.today`` answers when asked, this one speaks up unprompted —
"severe thunderstorm warning for your area until 9pm" is the companion
earning her keep at zero LLM cost.

Shape mirrors ``curator.step``: called every behavior tick, self-gates
on a kill switch + its own interval + the presence of a saved home
location (the SAME ``sources.home_location`` blob weather.today
learns — once she knows where home is, alerts come free).

Delivery rides the notifications pipeline (``publish_and_dispatch``):
persisted row + live WS banner + Web Push when no client is attached.
Because the store's dedupe-key semantics RE-surface an updated row
(read_at resets), repeat polls of a still-active alert must not
re-publish — an in-memory seen-set on the runtime guards that; a
restart re-notifying an ongoing severe alert once is acceptable.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)

_INTERVAL_S = 600.0          # poll cadence (providers cache beneath this)
_SEEN_CAP = 200              # ids remembered before oldest are dropped

_QUAKE_MIN_MAGNITUDE = 4.5
_QUAKE_RADIUS_KM = 300.0
_QUAKE_LOOKBACK_HOURS = 6.0


async def _load_home(runtime: CompanionRuntime, user_id: str) -> dict[str, Any] | None:
    app_state = getattr(runtime, "_app_state", None)
    store = getattr(app_state, "settings_store", None) if app_state else None
    if store is None:
        return None
    from augmentum.intent.builtin.weather import _load_home as load_home_blob
    return await load_home_blob(store, user_id)


def _seen(runtime: CompanionRuntime) -> dict[str, float]:
    seen = getattr(runtime, "_alert_watch_seen", None)
    if seen is None:
        seen = {}
        runtime._alert_watch_seen = seen
    if len(seen) > _SEEN_CAP:
        for key in sorted(seen, key=seen.get)[: len(seen) - _SEEN_CAP]:
            seen.pop(key, None)
    return seen


async def _notify(
    runtime: CompanionRuntime, *, user_id: str, dedupe_key: str,
    title: str, body: str, importance: int,
) -> None:
    app_state = getattr(runtime, "_app_state", None)
    hub = getattr(app_state, "notification_hub", None) if app_state else None
    if hub is None:
        from augmentum.notifications.hub import NotificationHub
        hub = NotificationHub()
        if app_state is not None:
            app_state.notification_hub = hub
    from augmentum.notifications.hub import publish_and_dispatch
    await publish_and_dispatch(
        runtime.backend.conn,
        hub=hub,
        user_id=user_id,
        channel_id="alerts.home",
        source="sources.alert_watch",
        title=title,
        body=body,
        importance=importance,
        dedupe_key=dedupe_key,
    )


async def step(runtime: CompanionRuntime) -> None:
    """One watcher iteration. Never raises (tick wraps anyway)."""
    from augmentum.config import settings
    if not getattr(settings, "companion_alert_watch_enabled", True):
        return

    now = time.time()
    last = float(getattr(runtime, "last_alert_watch_at", 0.0))
    if last and (now - last) < _INTERVAL_S:
        return
    runtime.last_alert_watch_at = now

    user_id = getattr(runtime, "owner_user_id", "") or ""
    if not user_id:
        return
    home = await _load_home(runtime, user_id)
    if not home:
        return  # she doesn't know where home is yet — weather.today teaches her

    lat = float(home.get("latitude") or 0.0)
    lon = float(home.get("longitude") or 0.0)
    place = str(home.get("name") or "home")
    seen = _seen(runtime)

    from augmentum.notifications.catalog import (
        IMPORTANCE_CRITICAL,
        IMPORTANCE_HIGH,
    )
    from augmentum.sources import nws, usgs

    # ── Severe weather (US homes only — NWS coverage) ────────────────
    if str(home.get("country_code") or "").upper() == "US":
        for alert in await nws.active_alerts(lat, lon):
            if alert["severity"] not in nws.NOTIFY_SEVERITIES:
                continue
            key = f"nws:{alert['id']}"
            if key in seen:
                continue
            seen[key] = now
            importance = (
                IMPORTANCE_CRITICAL
                if alert["severity"] == "Extreme"
                else IMPORTANCE_HIGH
            )
            body = alert["headline"] or alert["event"]
            if alert["instruction"]:
                body = f"{body}\n{alert['instruction']}"
            try:
                await _notify(
                    runtime, user_id=user_id, dedupe_key=key,
                    title=f"{alert['event']} — {place}",
                    body=body, importance=importance,
                )
                log.info(
                    "alert_watch_notified",
                    kind="nws", alert_event=alert["event"],
                    severity=alert["severity"], user_id=user_id,
                )
            except Exception:  # noqa: BLE001 — one failed publish ≠ dead watcher
                seen.pop(key, None)  # retry next poll
                log.warning("alert_watch_notify_failed", kind="nws", exc_info=True)

    # ── Significant quakes (global) ──────────────────────────────────
    for quake in await usgs.quakes_near(
        lat, lon,
        radius_km=_QUAKE_RADIUS_KM,
        min_magnitude=_QUAKE_MIN_MAGNITUDE,
        hours=_QUAKE_LOOKBACK_HOURS,
    ):
        key = f"usgs:{quake['id']}"
        if key in seen:
            continue
        seen[key] = now
        try:
            await _notify(
                runtime, user_id=user_id, dedupe_key=key,
                title=f"M{quake['magnitude']} earthquake near {place}",
                body=quake["place"] or "Details on the USGS event page.",
                importance=IMPORTANCE_HIGH,
            )
            log.info(
                "alert_watch_notified",
                kind="usgs", magnitude=quake["magnitude"], user_id=user_id,
            )
        except Exception:  # noqa: BLE001
            seen.pop(key, None)
            log.warning("alert_watch_notify_failed", kind="usgs", exc_info=True)
