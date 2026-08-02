"""NWS (weather.gov) provider — US severe-weather alerts, keyless.

A true public-service API: no key, no signup, county-level warnings as
GeoJSON. Tier A in the sources research. US-only by nature — callers
should skip it for non-US home locations.
"""
from __future__ import annotations

from typing import Any

from augmentum.sources.base import fetch_json

ALERTS_URL = "https://api.weather.gov/alerts/active"

# NWS caches responses ~60s server-side; alerts are checked on a
# multi-minute watcher cadence anyway.
_ALERTS_TTL_S = 300.0

# Severities worth interrupting someone over. "Minor"/"Unknown" stay
# in the feed but the watcher only notifies on these.
NOTIFY_SEVERITIES = ("Severe", "Extreme")


async def active_alerts(latitude: float, longitude: float) -> list[dict[str, Any]]:
    """Active alerts covering a point. Empty list on failure or none."""
    data = await fetch_json(
        "nws", ALERTS_URL,
        {"point": f"{round(latitude, 4)},{round(longitude, 4)}"},
        ttl_s=_ALERTS_TTL_S, timeout_s=10.0,
    )
    out: list[dict[str, Any]] = []
    for feature in (data or {}).get("features") or []:
        props = feature.get("properties") or {}
        alert_id = str(feature.get("id") or props.get("id") or "").strip()
        event = str(props.get("event") or "").strip()
        if not alert_id or not event:
            continue
        out.append({
            "id": alert_id,
            "event": event,
            "headline": str(props.get("headline") or "").strip(),
            "severity": str(props.get("severity") or "Unknown").strip(),
            "expires": str(props.get("expires") or ""),
            "instruction": str(props.get("instruction") or "").strip()[:300],
        })
    return out
