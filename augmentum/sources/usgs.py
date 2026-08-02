"""USGS provider — earthquakes near a point, keyless.

FDSN event service, GeoJSON. Responses are cached 60s server-side and
queries over 20k events 400 — neither matters at watcher scale.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

from augmentum.sources.base import fetch_json

QUERY_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

_QUAKES_TTL_S = 300.0


def _window_start(hours: float) -> str:
    """Start time for the lookback window, rounded DOWN to the hour so
    the cache key stays stable between watcher polls."""
    start = _dt.datetime.now(_dt.UTC) - _dt.timedelta(hours=hours)
    return start.strftime("%Y-%m-%dT%H:00:00")


async def quakes_near(
    latitude: float, longitude: float, *,
    radius_km: float = 300.0,
    min_magnitude: float = 4.5,
    hours: float = 6.0,
) -> list[dict[str, Any]]:
    """Recent significant quakes within radius. Empty on failure/none."""
    data = await fetch_json(
        "usgs", QUERY_URL,
        {
            "format": "geojson",
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "maxradiuskm": round(radius_km, 1),
            "minmagnitude": min_magnitude,
            "starttime": _window_start(hours),
            "orderby": "time",
        },
        ttl_s=_QUAKES_TTL_S, timeout_s=10.0,
    )
    out: list[dict[str, Any]] = []
    for feature in (data or {}).get("features") or []:
        props = feature.get("properties") or {}
        quake_id = str(feature.get("id") or "").strip()
        mag = props.get("mag")
        if not quake_id or mag is None:
            continue
        out.append({
            "id": quake_id,
            "magnitude": round(float(mag), 1),
            "place": str(props.get("place") or "").strip(),
            "time_ms": int(props.get("time") or 0),
            "url": str(props.get("url") or ""),
        })
    return out
