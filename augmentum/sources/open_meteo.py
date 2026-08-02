"""Open-Meteo provider — weather + geocoding, keyless.

The Tier-A anchor of the direct-sources layer: no key, no signup,
30+ national weather models, CC-BY data, and a bundled geocoder that
covers the entire "weather in <city>" use case without touching
Nominatim's strict public policy. Hosted free tier allows 10k
calls/day non-commercial — a personal box with our TTL cache uses a
rounding error of that.

Attribution: weather data by Open-Meteo.com (CC BY 4.0).
"""
from __future__ import annotations

from typing import Any

from augmentum.sources.base import fetch_json

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_GEOCODE_TTL_S = 7 * 24 * 3600.0   # places don't move
_FORECAST_TTL_S = 15 * 60.0        # model-update cadence

# WMO weather interpretation codes → spoken condition.
WMO_CODES: dict[int, str] = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "severe thunderstorms with hail",
}


def condition_for(code: Any) -> str:
    try:
        return WMO_CODES.get(int(code), "mixed conditions")
    except (TypeError, ValueError):
        return "mixed conditions"


# US state abbreviation → admin1 name, for "city, st" disambiguation.
# Without this, "springfield, il" prefers whichever US Springfield the
# geocoder ranks first (population), not the one in Illinois.
_US_STATES = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "co": "Colorado", "ct": "Connecticut",
    "de": "Delaware", "fl": "Florida", "ga": "Georgia", "hi": "Hawaii",
    "id": "Idaho", "il": "Illinois", "in": "Indiana", "ia": "Iowa",
    "ks": "Kansas", "ky": "Kentucky", "la": "Louisiana", "me": "Maine",
    "md": "Maryland", "ma": "Massachusetts", "mi": "Michigan",
    "mn": "Minnesota", "ms": "Mississippi", "mo": "Missouri",
    "mt": "Montana", "ne": "Nebraska", "nv": "Nevada",
    "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico",
    "ny": "New York", "nc": "North Carolina", "nd": "North Dakota",
    "oh": "Ohio", "ok": "Oklahoma", "or": "Oregon", "pa": "Pennsylvania",
    "ri": "Rhode Island", "sc": "South Carolina", "sd": "South Dakota",
    "tn": "Tennessee", "tx": "Texas", "ut": "Utah", "vt": "Vermont",
    "va": "Virginia", "wa": "Washington", "wv": "West Virginia",
    "wi": "Wisconsin", "wy": "Wyoming", "dc": "District of Columbia",
}


async def _geocode_query(name: str) -> list[dict[str, Any]]:
    data = await fetch_json(
        "open_meteo", GEOCODE_URL,
        {"name": name, "count": 5, "language": "en", "format": "json"},
        ttl_s=_GEOCODE_TTL_S,
    )
    return (data or {}).get("results") or []


def _normalize_place(r: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    return {
        "name": str(r.get("name") or fallback_name),
        "admin1": str(r.get("admin1") or ""),
        "country_code": str(r.get("country_code") or ""),
        "latitude": float(r.get("latitude") or 0.0),
        "longitude": float(r.get("longitude") or 0.0),
        "timezone": str(r.get("timezone") or "auto"),
    }


async def geocode(name: str) -> dict[str, Any] | None:
    """Resolve a place name to a normalized location dict, or None.

    Comma forms ("springfield, il" — the Settings→Location convention)
    need special handling: Open-Meteo's geocoder matches plain place
    NAMES, so the comma string matches nothing — observed live
    2026-06-11 as the settings-fallback resolving nowhere and the
    synthesis tier hallucinating a forecast. We query the head
    ("springfield") and, when the suffix looks like a US state
    abbreviation, prefer the US candidate among the top hits.
    """
    name = (name or "").strip()
    if not name:
        return None

    prefer_state = ""
    query = name
    if "," in name:
        head, _, tail = name.partition(",")
        query = head.strip()
        tail = tail.strip().lower()
        prefer_state = _US_STATES.get(tail, "")
        if not query:
            return None

    candidates = await _geocode_query(query)
    if not candidates and query != name:
        candidates = await _geocode_query(name)
    if not candidates:
        return None

    pick = candidates[0]
    if prefer_state:
        # Exact state match wins; any-US beats non-US as fallback.
        us_fallback = None
        for c in candidates:
            if str(c.get("country_code") or "").upper() != "US":
                continue
            if str(c.get("admin1") or "") == prefer_state:
                pick = c
                us_fallback = None
                break
            if us_fallback is None:
                us_fallback = c
        if us_fallback is not None:
            pick = us_fallback
    return _normalize_place(pick, query)


async def forecast(
    latitude: float, longitude: float, *, imperial: bool,
) -> dict[str, Any] | None:
    """Current conditions + 2-day daily forecast. None on failure."""
    params: dict[str, Any] = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "current": (
            "temperature_2m,apparent_temperature,weather_code,"
            "wind_speed_10m,relative_humidity_2m,precipitation"
        ),
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max"
        ),
        "forecast_days": 2,
        "timezone": "auto",
    }
    if imperial:
        params["temperature_unit"] = "fahrenheit"
        params["wind_speed_unit"] = "mph"
    return await fetch_json(
        "open_meteo", FORECAST_URL, params, ttl_s=_FORECAST_TTL_S,
    )


def _rnd(v: Any) -> int | None:
    try:
        return round(float(v))
    except (TypeError, ValueError):
        return None


def summarize(
    place: dict[str, Any], fc: dict[str, Any], *, imperial: bool,
) -> dict[str, Any]:
    """Fold a forecast response into spoken text + compact data.

    ``spoken`` is the conversational line; the structured fields ride
    the tool payload so follow-ups ("what about tomorrow?") answer
    from data already in context instead of refetching.
    """
    cur = fc.get("current") or {}
    daily = fc.get("daily") or {}
    unit = "°F" if imperial else "°C"

    now = {
        "temp": _rnd(cur.get("temperature_2m")),
        "feels_like": _rnd(cur.get("apparent_temperature")),
        "condition": condition_for(cur.get("weather_code")),
        "wind": _rnd(cur.get("wind_speed_10m")),
        "humidity": _rnd(cur.get("relative_humidity_2m")),
    }

    def _day(i: int) -> dict[str, Any]:
        def _at(key: str) -> Any:
            vals = daily.get(key) or []
            return vals[i] if i < len(vals) else None
        return {
            "high": _rnd(_at("temperature_2m_max")),
            "low": _rnd(_at("temperature_2m_min")),
            "precip_chance": _rnd(_at("precipitation_probability_max")),
            "condition": condition_for(_at("weather_code")),
        }

    today, tomorrow = _day(0), _day(1)

    place_name = place.get("name") or "your area"
    bits = []
    if now["temp"] is not None:
        feels = ""
        if (
            now["feels_like"] is not None
            and abs(now["feels_like"] - now["temp"]) >= 3
        ):
            feels = f", feels like {now['feels_like']}"
        bits.append(
            f"Right now in {place_name} it's {now['temp']}{unit} and "
            f"{now['condition']}{feels}."
        )
    if today["high"] is not None:
        rain = ""
        if today["precip_chance"] is not None and today["precip_chance"] >= 15:
            rain = f", {today['precip_chance']}% chance of precipitation"
        bits.append(
            f"Today: high {today['high']}, low {today['low']}, "
            f"{today['condition']}{rain}."
        )
    if tomorrow["high"] is not None:
        bits.append(
            f"Tomorrow: {tomorrow['high']} and {tomorrow['condition']}."
        )
    spoken = " ".join(bits) or (
        f"I reached the weather service but got nothing usable for "
        f"{place_name}."
    )

    return {
        "spoken": spoken,
        "place": place_name,
        "unit": unit,
        "now": now,
        "today": today,
        "tomorrow": tomorrow,
    }
