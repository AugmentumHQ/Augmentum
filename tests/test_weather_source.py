"""Direct-sources layer P1 — Open-Meteo provider + weather.today verb.

Pins: the seamless home-location ladder (arg → stored blob → Settings
Location field → honest ask), first-use home save without clobbering,
unit inference from country, summarize() output shape, and the base
fetch cache.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import augmentum.intent  # noqa: F401 — registers verbs
from augmentum.intent.action import SessionContext
from augmentum.intent.registry import REGISTRY
from augmentum.sources import base as source_base
from augmentum.sources import open_meteo

# ── Test doubles ──────────────────────────────────────────────────────

class _FakeStore:
    def __init__(self, values=None):
        self.values = dict(values or {})   # (user_id, key) → str
        self.global_values = {}

    async def get_user(self, user_id, key):
        return self.values.get((user_id, key))

    async def set_user(self, user_id, key, value):
        if value is None:
            self.values.pop((user_id, key), None)
        else:
            self.values[(user_id, key)] = value

    async def get_user_or_global(self, user_id, key):
        return self.values.get((user_id, key)) or self.global_values.get(key)


_PLACE = {
    "name": "Portland", "admin1": "Oregon", "country_code": "US",
    "latitude": 45.52, "longitude": -122.67, "timezone": "America/Los_Angeles",
}

_FORECAST = {
    "current": {
        "temperature_2m": 64.2, "apparent_temperature": 61.0,
        "weather_code": 2, "wind_speed_10m": 7.3,
        "relative_humidity_2m": 55, "precipitation": 0.0,
    },
    "daily": {
        "weather_code": [61, 1],
        "temperature_2m_max": [71.4, 75.0],
        "temperature_2m_min": [54.0, 56.1],
        "precipitation_probability_max": [40, 5],
    },
}


def _session(store, user_id="u_w1"):
    state = SimpleNamespace(settings_store=store)
    return SessionContext(
        user_id=user_id, session_id="s_w1", mode=None, app_state=state,
    )


def _patch_provider(monkeypatch, *, geocode=_PLACE, forecast=_FORECAST):
    async def _fake_geocode(name):
        return dict(geocode) if geocode else None

    async def _fake_forecast(lat, lon, *, imperial):
        _fake_forecast.last_imperial = imperial
        return dict(forecast) if forecast else None

    monkeypatch.setattr(open_meteo, "geocode", _fake_geocode)
    monkeypatch.setattr(open_meteo, "forecast", _fake_forecast)
    return _fake_forecast


# ── summarize() ───────────────────────────────────────────────────────

def test_summarize_spoken_and_data_shape():
    s = open_meteo.summarize(_PLACE, _FORECAST, imperial=True)
    assert "Portland" in s["spoken"]
    assert "64°F" in s["spoken"]
    assert "partly cloudy" in s["spoken"]
    assert "40% chance" in s["spoken"]
    assert s["now"]["temp"] == 64
    assert s["today"] == {
        "high": 71, "low": 54, "precip_chance": 40, "condition": "light rain",
    }
    assert s["tomorrow"]["condition"] == "mostly clear"


def test_summarize_skips_feels_like_when_close():
    fc = json.loads(json.dumps(_FORECAST))
    fc["current"]["apparent_temperature"] = 63.5  # within 3° of 64
    s = open_meteo.summarize(_PLACE, fc, imperial=True)
    assert "feels like" not in s["spoken"]


def test_wmo_unknown_code_degrades():
    assert open_meteo.condition_for(42) == "mixed conditions"
    assert open_meteo.condition_for(None) == "mixed conditions"
    assert open_meteo.condition_for(95) == "thunderstorms"


# ── Verb: resolution ladder ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_explicit_location_first_use_saves_home(monkeypatch):
    _patch_provider(monkeypatch)
    store = _FakeStore()
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(store), {"location": "portland"})
    assert "Portland" in result.speak
    saved = json.loads(store.values[("u_w1", "sources.home_location")])
    assert saved["name"] == "Portland"
    assert "[weather data]" in result.prompt_addendum


@pytest.mark.asyncio
async def test_one_off_location_does_not_clobber_home(monkeypatch):
    _patch_provider(monkeypatch)
    home = dict(_PLACE, name="Denver")
    store = _FakeStore({
        ("u_w1", "sources.home_location"): json.dumps(home),
    })
    action = REGISTRY.get("weather.today")
    await action.handler("", _session(store), {"location": "tokyo"})
    kept = json.loads(store.values[("u_w1", "sources.home_location")])
    assert kept["name"] == "Denver"


@pytest.mark.asyncio
async def test_remember_home_overwrites(monkeypatch):
    _patch_provider(monkeypatch)
    store = _FakeStore({
        ("u_w1", "sources.home_location"): json.dumps(dict(_PLACE, name="Denver")),
    })
    action = REGISTRY.get("weather.today")
    await action.handler(
        "", _session(store), {"location": "portland", "remember_home": "true"},
    )
    saved = json.loads(store.values[("u_w1", "sources.home_location")])
    assert saved["name"] == "Portland"


@pytest.mark.asyncio
async def test_no_arg_uses_stored_home(monkeypatch):
    fake_fc = _patch_provider(monkeypatch)

    async def _fail_geocode(name):  # home blob means geocode never runs
        raise AssertionError("geocode should not be called")
    monkeypatch.setattr(open_meteo, "geocode", _fail_geocode)

    store = _FakeStore({
        ("u_w1", "sources.home_location"): json.dumps(_PLACE),
    })
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(store), {})
    assert "Portland" in result.speak
    assert fake_fc.last_imperial is True  # US → imperial


@pytest.mark.asyncio
async def test_settings_location_field_promotes_to_home(monkeypatch):
    # USER-scoped Settings location → answers AND persists as home.
    _patch_provider(monkeypatch)
    store = _FakeStore({("u_w1", "location"): "Portland, OR"})
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(store), {})
    assert "Portland" in result.speak
    assert ("u_w1", "sources.home_location") in store.values


@pytest.mark.asyncio
async def test_global_location_answers_but_never_persists(monkeypatch):
    # Install-wide config location is a household default: it answers
    # the question but must NOT become this user's saved home —
    # companion_eval caught a fresh user inheriting the owner's city
    # as their persisted home through this fallback (2026-06-11).
    _patch_provider(monkeypatch)
    from augmentum.config import settings as app_settings
    monkeypatch.setattr(app_settings, "location", "Portland, OR", raising=False)
    store = _FakeStore()
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(store), {})
    assert "Portland" in result.speak
    assert ("u_w1", "sources.home_location") not in store.values


@pytest.mark.asyncio
async def test_nothing_known_asks_honestly(monkeypatch):
    _patch_provider(monkeypatch)
    # Ensure the install-wide config fallback is empty too.
    from augmentum.config import settings as app_settings
    monkeypatch.setattr(app_settings, "location", "", raising=False)
    store = _FakeStore()
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(store), {})
    assert "what city" in result.speak.lower()


@pytest.mark.asyncio
async def test_metric_units_outside_us(monkeypatch):
    fake_fc = _patch_provider(
        monkeypatch, geocode=dict(_PLACE, name="Berlin", country_code="DE"),
    )
    store = _FakeStore()
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(store), {"location": "berlin"})
    assert fake_fc.last_imperial is False
    assert "°C" in result.speak


@pytest.mark.asyncio
async def test_geocode_miss_is_honest(monkeypatch):
    _patch_provider(monkeypatch, geocode=None)
    store = _FakeStore()
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(store), {"location": "zzzzz"})
    assert "couldn't find" in result.speak.lower()


@pytest.mark.asyncio
async def test_forecast_failure_degrades(monkeypatch):
    _patch_provider(monkeypatch, forecast=None)
    store = _FakeStore({
        ("u_w1", "sources.home_location"): json.dumps(_PLACE),
    })
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(store), {})
    assert "couldn't reach" in result.speak.lower()


@pytest.mark.asyncio
async def test_signed_out_refuses(monkeypatch):
    _patch_provider(monkeypatch)
    action = REGISTRY.get("weather.today")
    result = await action.handler("", _session(_FakeStore(), user_id=""), {})
    assert "signed-out" in result.speak


# ── base.fetch_json cache ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_json_caches_within_ttl(monkeypatch):
    source_base.clear_cache()
    calls = {"n": 0}

    class _Resp:
        status_code = 200
        def json(self):
            return {"ok": calls["n"]}

    class _Client:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            calls["n"] += 1
            return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    r1 = await source_base.fetch_json("t", "https://x.test/a", {"q": "1"})
    r2 = await source_base.fetch_json("t", "https://x.test/a", {"q": "1"})
    r3 = await source_base.fetch_json("t", "https://x.test/a", {"q": "2"})
    assert calls["n"] == 2          # second call served from cache
    assert r1 == r2
    assert r3 == {"ok": 2}
    source_base.clear_cache()


@pytest.mark.asyncio
async def test_fetch_json_http_error_returns_none(monkeypatch):
    source_base.clear_cache()

    class _Resp:
        status_code = 503
        def json(self):
            return {}

    class _Client:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    assert await source_base.fetch_json("t", "https://x.test/b") is None
    source_base.clear_cache()


# ── Geocode comma-form handling (the comma+state live miss) ─────────

@pytest.mark.asyncio
async def test_geocode_comma_state_prefers_us(monkeypatch):
    queries = []

    async def _fake_fetch(provider, url, params=None, **kw):
        queries.append(params["name"])
        return {"results": [
            {"name": "Birmingham", "country_code": "GB",
             "latitude": 52.5, "longitude": -1.9, "timezone": "Europe/London"},
            {"name": "Birmingham", "admin1": "Alabama", "country_code": "US",
             "latitude": 33.5, "longitude": -86.8,
             "timezone": "America/Chicago"},
        ]}
    monkeypatch.setattr(open_meteo, "fetch_json", _fake_fetch)
    place = await open_meteo.geocode("birmingham, al")
    assert queries == ["birmingham"]  # comma string never sent verbatim
    assert place["country_code"] == "US"
    assert place["admin1"] == "Alabama"


@pytest.mark.asyncio
async def test_geocode_state_abbrev_picks_exact_state(monkeypatch):
    # "springfield, il" must pick Illinois even when another US
    # Springfield outranks it in the geocoder's population order.
    async def _fake_fetch(provider, url, params=None, **kw):
        return {"results": [
            {"name": "Springfield", "admin1": "Missouri",
             "country_code": "US", "latitude": 37.2, "longitude": -93.3,
             "timezone": "America/Chicago"},
            {"name": "Springfield", "admin1": "Illinois",
             "country_code": "US", "latitude": 39.8, "longitude": -89.6,
             "timezone": "America/Chicago"},
        ]}
    monkeypatch.setattr(open_meteo, "fetch_json", _fake_fetch)
    place = await open_meteo.geocode("springfield, il")
    assert place["admin1"] == "Illinois"


@pytest.mark.asyncio
async def test_geocode_state_abbrev_falls_back_to_any_us(monkeypatch):
    # Suffix names a state none of the candidates are in — any-US
    # still beats the non-US top hit.
    async def _fake_fetch(provider, url, params=None, **kw):
        return {"results": [
            {"name": "Birmingham", "country_code": "GB",
             "latitude": 52.5, "longitude": -1.9,
             "timezone": "Europe/London"},
            {"name": "Birmingham", "admin1": "Alabama",
             "country_code": "US", "latitude": 33.5, "longitude": -86.8,
             "timezone": "America/Chicago"},
        ]}
    monkeypatch.setattr(open_meteo, "fetch_json", _fake_fetch)
    place = await open_meteo.geocode("birmingham, mi")
    assert place["country_code"] == "US"


@pytest.mark.asyncio
async def test_geocode_plain_name_takes_top_hit(monkeypatch):
    async def _fake_fetch(provider, url, params=None, **kw):
        return {"results": [
            {"name": "Berlin", "country_code": "DE",
             "latitude": 52.5, "longitude": 13.4, "timezone": "Europe/Berlin"},
        ]}
    monkeypatch.setattr(open_meteo, "fetch_json", _fake_fetch)
    place = await open_meteo.geocode("berlin")
    assert place["country_code"] == "DE"


@pytest.mark.asyncio
async def test_geocode_no_results_is_none(monkeypatch):
    async def _fake_fetch(provider, url, params=None, **kw):
        return {"results": []}
    monkeypatch.setattr(open_meteo, "fetch_json", _fake_fetch)
    assert await open_meteo.geocode("zzzznowhere, xq") is None


def test_weather_today_is_artifact_delivery():
    # Numbers from a data source must be spoken VERBATIM — synthesis
    # hallucinated "42 degrees" over a geocode miss (2026-06-11).
    action = REGISTRY.get("weather.today")
    assert action.delivery == "artifact"
