"""Direct-sources P2+P3 — alert watch, NWS/USGS providers, rsshub:// expansion.

Pins: provider response parsing, the watcher's gates (kill switch,
interval, home presence, US-only NWS), severity filtering, seen-set
dedupe across polls, notify-failure retry, and the RSSHub shorthand
expansion in the discovery RSS fetcher.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from augmentum.companion_runtime import alert_watch
from augmentum.sources import base as source_base
from augmentum.sources import nws, usgs

# ── Provider parsing ──────────────────────────────────────────────────

_NWS_RESPONSE = {
    "features": [
        {
            "id": "urn:alert:1",
            "properties": {
                "event": "Severe Thunderstorm Warning",
                "headline": "Severe thunderstorm until 9PM",
                "severity": "Severe",
                "expires": "2026-06-11T21:00:00-06:00",
                "instruction": "Move indoors.",
            },
        },
        {
            "id": "urn:alert:2",
            "properties": {
                "event": "Air Quality Alert",
                "headline": "Smoke advisory",
                "severity": "Minor",
            },
        },
        {"id": "", "properties": {"event": "Ghost"}},  # dropped: no id
    ],
}

_USGS_RESPONSE = {
    "features": [
        {
            "id": "us7000aaaa",
            "properties": {
                "mag": 5.23, "place": "12 km W of Townsville",
                "time": 1765400000000, "url": "https://usgs.example/q",
            },
        },
        {"id": "us7000bbbb", "properties": {"mag": None}},  # dropped
    ],
}


@pytest.mark.asyncio
async def test_nws_parses_alerts(monkeypatch):
    async def _fake_fetch(provider, url, params=None, **kw):
        assert provider == "nws"
        assert params["point"] == "45.52,-122.67"
        return _NWS_RESPONSE
    monkeypatch.setattr(nws, "fetch_json", _fake_fetch)
    alerts = await nws.active_alerts(45.52, -122.67)
    assert len(alerts) == 2
    assert alerts[0]["event"] == "Severe Thunderstorm Warning"
    assert alerts[0]["severity"] == "Severe"
    assert alerts[1]["severity"] == "Minor"


@pytest.mark.asyncio
async def test_usgs_parses_quakes_and_rounds_window(monkeypatch):
    captured = {}
    async def _fake_fetch(provider, url, params=None, **kw):
        captured.update(params)
        return _USGS_RESPONSE
    monkeypatch.setattr(usgs, "fetch_json", _fake_fetch)
    quakes = await usgs.quakes_near(45.52, -122.67)
    assert len(quakes) == 1
    assert quakes[0]["magnitude"] == 5.2
    # Window start is rounded to the hour → stable cache key.
    assert captured["starttime"].endswith(":00:00")


@pytest.mark.asyncio
async def test_provider_failure_is_empty_list(monkeypatch):
    async def _none(*a, **kw):
        return None
    monkeypatch.setattr(nws, "fetch_json", _none)
    monkeypatch.setattr(usgs, "fetch_json", _none)
    assert await nws.active_alerts(1, 2) == []
    assert await usgs.quakes_near(1, 2) == []


# ── Watcher ───────────────────────────────────────────────────────────

_HOME_US = {
    "name": "Portland", "country_code": "US",
    "latitude": 45.52, "longitude": -122.67,
}


class _FakeStore:
    def __init__(self, home=None):
        self._home = home

    async def get_user(self, user_id, key):
        if key == "sources.home_location" and self._home:
            return json.dumps(self._home)
        return None


def _runtime(home=_HOME_US, owner="u_a1"):
    return SimpleNamespace(
        owner_user_id=owner,
        _app_state=SimpleNamespace(
            settings_store=_FakeStore(home), notification_hub=object(),
        ),
        backend=SimpleNamespace(conn=object()),
        last_alert_watch_at=0.0,
    )


def _patch_watch(monkeypatch, *, alerts=(), quakes=()):
    async def _alerts(lat, lon):
        return list(alerts)

    async def _quakes(lat, lon, **kw):
        return list(quakes)

    sent: list[dict] = []

    async def _notify(runtime, **kw):
        sent.append(kw)

    monkeypatch.setattr(nws, "active_alerts", _alerts)
    monkeypatch.setattr(usgs, "quakes_near", _quakes)
    monkeypatch.setattr(alert_watch, "_notify", _notify)
    return sent


_SEVERE = {
    "id": "a1", "event": "Tornado Warning", "headline": "Tornado nearby",
    "severity": "Extreme", "expires": "", "instruction": "Shelter now.",
}
_MINOR = {
    "id": "a2", "event": "Frost Advisory", "headline": "Chilly",
    "severity": "Minor", "expires": "", "instruction": "",
}
_QUAKE = {
    "id": "q1", "magnitude": 5.2, "place": "Near Portland",
    "time_ms": 0, "url": "",
}


@pytest.mark.asyncio
async def test_watch_notifies_severe_and_quake_once(monkeypatch):
    sent = _patch_watch(monkeypatch, alerts=[_SEVERE, _MINOR], quakes=[_QUAKE])
    rt = _runtime()
    await alert_watch.step(rt)
    assert len(sent) == 2  # Minor filtered out
    titles = [s["title"] for s in sent]
    assert any("Tornado Warning" in t for t in titles)
    assert any("M5.2 earthquake" in t for t in titles)
    # Extreme severity escalates to critical importance.
    tornado = next(s for s in sent if "Tornado" in s["title"])
    from augmentum.notifications.catalog import IMPORTANCE_CRITICAL
    assert tornado["importance"] == IMPORTANCE_CRITICAL
    assert "Shelter now." in tornado["body"]

    # Second poll: same events, nothing re-sent.
    rt.last_alert_watch_at = 0.0
    await alert_watch.step(rt)
    assert len(sent) == 2


@pytest.mark.asyncio
async def test_watch_skips_nws_for_non_us_home(monkeypatch):
    called = {"nws": False}

    async def _alerts(lat, lon):
        called["nws"] = True
        return [_SEVERE]
    monkeypatch.setattr(nws, "active_alerts", _alerts)
    sent = _patch_watch(monkeypatch, quakes=[_QUAKE])
    monkeypatch.setattr(nws, "active_alerts", _alerts)

    rt = _runtime(home=dict(_HOME_US, country_code="DE", name="Berlin"))
    await alert_watch.step(rt)
    assert called["nws"] is False
    assert len(sent) == 1  # quake still notifies globally


@pytest.mark.asyncio
async def test_watch_inert_without_home(monkeypatch):
    sent = _patch_watch(monkeypatch, alerts=[_SEVERE], quakes=[_QUAKE])
    rt = _runtime(home=None)
    await alert_watch.step(rt)
    assert sent == []


@pytest.mark.asyncio
async def test_watch_respects_kill_switch(monkeypatch):
    sent = _patch_watch(monkeypatch, alerts=[_SEVERE])
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_alert_watch_enabled", False,
                        raising=False)
    rt = _runtime()
    await alert_watch.step(rt)
    assert sent == []


@pytest.mark.asyncio
async def test_watch_interval_gate(monkeypatch):
    sent = _patch_watch(monkeypatch, quakes=[_QUAKE])
    rt = _runtime()
    await alert_watch.step(rt)
    assert len(sent) == 1
    # Immediately again — interval gate blocks the whole poll.
    sent_before = len(sent)
    await alert_watch.step(rt)
    assert len(sent) == sent_before


@pytest.mark.asyncio
async def test_failed_notify_retries_next_poll(monkeypatch):
    attempts = {"n": 0}

    async def _alerts(lat, lon):
        return [_SEVERE]

    async def _quakes(lat, lon, **kw):
        return []

    async def _notify(runtime, **kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("hub down")

    monkeypatch.setattr(nws, "active_alerts", _alerts)
    monkeypatch.setattr(usgs, "quakes_near", _quakes)
    monkeypatch.setattr(alert_watch, "_notify", _notify)

    rt = _runtime()
    await alert_watch.step(rt)          # fails → id released
    rt.last_alert_watch_at = 0.0
    await alert_watch.step(rt)          # retried
    assert attempts["n"] == 2
    rt.last_alert_watch_at = 0.0
    await alert_watch.step(rt)          # now seen → no third attempt
    assert attempts["n"] == 2


# ── rsshub:// expansion (P2) ─────────────────────────────────────────

def test_rsshub_shorthand_expands(monkeypatch):
    from augmentum.config import settings
    from augmentum.discovery.feeds import _expand_rsshub
    monkeypatch.setattr(settings, "rsshub_base_url", "http://rsshub:1200",
                        raising=False)
    out = _expand_rsshub([
        "rsshub://github/release/DIYgod/RSSHub",
        "https://example.com/feed.xml",
        "RSSHUB://youtube/user/@x",
    ])
    assert out == [
        "http://rsshub:1200/github/release/DIYgod/RSSHub",
        "https://example.com/feed.xml",
        "http://rsshub:1200/youtube/user/@x",
    ]


def test_rsshub_shorthand_dropped_when_base_empty(monkeypatch):
    from augmentum.config import settings
    from augmentum.discovery.feeds import _expand_rsshub
    monkeypatch.setattr(settings, "rsshub_base_url", "", raising=False)
    out = _expand_rsshub([
        "rsshub://github/release/DIYgod/RSSHub",
        "https://example.com/feed.xml",
    ])
    assert out == ["https://example.com/feed.xml"]


# ── base substrate throttle ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_json_min_interval_throttles(monkeypatch):
    source_base.clear_cache()
    times: list[float] = []

    class _Resp:
        status_code = 200
        def json(self):
            return {"ok": True}

    class _Client:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            import time as _t
            times.append(_t.monotonic())
            return _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    await source_base.fetch_json("th", "https://x.test/1",
                                 min_interval_s=0.15)
    await source_base.fetch_json("th", "https://x.test/2",
                                 min_interval_s=0.15)
    assert times[1] - times[0] >= 0.14
    source_base.clear_cache()
