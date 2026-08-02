"""Time-actions slice — timers that ring, end-actions, verb_fire kind,
briefing weather gather.

Pins the 2026-06-11 audit fixes: timer fire now DELIVERS (notification
path; the old bus event had zero consumers), then-actions are
stakes-gated at set time and re-checked at fire, verb_fire refuses
non-trivial verbs at every fire, and briefings can gather typed
weather instead of SearXNG snippets.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import augmentum.architect.primitives  # noqa: F401 — registers verbs
import augmentum.intent  # noqa: F401
from augmentum.architect.primitives import time_timer
from augmentum.companion_runtime.tool_protocol import ToolResult
from augmentum.intent.action import SessionContext
from augmentum.intent.registry import REGISTRY


def _session(user_id="u_t1"):
    return SessionContext(
        user_id=user_id, session_id="s_t1", mode=None,
        app_state=SimpleNamespace(companion_runtime=None),
    )


# ── then-action gating at SET time ────────────────────────────────────

@pytest.mark.asyncio
async def test_then_verb_unknown_refused_at_set():
    action = REGISTRY.get("time.set_timer")
    result = await action.handler("", _session(), {
        "duration": "10 minutes", "then_verb": "nuke.everything",
    })
    assert "don't have an action" in result.speak


@pytest.mark.asyncio
async def test_then_verb_non_trivial_stakes_refused(monkeypatch):
    action = REGISTRY.get("time.set_timer")
    # Find (or fake) a non-trivial verb: patch a registry lookup.
    target = REGISTRY.get("memory.save") or REGISTRY.get("note.create")
    assert target is not None
    monkeypatch.setattr(target, "stakes", "consequential", raising=False)
    try:
        result = await action.handler("", _session(), {
            "duration": "5 minutes", "then_verb": target.id,
        })
        assert "can't schedule" in result.speak
    finally:
        pass  # monkeypatch restores stakes


@pytest.mark.asyncio
async def test_timer_with_then_verb_confirms(monkeypatch):
    # Prevent the real sleep task from doing anything interesting.
    async def _noop_fire(*a, **kw):
        return None
    monkeypatch.setattr(time_timer, "_fire_timer", _noop_fire)
    action = REGISTRY.get("time.set_timer")
    result = await action.handler("", _session(), {
        "duration": "20 minutes", "then_verb": "media.pause",
    })
    assert "20 minutes" in result.speak
    assert "media pause" in result.speak


# ── fire path: delivery + end-action ──────────────────────────────────

@pytest.mark.asyncio
async def test_fire_runs_then_action_and_delivers(monkeypatch):
    ran = {}

    async def _fake_execute(call, runtime, **kw):
        ran["verb"] = call.name
        ran["args"] = call.args
        return ToolResult(ok=True, tool=call.name,
                          payload={"content": "Music paused."})

    delivered = {}

    async def _fake_deliver(runtime, **kw):
        delivered.update(kw)

    import augmentum.companion_runtime.tools as tool_bridge
    monkeypatch.setattr(tool_bridge, "execute_tool", _fake_execute)
    monkeypatch.setattr(time_timer, "_deliver_fire", _fake_deliver)

    rt = SimpleNamespace(bus=None, _app_state=None, backend=None)
    await time_timer._fire_timer(
        "u_t1", "t_x", 0, "pause music", rt,
        then_verb="media.pause", then_args={"x": "1"},
    )
    assert ran["verb"] == "media.pause"
    assert delivered["then_line"] == "Music paused."
    assert delivered["user_id"] == "u_t1"


@pytest.mark.asyncio
async def test_fire_without_then_still_delivers(monkeypatch):
    delivered = {}

    async def _fake_deliver(runtime, **kw):
        delivered.update(kw)
    monkeypatch.setattr(time_timer, "_deliver_fire", _fake_deliver)

    rt = SimpleNamespace(bus=None, _app_state=None, backend=None)
    await time_timer._fire_timer("u_t1", "t_y", 0, "tea", rt)
    assert delivered["label"] == "tea"
    assert delivered["then_line"] == ""


@pytest.mark.asyncio
async def test_fire_broken_then_action_still_rings(monkeypatch):
    async def _boom(call, runtime, **kw):
        raise RuntimeError("registry exploded")

    delivered = {}

    async def _fake_deliver(runtime, **kw):
        delivered.update(kw)

    import augmentum.companion_runtime.tools as tool_bridge
    monkeypatch.setattr(tool_bridge, "execute_tool", _boom)
    monkeypatch.setattr(time_timer, "_deliver_fire", _fake_deliver)

    rt = SimpleNamespace(bus=None, _app_state=None, backend=None)
    await time_timer._fire_timer(
        "u_t1", "t_z", 0, "", rt, then_verb="media.pause",
    )
    assert "couldn't run media.pause" in delivered["then_line"]


# ── verb_fire standing-task kind ──────────────────────────────────────

def _vf_runtime():
    return SimpleNamespace(_app_state=None, backend=None, bus=None)


@pytest.mark.asyncio
async def test_verb_fire_unknown_verb_honest():
    from augmentum.companion_runtime.standing_tasks import _TASK_KINDS
    out = await _TASK_KINDS["verb_fire"](
        _vf_runtime(), user_id="u_v1", params={"verb": "nope.never"},
    )
    assert out["noteworthy"] is True
    assert "don't have a tool" in out["summary"]


@pytest.mark.asyncio
async def test_verb_fire_stakes_refused_at_fire(monkeypatch):
    from augmentum.companion_runtime.standing_tasks import _TASK_KINDS
    target = REGISTRY.get("media.pause") or REGISTRY.get("note.create")
    assert target is not None
    monkeypatch.setattr(target, "stakes", "consequential", raising=False)
    out = await _TASK_KINDS["verb_fire"](
        _vf_runtime(), user_id="u_v1", params={"verb": target.id},
    )
    assert "refused" in out["summary"]


@pytest.mark.asyncio
async def test_verb_fire_success_uses_result_content(monkeypatch):
    from augmentum.companion_runtime.standing_tasks import _TASK_KINDS

    async def _fake_execute(call, runtime, **kw):
        return ToolResult(
            ok=True, tool=call.name,
            payload={"content": "Right now in Portland it's 64°F and clear."},
        )

    import augmentum.companion_runtime.tools as tool_bridge
    monkeypatch.setattr(tool_bridge, "execute_tool", _fake_execute)
    verb = "weather.today"
    assert REGISTRY.get(verb) is not None
    out = await _TASK_KINDS["verb_fire"](
        _vf_runtime(), user_id="u_v1",
        params={"verb": verb, "verb_args": {"location": "portland"}},
    )
    assert out["noteworthy"] is True
    assert "64°F" in out["summary"]


# ── briefing weather gather ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_briefing_weather_gather_uses_home(monkeypatch):
    from augmentum.companion_runtime.standing_tasks import (
        _gather_weather_for_briefing,
    )
    from augmentum.sources import nws, open_meteo

    home = {
        "name": "Portland", "country_code": "US",
        "latitude": 45.52, "longitude": -122.67,
    }

    class _Store:
        async def get_user(self, user_id, key):
            return json.dumps(home) if key == "sources.home_location" else None

    async def _fc(lat, lon, *, imperial):
        return {
            "current": {"temperature_2m": 64, "apparent_temperature": 64,
                        "weather_code": 0, "wind_speed_10m": 5,
                        "relative_humidity_2m": 50, "precipitation": 0},
            "daily": {"weather_code": [0, 1],
                      "temperature_2m_max": [70, 72],
                      "temperature_2m_min": [50, 52],
                      "precipitation_probability_max": [0, 0]},
        }

    async def _alerts(lat, lon):
        return [{"id": "a", "event": "Tornado Warning",
                 "headline": "Tornado warning until 9PM",
                 "severity": "Extreme", "expires": "", "instruction": ""}]

    monkeypatch.setattr(open_meteo, "forecast", _fc)
    monkeypatch.setattr(nws, "active_alerts", _alerts)

    rt = SimpleNamespace(_app_state=SimpleNamespace(settings_store=_Store()))
    wx = await _gather_weather_for_briefing(rt, user_id="u_b1")
    assert wx is not None
    assert "Portland" in wx["title"]
    assert "Tornado warning" in wx["snippet"]


@pytest.mark.asyncio
async def test_briefing_weather_gather_no_home_is_none():
    from augmentum.companion_runtime.standing_tasks import (
        _gather_weather_for_briefing,
    )

    class _Store:
        async def get_user(self, user_id, key):
            return None

    rt = SimpleNamespace(_app_state=SimpleNamespace(settings_store=_Store()))
    assert await _gather_weather_for_briefing(rt, user_id="u_b2") is None


def test_weather_gather_tool_recognized():
    from augmentum.tools.schedule_briefing import _normalize_gather_tools
    canonical, dropped, aliased = _normalize_gather_tools(
        ["weather", "forecast", "bogus_tool"],
    )
    assert canonical == ["weather"]
    assert ("forecast", "weather") in aliased
    assert dropped == ["bogus_tool"]


def test_timer_channel_in_catalog():
    from augmentum.notifications.catalog import DEFAULT_CHANNELS
    ids = {c.channel_id for c in DEFAULT_CHANNELS}
    assert "time.timer" in ids
    assert "alerts.home" in ids


# ── schedule_action tool ──────────────────────────────────────────────

def _sa_tool(monkeypatch, *, runtime="default"):
    from augmentum.config import settings as app_settings
    from augmentum.tools.schedule_action import ScheduleActionTool
    monkeypatch.setattr(app_settings, "companion_runtime_enabled", True,
                        raising=False)
    monkeypatch.setattr(app_settings, "companion_standing_tasks_enabled", True,
                        raising=False)
    rt = SimpleNamespace(
        backend=SimpleNamespace(conn=object()), companion_id="becca",
    ) if runtime == "default" else runtime
    return ScheduleActionTool(SimpleNamespace(companion_runtime=rt))


@pytest.mark.asyncio
async def test_schedule_action_unknown_verb(monkeypatch):
    tool = _sa_tool(monkeypatch)
    r = await tool.execute(verb="nope.never", local_time="5pm",
                           _context={"user_id": "u_s1"})
    assert r.success is False
    assert "unknown verb" in r.error


@pytest.mark.asyncio
async def test_schedule_action_stakes_refused(monkeypatch):
    tool = _sa_tool(monkeypatch)
    target = REGISTRY.get("media.pause")
    monkeypatch.setattr(target, "stakes", "irrevocable", raising=False)
    r = await tool.execute(verb="media.pause", local_time="5pm",
                           _context={"user_id": "u_s1"})
    assert r.success is False
    assert "needs you present" in r.error


@pytest.mark.asyncio
async def test_schedule_action_creates_verb_fire_task(monkeypatch):
    captured = {}

    async def _fake_add_task(conn, **kw):
        captured.update(kw)
        return SimpleNamespace(id=7, next_run_at="2026-06-11 17:00:00")

    async def _fake_tz(app_state, user_id):
        return "America/Los_Angeles"

    from augmentum.companion_runtime import standing_tasks
    monkeypatch.setattr(standing_tasks, "add_task", _fake_add_task)
    monkeypatch.setattr(standing_tasks, "_resolve_user_timezone", _fake_tz)

    tool = _sa_tool(monkeypatch)
    r = await tool.execute(
        verb="weather.today", verb_args={"location": "portland"},
        local_time="5pm", _context={"user_id": "u_s1"},
    )
    assert r.success is True
    assert captured["kind"] == "verb_fire"
    assert captured["params"]["verb"] == "weather.today"
    assert captured["params"]["verb_args"] == {"location": "portland"}
    assert captured["params"]["local_time"] == "17:00"
    assert captured["params"]["one_shot"] is True  # default
    assert captured["user_timezone"] == "America/Los_Angeles"
    assert "17:00" in r.output


@pytest.mark.asyncio
async def test_schedule_action_recurring_weekdays(monkeypatch):
    captured = {}

    async def _fake_add_task(conn, **kw):
        captured.update(kw)
        return SimpleNamespace(id=8, next_run_at="2026-06-12 09:00:00")

    async def _fake_tz(app_state, user_id):
        return ""

    from augmentum.companion_runtime import standing_tasks
    monkeypatch.setattr(standing_tasks, "add_task", _fake_add_task)
    monkeypatch.setattr(standing_tasks, "_resolve_user_timezone", _fake_tz)

    tool = _sa_tool(monkeypatch)
    r = await tool.execute(
        verb="weather.today", local_time="9am", one_shot=False,
        weekdays=["mon", "tue", "wed", "thu", "fri"],
        _context={"user_id": "u_s1"},
    )
    assert r.success is True
    assert "one_shot" not in captured["params"]
    assert captured["params"]["weekdays"] == [1, 2, 3, 4, 5]
    assert "Mon/Tue/Wed/Thu/Fri" in r.output


def test_manage_tools_cover_verb_fire():
    import inspect

    from augmentum.tools import manage_briefings
    src = inspect.getsource(manage_briefings)
    assert 'in ("briefing", "verb_fire")' in src
