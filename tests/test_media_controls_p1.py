"""Mid-experience media controls — wiring program Phase 1.

Covers the receiver resolution ladder (augmentum/intent/media_devices),
the media.volume / media.speed / media.sleep_timer verbs, the transport
retrofit (pause/next/previous route to a cast receiver when one is
live), and the receiver→presence feeder. Everything runs against a
fake DeviceRegistry — the drivers have their own tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.devices.invocation import InvocationResult
from augmentum.intent.action import SessionContext
from augmentum.intent.media_devices import (
    PlaybackTarget,
    resolve_playback_target,
    transport_on_target,
    volume_on_target,
)

UID = "user-p1"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSession(SimpleNamespace):
    pass


class FakeDevice(SimpleNamespace):
    pass


def _session(device_id="dev-tv", title="The Expanse", cap="media.video_play@1",
             sess_id="sess-1", extra=None):
    return FakeSession(
        id=sess_id, device_id=device_id, capability_id=cap, title=title,
        created_at=1000.0, last_event_at=1000.0, extra=dict(extra or {}),
    )


def _device(device_id="dev-tv", label="Living Room TV",
            caps=("media.video_play@1", "media.audio_play@1")):
    return FakeDevice(id=device_id, label=label, capabilities=list(caps))


class FakeRegistry:
    def __init__(self, sessions=(), devices=(), snapshots=None,
                 unsupported=()):
        self._sessions = list(sessions)
        self._devices = list(devices)
        self._snapshots = dict(snapshots or {})
        self._unsupported = set(unsupported)
        self.invocations: list[tuple] = []

    async def list_sessions(self, *, user_id):
        assert user_id == UID
        return list(self._sessions)

    async def list(self, *, user_id, only_online=False):
        return list(self._devices)

    async def get(self, device_id, *, user_id):
        return next((d for d in self._devices if d.id == device_id), None)

    async def snapshot(self, *, user_id, device_id, capability):
        return self._snapshots.get(device_id)

    async def invoke(self, *, user_id, device_id, capability, action, args=None):
        self.invocations.append((device_id, capability, action, dict(args or {})))
        if action in self._unsupported:
            return InvocationResult.failure("nope", code="unsupported_action")
        return InvocationResult.success()


def _app_state(reg):
    return SimpleNamespace(device_registry=reg)


def _ctx(reg):
    return SessionContext(user_id=UID, session_id="s1", app_state=_app_state(reg))


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ladder_no_registry_falls_to_tab():
    res = await resolve_playback_target(SimpleNamespace(), UID)
    assert res.target is None and not res.clarify and not res.miss


@pytest.mark.asyncio
async def test_ladder_single_session_wins():
    reg = FakeRegistry(sessions=[_session()], devices=[_device()])
    res = await resolve_playback_target(_app_state(reg), UID)
    assert res.target is not None
    assert res.target.device_label == "Living Room TV"
    assert res.target.session_id == "sess-1"


@pytest.mark.asyncio
async def test_ladder_multiple_sessions_clarify():
    reg = FakeRegistry(
        sessions=[_session(), _session(device_id="dev-br", sess_id="sess-2")],
        devices=[_device(), _device(device_id="dev-br", label="Bedroom TV")],
    )
    res = await resolve_playback_target(_app_state(reg), UID)
    assert res.target is None
    assert "Living Room TV" in res.clarify and "Bedroom TV" in res.clarify


@pytest.mark.asyncio
async def test_ladder_device_hint_overrides_session_count():
    reg = FakeRegistry(
        sessions=[_session(), _session(device_id="dev-br", sess_id="sess-2")],
        devices=[_device(), _device(device_id="dev-br", label="Bedroom TV")],
    )
    res = await resolve_playback_target(_app_state(reg), UID, device_hint="bedroom")
    assert res.target is not None and res.target.device_id == "dev-br"


@pytest.mark.asyncio
async def test_ladder_hint_matches_idle_saved_device():
    # Volume on a named device works even with nothing cast by us.
    reg = FakeRegistry(devices=[_device()])
    res = await resolve_playback_target(_app_state(reg), UID, device_hint="living room")
    assert res.target is not None and res.target.session_id == ""


@pytest.mark.asyncio
async def test_ladder_hint_miss_is_honest():
    reg = FakeRegistry(devices=[_device()])
    res = await resolve_playback_target(_app_state(reg), UID, device_hint="kitchen speaker")
    assert res.target is None and res.miss == "kitchen speaker"


# ---------------------------------------------------------------------------
# Volume on a receiver
# ---------------------------------------------------------------------------

def _target():
    return PlaybackTarget(
        device_id="dev-tv", device_label="Living Room TV",
        capability_id="media.video_play@1", session_id="sess-1",
    )


@pytest.mark.asyncio
async def test_volume_set_invokes_set_volume():
    reg = FakeRegistry()
    ok, detail = await volume_on_target(
        _app_state(reg), UID, _target(), direction="set", level=30,
    )
    assert ok and detail == "30"
    assert reg.invocations[-1][2] == "set_volume"
    assert reg.invocations[-1][3] == {"level": 30}


@pytest.mark.asyncio
async def test_volume_up_steps_from_snapshot():
    reg = FakeRegistry(snapshots={"dev-tv": {"volume_level": 40}})
    ok, detail = await volume_on_target(
        _app_state(reg), UID, _target(), direction="up",
    )
    assert ok and detail == "50"


@pytest.mark.asyncio
async def test_volume_up_without_readable_level_fails_honestly():
    reg = FakeRegistry(snapshots={"dev-tv": {}})
    ok, detail = await volume_on_target(
        _app_state(reg), UID, _target(), direction="up",
    )
    assert not ok and "Living Room TV" in detail
    assert all(call[2] != "set_volume" for call in reg.invocations)


@pytest.mark.asyncio
async def test_volume_mute_invokes_set_mute():
    reg = FakeRegistry()
    ok, _ = await volume_on_target(
        _app_state(reg), UID, _target(), direction="mute",
    )
    assert ok
    assert reg.invocations[-1][2] == "set_mute"
    assert reg.invocations[-1][3] == {"muted": True}


# ---------------------------------------------------------------------------
# media.volume verb
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_media_volume_receiver_path_speaks_device():
    from augmentum.intent.builtin.media import _media_volume
    reg = FakeRegistry(sessions=[_session()], devices=[_device()],
                       snapshots={"dev-tv": {"volume_level": 60}})
    res = await _media_volume("turn it down", _ctx(reg), {"direction": "down"})
    assert res.short_circuit
    assert "Living Room TV" in res.speak and "50" in res.speak
    assert res.surface_emit is None


@pytest.mark.asyncio
async def test_media_volume_no_cast_emits_in_tab_channel():
    from augmentum.intent.builtin.media import _media_volume
    reg = FakeRegistry()
    res = await _media_volume("turn it up", _ctx(reg), {"direction": "up"})
    assert res.surface_emit is not None
    assert res.surface_emit["channel"] == "media.volume"
    assert res.surface_emit["payload"]["direction"] == "up"


@pytest.mark.asyncio
async def test_media_volume_bare_level_implies_set():
    from augmentum.intent.builtin.media import _media_volume
    reg = FakeRegistry()
    res = await _media_volume("volume forty", _ctx(reg), {"level": 40})
    assert res.surface_emit["payload"] == {"direction": "set", "level": 40}


@pytest.mark.asyncio
async def test_media_volume_no_direction_clarifies():
    from augmentum.intent.builtin.media import _media_volume
    reg = FakeRegistry()
    res = await _media_volume("the volume", _ctx(reg), {})
    assert res.clarify is not None and "direction" in res.clarify["missing"]


# ---------------------------------------------------------------------------
# Transport retrofit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pause_routes_to_active_receiver():
    from augmentum.architect.primitives.media_control import _media_pause_handler
    reg = FakeRegistry(sessions=[_session()], devices=[_device()])
    res = await _media_pause_handler("pause it", _ctx(reg), {})
    assert res.surface_emit is None  # receiver took it — no tab emit
    assert reg.invocations[-1][2] == "pause"
    assert "Living Room TV" in res.speak


@pytest.mark.asyncio
async def test_pause_without_cast_keeps_tab_emit():
    from augmentum.architect.primitives.media_control import _media_pause_handler
    reg = FakeRegistry()
    res = await _media_pause_handler("pause it", _ctx(reg), {})
    assert res.surface_emit == {
        "channel": "media.transport", "payload": {"action": "pause"},
    }


@pytest.mark.asyncio
async def test_next_on_queueless_receiver_is_honest():
    from augmentum.architect.primitives.media_control import _media_next_handler
    reg = FakeRegistry(sessions=[_session()], devices=[_device()],
                       unsupported={"next"})
    res = await _media_next_handler("next", _ctx(reg), {})
    assert res.surface_emit is None
    assert "queue" in res.speak


@pytest.mark.asyncio
async def test_transport_on_target_uses_session_capability():
    reg = FakeRegistry()
    await transport_on_target(_app_state(reg), UID, _target(), "pause")
    assert reg.invocations[-1][1] == "media.video_play@1"


# ---------------------------------------------------------------------------
# media.speed / media.sleep_timer verbs (in-tab emits)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_media_speed_rate_emits_adjust():
    from augmentum.intent.builtin.media import _media_speed
    res = await _media_speed("1.5x", _ctx(FakeRegistry()), {"rate": 1.5})
    assert res.surface_emit["channel"] == "media.adjust"
    assert res.surface_emit["payload"] == {"action": "speed", "rate": 1.5}


@pytest.mark.asyncio
async def test_media_speed_step_and_clarify():
    from augmentum.intent.builtin.media import _media_speed
    res = await _media_speed("slower", _ctx(FakeRegistry()), {"step": "slower"})
    assert res.surface_emit["payload"] == {"action": "speed", "step": "slower"}
    res2 = await _media_speed("change speed", _ctx(FakeRegistry()), {})
    assert res2.clarify is not None


@pytest.mark.asyncio
async def test_sleep_timer_minutes_and_chapter_and_cancel():
    from augmentum.intent.builtin.media import _media_sleep_timer
    ctx = _ctx(FakeRegistry())
    res = await _media_sleep_timer("30 min", ctx, {"minutes": 30})
    assert res.surface_emit["payload"] == {"action": "sleep_timer", "minutes": 30}
    assert "30 minutes" in res.speak
    res = await _media_sleep_timer("chapter", ctx, {"end_of_chapter": True})
    assert res.surface_emit["payload"] == {
        "action": "sleep_timer", "end_of_chapter": True,
    }
    res = await _media_sleep_timer("cancel", ctx, {"cancel": True})
    assert res.surface_emit["payload"] == {"action": "sleep_timer", "cancel": True}


# ---------------------------------------------------------------------------
# Receiver → presence (playing slot)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_receiver_session_feeds_playing_slot():
    from augmentum.companion_runtime.presence_context import (
        now_context,
        prompt_lines,
    )
    reg = FakeRegistry(sessions=[_session()], devices=[_device()])
    snap = await now_context(None, UID, app_state=_app_state(reg))
    playing = snap["playing"]
    assert playing is not None
    assert playing["label"] == "The Expanse"
    assert playing["device"] == "Living Room TV"
    rendered = "\n".join(prompt_lines(snap))
    assert "on Living Room TV" in rendered


@pytest.mark.asyncio
async def test_no_receiver_session_leaves_playing_untouched():
    from augmentum.companion_runtime.presence_context import now_context
    reg = FakeRegistry()
    snap = await now_context(None, UID, app_state=_app_state(reg))
    assert snap["playing"] is None


@pytest.mark.asyncio
async def test_receiver_presence_carries_author_from_session_extra():
    from augmentum.companion_runtime.presence_context import (
        now_context,
        prompt_lines,
    )
    reg = FakeRegistry(
        sessions=[_session(
            title="Project Hail Mary",
            extra={"device_label": "Living Room TV", "author": "Andy Weir"},
        )],
    )
    snap = await now_context(None, UID, app_state=_app_state(reg))
    assert snap["playing"]["author"] == "Andy Weir"
    rendered = "\n".join(prompt_lines(snap))
    assert "Project Hail Mary by Andy Weir" in rendered


# ---------------------------------------------------------------------------
# context_peek playing — receiver leg (index and depth must agree)
# ---------------------------------------------------------------------------

def _peek_tool(reg):
    from augmentum.tools.context_peek import ContextPeekTool
    return ContextPeekTool(app_state=_app_state(reg))


@pytest.mark.asyncio
async def test_peek_playing_sees_receiver_cast_with_live_state():
    reg = FakeRegistry(
        sessions=[_session(
            title="Project Hail Mary",
            extra={"device_label": "Living Room TV", "author": "Andy Weir"},
        )],
        snapshots={"dev-tv": {
            "current_time_s": 1930, "duration_s": 7080,
            "is_paused": False, "volume_level": 40,
        }},
    )
    tool = _peek_tool(reg)
    res = await tool._peek_playing(UID, "s1")
    assert res.success
    assert "Casting to Living Room TV: Project Hail Mary by Andy Weir" in res.output
    assert "32m 10s of 1h 58m" in res.output
    assert "playing" in res.output
    assert "volume 40" in res.output


@pytest.mark.asyncio
async def test_peek_playing_paused_and_muted_render():
    reg = FakeRegistry(
        sessions=[_session()],
        snapshots={"dev-tv": {
            "current_time_s": 60, "is_paused": True, "is_muted": True,
        }},
    )
    res = await _peek_tool(reg)._peek_playing(UID, "s1")
    assert "paused" in res.output and "muted" in res.output


@pytest.mark.asyncio
async def test_peek_playing_no_cast_no_attention_is_honest():
    res = await _peek_tool(FakeRegistry())._peek_playing(UID, "s1")
    assert res.success
    assert "Nothing is playing" in res.output


def test_clock_formatting():
    from augmentum.tools.context_peek import _clock
    assert _clock(7080) == "1h 58m"
    assert _clock(1930) == "32m 10s"
    assert _clock(45) == "45s"
    assert _clock(None) == "?"


# ---------------------------------------------------------------------------
# notify verbs — gate behavior (store round-trips live in the
# notifications substrate's own tests)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notify_mute_respects_disabled_gate(monkeypatch):
    from augmentum.config import settings
    from augmentum.intent.builtin.notify import _notify_mute
    monkeypatch.setattr(settings, "notifications_enabled", False, raising=False)
    res = await _notify_mute("mute it", _ctx(FakeRegistry()), {"channel": "jobs"})
    assert "turned off" in res.speak


@pytest.mark.asyncio
async def test_notify_dismiss_needs_store(monkeypatch):
    from augmentum.config import settings
    from augmentum.intent.builtin.notify import _notify_dismiss
    monkeypatch.setattr(settings, "notifications_enabled", True, raising=False)
    # app_state without a SQLite-backed state_manager → honest refusal.
    res = await _notify_dismiss("dismiss", _ctx(FakeRegistry()), {})
    assert "can't reach" in res.speak


# ---------------------------------------------------------------------------
# Manifest / registry pins
# ---------------------------------------------------------------------------

def test_new_verbs_registered_and_bucketed():
    import augmentum.intent  # noqa: F401 — trigger builtin imports
    from augmentum.intent.manifest import (
        VOICE_TOOLS_CORE,
        VOICE_TOOLS_DISRUPTIVE,
    )
    from augmentum.intent.registry import REGISTRY
    ids = {a.id for a in REGISTRY.all()}
    for verb in ("media.volume", "media.speed", "media.sleep_timer",
                 "notify.mute", "notify.dismiss"):
        assert verb in ids, f"{verb} not registered"
    for verb in ("media.volume", "media.speed", "media.sleep_timer"):
        assert verb in VOICE_TOOLS_DISRUPTIVE
    for verb in ("notify.mute", "notify.dismiss"):
        assert verb in VOICE_TOOLS_CORE
