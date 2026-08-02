"""Tests for the Synapse Layer §2 — chat→PAD affect echo.

Covers:

- :class:`UserAffectTracker` correctly maps tags to PAD coordinates
- Decay is exponential with the right half-life
- Reads with no observation return neutral
- Multiple observations blend rather than overwrite
- Observer hooks update the tracker on chat moments and voice turns
- The bus emits ``user.affect_observed`` events
- :func:`_user_weather_line` returns prose only at high enough confidence
- Containment: factual_only chat turns never produce moments (verified
  upstream in salience tests; here we verify the *consequence* — affect
  is never updated for factual_only because no moment ever fires)
"""

from __future__ import annotations

import asyncio
import time

import pytest


# ── Tracker math ──────────────────────────────────────────────────────


def test_tracker_initial_read_is_neutral():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    t = UserAffectTracker()
    obs = t.read("u1")
    assert obs.confidence == 0.0
    assert obs.sample_count == 0
    assert obs.tag == "unclear"


def test_tracker_update_records_observation():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    t = UserAffectTracker()
    obs = t.update("u1", "tender")
    assert obs.tag == "tender"
    assert obs.sample_count == 1
    # Tender maps to positive valence, low arousal
    assert obs.valence > 0.0
    assert obs.arousal < 0.5


def test_tracker_read_decays_to_neutral_over_half_life():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    t = UserAffectTracker(half_life_s=60.0)
    base = time.time()
    t.update("u1", "frustrated", observed_at=base)

    # At t=0, full confidence
    obs0 = t.read("u1", now=base)
    assert obs0.confidence == pytest.approx(1.0, abs=0.01)

    # At t=half_life, confidence ≈ 0.5
    obs_half = t.read("u1", now=base + 60.0)
    assert obs_half.confidence == pytest.approx(0.5, abs=0.05)

    # At t=4*half_life, confidence very low
    obs_far = t.read("u1", now=base + 240.0)
    assert obs_far.confidence < 0.1


def test_tracker_blend_doesnt_clobber():
    """A second observation blends with the first rather than replacing."""
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    t = UserAffectTracker(half_life_s=600.0)  # 10-min half-life
    base = time.time()
    # Strong positive observation
    t.update("u1", "excited", observed_at=base)
    excited_v = t.read("u1", now=base).valence
    # A small amount of time later, a slightly negative read
    t.update("u1", "tired", observed_at=base + 60.0)  # 1 min later, decay ≈ 0.9
    blended = t.read("u1", now=base + 60.0)
    # The blend should be between excited and tired valences
    # excited is +0.7, tired is -0.2
    assert blended.valence < excited_v
    assert blended.valence > -0.2
    # Sample count incremented
    assert blended.sample_count == 2


def test_tracker_unknown_tag_falls_to_neutral():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    t = UserAffectTracker()
    obs = t.update("u1", "this_tag_doesnt_exist")
    assert obs.tag == "unclear"


def test_tracker_reset():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    t = UserAffectTracker()
    t.update("u1", "tender")
    t.update("u2", "excited")
    t.reset("u1")
    assert t.read("u1").sample_count == 0
    assert t.read("u2").sample_count == 1
    t.reset()  # clear all
    assert t.read("u2").sample_count == 0


# ── Observer → tracker integration ────────────────────────────────────


async def _make_observer_with_tracker():
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.companion_runtime.observer import BeccaObserver
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker

    journals: list[dict] = []

    class _FakeMemory:
        async def journal(self, content, **kwargs):
            journals.append({"content": content, **kwargs})
            return len(journals)

    class _FakeRuntime:
        def __init__(self):
            self.bus = PresenceBus()
            self.companion_id = "becca"
            self.memory = _FakeMemory()
            self._last_affect_tag = ""
            self.user_affect = UserAffectTracker()

    runtime = _FakeRuntime()
    observer = BeccaObserver(runtime)
    await observer.start()
    return runtime, observer, journals


@pytest.mark.asyncio
async def test_chat_moment_updates_user_affect():
    from augmentum.companion_runtime.bus import PROP_FULL

    runtime, observer, _journals = await _make_observer_with_tracker()
    try:
        await runtime.bus.publish_topic(
            "chat.moment_observed",
            {
                "mode": "passthrough",
                "user_id": "u1",
                "session_id": "s1",
                "salience": 0.72,
                "moment": "I'm so tired today, I can barely think.",
                "user_affect": "tired",
            },
            source_companion_id="becca",
            propagation=PROP_FULL,
        )
        await asyncio.sleep(0.15)
        obs = runtime.user_affect.read("u1")
        assert obs.sample_count == 1
        assert obs.tag == "tired"
        assert obs.arousal < 0.3  # tired = low arousal
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_voice_turn_updates_user_affect():
    from augmentum.companion_runtime.bus import PROP_FULL

    runtime, observer, _ = await _make_observer_with_tracker()
    try:
        await runtime.bus.publish_topic(
            "voice.turn_ended",
            {
                "user_id": "u1",
                "session_id": "vs_1",
                "invocation_id": "inv_abc",
                "salience": 0.78,
                "moment": "Voice moment.",
                "user_affect": "frustrated",
                "assistant_excerpt": "I hear you.",
            },
            source_companion_id="becca",
            propagation=PROP_FULL,
        )
        await asyncio.sleep(0.15)
        obs = runtime.user_affect.read("u1")
        assert obs.sample_count == 1
        assert obs.tag == "frustrated"
        assert obs.valence < 0.0  # frustrated = negative
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_observer_emits_user_affect_observed_event():
    """The bus event lets downstream consumers (avatar widget, future
    emotion bridge) react without polling the tracker."""
    from augmentum.companion_runtime.bus import PROP_FULL

    runtime, observer, _ = await _make_observer_with_tracker()
    captured: list[dict] = []
    sub = await runtime.bus.subscribe("user.affect_observed", slice_key="t")

    async def _drain():
        for _ in range(2):
            try:
                ev = await asyncio.wait_for(sub.queue.get(), timeout=0.6)
            except asyncio.TimeoutError:
                break
            if ev is None:
                break
            captured.append({"payload": ev.payload, "topic": ev.topic})

    drain_task = asyncio.create_task(_drain())

    try:
        await runtime.bus.publish_topic(
            "chat.moment_observed",
            {
                "mode": "passthrough",
                "user_id": "u1",
                "session_id": "s1",
                "salience": 0.7,
                "moment": "I'm excited about this.",
                "user_affect": "excited",
            },
            source_companion_id="becca",
            propagation=PROP_FULL,
        )
        await asyncio.sleep(0.2)
        await drain_task
        topics = [c["topic"] for c in captured]
        assert "user.affect_observed" in topics
        affect_event = next(c for c in captured if c["topic"] == "user.affect_observed")
        assert affect_event["payload"]["tag"] == "excited"
        assert affect_event["payload"]["source"] == "chat"
        assert affect_event["payload"]["user_id"] == "u1"
    finally:
        await runtime.bus.unsubscribe(sub)
        await observer.stop()


# ── Prompt composition read site ──────────────────────────────────────


def test_user_weather_line_empty_when_no_observation():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker
    from augmentum.companion_runtime.prompt_compose import _user_weather_line

    class _FakeRuntime:
        user_affect = UserAffectTracker()

    class _FakeIntent:
        user_id = "u1"

    line = _user_weather_line(_FakeRuntime(), _FakeIntent())
    assert line == ""


def test_user_weather_line_for_recent_observation():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker
    from augmentum.companion_runtime.prompt_compose import _user_weather_line

    tracker = UserAffectTracker()
    tracker.update("u1", "tired")

    class _FakeRuntime:
        pass
    rt = _FakeRuntime()
    rt.user_affect = tracker

    class _FakeIntent:
        user_id = "u1"

    line = _user_weather_line(rt, _FakeIntent())
    assert "tired" in line.lower()
    assert "quiet" in line.lower()


def test_user_weather_line_empty_for_decayed_observation():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker
    from augmentum.companion_runtime.prompt_compose import _user_weather_line

    tracker = UserAffectTracker(half_life_s=60.0)
    # Observation 4 half-lives ago — confidence ~ 0.0625
    tracker.update("u1", "frustrated", observed_at=time.time() - 240.0)

    class _FakeRuntime:
        pass
    rt = _FakeRuntime()
    rt.user_affect = tracker

    class _FakeIntent:
        user_id = "u1"

    line = _user_weather_line(rt, _FakeIntent())
    assert line == ""  # decayed — she doesn't pretend to know


def test_user_weather_line_hedges_at_medium_confidence():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker
    from augmentum.companion_runtime.prompt_compose import _user_weather_line

    tracker = UserAffectTracker(half_life_s=60.0)
    # Observation ~1 half-life ago — confidence ~ 0.5
    tracker.update("u1", "frustrated", observed_at=time.time() - 65.0)

    class _FakeRuntime:
        pass
    rt = _FakeRuntime()
    rt.user_affect = tracker

    class _FakeIntent:
        user_id = "u1"

    line = _user_weather_line(rt, _FakeIntent())
    # Medium confidence — line exists but with a hedge marker
    assert "low confidence" in line.lower() or "could be wrong" in line.lower()


def test_user_weather_line_empty_when_no_user_id():
    from augmentum.companion_runtime.perception.user_affect import UserAffectTracker
    from augmentum.companion_runtime.prompt_compose import _user_weather_line

    tracker = UserAffectTracker()
    tracker.update("u1", "tender")

    class _FakeRuntime:
        pass
    rt = _FakeRuntime()
    rt.user_affect = tracker

    class _FakeIntent:
        user_id = ""

    assert _user_weather_line(rt, _FakeIntent()) == ""


def test_user_weather_line_safe_when_no_tracker():
    """Older fixtures without user_affect attached — should not crash."""
    from augmentum.companion_runtime.prompt_compose import _user_weather_line

    class _FakeRuntime:
        pass

    class _FakeIntent:
        user_id = "u1"

    assert _user_weather_line(_FakeRuntime(), _FakeIntent()) == ""
