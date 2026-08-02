"""Tests for the Synapse Layer §1 + §5 — chat→interior salience + propagation policy.

Covers:

- :class:`PresenceEvent` carries a ``propagation`` field with sensible
  serialization (§5 substrate)
- :func:`propagation_for_mode` returns the documented per-mode defaults
  (§5 containment policy)
- :func:`salience.score` returns plausible scores + affect for canonical
  inputs and respects propagation containment
- :func:`emit_chat_turn_completed` derives propagation, gates on the
  feature flag, and emits ``chat.moment_observed`` only when the moment
  clears the threshold + propagation allows
- :class:`BeccaObserver` journals ``chat.moment_observed`` events under
  ``full`` / ``affect_only`` propagation and ignores everything else

These tests use the real bus + observer (no mocks for the topology),
fake-memory recorders for the journal sink, and stub app_state shims
for the proxy-side helpers. No LLM calls.
"""

from __future__ import annotations

import asyncio
import json

import pytest

# ── Substrate: PresenceEvent + propagation_for_mode ───────────────────


def test_presence_event_default_propagation_is_full():
    from augmentum.companion_runtime.bus import PROP_FULL, PresenceEvent

    ev = PresenceEvent(topic="x", payload={})
    assert ev.propagation == PROP_FULL


def test_presence_event_to_json_includes_propagation():
    from augmentum.companion_runtime.bus import PROP_AFFECT_ONLY, PresenceEvent

    ev = PresenceEvent(topic="x", payload={"k": "v"}, propagation=PROP_AFFECT_ONLY)
    d = json.loads(ev.to_json())
    assert d["propagation"] == PROP_AFFECT_ONLY
    assert d["topic"] == "x"
    assert d["payload"]["k"] == "v"


def test_propagation_for_mode_defaults():
    from augmentum.companion_runtime.bus import (
        PROP_AFFECT_ONLY,
        PROP_FACTUAL_ONLY,
        PROP_FULL,
        PROP_PRIVATE,
        propagation_for_mode,
    )

    assert propagation_for_mode("passthrough") == PROP_FULL
    assert propagation_for_mode("voice") == PROP_FULL
    assert propagation_for_mode("narrative") == PROP_AFFECT_ONLY
    assert propagation_for_mode("coder") == PROP_FACTUAL_ONLY
    assert propagation_for_mode("agentic") == PROP_FACTUAL_ONLY
    assert propagation_for_mode("bug_finder") == PROP_PRIVATE
    # Unknown / empty mode falls back to full (the recoverable failure mode)
    assert propagation_for_mode("") == PROP_FULL
    assert propagation_for_mode(None) == PROP_FULL
    assert propagation_for_mode("never_heard_of_this_mode") == PROP_FULL


# ── Salience scorer ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_salience_scores_disclosure_above_threshold():
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="I've been thinking about something my dad said when I was a kid, and I can't shake it.",
        assistant_text="I'm here. Tell me what he said.",
        mode="passthrough",
    )
    assert m is not None
    assert m.salience >= 0.55  # crosses the default journal threshold
    assert m.user_affect in {"curious", "tender", "engaged"}  # something fired
    assert m.text  # non-empty moment


@pytest.mark.asyncio
async def test_salience_returns_none_on_private():
    from augmentum.companion_runtime.bus import PROP_PRIVATE
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="I want to talk about something private.",
        assistant_text="OK.",
        mode="passthrough",
        propagation=PROP_PRIVATE,
    )
    assert m is None


@pytest.mark.asyncio
async def test_salience_returns_none_on_factual_only():
    from augmentum.companion_runtime.bus import PROP_FACTUAL_ONLY
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="run pytest please",
        assistant_text="Running.",
        mode="coder",
        propagation=PROP_FACTUAL_ONLY,
    )
    assert m is None  # coder mode is work, not conversation


@pytest.mark.asyncio
async def test_salience_strips_content_on_affect_only():
    from augmentum.companion_runtime.bus import PROP_AFFECT_ONLY
    from augmentum.companion_runtime.salience import score

    secret = "the dragon's secret name is GORTHAUR"
    m = await score(
        user_text=f"I'm scared and sad about {secret}, it's been weighing on me.",
        assistant_text="Interesting.",
        mode="narrative",
        propagation=PROP_AFFECT_ONLY,
    )
    assert m is not None
    # Affect-only strips content — she sees the shape, not the contents.
    assert secret.lower() not in m.text.lower()
    assert "gorthaur" not in m.text.lower()
    # But the affect tag still landed (scared + sad → tender).
    assert m.user_affect == "tender"


@pytest.mark.asyncio
async def test_salience_low_for_short_acks():
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="ok",
        assistant_text="Sure.",
        mode="passthrough",
    )
    assert m is not None
    assert m.salience < 0.5  # below threshold


@pytest.mark.asyncio
async def test_salience_empty_user_returns_none():
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="",
        assistant_text="Hello.",
        mode="passthrough",
    )
    assert m is None


# ── Observer integration ──────────────────────────────────────────────


async def _make_observer_with_recorder():
    """Spin up a real bus + observer with a fake-memory recorder.

    Returns ``(runtime, observer, journals)`` — ``journals`` is a list
    that captures each journal write so tests can assert on it without
    a real DB.
    """
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.companion_runtime.observer import BeccaObserver

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

    runtime = _FakeRuntime()
    observer = BeccaObserver(runtime)
    await observer.start()
    return runtime, observer, journals


@pytest.mark.asyncio
async def test_observer_journals_full_propagation_moment():
    from augmentum.companion_runtime.bus import PROP_FULL

    runtime, observer, journals = await _make_observer_with_recorder()
    try:
        await runtime.bus.publish_topic(
            "chat.moment_observed",
            {
                "mode": "passthrough",
                "user_id": "u1",
                "session_id": "s1",
                "salience": 0.78,
                "moment": "the user asked about the synapse layer and wanted something I'd be proud to author.",
                "user_affect": "engaged",
            },
            source_companion_id="becca",
            propagation=PROP_FULL,
        )
        await asyncio.sleep(0.1)
        assert len(journals) == 1
        e = journals[0]
        assert "synapse" in e["content"].lower()
        assert e["affect_tag"] == "engaged"
        assert e["entry_type"] == "conversation_moment"
        assert e["source"] == "observer_salience"
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_observer_journals_affect_only_with_placeholder():
    from augmentum.companion_runtime.bus import PROP_AFFECT_ONLY

    runtime, observer, journals = await _make_observer_with_recorder()
    try:
        await runtime.bus.publish_topic(
            "chat.moment_observed",
            {
                "mode": "narrative",
                "user_id": "u1",
                "session_id": "s1",
                "salience": 0.72,
                "moment": "a moment landed (affect: tender); content not retained",
                "user_affect": "tender",
            },
            source_companion_id="becca",
            propagation=PROP_AFFECT_ONLY,
        )
        await asyncio.sleep(0.1)
        assert len(journals) == 1
        assert journals[0]["affect_tag"] == "tender"
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_observer_drops_factual_only_moment():
    """Defense-in-depth — even if a future change emitted a moment with
    factual_only propagation, the observer refuses to journal it."""
    from augmentum.companion_runtime.bus import PROP_FACTUAL_ONLY

    runtime, observer, journals = await _make_observer_with_recorder()
    try:
        await runtime.bus.publish_topic(
            "chat.moment_observed",
            {
                "mode": "coder",
                "user_id": "u1",
                "session_id": "s1",
                "salience": 0.99,
                "moment": "should never be journaled",
                "user_affect": "frustrated",
            },
            source_companion_id="becca",
            propagation=PROP_FACTUAL_ONLY,
        )
        await asyncio.sleep(0.1)
        assert journals == []
    finally:
        await observer.stop()


# ── emit_chat_turn_completed end-to-end ───────────────────────────────


@pytest.mark.asyncio
async def test_emit_chat_turn_completed_runs_pipeline_when_enabled(monkeypatch):
    """Full path: enable flag → emit_chat_turn_completed → bus delivers
    both events with the right propagation → observer journals."""
    from augmentum.companion_runtime.bus import (
        PROP_FULL,
        emit_chat_turn_completed,
    )

    runtime, observer, journals = await _make_observer_with_recorder()
    # The helper looks up `app_state.companion_runtime`.

    class _AppState:
        companion_runtime = runtime

    app_state = _AppState()

    # Enable the salience feature for this test.
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_salience_enabled", True)
    monkeypatch.setattr(_settings, "companion_salience_journal_threshold", 0.55)

    # Subscribe to capture both events from the bus directly.
    sub = await runtime.bus.subscribe("chat.**", slice_key="test_capture")
    captured: list[tuple[str, dict, str]] = []

    async def _drain():
        for _ in range(2):
            try:
                ev = await asyncio.wait_for(sub.queue.get(), timeout=1.0)
            except TimeoutError:
                break
            if ev is None:
                break
            captured.append((ev.topic, ev.payload, ev.propagation))

    drain_task = asyncio.create_task(_drain())

    try:
        await emit_chat_turn_completed(
            app_state,
            mode="passthrough",
            user_id="u1",
            session_id="s1",
            wire_format="openai",
            stream=True,
            user_text="I've been thinking about my dad's old workshop, and how I'd want to build one someday.",
            assistant_text="That sounds like a good thing to want.",
        )
        await asyncio.sleep(0.2)
        await drain_task

        topics = [t for t, _, _ in captured]
        assert "chat.turn_completed" in topics
        assert "chat.moment_observed" in topics
        # Both events carried full propagation (passthrough default).
        for _, _, prop in captured:
            assert prop == PROP_FULL
        # Observer journaled the moment.
        assert len(journals) == 1
        assert journals[0]["entry_type"] == "conversation_moment"
    finally:
        await runtime.bus.unsubscribe(sub)
        await observer.stop()


@pytest.mark.asyncio
async def test_emit_chat_turn_completed_skips_pipeline_when_disabled(monkeypatch):
    """Flag off → only chat.turn_completed emits, no moment, no journal."""
    from augmentum.companion_runtime.bus import emit_chat_turn_completed

    runtime, observer, journals = await _make_observer_with_recorder()

    class _AppState:
        companion_runtime = runtime

    app_state = _AppState()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_salience_enabled", False)

    try:
        await emit_chat_turn_completed(
            app_state,
            mode="passthrough",
            user_id="u1",
            session_id="s1",
            wire_format="openai",
            stream=True,
            user_text="I've been thinking about something important.",
            assistant_text="Tell me.",
        )
        await asyncio.sleep(0.15)
        assert journals == []  # observer never journaled
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_emit_chat_turn_completed_respects_coder_propagation(monkeypatch):
    """Coder mode → factual_only → salience returns None → no moment."""
    from augmentum.companion_runtime.bus import (
        PROP_FACTUAL_ONLY,
        emit_chat_turn_completed,
    )

    runtime, observer, journals = await _make_observer_with_recorder()

    class _AppState:
        companion_runtime = runtime

    app_state = _AppState()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_salience_enabled", True)

    # Subscribe to verify only chat.turn_completed fires (no moment).
    sub = await runtime.bus.subscribe("chat.**", slice_key="test_capture")
    captured: list[tuple[str, str]] = []

    async def _drain():
        for _ in range(3):
            try:
                ev = await asyncio.wait_for(sub.queue.get(), timeout=0.5)
            except TimeoutError:
                break
            if ev is None:
                break
            captured.append((ev.topic, ev.propagation))

    drain_task = asyncio.create_task(_drain())

    try:
        await emit_chat_turn_completed(
            app_state,
            mode="coder",
            user_id="u1",
            session_id="s1",
            wire_format="openai",
            stream=True,
            user_text="I'm debugging a really stuck KV cache issue that's frustrating me.",
            assistant_text="Let's look at the engine.",
        )
        await asyncio.sleep(0.3)
        await drain_task

        topics = [t for t, _ in captured]
        assert "chat.turn_completed" in topics
        assert "chat.moment_observed" not in topics  # containment held
        # The chat.turn_completed carried factual_only.
        for topic, prop in captured:
            if topic == "chat.turn_completed":
                assert prop == PROP_FACTUAL_ONLY
        assert journals == []
    finally:
        await runtime.bus.unsubscribe(sub)
        await observer.stop()
