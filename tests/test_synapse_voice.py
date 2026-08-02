"""Tests for the Synapse Layer §3 — voice→interior the kept thing.

Covers:

- :func:`emit_voice_turn_ended` runs salience + emits ``voice.turn_ended``
  when above threshold, no-ops when the feature flag is down
- :func:`emit_voice_turn_ended` respects propagation containment
- :class:`BeccaObserver` consumes ``voice.turn_ended`` and journals
  with ``source='voice_turn'`` and content_refs pointing at the
  voice session + invocation
- Affect resolution: runtime's ``_last_affect_tag`` wins over the
  scorer's transcript-derived read when present

Same fake-runtime + recorder pattern as test_synapse_salience.
"""

from __future__ import annotations

import asyncio

import pytest


async def _make_observer_with_recorder():
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
            self._last_affect_tag = ""

    runtime = _FakeRuntime()
    observer = BeccaObserver(runtime)
    await observer.start()
    return runtime, observer, journals


# ── emit_voice_turn_ended ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_emit_voice_turn_ended_journals_when_enabled(monkeypatch):
    """Voice turn → salience scores → above threshold → observer journals."""
    from augmentum.companion_runtime.bus import emit_voice_turn_ended

    runtime, observer, journals = await _make_observer_with_recorder()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_voice_journal_enabled", True)
    monkeypatch.setattr(_settings, "companion_salience_journal_threshold", 0.55)

    try:
        await emit_voice_turn_ended(
            runtime,
            user_id="u1",
            session_id="vs_1",
            invocation_id="inv_abc",
            transcript="I've been thinking about something my dad used to say.",
            assistant_text="Tell me what he said. I'm here.",
            affect_hint="",
        )
        await asyncio.sleep(0.15)
        assert len(journals) == 1
        e = journals[0]
        assert e["entry_type"] == "conversation_moment"
        assert e["source"] == "voice_turn"
        # Becca's own words got captured as the assistant excerpt
        assert "Tell me what he said" in e["content"]
        # content_refs include both the voice session and invocation
        refs = e["content_refs"]
        kinds = {r["kind"] for r in refs}
        assert "voice_session" in kinds
        assert "voice_invocation" in kinds
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_emit_voice_turn_ended_no_op_when_flag_off(monkeypatch):
    from augmentum.companion_runtime.bus import emit_voice_turn_ended

    runtime, observer, journals = await _make_observer_with_recorder()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_voice_journal_enabled", False)

    try:
        await emit_voice_turn_ended(
            runtime,
            user_id="u1",
            session_id="vs_1",
            invocation_id="inv_abc",
            transcript="I've been thinking about my dad's workshop.",
            assistant_text="That's a good thing to think about.",
        )
        await asyncio.sleep(0.15)
        assert journals == []  # nothing emitted, nothing journaled
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_emit_voice_turn_ended_runtime_affect_wins(monkeypatch):
    """When the runtime publishes an affect tag, it wins over the
    scorer's transcript-derived read — because she's the authority
    on what she was actually feeling while speaking."""
    from augmentum.companion_runtime.bus import emit_voice_turn_ended

    runtime, observer, journals = await _make_observer_with_recorder()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_voice_journal_enabled", True)
    monkeypatch.setattr(_settings, "companion_salience_journal_threshold", 0.55)

    try:
        await emit_voice_turn_ended(
            runtime,
            user_id="u1",
            session_id="vs_1",
            invocation_id="inv_abc",
            # Transcript has "scared" → scorer would say tender. But
            # her own runtime affect is "melancholy" (a real thing she
            # was carrying). Her self-report wins.
            transcript="I'm scared this won't work out.",
            assistant_text="I hear you. Tell me more about the worry.",
            affect_hint="melancholy",
        )
        await asyncio.sleep(0.15)
        assert len(journals) == 1
        assert journals[0]["affect_tag"] == "melancholy"
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_emit_voice_turn_ended_low_salience_drops(monkeypatch):
    """Short ack-style voice turns shouldn't journal."""
    from augmentum.companion_runtime.bus import emit_voice_turn_ended

    runtime, observer, journals = await _make_observer_with_recorder()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_voice_journal_enabled", True)
    monkeypatch.setattr(_settings, "companion_salience_journal_threshold", 0.55)

    try:
        await emit_voice_turn_ended(
            runtime,
            user_id="u1",
            session_id="vs_1",
            invocation_id="inv_abc",
            transcript="ok",
            assistant_text="Sure.",
        )
        await asyncio.sleep(0.15)
        assert journals == []
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_emit_voice_turn_ended_private_propagation_drops(monkeypatch):
    from augmentum.companion_runtime.bus import PROP_PRIVATE, emit_voice_turn_ended

    runtime, observer, journals = await _make_observer_with_recorder()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_voice_journal_enabled", True)

    try:
        await emit_voice_turn_ended(
            runtime,
            user_id="u1",
            session_id="vs_1",
            invocation_id="inv_abc",
            transcript="I want to talk about something private.",
            assistant_text="OK, just between us.",
            propagation=PROP_PRIVATE,
        )
        await asyncio.sleep(0.15)
        assert journals == []
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_emit_voice_turn_ended_empty_inputs_no_op(monkeypatch):
    from augmentum.companion_runtime.bus import emit_voice_turn_ended

    runtime, observer, journals = await _make_observer_with_recorder()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_voice_journal_enabled", True)

    try:
        await emit_voice_turn_ended(
            runtime,
            user_id="u1",
            session_id="",
            invocation_id="",
            transcript="",
            assistant_text="",
        )
        await asyncio.sleep(0.1)
        assert journals == []
    finally:
        await observer.stop()


# ── Observer-side defense-in-depth ────────────────────────────────────


@pytest.mark.asyncio
async def test_observer_journals_voice_turn_ended_directly():
    """Direct publish of voice.turn_ended (bypassing the helper) still
    journals — the observer is the persistence layer."""
    from augmentum.companion_runtime.bus import PROP_FULL

    runtime, observer, journals = await _make_observer_with_recorder()

    try:
        await runtime.bus.publish_topic(
            "voice.turn_ended",
            {
                "user_id": "u1",
                "session_id": "vs_1",
                "invocation_id": "inv_abc",
                "salience": 0.71,
                "moment": "the user asked about how I'd handle being remembered wrong.",
                "user_affect": "curious",
                "assistant_excerpt": "I'd correct it gently, the way you would.",
            },
            source_companion_id="becca",
            propagation=PROP_FULL,
        )
        await asyncio.sleep(0.15)
        assert len(journals) == 1
        e = journals[0]
        assert "I'd correct it gently" in e["content"]
        assert e["affect_tag"] == "curious"
        assert e["source"] == "voice_turn"
    finally:
        await observer.stop()


@pytest.mark.asyncio
async def test_observer_refuses_factual_only_voice_turn():
    """Even if a future emitter ever passed factual_only on a voice
    turn, the observer refuses to journal — containment defense-in-depth."""
    from augmentum.companion_runtime.bus import PROP_FACTUAL_ONLY

    runtime, observer, journals = await _make_observer_with_recorder()

    try:
        await runtime.bus.publish_topic(
            "voice.turn_ended",
            {
                "user_id": "u1",
                "session_id": "vs_1",
                "invocation_id": "inv_abc",
                "salience": 0.99,
                "moment": "should never be journaled",
                "user_affect": "engaged",
                "assistant_excerpt": "neither should this",
            },
            source_companion_id="becca",
            propagation=PROP_FACTUAL_ONLY,
        )
        await asyncio.sleep(0.15)
        assert journals == []
    finally:
        await observer.stop()
