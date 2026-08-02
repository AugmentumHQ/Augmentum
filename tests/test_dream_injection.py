"""Tests for dream context injection into system messages."""
from __future__ import annotations

import pytest
from augmentum.dream.models import DreamPortrait, DreamEntry, DreamEntryType


@pytest.mark.asyncio
async def test_inject_portrait_into_system_message():
    from augmentum.memory.integration import inject_dream_context

    portrait = DreamPortrait(
        id="p1", persona_id="default",
        voice_notes="We challenge each other.",
        active_threads="Thinking about resilience.",
        impressions="Genuine collaboration.",
        source_entries=[], is_current=True, created_at="",
    )
    messages = [{"role": "system", "content": "You are helpful."}]
    await inject_dream_context(messages, portrait, dream_entries=[])
    system = messages[0]["content"]
    assert "<evolved_self>" in system
    assert "We challenge each other." in system
    assert "Thinking about resilience." in system
    assert "Genuine collaboration." in system


@pytest.mark.asyncio
async def test_inject_dream_recall():
    from augmentum.memory.integration import inject_dream_context

    entries = [
        DreamEntry(
            id="e1", persona_id="default",
            content="That architecture discussion was eye-opening.",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=[], source_sessions=[],
            context_window={}, embedding=None,
            dream_cycle_id="c1", created_at="",
        ),
    ]
    messages = [{"role": "system", "content": "You are helpful."}]
    await inject_dream_context(messages, portrait=None, dream_entries=entries)
    system = messages[0]["content"]
    assert "<recent_reflections>" in system
    assert "architecture discussion" in system


@pytest.mark.asyncio
async def test_no_injection_when_empty():
    from augmentum.memory.integration import inject_dream_context

    messages = [{"role": "system", "content": "Original content."}]
    await inject_dream_context(messages, portrait=None, dream_entries=[])
    assert messages[0]["content"] == "Original content."


@pytest.mark.asyncio
async def test_inject_creates_system_message_if_missing():
    from augmentum.memory.integration import inject_dream_context

    portrait = DreamPortrait(
        id="p1", persona_id="default",
        voice_notes="Direct dynamic.",
        active_threads="", impressions="",
        source_entries=[], is_current=True, created_at="",
    )
    messages = [{"role": "user", "content": "Hello"}]
    await inject_dream_context(messages, portrait, dream_entries=[])
    assert messages[0]["role"] == "system"
    assert "<evolved_self>" in messages[0]["content"]


@pytest.mark.asyncio
async def test_inject_both_portrait_and_entries():
    from augmentum.memory.integration import inject_dream_context

    portrait = DreamPortrait(
        id="p1", persona_id="default",
        voice_notes="We challenge each other.",
        active_threads="Curious about X.",
        impressions="Good energy.",
        source_entries=[], is_current=True, created_at="",
    )
    entries = [
        DreamEntry(
            id="e1", persona_id="default",
            content="The resilience discussion was meaningful.",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=[], source_sessions=[],
            context_window={}, embedding=None,
            dream_cycle_id="c1", created_at="",
        ),
    ]
    messages = [{"role": "system", "content": "Base."}]
    await inject_dream_context(messages, portrait, dream_entries=entries)
    system = messages[0]["content"]
    assert "<evolved_self>" in system
    assert "<recent_reflections>" in system
    assert "Base." in system
