"""Tests for dream generation engine."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from augmentum.dream.engine import DreamEngine
from augmentum.dream.models import DreamEntryType


def _make_engine():
    """Create a DreamEngine with mocked dependencies."""
    engine = DreamEngine.__new__(DreamEngine)
    engine._journal = AsyncMock()
    engine._memory_store = AsyncMock()
    engine._state_manager = AsyncMock()
    engine._embedding_service = None
    engine._portrait_manager = None
    engine._settings = {}
    engine._context_builder = MagicMock()
    return engine


def test_parse_dream_response_valid():
    engine = _make_engine()
    response = json.dumps({
        "reflections": [
            {"type": "reflection", "content": "That conversation about resilience stuck with me."},
            {"type": "voice_note", "content": "We work well when challenging each other."},
        ]
    })
    entries = engine._parse_dream_response(response, "cycle_1", "default", [], [])
    assert len(entries) == 2
    assert entries[0].entry_type == DreamEntryType.REFLECTION
    assert entries[1].entry_type == DreamEntryType.VOICE_NOTE


def test_parse_dream_response_fallback_on_invalid_json():
    engine = _make_engine()
    entries = engine._parse_dream_response(
        "This was a meaningful moment for me.", "cycle_1", "default", [], [],
    )
    assert len(entries) == 1
    assert entries[0].entry_type == DreamEntryType.REFLECTION
    assert "meaningful moment" in entries[0].content


def test_filter_anti_patterns():
    engine = _make_engine()
    response = json.dumps({
        "reflections": [
            {"type": "reflection", "content": "As an AI language model, I found this interesting."},
            {"type": "impression", "content": "There's warmth in how we collaborate."},
        ]
    })
    entries = engine._parse_dream_response(response, "cycle_1", "default", [], [])
    assert len(entries) == 1
    assert "warmth" in entries[0].content


def test_parse_empty_reflections():
    engine = _make_engine()
    response = json.dumps({"reflections": []})
    entries = engine._parse_dream_response(response, "cycle_1", "default", [], [])
    assert len(entries) == 0


def test_parse_unknown_entry_type_defaults_to_reflection():
    engine = _make_engine()
    response = json.dumps({
        "reflections": [{"type": "unknown_type", "content": "Some thought."}]
    })
    entries = engine._parse_dream_response(response, "cycle_1", "default", [], [])
    assert len(entries) == 1
    assert entries[0].entry_type == DreamEntryType.REFLECTION


def test_parse_skips_empty_content():
    engine = _make_engine()
    response = json.dumps({
        "reflections": [
            {"type": "reflection", "content": ""},
            {"type": "reflection", "content": "   "},
            {"type": "reflection", "content": "Valid content."},
        ]
    })
    entries = engine._parse_dream_response(response, "cycle_1", "default", [], [])
    assert len(entries) == 1
    assert entries[0].content == "Valid content."


@pytest.mark.asyncio
async def test_select_dream_material_gates_on_tier_not_approval():
    """Eligibility = active/core tier, regardless of user_approved flag.

    The engine used to require user_approved=1, but that flag is only
    set by the manual notification-approval UI which most users never
    touch — it produced empty cycles for almost everyone. Now the tier
    filter is the actual quality gate (provisional/expired never reach
    it), and any tier-active/core memory qualifies whether or not the
    user explicitly approved it via the notifications panel.
    """
    engine = _make_engine()
    engine._get_dream_eligible_memories = AsyncMock(return_value=[
        {"id": "m1", "user_approved": 1, "tier": "active", "source_message_id": "msg1", "session_id": "s1", "created_at": "2026-01-01"},
        {"id": "m2", "user_approved": 0, "tier": "active", "source_message_id": "msg2", "session_id": "s1", "created_at": "2026-01-01"},
        {"id": "m3", "user_approved": 1, "tier": "provisional", "source_message_id": "msg3", "session_id": "s1", "created_at": "2026-01-01"},
    ])
    engine._dreamed_memory_ids = AsyncMock(return_value=set())

    result = await engine._select_dream_material("default")
    ids = {m["id"] for m in result}
    assert ids == {"m1", "m2"}, "tier=active rows qualify regardless of user_approved"
    assert "m3" not in ids, "tier=provisional must still be excluded"
