"""Tests for dream portrait synthesis and management."""
from __future__ import annotations

import json
import pytest
from datetime import datetime, timedelta, timezone
from augmentum.dream.portrait import PortraitManager
from augmentum.dream.models import DreamPortrait, DreamEntry, DreamEntryType


def test_parse_portrait_response_valid():
    manager = PortraitManager.__new__(PortraitManager)
    response = json.dumps({
        "voice_notes": "We have a direct, challenging dynamic.",
        "active_threads": "Curious about their resilience patterns.",
        "impressions": "Genuine collaborative energy.",
    })
    portrait = manager._parse_portrait_response(response, "default", ["e1", "e2"])
    assert portrait.voice_notes == "We have a direct, challenging dynamic."
    assert portrait.active_threads == "Curious about their resilience patterns."
    assert portrait.impressions == "Genuine collaborative energy."
    assert portrait.is_current is True


def test_parse_portrait_response_fallback():
    manager = PortraitManager.__new__(PortraitManager)
    portrait = manager._parse_portrait_response("We work well together.", "default", ["e1"])
    assert "work well" in portrait.voice_notes
    assert portrait.active_threads == ""
    assert portrait.impressions == ""


def test_parse_portrait_response_empty():
    manager = PortraitManager.__new__(PortraitManager)
    portrait = manager._parse_portrait_response("", "default", [])
    assert portrait is None


def test_weight_entries():
    manager = PortraitManager.__new__(PortraitManager)
    now = datetime.now(timezone.utc)
    entries = [
        DreamEntry(
            id="e1", persona_id="default", content="Recent pinned",
            entry_type=DreamEntryType.REFLECTION, source_memories=[],
            source_sessions=[], context_window={}, embedding=None,
            weight=1.0, pinned=True, dream_cycle_id="c1",
            created_at=(now - timedelta(days=1)).isoformat(),
        ),
        DreamEntry(
            id="e2", persona_id="default", content="Old unpinned",
            entry_type=DreamEntryType.REFLECTION, source_memories=[],
            source_sessions=[], context_window={}, embedding=None,
            weight=1.0, pinned=False, dream_cycle_id="c1",
            created_at=(now - timedelta(days=45)).isoformat(),
        ),
    ]
    weighted = manager._weight_entries(entries, now)
    assert weighted[0][0] == "e1"
    assert weighted[0][1] > weighted[1][1]


def test_weight_entries_decay_tiers():
    """Test the three decay tiers: <7 days (1.0), 7-30 days (0.7), >30 days (0.4)."""
    manager = PortraitManager.__new__(PortraitManager)
    now = datetime.now(timezone.utc)
    entries = [
        DreamEntry(id="recent", persona_id="default", content="R",
                   entry_type=DreamEntryType.REFLECTION, source_memories=[],
                   source_sessions=[], context_window={}, embedding=None,
                   weight=1.0, pinned=False, dream_cycle_id="c1",
                   created_at=(now - timedelta(days=3)).isoformat()),
        DreamEntry(id="aging", persona_id="default", content="A",
                   entry_type=DreamEntryType.REFLECTION, source_memories=[],
                   source_sessions=[], context_window={}, embedding=None,
                   weight=1.0, pinned=False, dream_cycle_id="c1",
                   created_at=(now - timedelta(days=15)).isoformat()),
        DreamEntry(id="old", persona_id="default", content="O",
                   entry_type=DreamEntryType.REFLECTION, source_memories=[],
                   source_sessions=[], context_window={}, embedding=None,
                   weight=1.0, pinned=False, dream_cycle_id="c1",
                   created_at=(now - timedelta(days=45)).isoformat()),
    ]
    weighted = manager._weight_entries(entries, now)
    scores = {w[0]: w[1] for w in weighted}
    assert scores["recent"] == pytest.approx(1.0)
    assert scores["aging"] == pytest.approx(0.7)
    assert scores["old"] == pytest.approx(0.4)


def test_enforce_token_budget():
    """Portrait sections should be truncated to token budget by sentence boundary."""
    manager = PortraitManager.__new__(PortraitManager)
    portrait = DreamPortrait(
        id="p1", persona_id="default",
        voice_notes="First sentence about voice. " * 30,
        active_threads="First thread thought. " * 30,
        impressions="First impression note. " * 30,
        source_entries=[], is_current=True, created_at="",
    )
    trimmed = manager._enforce_token_budget(portrait)
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(trimmed.voice_notes)) <= 150
    assert len(enc.encode(trimmed.active_threads)) <= 150
    assert len(enc.encode(trimmed.impressions)) <= 100
    assert trimmed.voice_notes.rstrip().endswith(".")
    assert trimmed.active_threads.rstrip().endswith(".")
    assert trimmed.impressions.rstrip().endswith(".")
