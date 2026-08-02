"""Tests for Dream System data models."""
from __future__ import annotations

import pytest

from augmentum.dream.models import (
    ContextSegment,
    DreamCycle,
    DreamEntry,
    DreamEntryType,
    DreamPortrait,
)


class TestDreamEntryType:
    def test_enum_values(self):
        assert DreamEntryType.REFLECTION == "reflection"
        assert DreamEntryType.VOICE_NOTE == "voice_note"
        assert DreamEntryType.ACTIVE_THREAD == "active_thread"
        assert DreamEntryType.IMPRESSION == "impression"

    def test_is_string(self):
        """DreamEntryType inherits from str so it behaves as a string."""
        assert isinstance(DreamEntryType.REFLECTION, str)

    def test_string_comparison(self):
        assert DreamEntryType.REFLECTION == "reflection"
        assert DreamEntryType.VOICE_NOTE != "reflection"

    def test_from_value(self):
        et = DreamEntryType("voice_note")
        assert et is DreamEntryType.VOICE_NOTE


class TestDreamEntry:
    def _make(self, **overrides):
        defaults = dict(
            id="entry-001",
            persona_id="default",
            content="I have been reflecting on recent conversations.",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=["mem-1", "mem-2"],
            source_sessions=["sess-abc"],
            context_window={"token_count": 512},
            embedding=None,
        )
        defaults.update(overrides)
        return DreamEntry(**defaults)

    def test_construction_all_fields(self):
        entry = self._make(
            weight=2.5,
            pinned=True,
            dream_cycle_id="cycle-42",
            created_at="2026-03-25T10:00:00",
            expires_at="2026-04-25T10:00:00",
            embedding=b"\x00\x01\x02",
        )
        assert entry.id == "entry-001"
        assert entry.persona_id == "default"
        assert entry.content == "I have been reflecting on recent conversations."
        assert entry.entry_type == DreamEntryType.REFLECTION
        assert entry.source_memories == ["mem-1", "mem-2"]
        assert entry.source_sessions == ["sess-abc"]
        assert entry.context_window == {"token_count": 512}
        assert entry.embedding == b"\x00\x01\x02"
        assert entry.weight == 2.5
        assert entry.pinned is True
        assert entry.dream_cycle_id == "cycle-42"
        assert entry.created_at == "2026-03-25T10:00:00"
        assert entry.expires_at == "2026-04-25T10:00:00"

    def test_defaults(self):
        entry = self._make()
        assert entry.weight == 1.0
        assert entry.pinned is False
        assert entry.dream_cycle_id == ""
        assert entry.created_at == ""
        assert entry.expires_at is None

    def test_embedding_none(self):
        entry = self._make(embedding=None)
        assert entry.embedding is None

    def test_embedding_bytes(self):
        data = bytes([0] * 768 * 4)
        entry = self._make(embedding=data)
        assert len(entry.embedding) == 768 * 4


class TestDreamPortrait:
    def _make(self, **overrides):
        defaults = dict(
            id="portrait-001",
            persona_id="default",
            voice_notes="I speak with warmth and curiosity.",
            active_threads="Thread A\nThread B",
            impressions="Users appreciate clear explanations.",
            source_entries=["entry-001", "entry-002"],
        )
        defaults.update(overrides)
        return DreamPortrait(**defaults)

    def test_construction(self):
        portrait = self._make()
        assert portrait.id == "portrait-001"
        assert portrait.persona_id == "default"
        assert portrait.voice_notes == "I speak with warmth and curiosity."
        assert portrait.active_threads == "Thread A\nThread B"
        assert portrait.impressions == "Users appreciate clear explanations."
        assert portrait.source_entries == ["entry-001", "entry-002"]

    def test_default_is_current_true(self):
        portrait = self._make()
        assert portrait.is_current is True

    def test_is_current_false(self):
        portrait = self._make(is_current=False)
        assert portrait.is_current is False

    def test_checkpoint_name_default_none(self):
        portrait = self._make()
        assert portrait.checkpoint_name is None

    def test_checkpoint_name_set(self):
        portrait = self._make(checkpoint_name="pre-experiment", created_at="2026-03-25T00:00:00")
        assert portrait.checkpoint_name == "pre-experiment"
        assert portrait.created_at == "2026-03-25T00:00:00"


class TestDreamCycle:
    def _make(self, **overrides):
        defaults = dict(
            id="cycle-001",
            persona_id="default",
            trigger_reason="memory_threshold_reached",
        )
        defaults.update(overrides)
        return DreamCycle(**defaults)

    def test_construction(self):
        cycle = self._make()
        assert cycle.id == "cycle-001"
        assert cycle.persona_id == "default"
        assert cycle.trigger_reason == "memory_threshold_reached"

    def test_default_status_pending(self):
        cycle = self._make()
        assert cycle.status == "pending"

    def test_defaults(self):
        cycle = self._make()
        assert cycle.memories_count == 0
        assert cycle.entries_count == 0
        assert cycle.model_used is None
        assert cycle.tokens_used == 0
        assert cycle.duration_ms == 0
        assert cycle.error is None
        assert cycle.started_at == ""
        assert cycle.completed_at is None

    def test_completed_cycle(self):
        cycle = self._make(
            memories_count=25,
            entries_count=4,
            model_used="mistral",
            tokens_used=3500,
            duration_ms=12000,
            status="completed",
            started_at="2026-03-25T10:00:00",
            completed_at="2026-03-25T10:00:12",
        )
        assert cycle.status == "completed"
        assert cycle.memories_count == 25
        assert cycle.entries_count == 4
        assert cycle.model_used == "mistral"
        assert cycle.tokens_used == 3500
        assert cycle.duration_ms == 12000
        assert cycle.completed_at == "2026-03-25T10:00:12"

    def test_failed_cycle(self):
        cycle = self._make(status="failed", error="LLM unavailable")
        assert cycle.status == "failed"
        assert cycle.error == "LLM unavailable"


class TestContextSegment:
    def test_construction(self):
        seg = ContextSegment(
            memories=[{"id": "mem-1", "content": "User prefers bullet points"}],
            messages=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}],
            session_id="sess-abc",
            timestamp_range=("2026-03-24T08:00:00", "2026-03-24T09:00:00"),
            relative_age="1 day ago",
        )
        assert seg.session_id == "sess-abc"
        assert len(seg.memories) == 1
        assert len(seg.messages) == 2
        assert seg.timestamp_range == ("2026-03-24T08:00:00", "2026-03-24T09:00:00")
        assert seg.relative_age == "1 day ago"

    def test_empty_memories_and_messages(self):
        seg = ContextSegment(
            memories=[],
            messages=[],
            session_id="sess-empty",
            timestamp_range=("2026-03-25T00:00:00", "2026-03-25T00:00:00"),
            relative_age="just now",
        )
        assert seg.memories == []
        assert seg.messages == []
