"""Tests for dream modules -- models, scheduler, journal, context, portrait."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.dream.context import DreamContextBuilder
from augmentum.dream.models import (
    ContextSegment,
    DreamCycle,
    DreamEntry,
    DreamEntryType,
    DreamPortrait,
)
from augmentum.dream.scheduler import DreamScheduler


class TestDreamModels:
    def test_dream_entry_construct(self):
        entry = DreamEntry(
            id="e1", persona_id="default", content="A quiet thought",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=["m1"], source_sessions=["s1"],
            context_window={}, embedding=None,
        )
        assert entry.id == "e1"
        assert entry.weight == 1.0
        assert entry.pinned is False

    def test_dream_entry_types(self):
        assert DreamEntryType.REFLECTION == "reflection"
        assert DreamEntryType.VOICE_NOTE == "voice_note"
        assert DreamEntryType.ACTIVE_THREAD == "active_thread"
        assert DreamEntryType.IMPRESSION == "impression"

    def test_dream_cycle_construct(self):
        cycle = DreamCycle(id="c1", persona_id="default", trigger_reason="threshold")
        assert cycle.status == "pending"
        assert cycle.entries_count == 0
        assert cycle.memories_count == 0

    def test_dream_portrait_construct(self):
        portrait = DreamPortrait(
            id="p1", persona_id="default",
            voice_notes="Thoughtful tone",
            active_threads="Exploring music",
            impressions="Curious nature",
            source_entries=["e1", "e2"],
        )
        assert portrait.is_current is True
        assert len(portrait.source_entries) == 2

    def test_context_segment_construct(self):
        seg = ContextSegment(
            memories=[{"id": "m1"}], messages=[],
            session_id="s1", timestamp_range=("t1", "t2"),
            relative_age="recently",
        )
        assert seg.session_id == "s1"


class TestDreamScheduler:
    def test_default_construction(self):
        engine = MagicMock()
        store = MagicMock()
        sched = DreamScheduler(engine, store)
        assert sched._enabled is True
        assert sched._message_threshold == 6
        assert sched._idle_minutes == 30
        assert sched._cooldown_minutes == 60

    def test_not_eligible_when_disabled(self):
        sched = DreamScheduler(MagicMock(), MagicMock(), enabled=False)
        assert sched._is_eligible() is False

    def test_not_eligible_below_threshold(self):
        sched = DreamScheduler(MagicMock(), MagicMock(), message_threshold=10)
        sched._messages_since_dream = 5
        sched._approved_since_dream = 1
        assert sched._is_eligible() is False

    def test_not_eligible_no_approvals(self):
        sched = DreamScheduler(MagicMock(), MagicMock(), message_threshold=5)
        sched._messages_since_dream = 10
        sched._approved_since_dream = 0
        assert sched._is_eligible() is False

    def test_not_eligible_not_idle(self):
        sched = DreamScheduler(MagicMock(), MagicMock(),
                               message_threshold=5, idle_minutes=30)
        sched._messages_since_dream = 10
        sched._approved_since_dream = 1
        sched._last_request_at = datetime.now(timezone.utc)  # just now
        assert sched._is_eligible() is False

    def test_eligible_all_conditions_met(self):
        sched = DreamScheduler(MagicMock(), MagicMock(),
                               message_threshold=5, idle_minutes=1, cooldown_minutes=1)
        sched._messages_since_dream = 10
        sched._approved_since_dream = 2
        sched._last_request_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        sched._last_dream_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert sched._is_eligible() is True

    def test_not_eligible_during_cooldown(self):
        sched = DreamScheduler(MagicMock(), MagicMock(),
                               message_threshold=5, idle_minutes=1, cooldown_minutes=60)
        sched._messages_since_dream = 10
        sched._approved_since_dream = 2
        sched._last_request_at = datetime.now(timezone.utc) - timedelta(minutes=5)
        sched._last_dream_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        assert sched._is_eligible() is False

    def test_notify_message_increments(self):
        sched = DreamScheduler(MagicMock(), MagicMock())
        sched.notify_message()
        sched.notify_message()
        assert sched._messages_since_dream == 2

    def test_notify_approval_increments(self):
        sched = DreamScheduler(MagicMock(), MagicMock())
        sched.notify_approval("mem1")
        assert sched._approved_since_dream == 1

    def test_get_status(self):
        sched = DreamScheduler(MagicMock(), MagicMock())
        status = sched.get_status()
        assert "enabled" in status
        assert "messages_since_dream" in status
        assert "running" in status

    def test_reset_counters(self):
        sched = DreamScheduler(MagicMock(), MagicMock())
        sched._messages_since_dream = 20
        sched._approved_since_dream = 5
        sched._reset_counters()
        assert sched._messages_since_dream == 0
        assert sched._approved_since_dream == 0


class TestDreamContextBuilder:
    def test_extract_window_centered(self):
        builder = DreamContextBuilder()
        path = [{"id": f"m{i}"} for i in range(10)]
        window = builder.extract_window(path, "m5", pairs=2)
        assert len(window) <= 4
        ids = [m["id"] for m in window]
        assert "m5" in ids

    def test_extract_window_target_not_found(self):
        builder = DreamContextBuilder()
        path = [{"id": "m1"}, {"id": "m2"}]
        window = builder.extract_window(path, "m99", pairs=2)
        assert window == []

    def test_cluster_by_proximity_single_session(self):
        builder = DreamContextBuilder()
        memories = [
            {"id": "m1", "session_id": "s1", "source_message_id": "msg1"},
            {"id": "m2", "session_id": "s1", "source_message_id": "msg2"},
            {"id": "m3", "session_id": "s1", "source_message_id": "msg100"},
        ]
        clusters = builder.cluster_by_proximity(memories)
        assert len(clusters) >= 1
        # All should be clustered (or split based on proximity)
        total = sum(len(c["memories"]) for c in clusters)
        assert total == 3

    def test_cluster_by_proximity_multiple_sessions(self):
        builder = DreamContextBuilder()
        memories = [
            {"id": "m1", "session_id": "s1", "source_message_id": "msg1"},
            {"id": "m2", "session_id": "s2", "source_message_id": "msg1"},
        ]
        clusters = builder.cluster_by_proximity(memories)
        assert len(clusters) == 2

    def test_humanize_age_just_now(self):
        builder = DreamContextBuilder()
        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        assert builder.humanize_age(ts, now) == "just now"

    def test_humanize_age_hours(self):
        builder = DreamContextBuilder()
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(hours=3)).isoformat()
        assert "3 hours ago" == builder.humanize_age(ts, now)

    def test_humanize_age_yesterday(self):
        builder = DreamContextBuilder()
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(days=1)).isoformat()
        assert builder.humanize_age(ts, now) == "yesterday"

    def test_humanize_age_invalid(self):
        builder = DreamContextBuilder()
        assert builder.humanize_age("not-a-date") == "some time ago"


class TestJournalPersistence:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_entry(self, tmp_path):
        import aiosqlite
        from augmentum.dream.journal import DreamJournal

        db_path = str(tmp_path / "dream_test.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE dream_entries (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT,
                    content TEXT,
                    entry_type TEXT,
                    source_memories TEXT,
                    source_sessions TEXT,
                    context_window TEXT,
                    embedding BLOB,
                    weight REAL DEFAULT 1.0,
                    pinned INTEGER DEFAULT 0,
                    dream_cycle_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            await conn.commit()

        journal = DreamJournal(db_path)
        await journal.initialize.__wrapped__(journal) if hasattr(journal.initialize, '__wrapped__') else None
        # Open manual connection since initialize checks for tables
        journal._db = await __import__("aiosqlite").connect(db_path)

        entry_id = await journal.store_entry(
            persona_id="default",
            content="A moment of reflection",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=["m1"],
            source_sessions=["s1"],
            context_window={},
            dream_cycle_id="cycle1",
        )

        assert isinstance(entry_id, str)
        assert len(entry_id) > 0

        entry = await journal.get_entry(entry_id)
        assert entry is not None
        assert entry.content == "A moment of reflection"
        assert entry.entry_type == DreamEntryType.REFLECTION

        await journal.close()

    @pytest.mark.asyncio
    async def test_list_entries(self, tmp_path):
        import aiosqlite
        from augmentum.dream.journal import DreamJournal

        db_path = str(tmp_path / "dream_list.db")
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE dream_entries (
                    id TEXT PRIMARY KEY,
                    persona_id TEXT,
                    content TEXT,
                    entry_type TEXT,
                    source_memories TEXT,
                    source_sessions TEXT,
                    context_window TEXT,
                    embedding BLOB,
                    weight REAL DEFAULT 1.0,
                    pinned INTEGER DEFAULT 0,
                    dream_cycle_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            await conn.commit()

        journal = DreamJournal(db_path)
        journal._db = await __import__("aiosqlite").connect(db_path)

        await journal.store_entry(
            persona_id="default", content="First thought",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=[], source_sessions=[],
            context_window={}, dream_cycle_id="c1",
        )
        await journal.store_entry(
            persona_id="default", content="Second thought",
            entry_type=DreamEntryType.VOICE_NOTE,
            source_memories=[], source_sessions=[],
            context_window={}, dream_cycle_id="c1",
        )

        entries, total = await journal.list_entries("default")
        assert total == 2
        assert len(entries) == 2

        await journal.close()
