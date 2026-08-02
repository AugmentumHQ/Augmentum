"""Tests for Dream Journal CRUD operations."""
from __future__ import annotations

import pytest
import aiosqlite
from augmentum.dream.journal import DreamJournal
from augmentum.dream.models import DreamEntry, DreamEntryType


UID = "user_test"


@pytest.fixture
async def journal(tmp_path):
    """Create a DreamJournal backed by a temp SQLite database with migration applied."""
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        # Create the tables the DreamJournal initializer expects.
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS dream_entries (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL DEFAULT 'default',
                content TEXT NOT NULL,
                entry_type TEXT NOT NULL DEFAULT 'reflection',
                source_memories TEXT NOT NULL DEFAULT '[]',
                source_sessions TEXT NOT NULL DEFAULT '[]',
                context_window TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                weight REAL NOT NULL DEFAULT 1.0,
                pinned INTEGER NOT NULL DEFAULT 0,
                dream_cycle_id TEXT NOT NULL,
                user_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_dream_entries_persona
                ON dream_entries(persona_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS dream_portraits (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL,
                voice_notes TEXT,
                active_threads TEXT,
                impressions TEXT,
                source_entries TEXT,
                is_current INTEGER NOT NULL DEFAULT 0,
                checkpoint_name TEXT,
                user_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS dream_cycles (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL,
                trigger_reason TEXT,
                memories_count INTEGER DEFAULT 0,
                entries_count INTEGER DEFAULT 0,
                model_used TEXT,
                tokens_used INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                error TEXT,
                started_at TEXT,
                completed_at TEXT,
                user_id TEXT
            );
            CREATE TABLE IF NOT EXISTS dream_memory_log (
                memory_id TEXT NOT NULL,
                dream_cycle_id TEXT NOT NULL,
                persona_id TEXT NOT NULL,
                user_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (memory_id, dream_cycle_id)
            );
        """)
        await db.commit()
    journal = DreamJournal(db_path)
    await journal.initialize()
    return journal


@pytest.mark.asyncio
async def test_store_and_retrieve_entry(journal):
    entry_id = await journal.store_entry(
        persona_id="default",
        content="I found our discussion about architecture fascinating.",
        entry_type=DreamEntryType.REFLECTION,
        source_memories=["mem_1"],
        source_sessions=["sess_1"],
        context_window={"messages": []},
        dream_cycle_id="cycle_1",
        user_id=UID,
    )
    assert entry_id is not None
    entry = await journal.get_entry(entry_id)
    assert entry is not None
    assert entry.content == "I found our discussion about architecture fascinating."
    assert entry.entry_type == DreamEntryType.REFLECTION
    assert entry.persona_id == "default"
    assert entry.source_memories == ["mem_1"]


@pytest.mark.asyncio
async def test_list_entries_pagination(journal):
    for i in range(5):
        await journal.store_entry(
            persona_id="default", content=f"Reflection {i}",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=[], source_sessions=[],
            context_window={}, dream_cycle_id="cycle_1",
            user_id=UID,
        )
    entries, total = await journal.list_entries("default", limit=3, offset=0)
    assert len(entries) == 3
    assert total == 5


@pytest.mark.asyncio
async def test_update_entry_weight_and_pin(journal):
    entry_id = await journal.store_entry(
        persona_id="default", content="Test",
        entry_type=DreamEntryType.REFLECTION,
        source_memories=[], source_sessions=[],
        context_window={}, dream_cycle_id="cycle_1",
        user_id=UID,
    )
    await journal.update_entry(entry_id, weight=1.5, pinned=True)
    entry = await journal.get_entry(entry_id)
    assert entry.weight == 1.5
    assert entry.pinned is True


@pytest.mark.asyncio
async def test_delete_entry(journal):
    entry_id = await journal.store_entry(
        persona_id="default", content="To delete",
        entry_type=DreamEntryType.REFLECTION,
        source_memories=[], source_sessions=[],
        context_window={}, dream_cycle_id="cycle_1",
        user_id=UID,
    )
    await journal.delete_entry(entry_id)
    entry = await journal.get_entry(entry_id)
    assert entry is None


@pytest.mark.asyncio
async def test_compact_skips_pinned(journal):
    pinned_id = await journal.store_entry(
        persona_id="default", content="Pinned reflection",
        entry_type=DreamEntryType.REFLECTION,
        source_memories=[], source_sessions=[],
        context_window={}, dream_cycle_id="cycle_1",
        user_id=UID,
    )
    await journal.update_entry(pinned_id, pinned=True)
    unpinned_id = await journal.store_entry(
        persona_id="default", content="Old unpinned",
        entry_type=DreamEntryType.REFLECTION,
        source_memories=[], source_sessions=[],
        context_window={}, dream_cycle_id="cycle_1",
        user_id=UID,
    )
    stats = await journal.compact_journal("default", max_age_days=0)
    pinned = await journal.get_entry(pinned_id)
    assert pinned is not None  # survived compaction


@pytest.mark.asyncio
async def test_list_entries_filter_by_type(journal):
    await journal.store_entry(
        persona_id="default", content="A reflection",
        entry_type=DreamEntryType.REFLECTION,
        source_memories=[], source_sessions=[],
        context_window={}, dream_cycle_id="cycle_1",
        user_id=UID,
    )
    await journal.store_entry(
        persona_id="default", content="A voice note",
        entry_type=DreamEntryType.VOICE_NOTE,
        source_memories=[], source_sessions=[],
        context_window={}, dream_cycle_id="cycle_1",
        user_id=UID,
    )
    entries, total = await journal.list_entries("default", entry_type=DreamEntryType.VOICE_NOTE)
    assert total == 1
    assert entries[0].entry_type == DreamEntryType.VOICE_NOTE
