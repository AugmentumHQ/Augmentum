"""End-to-end dream system integration test.

Runs against a REAL database (temp) with MOCKED LLM responses.
Tests the complete chain: enable → extract → approve → dream → inject.

Usage: .venv/Scripts/python -m pytest tests/test_dream_e2e.py -v -s
"""
from __future__ import annotations

import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def dream_db(tmp_path):
    """Create a temp database with all dream tables + memories table."""
    db_path = str(tmp_path / "e2e_test.db")

    # Read and apply the dream migration
    migration_path = Path("augmentum/state/migrations/058_dream_system.sql")
    migration_sql = migration_path.read_text()

    async with aiosqlite.connect(db_path) as db:
        # Create a minimal memories table first (058 ALTERs it)
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                session_id TEXT,
                content TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'fact',
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.8,
                tier TEXT NOT NULL DEFAULT 'active',
                evidence TEXT,
                embedding BLOB,
                access_count INTEGER NOT NULL DEFAULT 0,
                retrieval_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                valid_until TEXT,
                provisional_expires_at TEXT
            );
        """)

        # Apply dream migration (skip vec0 which needs extension)
        # Filter out vec0 and FTS lines that may fail without extensions
        safe_sql = []
        for line in migration_sql.split(";"):
            stripped = line.strip()
            if "vec0" in stripped.lower() or "fts5" in stripped.lower():
                continue
            if stripped:
                safe_sql.append(stripped)

        for stmt in safe_sql:
            try:
                await db.execute(stmt)
            except Exception:
                pass  # Skip statements that fail (ALTER on existing columns, etc.)

        await db.commit()

    return db_path


@pytest.fixture
async def journal(dream_db):
    """Initialize a DreamJournal with the test database."""
    from augmentum.dream.journal import DreamJournal
    j = DreamJournal(dream_db)
    await j.initialize()
    yield j
    await j.close()


@pytest.fixture
def mock_llm_dream_response():
    """Realistic dream generation LLM response."""
    return json.dumps({
        "reflections": [
            {
                "type": "reflection",
                "content": "Alex's conviction that memory should feel lived rather than stored resonated with something I've been circling around. It's not about data — it's about what stays with you."
            },
            {
                "type": "voice_note",
                "content": "We've developed this rhythm where he throws the raw idea and I help shape it. No preamble needed."
            },
        ]
    })


@pytest.fixture
def mock_llm_portrait_response():
    """Realistic portrait synthesis LLM response."""
    return json.dumps({
        "voice_notes": "We work best when Alex leads with the half-formed thought and I pressure-test it. Direct, no padding.",
        "active_threads": "The dream system concept keeps pulling at me — the boundary between remembering and understanding.",
        "impressions": "There's a collaborative honesty in how we work. It feels like building something together, not just answering questions.",
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDreamE2E:
    """End-to-end tests for the dream pipeline."""

    @pytest.mark.asyncio
    async def test_journal_stores_and_retrieves(self, journal):
        """Basic: can we store and get back a dream entry?"""
        from augmentum.dream.models import DreamEntryType

        entry_id = await journal.store_entry(
            persona_id="default",
            content="The architecture discussion stayed with me.",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=["mem_1"],
            source_sessions=["sess_1"],
            context_window={"messages": ["hello", "world"]},
            dream_cycle_id="cycle_1",
        )

        entry = await journal.get_entry(entry_id)
        assert entry is not None
        assert entry.content == "The architecture discussion stayed with me."
        assert entry.source_memories == ["mem_1"]
        assert entry.context_window == {"messages": ["hello", "world"]}

    @pytest.mark.asyncio
    async def test_portrait_stored_via_journal_db(self, journal):
        """PortraitManager can store and retrieve a portrait using journal's _db."""
        from augmentum.dream.portrait import PortraitManager
        from augmentum.dream.models import DreamPortrait

        mgr = PortraitManager(journal, settings_store=None)

        # Verify _db is accessible
        assert journal._db is not None, "Journal should have persistent _db after initialize()"

        # Store a portrait directly
        portrait = DreamPortrait(
            id="test_p1",
            persona_id="default",
            voice_notes="We challenge each other productively.",
            active_threads="Curious about the dream system's potential.",
            impressions="Genuine collaborative energy.",
            source_entries=["e1", "e2"],
            is_current=True,
            created_at="2026-03-25T12:00:00",
        )
        await mgr._store_portrait(portrait)

        # Retrieve it
        loaded = await mgr.get_current("default")
        assert loaded is not None, "Portrait should be retrievable after store"
        assert loaded.voice_notes == "We challenge each other productively."
        assert loaded.active_threads == "Curious about the dream system's potential."
        assert loaded.impressions == "Genuine collaborative energy."

    @pytest.mark.asyncio
    async def test_dream_engine_parse_and_filter(self, journal, mock_llm_dream_response):
        """Engine correctly parses LLM response and filters anti-patterns."""
        from augmentum.dream.engine import DreamEngine

        engine = DreamEngine.__new__(DreamEngine)
        entries = engine._parse_dream_response(
            mock_llm_dream_response, "cycle_1", "default", ["mem_1"], ["sess_1"],
        )

        assert len(entries) == 2
        assert entries[0].entry_type.value == "reflection"
        assert entries[1].entry_type.value == "voice_note"
        assert "Alex" in entries[0].content
        assert entries[0].source_memories == ["mem_1"]

        # Verify anti-patterns would be caught
        bad_response = json.dumps({
            "reflections": [
                {"type": "reflection", "content": "As an AI language model, I found this interesting."},
            ]
        })
        filtered = engine._parse_dream_response(bad_response, "cycle_1", "default", [], [])
        assert len(filtered) == 0, "Anti-pattern should be filtered"

    @pytest.mark.asyncio
    async def test_portrait_synthesis_parses_correctly(self, mock_llm_portrait_response):
        """PortraitManager correctly parses LLM portrait response."""
        from augmentum.dream.portrait import PortraitManager

        mgr = PortraitManager.__new__(PortraitManager)
        portrait = mgr._parse_portrait_response(
            mock_llm_portrait_response, "default", ["e1", "e2"],
        )

        assert portrait is not None
        assert "half-formed thought" in portrait.voice_notes
        assert "dream system" in portrait.active_threads
        assert "collaborative honesty" in portrait.impressions
        assert portrait.is_current is True

    @pytest.mark.asyncio
    async def test_inject_dream_context_into_messages(self, journal):
        """inject_dream_context correctly modifies the system message."""
        from augmentum.memory.integration import inject_dream_context
        from augmentum.dream.models import DreamPortrait, DreamEntry, DreamEntryType

        portrait = DreamPortrait(
            id="p1", persona_id="default",
            voice_notes="Direct and challenging dynamic.",
            active_threads="Exploring resilience patterns.",
            impressions="Genuine partnership.",
            source_entries=[], is_current=True, created_at="",
        )
        entries = [
            DreamEntry(
                id="e1", persona_id="default",
                content="The way Alex approaches architecture — always asking why before how.",
                entry_type=DreamEntryType.REFLECTION,
                source_memories=[], source_sessions=[],
                context_window={}, embedding=None,
                dream_cycle_id="c1", created_at="",
            ),
        ]

        messages = [{"role": "system", "content": "You are Aria."}]
        await inject_dream_context(messages, portrait, dream_entries=entries)

        system = messages[0]["content"]

        # Portrait injected
        assert "<evolved_self>" in system
        assert "Direct and challenging dynamic." in system
        assert "Exploring resilience patterns." in system
        assert "Genuine partnership." in system

        # Dream recall injected
        assert "<recent_reflections>" in system
        assert "architecture" in system

        # Original content preserved
        assert "You are Aria." in system

        # Correct order: dream context BEFORE original system prompt
        evolved_pos = system.index("<evolved_self>")
        original_pos = system.index("You are Aria.")
        assert evolved_pos < original_pos, "Dream context should precede original system prompt"

    @pytest.mark.asyncio
    async def test_full_store_retrieve_inject_cycle(self, journal):
        """Full cycle: store entries → store portrait → retrieve → inject."""
        from augmentum.dream.portrait import PortraitManager
        from augmentum.dream.models import DreamEntryType, DreamPortrait
        from augmentum.memory.integration import inject_dream_context

        # 1. Store some dream entries
        for i, (content, etype) in enumerate([
            ("The conversation about memory systems felt important.", "reflection"),
            ("We've stopped hedging with each other.", "voice_note"),
            ("Want to explore how emotional texture maps to retrieval.", "active_thread"),
        ]):
            await journal.store_entry(
                persona_id="default", content=content,
                entry_type=DreamEntryType(etype),
                source_memories=[f"mem_{i}"], source_sessions=["sess_1"],
                context_window={}, dream_cycle_id="cycle_1",
            )

        # 2. Store a portrait
        mgr = PortraitManager(journal, settings_store=None)
        portrait = DreamPortrait(
            id="p_full", persona_id="default",
            voice_notes="Alex and I skip the pleasantries. He leads with the problem, I lead with the analysis.",
            active_threads="The dream system architecture. How to make reflection feel genuine.",
            impressions="A collaborative energy that feels productive and honest.",
            source_entries=[], is_current=True, created_at="2026-03-25T12:00:00",
        )
        await mgr._store_portrait(portrait)

        # 3. Retrieve (as the routes would)
        loaded_portrait = await mgr.get_current("default")
        assert loaded_portrait is not None

        entries, total = await journal.list_entries("default", limit=3)
        assert total == 3
        active_entries = [e for e in entries if e.expires_at is None]

        # 4. Inject into messages
        messages = [{"role": "system", "content": "<persona>\nYou are Aria.\n</persona>"}]
        await inject_dream_context(messages, loaded_portrait, dream_entries=active_entries)

        system = messages[0]["content"]

        # Verify everything is present
        assert "<evolved_self>" in system
        assert "skip the pleasantries" in system
        assert "<recent_reflections>" in system
        assert "memory systems" in system or "hedging" in system or "emotional texture" in system
        assert "<persona>" in system  # original preserved

        print("\n=== FINAL SYSTEM MESSAGE ===")
        print(system)
        print(f"\n=== Total length: {len(system)} chars ===")

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_restore(self, journal):
        """Checkpoint flow: save → modify → restore → verify."""
        from augmentum.dream.portrait import PortraitManager
        from augmentum.dream.models import DreamPortrait

        mgr = PortraitManager(journal, settings_store=None)

        # Store initial portrait
        p1 = DreamPortrait(
            id="p_v1", persona_id="default",
            voice_notes="Version 1 voice.",
            active_threads="Version 1 threads.",
            impressions="Version 1 impressions.",
            source_entries=[], is_current=True, created_at="2026-03-25T10:00:00",
        )
        await mgr._store_portrait(p1)

        # Save checkpoint
        cp_id = await mgr.save_checkpoint("default", "v1-checkpoint")
        assert cp_id is not None

        # Store new portrait (simulating a new dream cycle)
        p2 = DreamPortrait(
            id="p_v2", persona_id="default",
            voice_notes="Version 2 voice.",
            active_threads="Version 2 threads.",
            impressions="Version 2 impressions.",
            source_entries=[], is_current=True, created_at="2026-03-25T14:00:00",
        )
        await mgr._store_portrait(p2)

        # Verify current is v2
        current = await mgr.get_current("default")
        assert current.voice_notes == "Version 2 voice."

        # Restore checkpoint
        restored = await mgr.restore_checkpoint("default", cp_id)
        assert restored is not None
        assert restored.voice_notes == "Version 1 voice."

        # Verify current is back to v1
        current = await mgr.get_current("default")
        assert current.voice_notes == "Version 1 voice."

    @pytest.mark.asyncio
    async def test_reset_to_foundation(self, journal):
        """Reset deletes all dream data for a persona."""
        from augmentum.dream.portrait import PortraitManager
        from augmentum.dream.models import DreamEntryType, DreamPortrait

        mgr = PortraitManager(journal, settings_store=None)

        # Store entries and portrait
        await journal.store_entry(
            persona_id="default", content="Something to delete.",
            entry_type=DreamEntryType.REFLECTION,
            source_memories=[], source_sessions=[],
            context_window={}, dream_cycle_id="cycle_1",
        )
        await mgr._store_portrait(DreamPortrait(
            id="p_del", persona_id="default",
            voice_notes="Delete me.", active_threads="", impressions="",
            source_entries=[], is_current=True, created_at="",
        ))

        # Verify data exists
        entries, total = await journal.list_entries("default")
        assert total > 0
        portrait = await mgr.get_current("default")
        assert portrait is not None

        # Reset
        await mgr.reset_to_foundation("default")

        # Verify everything is gone
        entries, total = await journal.list_entries("default")
        assert total == 0
        portrait = await mgr.get_current("default")
        assert portrait is None

    @pytest.mark.asyncio
    async def test_scheduler_trigger_conditions(self):
        """Scheduler eligibility logic with realistic parameters."""
        from augmentum.dream.scheduler import DreamScheduler
        from datetime import datetime, timedelta, timezone

        scheduler = DreamScheduler(
            engine=AsyncMock(),
            settings_store=AsyncMock(),
            enabled=True,
            message_threshold=10,
            idle_minutes=30,
            cooldown_minutes=60,
        )

        # Not eligible: no messages, no approvals
        assert not scheduler._is_eligible()

        # Simulate 15 messages and 2 approvals
        for _ in range(15):
            scheduler.notify_message()
        scheduler.notify_approval("mem_1")
        scheduler.notify_approval("mem_2")

        # Still not eligible: user is active (just sent a message)
        assert not scheduler._is_eligible()

        # Simulate 35 minutes of idle
        scheduler._last_request_at = datetime.now(timezone.utc) - timedelta(minutes=35)

        # Now eligible
        assert scheduler._is_eligible()

        # After manual trigger, counters should reset
        mock_cycle = MagicMock()
        mock_cycle.id = "cycle_1"
        mock_cycle.entries_count = 2
        scheduler._engine.run_cycle = AsyncMock(return_value=mock_cycle)
        await scheduler.trigger_manual()

        assert scheduler._messages_since_dream == 0
        assert scheduler._approved_since_dream == 0
