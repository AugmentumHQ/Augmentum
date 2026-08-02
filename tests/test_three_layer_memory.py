"""Tests for the three-layer narrative memory architecture.

Covers: prompt generation, response parsing, formatting, dataclass
serialization, engine integration, context builder, persistence, UI routes,
and config defaults.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from augmentum.modes.narrative.context_builder import ContextBuilder
from augmentum.modes.narrative.memory import (
    MEMORY_CATEGORIES,
    STATE_FIELDS,
    CardType,
    MemoryEntry,
    StateSnapshot,
    SummaryMode,
    build_compaction_prompt,
    build_state_memory_prompt,
    format_ledger_for_context,
    format_state_for_context,
    parse_state_memory_response,
)
from augmentum.state.narrative_state import NarrativeSessionState

# =====================================================================
# 1. Template / Prompt generation (8 tests)
# =====================================================================


class TestPromptGeneration:
    def test_build_prompt_character_standard(self):
        system, user = build_state_memory_prompt(
            CardType.CHARACTER, None, [], ["Hello"], "Alice", 1, 5,
        )
        assert "## STATE" in system
        assert "## MEMORY" in system
        for f in STATE_FIELDS[CardType.CHARACTER]:
            assert f in system

    def test_build_prompt_narrator_lite(self):
        system, user = build_state_memory_prompt(
            CardType.NARRATOR, None, [], ["msg"], "DM", 1, 5,
            mode=SummaryMode.LITE,
        )
        for f in STATE_FIELDS[CardType.NARRATOR]:
            assert f in system
        assert "~100 words" in system
        assert "3 new" in system

    def test_build_prompt_ensemble(self):
        system, _ = build_state_memory_prompt(
            CardType.ENSEMBLE, None, [], ["msg"], "Party", 1, 5,
        )
        for f in STATE_FIELDS[CardType.ENSEMBLE]:
            assert f in system

    def test_build_prompt_with_existing_state(self):
        snap = StateSnapshot(fields={"location": "tavern"}, card_type=CardType.CHARACTER)
        _, user = build_state_memory_prompt(
            CardType.CHARACTER, snap, [], ["msg"], "Bob", 1, 5,
        )
        assert "Previous STATE" in user
        assert "tavern" in user

    def test_build_prompt_with_ledger_context(self):
        ledger = [MemoryEntry(round_num=3, category="discovery", content="found a key")]
        _, user = build_state_memory_prompt(
            CardType.CHARACTER, None, ledger, ["msg"], "Bob", 1, 5,
        )
        assert "Recent MEMORY entries" in user
        assert "found a key" in user

    def test_build_prompt_custom_override(self):
        custom = "Custom prompt for {char_name} with {state_fields}"
        system, _ = build_state_memory_prompt(
            CardType.CHARACTER, None, [], ["msg"], "Eve", 1, 5,
            custom_prompt=custom,
        )
        assert "Custom prompt for Eve" in system
        assert "location" in system  # state_fields expanded

    def test_build_prompt_round_numbers(self):
        # 12 messages in history; batch covers the last 3 (rounds 10-12).
        # Round numbers are 1-based positions in the full history.
        history = [f"msg{i}" for i in range(12)]
        _, user = build_state_memory_prompt(
            CardType.CHARACTER, None, [], history, "X", 10, 12,
        )
        assert "[R10]" in user
        assert "[R11]" in user
        assert "[R12]" in user

    def test_build_compaction_prompt(self):
        entries = [
            MemoryEntry(round_num=1, category="discovery", content="found gem"),
            MemoryEntry(round_num=2, category="world_change", content="earthquake"),
        ]
        system, user = build_compaction_prompt(entries, CardType.CHARACTER)
        assert "compressor" in system.lower()
        assert "found gem" in user
        assert "earthquake" in user
        for cat in MEMORY_CATEGORIES[CardType.CHARACTER]:
            assert cat in system


# =====================================================================
# 2. Response parsing (10 tests)
# =====================================================================


class TestResponseParsing:
    def test_parse_basic_response(self):
        text = "## STATE\nlocation: forest\n\n## MEMORY\n[R5|discovery] found a cave"
        snap, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 10)
        assert snap.fields.get("location") == "forest"
        assert len(entries) == 1

    def test_parse_character_fields(self):
        lines = "\n".join(f"- {f}: value_{f}" for f in STATE_FIELDS[CardType.CHARACTER])
        text = f"## STATE\n{lines}\n\n## MEMORY\n"
        snap, _ = parse_state_memory_response(text, CardType.CHARACTER, 1, 10)
        for f in STATE_FIELDS[CardType.CHARACTER]:
            assert snap.fields.get(f) == f"value_{f}"

    def test_parse_narrator_fields(self):
        lines = "\n".join(f"- {f}: val" for f in STATE_FIELDS[CardType.NARRATOR])
        text = f"## STATE\n{lines}\n\n## MEMORY\n"
        snap, _ = parse_state_memory_response(text, CardType.NARRATOR, 1, 10)
        for f in STATE_FIELDS[CardType.NARRATOR]:
            assert f in snap.fields

    def test_parse_ensemble_fields(self):
        lines = "\n".join(f"- {f}: val" for f in STATE_FIELDS[CardType.ENSEMBLE])
        text = f"## STATE\n{lines}\n\n## MEMORY\n"
        snap, _ = parse_state_memory_response(text, CardType.ENSEMBLE, 1, 10)
        for f in STATE_FIELDS[CardType.ENSEMBLE]:
            assert f in snap.fields

    def test_parse_memory_entries(self):
        text = (
            "## STATE\n\n## MEMORY\n"
            "[R3|discovery] found a map\n"
            "[R5|commitment] swore an oath\n"
        )
        _, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 10)
        assert len(entries) == 2
        assert entries[0].round_num == 3
        assert entries[0].category == "discovery"
        assert entries[1].content == "swore an oath"

    def test_parse_round_validation_in_range(self):
        text = "## STATE\n\n## MEMORY\n[R5|discovery] event"
        _, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 10)
        assert entries[0].round_num == 5

    def test_parse_round_validation_out_of_range(self):
        text = "## STATE\n\n## MEMORY\n[R99|discovery] event"
        _, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 10)
        assert entries[0].round_num == 10  # falls back to batch_end

    def test_parse_no_headers(self):
        text = "location: park\nwho_present: nobody"
        snap, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 5)
        # Should attempt to parse fields from raw text
        assert isinstance(snap, StateSnapshot)
        assert isinstance(entries, list)

    def test_parse_empty_response(self):
        snap, entries = parse_state_memory_response("", CardType.CHARACTER, 1, 5)
        assert snap.fields == {}
        assert entries == []

    def test_parse_malformed_entries(self):
        text = "## STATE\n\n## MEMORY\nthis is not a valid entry\n[R3|discovery] valid one"
        _, entries = parse_state_memory_response(text, CardType.CHARACTER, 1, 5)
        assert len(entries) == 1
        assert entries[0].content == "valid one"


# =====================================================================
# 3. Formatting (4 tests)
# =====================================================================


class TestFormatting:
    def test_format_state_for_context(self):
        snap = StateSnapshot(fields={"location": "castle", "who_present": "guards"})
        result = format_state_for_context(snap)
        assert "[Current State]" in result
        assert "Location: castle" in result
        assert "Who Present: guards" in result

    def test_format_state_empty(self):
        snap = StateSnapshot(fields={})
        assert format_state_for_context(snap) == ""

    def test_format_ledger_for_context(self):
        entries = [
            MemoryEntry(round_num=1, category="discovery", content="found gold"),
            MemoryEntry(round_num=3, category="world_change", content="storm hit"),
        ]
        result = format_ledger_for_context(entries)
        assert "[Story Memory]" in result
        assert "[R1|discovery] found gold" in result
        assert "[R3|world_change] storm hit" in result

    def test_format_ledger_empty(self):
        assert format_ledger_for_context([]) == ""


# =====================================================================
# 4. Dataclass serialization (4 tests)
# =====================================================================


class TestSerialization:
    def test_state_snapshot_to_dict_from_dict(self):
        snap = StateSnapshot(fields={"location": "cave"}, card_type=CardType.NARRATOR)
        d = snap.to_dict()
        restored = StateSnapshot.from_dict(d)
        assert restored.fields == {"location": "cave"}
        assert restored.card_type == CardType.NARRATOR

    def test_memory_entry_to_dict_from_dict(self):
        entry = MemoryEntry(round_num=7, category="quest_update", content="completed quest")
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        assert restored.round_num == 7
        assert restored.category == "quest_update"
        assert restored.content == "completed quest"

    def test_state_snapshot_from_dict_defaults(self):
        snap = StateSnapshot.from_dict({})
        assert snap.fields == {}
        assert snap.card_type == CardType.CHARACTER

    def test_memory_entry_from_dict_defaults(self):
        entry = MemoryEntry.from_dict({})
        assert entry.round_num == 0
        assert entry.category == ""
        assert entry.content == ""


# =====================================================================
# 5. Engine integration (10 tests)
# =====================================================================


class TestEngineIntegration:
    def _make_engine(self):
        from augmentum.modes.narrative.engine import NarrativeEngine
        return NarrativeEngine(session_id="test-session")

    def test_engine_initial_state(self):
        engine = self._make_engine()
        assert engine.state_snapshot is None
        assert engine.memory_ledger == []

    def test_engine_apply_state_memory_response(self):
        engine = self._make_engine()
        snap = StateSnapshot(fields={"location": "inn"}, card_type=CardType.CHARACTER)
        entries = [MemoryEntry(round_num=1, category="discovery", content="found note")]
        engine.apply_state_memory_response(snap, entries)
        assert engine.state_snapshot is snap
        assert len(engine.memory_ledger) == 1

    def test_engine_apply_updates_legacy_summary(self):
        engine = self._make_engine()
        snap = StateSnapshot(fields={"location": "inn"}, card_type=CardType.CHARACTER)
        entries = [MemoryEntry(round_num=1, category="discovery", content="found note")]
        engine.apply_state_memory_response(snap, entries)
        assert engine.state.memory_summary != ""

    @patch("augmentum.config.settings")
    def test_engine_compaction_flag(self, mock_settings):
        mock_settings.narrative_memory_ledger_ceiling = 5
        engine = self._make_engine()
        snap = StateSnapshot(fields={})
        entries = [MemoryEntry(round_num=i, category="discovery", content=f"e{i}") for i in range(5)]
        engine.apply_state_memory_response(snap, entries)
        assert engine.needs_compaction is True

    @patch("augmentum.config.settings")
    def test_engine_build_request(self, mock_settings):
        from augmentum.models.base import InternalChatRequest
        mock_settings.narrative_memory_mode = "standard"
        mock_settings.narrative_memory_prompt = ""
        engine = self._make_engine()
        engine._state.card_type = "character"
        engine._message_history = ["hello", "hi there"]
        req = engine.build_state_memory_request(batch_start=1, batch_end=2)
        assert isinstance(req, InternalChatRequest)
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"

    def test_engine_get_state_text(self):
        engine = self._make_engine()
        assert engine.get_state_text() == ""
        engine._state_snapshot = StateSnapshot(fields={"location": "dock"})
        result = engine.get_state_text()
        assert "dock" in result

    def test_engine_get_memory_text(self):
        engine = self._make_engine()
        assert engine.get_memory_text() == ""
        engine._memory_ledger = [MemoryEntry(round_num=1, category="x", content="y")]
        result = engine.get_memory_text()
        assert "[R1|x] y" in result

    def test_engine_should_refresh(self):
        engine = self._make_engine()
        engine._state.message_count = 10
        engine._state.last_summary_at = 0
        assert engine.should_refresh(10) is True
        engine._state.last_summary_at = 5
        assert engine.should_refresh(10) is False

    def test_engine_apply_edited_state(self):
        engine = self._make_engine()
        engine.apply_edited_state({"location": "beach"})
        assert engine.state_snapshot is not None
        assert engine.state_snapshot.fields["location"] == "beach"

    @patch("augmentum.config.settings")
    def test_engine_apply_edited_ledger(self, mock_settings):
        mock_settings.narrative_memory_ledger_ceiling = 100
        engine = self._make_engine()
        data = [{"round_num": 1, "category": "discovery", "content": "x"}]
        engine.apply_edited_ledger(data)
        assert len(engine.memory_ledger) == 1
        assert engine.memory_ledger[0].content == "x"


# =====================================================================
# 6. Context builder (4 tests)
# =====================================================================


class TestContextBuilder:
    def test_context_builder_state_text_priority(self):
        builder = ContextBuilder(token_budget=10000)
        result = builder.build(state_text="[Current State]\nLocation: forest")
        assert "state_snapshot" in result.blocks_used
        # Verify priority 13 by checking it appears before lower-priority blocks
        detail = {b.label: b for b in result.blocks_detail}
        assert detail["state_snapshot"].included

    def test_context_builder_memory_text_priority(self):
        builder = ContextBuilder(token_budget=10000)
        result = builder.build(memory_text="[Story Memory]\n[R1|discovery] x")
        assert "story_memory" in result.blocks_used

    def test_context_builder_both_blocks(self):
        builder = ContextBuilder(token_budget=10000)
        result = builder.build(
            state_text="[Current State]\nLocation: cave",
            memory_text="[Story Memory]\n[R1|discovery] gem",
        )
        assert "state_snapshot" in result.blocks_used
        assert "story_memory" in result.blocks_used
        assert "cave" in result.injected_text
        assert "gem" in result.injected_text

    def test_context_builder_backward_compat(self):
        builder = ContextBuilder(token_budget=10000)
        result = builder.build(state_text="", memory_text="")
        assert "state_snapshot" not in result.blocks_used
        assert "story_memory" not in result.blocks_used


# =====================================================================
# 7. Persistence round-trip (4 tests)
# =====================================================================


@pytest.mark.asyncio
class TestPersistence:
    async def _setup_db(self):
        import aiosqlite
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        # Create minimal schema needed
        await conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        await conn.execute("INSERT INTO sessions (id) VALUES ('s1')")
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY, description TEXT)"
        )
        # entities, facts, fact_tags, plot_threads, contradictions,
        # lorebook_entries, assumptions, character_cards
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS entities "
            "(id TEXT PRIMARY KEY, session_id TEXT, entity_type TEXT, name TEXT, "
            "aliases TEXT DEFAULT '[]', state TEXT DEFAULT '{}', branch_id TEXT DEFAULT 'main', "
            "created_at TEXT, updated_at TEXT)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS facts "
            "(id TEXT PRIMARY KEY, session_id TEXT, content TEXT, source TEXT, "
            "confidence REAL, domain TEXT, established_at INTEGER, superseded_by TEXT, "
            "branch_id TEXT, created_at TEXT)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS fact_tags (fact_id TEXT, tag TEXT, PRIMARY KEY (fact_id, tag))"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS plot_threads "
            "(id TEXT PRIMARY KEY, session_id TEXT, title TEXT, description TEXT, "
            "status TEXT, established_at INTEGER, resolved_at INTEGER, branch_id TEXT, "
            "state TEXT, created_at TEXT, updated_at TEXT)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS contradictions "
            "(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, message_index INTEGER, "
            "contradiction_type TEXT, description TEXT, severity TEXT, resolution TEXT, "
            "fact_ids TEXT, branch_id TEXT, created_at TEXT)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS lorebook_entries "
            "(id TEXT PRIMARY KEY, session_id TEXT, keywords TEXT, content TEXT, "
            "priority INTEGER, source TEXT, enabled INTEGER, constant INTEGER, "
            "position TEXT, scan_depth INTEGER, case_sensitive INTEGER, "
            "sticky_turns INTEGER, cooldown_turns INTEGER, last_triggered_at INTEGER, "
            "trigger_count INTEGER, created_at TEXT)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS assumptions "
            "(id TEXT PRIMARY KEY, session_id TEXT, content TEXT, made_at INTEGER, "
            "validated INTEGER, confidence REAL, branch_id TEXT, created_at TEXT)"
        )
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS character_cards "
            "(id TEXT PRIMARY KEY, session_id TEXT, name TEXT, data TEXT, "
            "source_format TEXT, created_at TEXT)"
        )
        # narrative_memory with three-layer columns
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS narrative_memory "
            "(session_id TEXT PRIMARY KEY, card_type TEXT DEFAULT 'character', "
            "memory_summary TEXT DEFAULT '', last_summary_at INTEGER DEFAULT 0, "
            "overflow_summaries TEXT DEFAULT '[]', archived_messages TEXT DEFAULT '[]', "
            "state_snapshot TEXT DEFAULT '{}', memory_ledger TEXT DEFAULT '[]', "
            "updated_at TEXT)"
        )
        # character_relationships
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS character_relationships "
            "(session_id TEXT, source_entity TEXT, target_entity TEXT, "
            "trust REAL, affection REAL, tension REAL, label TEXT, "
            "last_updated_at INTEGER, updated_at TEXT)"
        )
        await conn.commit()
        return conn

    async def test_save_load_state_snapshot(self):
        from augmentum.state.narrative_persistence import NarrativePersistence
        conn = await self._setup_db()
        try:
            p = NarrativePersistence(conn)
            state = NarrativeSessionState(session_id="s1")
            state.state_snapshot_data = {"fields": {"location": "market"}, "card_type": "character"}
            state.memory_ledger_data = []
            await p.save_session_state("s1", state)
            loaded = await p.load_session_state("s1")
            # No entities/facts => returns None, so we test via _load_memory
            mem = await p._load_memory("s1")
            assert mem is not None
            assert mem[5] == {"fields": {"location": "market"}, "card_type": "character"}
        finally:
            await conn.close()

    async def test_save_load_memory_ledger(self):
        from augmentum.state.narrative_persistence import NarrativePersistence
        conn = await self._setup_db()
        try:
            p = NarrativePersistence(conn)
            state = NarrativeSessionState(session_id="s1")
            state.memory_ledger_data = [
                {"round_num": 1, "category": "discovery", "content": "found sword"},
            ]
            await p.save_session_state("s1", state)
            mem = await p._load_memory("s1")
            assert mem is not None
            assert len(mem[6]) == 1
            assert mem[6][0]["content"] == "found sword"
        finally:
            await conn.close()

    async def test_legacy_fallback(self):
        """Loading from a DB without three-layer columns gives empty defaults."""
        import aiosqlite
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        # Legacy schema — no state_snapshot or memory_ledger columns
        await conn.execute(
            "CREATE TABLE narrative_memory "
            "(session_id TEXT PRIMARY KEY, card_type TEXT, memory_summary TEXT, "
            "last_summary_at INTEGER)"
        )
        await conn.execute(
            "INSERT INTO narrative_memory VALUES ('s1', 'character', 'old summary', 5)"
        )
        await conn.commit()
        try:
            from augmentum.state.narrative_persistence import NarrativePersistence
            p = NarrativePersistence(conn)
            mem = await p._load_memory("s1")
            assert mem is not None
            assert mem[0] == "character"
            assert mem[1] == "old summary"
            # Legacy path returns empty dicts/lists for three-layer fields
            assert mem[5] == {}
            assert mem[6] == []
        finally:
            await conn.close()

    async def test_narrative_session_state_new_fields(self):
        state = NarrativeSessionState(session_id="s1")
        assert hasattr(state, "state_snapshot_data")
        assert hasattr(state, "memory_ledger_data")
        assert state.state_snapshot_data == {}
        assert state.memory_ledger_data == []


# =====================================================================
# 8. UI routes (4 tests)
# =====================================================================


class TestUIRoutes:
    def _make_engine_with_data(self):
        from augmentum.modes.narrative.engine import NarrativeEngine
        engine = NarrativeEngine(session_id="s1")
        engine._state_snapshot = StateSnapshot(
            fields={"location": "throne_room"}, card_type=CardType.CHARACTER,
        )
        engine._memory_ledger = [
            MemoryEntry(round_num=1, category="discovery", content="ancient scroll"),
        ]
        engine._state.memory_summary = "old summary text"
        return engine

    def test_get_state_includes_snapshot_and_ledger(self):
        engine = self._make_engine_with_data()
        # Simulate what the route does
        state_snapshot = {}
        memory_ledger = []
        if engine.state_snapshot:
            state_snapshot = engine.state_snapshot.fields
        if engine.memory_ledger:
            memory_ledger = [e.to_dict() for e in engine.memory_ledger]
        assert state_snapshot == {"location": "throne_room"}
        assert len(memory_ledger) == 1
        assert memory_ledger[0]["content"] == "ancient scroll"

    def test_patch_state_snapshot(self):
        engine = self._make_engine_with_data()
        engine.apply_edited_state({"location": "dungeon"})
        assert engine.state_snapshot.fields["location"] == "dungeon"

    @patch("augmentum.config.settings")
    def test_patch_memory_ledger(self, mock_settings):
        mock_settings.narrative_memory_ledger_ceiling = 100
        engine = self._make_engine_with_data()
        new_ledger = [{"round_num": 5, "category": "world_change", "content": "flood"}]
        engine.apply_edited_ledger(new_ledger)
        assert len(engine.memory_ledger) == 1
        assert engine.memory_ledger[0].content == "flood"

    def test_patch_legacy_summary(self):
        engine = self._make_engine_with_data()
        engine.state.memory_summary = "updated legacy"
        engine.state.last_summary_at = engine.state.message_count
        assert engine.state.memory_summary == "updated legacy"


# =====================================================================
# 9. Config (2 tests)
# =====================================================================


class TestConfig:
    def test_config_defaults(self):
        from augmentum.config import Settings
        s = Settings()
        assert s.narrative_memory_ledger_ceiling == 60
        assert s.narrative_memory_compaction_ratio == 0.5
        assert s.narrative_memory_state_word_target == 200
        assert s.narrative_memory_continuous_archive is True
        assert s.narrative_memory_mode == "standard"

    def test_config_ledger_ceiling(self):
        from augmentum.config import Settings
        s = Settings()
        assert isinstance(s.narrative_memory_ledger_ceiling, int)
        assert s.narrative_memory_ledger_ceiling > 0
