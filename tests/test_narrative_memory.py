"""Tests for narrative long-term memory — card type detection, summary prompts,
engine integration, context builder memory injection, handler refresh, and persistence.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import aiosqlite
import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    Message,
)
from augmentum.modes.narrative.card_parser import CharacterCard
from augmentum.modes.narrative.context_builder import ContextBuilder
from augmentum.modes.narrative.engine import NarrativeEngine
from augmentum.modes.narrative.handler import NarrativeHandler
from augmentum.modes.narrative.memory import (
    CardType,
    build_summary_prompt,
    detect_card_type,
)
from augmentum.state.narrative_persistence import NarrativePersistence
from augmentum.state.narrative_state import (
    Entity,
    EntityType,
    Fact,
    NarrativeSessionState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PERSISTENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'passthrough',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    message_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'extracted',
    confidence REAL NOT NULL DEFAULT 0.8,
    domain TEXT NOT NULL DEFAULT 'general',
    established_at INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT,
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS fact_tags (
    fact_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (fact_id, tag)
);
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases TEXT DEFAULT '[]',
    state TEXT DEFAULT '{}',
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS entity_state_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    delta TEXT NOT NULL DEFAULT '{}',
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS plot_threads (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    established_at INTEGER NOT NULL DEFAULT 0,
    resolved_at INTEGER,
    branch_id TEXT NOT NULL DEFAULT 'main',
    state TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    contradiction_type TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'minor',
    resolution TEXT,
    fact_ids TEXT DEFAULT '[]',
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS lorebook_entries (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '[]',
    content TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    source TEXT NOT NULL DEFAULT 'character_book',
    enabled INTEGER NOT NULL DEFAULT 1,
    constant INTEGER NOT NULL DEFAULT 0,
    position TEXT NOT NULL DEFAULT 'before_char',
    scan_depth INTEGER NOT NULL DEFAULT 5,
    case_sensitive INTEGER NOT NULL DEFAULT 0,
    sticky_turns INTEGER NOT NULL DEFAULT 0,
    cooldown_turns INTEGER NOT NULL DEFAULT 0,
    last_triggered_at INTEGER,
    trigger_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS assumptions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    made_at INTEGER NOT NULL DEFAULT 0,
    validated INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS character_cards (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    name TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    source_format TEXT NOT NULL DEFAULT 'unknown',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS narrative_memory (
    session_id TEXT PRIMARY KEY,
    card_type TEXT NOT NULL DEFAULT 'character',
    memory_summary TEXT NOT NULL DEFAULT '',
    last_summary_at INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SESSION_ID = "test-memory-session"


def _make_card(**kwargs) -> CharacterCard:
    """Create a CharacterCard with sensible defaults."""
    defaults = {
        "name": "TestChar",
        "personality": "",
        "description": "",
        "scenario": "",
        "system_prompt": "",
    }
    defaults.update(kwargs)
    return CharacterCard(**defaults)


def _make_request(user: str, system: str = "") -> InternalChatRequest:
    msgs = []
    if system:
        msgs.append(Message(role="system", content=system))
    msgs.append(Message(role="user", content=user))
    return InternalChatRequest(model="test", messages=msgs)


# ---------------------------------------------------------------------------
# CardType Detection
# ---------------------------------------------------------------------------


class TestCardTypeDetection:
    """Test heuristic card type classification."""

    def test_default_is_character(self):
        card = _make_card(name="Luna", description="A friendly witch.")
        assert detect_card_type(card) == CardType.CHARACTER

    def test_narrator_from_name(self):
        card = _make_card(name="The Narrator", description="Tells the story.")
        assert detect_card_type(card) == CardType.NARRATOR

    def test_narrator_from_dungeon_master(self):
        card = _make_card(
            name="Game Master",
            description="Runs the campaign.",
        )
        assert detect_card_type(card) == CardType.NARRATOR

    def test_narrator_from_description_keywords(self):
        card = _make_card(
            name="Cyraeth",
            description="The world of Cyraeth is vast and dangerous.",
            scenario="You are a narrator guiding the player through the realm.",
        )
        assert detect_card_type(card) == CardType.NARRATOR

    def test_ensemble_from_group_pattern(self):
        card = _make_card(
            name="The Party",
            description="Characters: Aldric the warrior, Luna the witch, and Finn the rogue.",
            system_prompt="Play the group of companions on their adventure.",
        )
        assert detect_card_type(card) == CardType.ENSEMBLE

    def test_empty_card_defaults_to_character(self):
        card = _make_card(name="", description="", personality="")
        assert detect_card_type(card) == CardType.CHARACTER

    def test_case_insensitive_narrator_detection(self):
        card = _make_card(
            name="STORYTELLER",
            description="Narrates the world.",
        )
        assert detect_card_type(card) == CardType.NARRATOR


# ---------------------------------------------------------------------------
# Summary Prompt Building
# ---------------------------------------------------------------------------


class TestBuildSummaryPrompt:
    """Test summary prompt construction per card type."""

    def test_character_prompt_includes_char_name(self):
        system, user = build_summary_prompt(
            card_type=CardType.CHARACTER,
            previous_summary="",
            recent_messages=["Hello!", "Hi there!"],
            char_name="Luna",
        )
        assert "Luna" in system
        assert "relationship" in system.lower()

    def test_narrator_prompt_focuses_on_world(self):
        system, user = build_summary_prompt(
            card_type=CardType.NARRATOR,
            previous_summary="",
            recent_messages=["The kingdom falls."],
            char_name="Cyraeth",
        )
        assert "world" in system.lower() or "quest" in system.lower()

    def test_ensemble_prompt_focuses_on_dynamics(self):
        system, user = build_summary_prompt(
            card_type=CardType.ENSEMBLE,
            previous_summary="",
            recent_messages=["Aldric attacks!"],
            char_name="The Party",
        )
        assert "dynamics" in system.lower() or "group" in system.lower()

    def test_first_summary_uses_creation_phrasing(self):
        system, _user = build_summary_prompt(
            card_type=CardType.CHARACTER,
            previous_summary="",
            recent_messages=["Hello"],
            char_name="Luna",
        )
        assert "initial summary" in system.lower() or "create" in system.lower()

    def test_update_includes_previous_summary(self):
        system, _user = build_summary_prompt(
            card_type=CardType.CHARACTER,
            previous_summary="Luna and the user have been exploring the forest.",
            recent_messages=["Let's go deeper."],
            char_name="Luna",
        )
        assert "Luna and the user have been exploring the forest" in system

    def test_truncates_to_max_20_messages(self):
        messages = [f"Message {i}" for i in range(30)]
        _system, user = build_summary_prompt(
            card_type=CardType.CHARACTER,
            previous_summary="",
            recent_messages=messages,
            char_name="Luna",
        )
        # Should only include last 20 messages
        assert "Message 29" in user
        assert "Message 10" in user
        assert "Message 9" not in user


# ---------------------------------------------------------------------------
# Engine Integration
# ---------------------------------------------------------------------------


class TestEngineMemoryIntegration:
    """Test NarrativeEngine memory helper methods."""

    def test_should_refresh_at_interval(self):
        engine = NarrativeEngine(session_id="test")
        engine._state.message_count = 10
        engine._state.last_summary_at = 0
        assert engine.should_refresh_summary(10) is True

    def test_should_not_refresh_before_interval(self):
        engine = NarrativeEngine(session_id="test")
        engine._state.message_count = 5
        engine._state.last_summary_at = 0
        assert engine.should_refresh_summary(10) is False

    def test_should_not_refresh_at_zero(self):
        engine = NarrativeEngine(session_id="test")
        engine._state.message_count = 0
        assert engine.should_refresh_summary(10) is False

    def test_should_refresh_after_second_interval(self):
        engine = NarrativeEngine(session_id="test")
        engine._state.message_count = 20
        engine._state.last_summary_at = 10
        assert engine.should_refresh_summary(10) is True

    def test_build_summary_request_returns_valid_request(self):
        engine = NarrativeEngine(session_id="test")
        engine._state.character_card_name = "Luna"
        engine._state.card_type = "character"
        engine._message_history = ["Hello", "Hi there", "How are you?"]

        request = engine.build_summary_request()
        assert isinstance(request, InternalChatRequest)
        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"
        assert request.stream is False
        assert "Luna" in request.messages[0].content

    def test_update_summary_sets_state(self):
        engine = NarrativeEngine(session_id="test")
        engine._state.message_count = 15

        engine.update_summary("Luna and the user explored the cave.")

        assert engine.state.memory_summary == "Luna and the user explored the cave."
        assert engine.state.last_summary_at == 15


# ---------------------------------------------------------------------------
# Context Builder Memory Block
# ---------------------------------------------------------------------------


class TestContextBuilderMemory:
    """Test memory block injection and fact suppression."""

    def test_memory_block_injected(self):
        builder = ContextBuilder(token_budget=2048)
        result = builder.build(memory_summary="Luna trusts the user deeply.")
        assert "narrative_memory" in result.blocks_used
        assert "Luna trusts the user deeply." in result.injected_text

    def test_facts_suppressed_when_memory_present(self):
        builder = ContextBuilder(token_budget=2048)
        facts = [
            Fact(content="Luna is a witch.", established_at=0),
            Fact(content="The forest is dark.", established_at=1),
        ]
        result = builder.build(
            recent_facts=facts,
            memory_summary="Luna the witch explored the dark forest.",
        )
        assert "narrative_memory" in result.blocks_used
        assert "established_facts" not in result.blocks_used

    def test_facts_included_when_no_memory(self):
        builder = ContextBuilder(token_budget=2048)
        facts = [
            Fact(content="Luna is a witch.", established_at=0),
        ]
        result = builder.build(recent_facts=facts)
        assert "established_facts" in result.blocks_used
        assert "narrative_memory" not in result.blocks_used

    def test_no_memory_block_when_empty(self):
        builder = ContextBuilder(token_budget=2048)
        result = builder.build(memory_summary="")
        assert "narrative_memory" not in result.blocks_used

    def test_memory_priority_between_card_and_consistency(self):
        """Memory block priority=15 should appear after card=10 but before consistency=20."""
        builder = ContextBuilder(token_budget=4096)

        from augmentum.state.narrative_state import Contradiction, ContradictionSeverity

        contradictions = [
            Contradiction(
                session_id="t",
                message_index=1,
                contradiction_type="test",
                description="A contradiction",
                severity=ContradictionSeverity.MINOR,
            ),
        ]

        result = builder.build(
            character_card_summary="Name: Luna\nSpecies: Witch",
            memory_summary="Luna explored the forest.",
            contradictions=contradictions,
        )

        # All three blocks should be present
        assert "character_card" in result.blocks_used
        assert "narrative_memory" in result.blocks_used
        assert "consistency_warnings" in result.blocks_used

        # Check order: card < memory < consistency
        card_idx = result.blocks_used.index("character_card")
        mem_idx = result.blocks_used.index("narrative_memory")
        warn_idx = result.blocks_used.index("consistency_warnings")
        assert card_idx < mem_idx < warn_idx


# ---------------------------------------------------------------------------
# Handler Memory Refresh
# ---------------------------------------------------------------------------


class TestHandlerMemoryRefresh:
    """Test handler triggers memory refresh at the right time."""

    @pytest.mark.asyncio
    async def test_refresh_fires_when_interval_reached(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(role="assistant", content="Summary here."),
            model="test-model",
        ))

        engine = NarrativeEngine(session_id="test")
        engine._state.message_count = 12
        engine._state.last_summary_at = 0
        engine._state.character_card_name = "Luna"
        engine._state.card_type = "character"
        engine._message_history = ["Hello"] * 12
        engine._initialized = True

        from augmentum.modes.narrative.memory_settings import SessionMemorySettings
        engine.state.memory_settings = SessionMemorySettings(memory_enabled=True, memory_interval=10)
        handler = NarrativeHandler(
            backend=backend,
            engine=engine,
        )

        task = handler._maybe_refresh_summary()
        assert task is not None
        # Wait for the background task to complete
        await task

        # Summary should have been updated
        assert engine.state.memory_summary == "Summary here."
        assert engine.state.last_summary_at == 12

    @pytest.mark.asyncio
    async def test_refresh_skipped_when_disabled(self):
        engine = NarrativeEngine(session_id="test")
        engine._state.message_count = 100  # Well past interval

        from augmentum.modes.narrative.memory_settings import SessionMemorySettings
        engine.state.memory_settings = SessionMemorySettings(memory_enabled=False, memory_interval=10)
        handler = NarrativeHandler(
            backend=AsyncMock(),
            engine=engine,
        )

        task = handler._maybe_refresh_summary()
        assert task is None


# ---------------------------------------------------------------------------
# Memory Persistence
# ---------------------------------------------------------------------------


class TestMemoryPersistence:
    """Test save/load roundtrip for narrative memory."""

    @pytest.fixture
    async def db(self):
        conn = await aiosqlite.connect(":memory:")
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys=OFF")
        await conn.executescript(_PERSISTENCE_SCHEMA)
        await conn.execute(
            "INSERT INTO sessions (id, mode) VALUES (?, ?)",
            (SESSION_ID, "narrative"),
        )
        await conn.commit()
        yield conn
        await conn.close()

    @pytest.fixture
    async def persistence(self, db):
        return NarrativePersistence(db)

    @pytest.mark.asyncio
    async def test_memory_roundtrip(self, persistence):
        state = NarrativeSessionState(
            session_id=SESSION_ID,
            entities={
                "e1": Entity(
                    id="e1",
                    session_id=SESSION_ID,
                    entity_type=EntityType.CHARACTER,
                    name="Luna",
                ),
            },
            memory_summary="Luna and the user have built a strong bond.",
            card_type="character",
            last_summary_at=10,
        )

        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        assert loaded.memory_summary == "Luna and the user have built a strong bond."
        assert loaded.card_type == "character"
        assert loaded.last_summary_at == 10

    @pytest.mark.asyncio
    async def test_memory_defaults_when_empty(self, persistence):
        state = NarrativeSessionState(
            session_id=SESSION_ID,
            entities={
                "e1": Entity(
                    id="e1",
                    session_id=SESSION_ID,
                    entity_type=EntityType.CHARACTER,
                    name="Test",
                ),
            },
        )

        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        assert loaded.memory_summary == ""
        assert loaded.card_type == "character"
        assert loaded.last_summary_at == 0

    @pytest.mark.asyncio
    async def test_memory_update_persists(self, persistence):
        """Saving state, updating memory, re-saving — memory should update."""
        state = NarrativeSessionState(
            session_id=SESSION_ID,
            entities={
                "e1": Entity(
                    id="e1",
                    session_id=SESSION_ID,
                    entity_type=EntityType.CHARACTER,
                    name="Luna",
                ),
            },
            memory_summary="First summary.",
            card_type="narrator",
            last_summary_at=5,
        )

        await persistence.save_session_state(SESSION_ID, state)

        # Update memory
        state.memory_summary = "Updated summary after more events."
        state.last_summary_at = 15
        await persistence.save_session_state(SESSION_ID, state)

        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        assert loaded.memory_summary == "Updated summary after more events."
        assert loaded.card_type == "narrator"
        assert loaded.last_summary_at == 15
