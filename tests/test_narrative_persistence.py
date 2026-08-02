"""Tests for narrative state persistence — save/load roundtrips via SQLite."""

from __future__ import annotations

import aiosqlite
import pytest

from augmentum.state.narrative_persistence import NarrativePersistence

try:
    from augmentum.state.narrative_state import (
        Assumption,
        Contradiction,
        ContradictionSeverity,
        Entity,
        EntityState,
        EntityType,
        Fact,
        LorebookEntry,
        LorebookPosition,
        NarrativeSessionState,
        PlotStatus,
        PlotThread,
    )
except ImportError as _import_exc:
    import pytest as _pytest_skip  # noqa: E402
    _pytest_skip.skip(f"augmentum.state.narrative_state not importable in this build: {_import_exc}", allow_module_level=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# SQL schemas needed for tests (combined from 001 + 002 migrations)
_SCHEMA_SQL = """
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

CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    parent_id INTEGER REFERENCES session_messages(id),
    role TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    branch_id TEXT NOT NULL DEFAULT 'main',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'extracted',
    confidence REAL NOT NULL DEFAULT 0.8,
    domain TEXT NOT NULL DEFAULT 'general',
    established_at INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT REFERENCES facts(id),
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fact_tags (
    fact_id TEXT NOT NULL REFERENCES facts(id),
    tag TEXT NOT NULL,
    PRIMARY KEY (fact_id, tag)
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
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
    entity_id TEXT NOT NULL REFERENCES entities(id),
    message_index INTEGER NOT NULL,
    delta TEXT NOT NULL DEFAULT '{}',
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS plot_threads (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
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
    session_id TEXT NOT NULL REFERENCES sessions(id),
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
    session_id TEXT NOT NULL REFERENCES sessions(id),
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
    session_id TEXT NOT NULL REFERENCES sessions(id),
    content TEXT NOT NULL,
    made_at INTEGER NOT NULL DEFAULT 0,
    validated INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS character_cards (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
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

SESSION_ID = "test-session-1"


@pytest.fixture
async def db():
    """Create an in-memory SQLite database with the full schema."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA foreign_keys=OFF")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute(
        "INSERT INTO sessions (id, mode) VALUES (?, ?)",
        (SESSION_ID, "narrative"),
    )
    await conn.commit()
    yield conn
    await conn.close()


@pytest.fixture
async def persistence(db):
    """Create a NarrativePersistence wired to the in-memory DB."""
    return NarrativePersistence(db)


def _make_full_state() -> NarrativeSessionState:
    """Build a richly populated NarrativeSessionState for roundtrip tests."""
    entities = {
        "ent1": Entity(
            id="ent1",
            session_id=SESSION_ID,
            entity_type=EntityType.CHARACTER,
            name="Lyra",
            aliases=["The Sorceress", "Moonwhisper"],
            state=EntityState(
                location="tower",
                emotional_state="calm",
                physical_state="standing",
                inventory=["staff", "spellbook", "crystal"],
                relationships={"Aldric": "friend", "Dragon": "enemy"},
                custom={"mana": 100, "title": "Archmage"},
            ),
            branch_id="main",
        ),
        "ent2": Entity(
            id="ent2",
            session_id=SESSION_ID,
            entity_type=EntityType.LOCATION,
            name="Whispering Forest",
            aliases=["The Forest"],
            state=EntityState(
                custom={"danger_level": "high", "inhabitants": ["wolves", "sprites"]},
            ),
            branch_id="main",
        ),
        "ent3": Entity(
            id="ent3",
            session_id=SESSION_ID,
            entity_type=EntityType.ITEM,
            name="Crystal Orb",
            state=EntityState(
                location="tower",
                custom={"power": "divination"},
            ),
            branch_id="main",
        ),
    }

    facts = [
        Fact(
            id="fact1",
            session_id=SESSION_ID,
            content="The king is alive and rules from the capital.",
            source="extracted",
            confidence=0.95,
            domain="politics",
            established_at=2,
            branch_id="main",
            tags=["royalty", "politics", "alive"],
        ),
        Fact(
            id="fact2",
            session_id=SESSION_ID,
            content="Lyra has been studying magic for 200 years.",
            source="character_card",
            confidence=1.0,
            domain="characters",
            established_at=0,
            branch_id="main",
            tags=["lyra", "magic", "backstory"],
        ),
        Fact(
            id="fact3",
            session_id=SESSION_ID,
            content="The forest is haunted at night.",
            source="extracted",
            confidence=0.7,
            domain="world",
            established_at=5,
            superseded_by=None,
            branch_id="main",
            tags=[],
        ),
    ]

    plot_threads = [
        PlotThread(
            id="plot1",
            session_id=SESSION_ID,
            title="Find the lost artifact",
            description="An ancient sword hidden in the temple.",
            status=PlotStatus.ACTIVE,
            established_at=1,
            branch_id="main",
            state={"progressions": [{"message_index": 3, "note": "Found a clue"}]},
        ),
        PlotThread(
            id="plot2",
            session_id=SESSION_ID,
            title="Defeat the dragon",
            description="A dragon terrorizes the valley.",
            status=PlotStatus.RESOLVED,
            established_at=0,
            resolved_at=8,
            branch_id="main",
            state={},
        ),
    ]

    contradictions = [
        Contradiction(
            session_id=SESSION_ID,
            message_index=5,
            contradiction_type="time_paradox",
            description="Time went backwards: afternoon to morning",
            severity=ContradictionSeverity.MINOR,
            resolution=None,
            fact_ids=["fact1"],
            branch_id="main",
        ),
        Contradiction(
            session_id=SESSION_ID,
            message_index=7,
            contradiction_type="fact_conflict",
            description="Conflicting facts about the king",
            severity=ContradictionSeverity.MAJOR,
            resolution="Resolved by retcon",
            fact_ids=["fact1", "fact2"],
            branch_id="main",
        ),
    ]

    lorebook = [
        LorebookEntry(
            id="lore1",
            session_id=SESSION_ID,
            keywords=["dragon", "wyrm"],
            content="Dragons are ancient magical creatures.",
            priority=10,
            source="character_book",
            enabled=True,
            constant=False,
            position=LorebookPosition.BEFORE_CHAR,
            scan_depth=5,
            case_sensitive=False,
            sticky_turns=2,
            cooldown_turns=1,
            last_triggered_at=3,
            trigger_count=2,
        ),
        LorebookEntry(
            id="lore2",
            session_id=SESSION_ID,
            keywords=["Crimson Wave", "ship"],
            content="The Crimson Wave is a three-masted brigantine.",
            priority=20,
            source="character_book",
            enabled=True,
            constant=True,
            position=LorebookPosition.AFTER_CHAR,
            scan_depth=10,
            case_sensitive=True,
            sticky_turns=0,
            cooldown_turns=0,
            last_triggered_at=None,
            trigger_count=0,
        ),
    ]

    assumptions = [
        Assumption(
            id="asm1",
            session_id=SESSION_ID,
            content="The traveler is friendly.",
            made_at=1,
            validated=False,
            confidence=0.6,
            branch_id="main",
        ),
    ]

    return NarrativeSessionState(
        session_id=SESSION_ID,
        branch_id="main",
        message_count=10,
        character_card_name="Lyra",
        entities=entities,
        facts=facts,
        plot_threads=plot_threads,
        contradictions=contradictions,
        lorebook=lorebook,
        assumptions=assumptions,
        scene_context={"location": "tower", "weather": "stormy"},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullRoundtrip:
    """Save a full state and load it back — every field must survive."""

    @pytest.mark.asyncio
    async def test_full_roundtrip(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        assert loaded.session_id == SESSION_ID
        assert loaded.branch_id == "main"
        # message_count is reconstructed from max timestamps
        assert loaded.message_count > 0

        # Entities
        assert len(loaded.entities) == 3
        assert "ent1" in loaded.entities
        assert "ent2" in loaded.entities
        assert "ent3" in loaded.entities

        # Facts
        assert len(loaded.facts) == 3

        # Plot threads
        assert len(loaded.plot_threads) == 2

        # Contradictions
        assert len(loaded.contradictions) == 2

        # Lorebook
        assert len(loaded.lorebook) == 2

        # Assumptions
        assert len(loaded.assumptions) == 1


class TestEntityRoundtrip:
    """Entities with complex EntityState must survive roundtrip."""

    @pytest.mark.asyncio
    async def test_entity_with_inventory_and_relationships(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        lyra = loaded.entities["ent1"]
        assert lyra.name == "Lyra"
        assert lyra.entity_type == EntityType.CHARACTER
        assert lyra.aliases == ["The Sorceress", "Moonwhisper"]
        assert lyra.branch_id == "main"

        # EntityState fields
        assert lyra.state.location == "tower"
        assert lyra.state.emotional_state == "calm"
        assert lyra.state.physical_state == "standing"
        assert lyra.state.inventory == ["staff", "spellbook", "crystal"]
        assert lyra.state.relationships == {"Aldric": "friend", "Dragon": "enemy"}
        assert lyra.state.custom == {"mana": 100, "title": "Archmage"}

    @pytest.mark.asyncio
    async def test_location_entity(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        forest = loaded.entities["ent2"]
        assert forest.name == "Whispering Forest"
        assert forest.entity_type == EntityType.LOCATION
        assert forest.state.custom["danger_level"] == "high"
        assert forest.state.custom["inhabitants"] == ["wolves", "sprites"]

    @pytest.mark.asyncio
    async def test_item_entity(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        orb = loaded.entities["ent3"]
        assert orb.name == "Crystal Orb"
        assert orb.entity_type == EntityType.ITEM
        assert orb.state.custom["power"] == "divination"


class TestFactRoundtrip:
    """Facts with tags must survive roundtrip."""

    @pytest.mark.asyncio
    async def test_facts_with_tags(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        # Find fact1 by id
        fact1 = next(f for f in loaded.facts if f.id == "fact1")
        assert fact1.content == "The king is alive and rules from the capital."
        assert fact1.source == "extracted"
        assert fact1.confidence == pytest.approx(0.95)
        assert fact1.domain == "politics"
        assert fact1.established_at == 2
        assert set(fact1.tags) == {"royalty", "politics", "alive"}

    @pytest.mark.asyncio
    async def test_fact_without_tags(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        fact3 = next(f for f in loaded.facts if f.id == "fact3")
        assert fact3.tags == []

    @pytest.mark.asyncio
    async def test_fact_with_superseded_by(self, persistence):
        """A fact with superseded_by=None should roundtrip correctly."""
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        fact2 = next(f for f in loaded.facts if f.id == "fact2")
        assert fact2.superseded_by is None
        assert fact2.source == "character_card"
        assert fact2.confidence == pytest.approx(1.0)


class TestPlotThreadRoundtrip:
    """Plot threads must survive roundtrip."""

    @pytest.mark.asyncio
    async def test_active_plot_thread(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        plot1 = next(p for p in loaded.plot_threads if p.id == "plot1")
        assert plot1.title == "Find the lost artifact"
        assert plot1.description == "An ancient sword hidden in the temple."
        assert plot1.status == PlotStatus.ACTIVE
        assert plot1.established_at == 1
        assert plot1.resolved_at is None
        assert plot1.state["progressions"][0]["message_index"] == 3

    @pytest.mark.asyncio
    async def test_resolved_plot_thread(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        plot2 = next(p for p in loaded.plot_threads if p.id == "plot2")
        assert plot2.status == PlotStatus.RESOLVED
        assert plot2.resolved_at == 8


class TestLorebookRoundtrip:
    """Lorebook entries must survive roundtrip with all fields."""

    @pytest.mark.asyncio
    async def test_lorebook_full_fields(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        lore1 = next(e for e in loaded.lorebook if e.id == "lore1")
        assert lore1.keywords == ["dragon", "wyrm"]
        assert lore1.content == "Dragons are ancient magical creatures."
        assert lore1.priority == 10
        assert lore1.source == "character_book"
        assert lore1.enabled is True
        assert lore1.constant is False
        assert lore1.position == LorebookPosition.BEFORE_CHAR
        assert lore1.scan_depth == 5
        assert lore1.case_sensitive is False
        assert lore1.sticky_turns == 2
        assert lore1.cooldown_turns == 1
        assert lore1.last_triggered_at == 3
        assert lore1.trigger_count == 2

    @pytest.mark.asyncio
    async def test_lorebook_constant_entry(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        lore2 = next(e for e in loaded.lorebook if e.id == "lore2")
        assert lore2.constant is True
        assert lore2.position == LorebookPosition.AFTER_CHAR
        assert lore2.case_sensitive is True
        assert lore2.last_triggered_at is None
        assert lore2.trigger_count == 0


class TestContradictionRoundtrip:
    """Contradictions must survive roundtrip."""

    @pytest.mark.asyncio
    async def test_contradictions(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        assert len(loaded.contradictions) == 2

        minor = next(
            c for c in loaded.contradictions
            if c.severity == ContradictionSeverity.MINOR
        )
        assert minor.contradiction_type == "time_paradox"
        assert minor.description == "Time went backwards: afternoon to morning"
        assert minor.resolution is None
        assert minor.fact_ids == ["fact1"]

        major = next(
            c for c in loaded.contradictions
            if c.severity == ContradictionSeverity.MAJOR
        )
        assert major.resolution == "Resolved by retcon"
        assert set(major.fact_ids) == {"fact1", "fact2"}


class TestIncrementalSave:
    """Incremental save should only write new/changed records."""

    @pytest.mark.asyncio
    async def test_incremental_adds_new_facts(self, persistence, db):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)

        # Add a new fact at message_index=10
        state.facts.append(
            Fact(
                id="fact_new",
                session_id=SESSION_ID,
                content="A new discovery!",
                source="extracted",
                confidence=0.9,
                domain="world",
                established_at=10,
                branch_id="main",
                tags=["discovery"],
            )
        )

        await persistence.save_incremental(SESSION_ID, state, message_index=10)

        # Verify the new fact was saved
        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        assert len(loaded.facts) == 4
        new_fact = next(f for f in loaded.facts if f.id == "fact_new")
        assert new_fact.content == "A new discovery!"
        assert new_fact.tags == ["discovery"]

    @pytest.mark.asyncio
    async def test_incremental_updates_entities(self, persistence, db):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)

        # Modify an entity's state
        state.entities["ent1"].state.emotional_state = "excited"
        state.entities["ent1"].state.inventory.append("healing potion")

        await persistence.save_incremental(SESSION_ID, state, message_index=10)

        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        lyra = loaded.entities["ent1"]
        assert lyra.state.emotional_state == "excited"
        assert "healing potion" in lyra.state.inventory

    @pytest.mark.asyncio
    async def test_incremental_skips_old_facts(self, persistence, db):
        """Facts before message_index should not be re-written."""
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)

        # Mutate an old fact in memory (established_at=2) — this should NOT
        # be re-saved during incremental with message_index=10
        old_fact = next(f for f in state.facts if f.id == "fact1")
        old_fact.content = "MUTATED — should not persist"

        # Add a genuinely new fact
        state.facts.append(
            Fact(
                id="fact_new2",
                session_id=SESSION_ID,
                content="Brand new fact",
                established_at=11,
                branch_id="main",
            )
        )

        await persistence.save_incremental(SESSION_ID, state, message_index=10)

        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        # Old fact should still have original content (from full save)
        old_loaded = next(f for f in loaded.facts if f.id == "fact1")
        assert old_loaded.content == "The king is alive and rules from the capital."
        # New fact should exist
        assert any(f.id == "fact_new2" for f in loaded.facts)


class TestLoadNonExistent:
    """Loading a non-existent session should return None."""

    @pytest.mark.asyncio
    async def test_returns_none(self, persistence):
        result = await persistence.load_session_state("nonexistent-session")
        assert result is None


class TestCharacterCard:
    """Character card persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load_character_card_name(self, persistence):
        # Save a character card
        await persistence.save_character_card(
            session_id=SESSION_ID,
            card_id="card1",
            name="Captain Aria",
            data={"personality": "Bold"},
            source_format="v2_json",
        )

        # Save state so load picks up the character card name
        state = NarrativeSessionState(
            session_id=SESSION_ID,
            entities={
                "e1": Entity(
                    id="e1",
                    session_id=SESSION_ID,
                    entity_type=EntityType.CHARACTER,
                    name="Captain Aria",
                ),
            },
        )
        await persistence.save_session_state(SESSION_ID, state)

        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        assert loaded.character_card_name == "Captain Aria"


class TestAssumptionRoundtrip:
    """Assumptions must survive roundtrip."""

    @pytest.mark.asyncio
    async def test_assumption_fields(self, persistence):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)

        assert loaded is not None
        assert len(loaded.assumptions) == 1
        asm = loaded.assumptions[0]
        assert asm.id == "asm1"
        assert asm.content == "The traveler is friendly."
        assert asm.made_at == 1
        assert asm.validated is False
        assert asm.confidence == pytest.approx(0.6)
        assert asm.branch_id == "main"


class TestUpsertBehavior:
    """Saving the same state twice should not create duplicates."""

    @pytest.mark.asyncio
    async def test_double_save_no_duplicates(self, persistence, db):
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)
        await persistence.save_session_state(SESSION_ID, state)

        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        assert len(loaded.entities) == 3
        assert len(loaded.facts) == 3
        assert len(loaded.plot_threads) == 2
        assert len(loaded.lorebook) == 2

    @pytest.mark.asyncio
    async def test_update_existing_entity(self, persistence, db):
        """Saving modified state should update existing records."""
        state = _make_full_state()
        await persistence.save_session_state(SESSION_ID, state)

        # Modify entity in-place
        state.entities["ent1"].state.location = "library"
        state.entities["ent1"].name = "Lyra Moonwhisper"
        await persistence.save_session_state(SESSION_ID, state)

        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        lyra = loaded.entities["ent1"]
        assert lyra.name == "Lyra Moonwhisper"
        assert lyra.state.location == "library"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_state_returns_none(self, persistence):
        """Saving a state with no data and loading returns None."""
        empty_state = NarrativeSessionState(session_id=SESSION_ID)
        await persistence.save_session_state(SESSION_ID, empty_state)
        loaded = await persistence.load_session_state(SESSION_ID)
        # No entities, facts, etc. were saved, so should return None
        assert loaded is None

    @pytest.mark.asyncio
    async def test_entity_with_empty_state(self, persistence):
        """An entity with default EntityState should roundtrip."""
        state = NarrativeSessionState(
            session_id=SESSION_ID,
            entities={
                "e1": Entity(
                    id="e1",
                    session_id=SESSION_ID,
                    entity_type=EntityType.CHARACTER,
                    name="Nobody",
                    state=EntityState(),
                ),
            },
        )
        await persistence.save_session_state(SESSION_ID, state)
        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        assert loaded.entities["e1"].state.location == ""
        assert loaded.entities["e1"].state.inventory == []
        assert loaded.entities["e1"].state.relationships == {}

    @pytest.mark.asyncio
    async def test_fact_tags_updated_on_resave(self, persistence, db):
        """When a fact's tags change, the new tags should replace old ones."""
        state = NarrativeSessionState(
            session_id=SESSION_ID,
            facts=[
                Fact(
                    id="f1",
                    session_id=SESSION_ID,
                    content="A fact",
                    established_at=0,
                    tags=["old_tag1", "old_tag2"],
                ),
            ],
        )
        await persistence.save_session_state(SESSION_ID, state)

        # Change tags
        state.facts[0].tags = ["new_tag"]
        await persistence.save_session_state(SESSION_ID, state)

        loaded = await persistence.load_session_state(SESSION_ID)
        assert loaded is not None
        assert loaded.facts[0].tags == ["new_tag"]
