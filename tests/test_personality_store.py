"""Tests for augmentum.personality.store — Hebbian cooccurrence + user_id isolation.

Verifies the schema-level operations of `PersonalityStore`: vocabulary seeding,
activation recording, cooccurrence updates, cross-table memory associations,
decay, and multi-tenant scoping.
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from augmentum.personality.models import FacetCategory
from augmentum.personality.store import (
    COOCCURRENCE_FLOOR,
    DECAY_FACTOR,
    PersonalityStore,
)
from augmentum.personality.vocabulary import SEED_FACETS

_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "augmentum"
    / "state"
    / "migrations"
    / "160_personality_facets.sql"
)


class _StubBackend:
    """Minimal SQLiteBackend stand-in — only `.conn` is exercised by the store."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self.conn = conn


@pytest_asyncio.fixture
async def store():
    conn = await aiosqlite.connect(":memory:")
    # schema_version is a dependency of the migration's final INSERT
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, description TEXT)"
    )
    sql = _MIGRATION.read_text(encoding="utf-8")
    await conn.executescript(sql)
    await conn.commit()
    backend = _StubBackend(conn)
    store_ = PersonalityStore(backend)
    await store_.seed_vocabulary()
    yield store_
    await conn.close()


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_seed_vocabulary_inserts_all_seed_facets(store):
    facets = await store.list_facets()
    assert len(facets) == len(SEED_FACETS)
    names = {f.name for f in facets}
    assert {seed.name for seed in SEED_FACETS} == names


@pytest.mark.asyncio
async def test_seed_vocabulary_is_idempotent(store):
    inserted_again = await store.seed_vocabulary()
    assert inserted_again == 0
    facets = await store.list_facets()
    assert len(facets) == len(SEED_FACETS)


@pytest.mark.asyncio
async def test_list_facets_by_category(store):
    affect_facets = await store.list_facets(category=FacetCategory.AFFECT)
    assert all(f.category == FacetCategory.AFFECT for f in affect_facets)
    assert any(f.name == "tender" for f in affect_facets)


@pytest.mark.asyncio
async def test_first_class_affect_facets_present(store):
    """Commitment #6: `unsure` and `not_okay` must be first-class facets."""
    assert await store.get_facet("unsure") is not None
    assert await store.get_facet("not_okay") is not None


# ----------------------------------------------------------------------
# Activations + cooccurrence (the Hebbian write path)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_activations_requires_user_id(store):
    with pytest.raises(ValueError, match="user_id"):
        await store.record_activations([("warm", 0.7)], user_id="", companion_id="c1")


@pytest.mark.asyncio
async def test_record_activations_requires_companion_id(store):
    with pytest.raises(ValueError, match="companion_id"):
        await store.record_activations([("warm", 0.7)], user_id="u1", companion_id="")


@pytest.mark.asyncio
async def test_record_activations_inserts_one_row_per_facet(store):
    ids = await store.record_activations(
        [("warm", 0.7), ("patient", 0.5), ("tender", 0.6)],
        user_id="u1",
        companion_id="c1",
    )
    assert len(ids) == 3
    recent = await store.query_recent_activations(user_id="u1", companion_id="c1")
    assert len(recent) == 3


@pytest.mark.asyncio
async def test_record_activations_writes_pairwise_cooccurrence(store):
    """N active facets → C(N,2) cooccurrence rows."""
    await store.record_activations(
        [("warm", 1.0), ("patient", 1.0), ("tender", 1.0)],
        user_id="u1",
        companion_id="c1",
    )
    # 3 facets → 3 pairs
    cursor = await store._conn.execute(
        "SELECT COUNT(*) FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    count = (await cursor.fetchone())[0]
    assert count == 3


@pytest.mark.asyncio
async def test_record_activations_drops_unknown_facets(store):
    """Hallucinated facet names (e.g., labeler typo) are silently dropped."""
    ids = await store.record_activations(
        [("warm", 0.7), ("not_a_real_facet", 0.5)],
        user_id="u1",
        companion_id="c1",
    )
    # Only the valid facet is recorded.
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_record_activations_canonical_pair_ordering(store):
    """warm-tender and tender-warm collapse to one row."""
    await store.record_activations(
        [("tender", 1.0), ("warm", 1.0)],
        user_id="u1",
        companion_id="c1",
    )
    # Then same facets in different order — should hit the same row.
    await store.record_activations(
        [("warm", 1.0), ("tender", 1.0)],
        user_id="u1",
        companion_id="c1",
    )
    cursor = await store._conn.execute(
        "SELECT facet_a, facet_b, count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    rows = await cursor.fetchall()
    assert len(rows) == 1
    facet_a, facet_b, count = rows[0]
    assert facet_a == "tender"  # alphabetically first
    assert facet_b == "warm"
    assert count == 2


@pytest.mark.asyncio
async def test_record_activations_cooccurrence_idempotent_increment(store):
    """Recording the same pair twice → count = 2, not 2 rows."""
    for _ in range(5):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0)],
            user_id="u1",
            companion_id="c1",
        )
    cursor = await store._conn.execute(
        "SELECT count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    row = await cursor.fetchone()
    assert row[0] == 5


# ----------------------------------------------------------------------
# Query / retrieval
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_query_cooccurrent_facets_respects_floor(store):
    """Pairs below COOCCURRENCE_FLOOR are noise — excluded from queries."""
    # Record warm-patient twice (count=2, below floor of 3).
    for _ in range(2):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0)],
            user_id="u1",
            companion_id="c1",
        )
    result = await store.query_cooccurrent_facets(
        ["warm"], user_id="u1", companion_id="c1"
    )
    assert result == []

    # Now bring it above the floor.
    for _ in range(COOCCURRENCE_FLOOR):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0)],
            user_id="u1",
            companion_id="c1",
        )
    result = await store.query_cooccurrent_facets(
        ["warm"], user_id="u1", companion_id="c1"
    )
    assert any(name == "patient" for name, _ in result)


@pytest.mark.asyncio
async def test_query_cooccurrent_facets_excludes_inputs(store):
    """The query returns facets ASSOCIATED with the inputs, not the inputs themselves."""
    for _ in range(COOCCURRENCE_FLOOR + 1):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0), ("tender", 1.0)],
            user_id="u1",
            companion_id="c1",
        )
    result = await store.query_cooccurrent_facets(
        ["warm", "patient"], user_id="u1", companion_id="c1"
    )
    returned_names = {name for name, _ in result}
    assert "warm" not in returned_names
    assert "patient" not in returned_names
    assert "tender" in returned_names


# ----------------------------------------------------------------------
# Multi-tenant isolation (the load-bearing security invariant)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_isolation_in_activations(store):
    """User A's activations are invisible to User B."""
    await store.record_activations(
        [("warm", 1.0)], user_id="u1", companion_id="c1"
    )
    a_results = await store.query_recent_activations(user_id="u1", companion_id="c1")
    b_results = await store.query_recent_activations(user_id="u2", companion_id="c1")
    assert len(a_results) == 1
    assert len(b_results) == 0


@pytest.mark.asyncio
async def test_companion_isolation_within_same_user(store):
    """Different companions for the same user have independent facet graphs."""
    for _ in range(COOCCURRENCE_FLOOR + 1):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0)],
            user_id="u1",
            companion_id="becca",
        )
    becca_assoc = await store.query_cooccurrent_facets(
        ["warm"], user_id="u1", companion_id="becca"
    )
    other_assoc = await store.query_cooccurrent_facets(
        ["warm"], user_id="u1", companion_id="other"
    )
    assert any(name == "patient" for name, _ in becca_assoc)
    assert other_assoc == []


# ----------------------------------------------------------------------
# Memory associations (the cross-table)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_record_memory_associations_writes_pairs(store):
    """N memories × M facets → N*M rows."""
    count = await store.record_memory_associations(
        memory_ids=["m1", "m2"],
        facets=["warm", "patient"],
        user_id="u1",
        companion_id="c1",
    )
    assert count == 4


@pytest.mark.asyncio
async def test_record_memory_associations_idempotent(store):
    """Repeated calls strengthen the association, don't multiply rows."""
    for _ in range(3):
        await store.record_memory_associations(
            memory_ids=["m1"],
            facets=["warm"],
            user_id="u1",
            companion_id="c1",
        )
    cursor = await store._conn.execute(
        "SELECT count FROM personality_memory_associations "
        "WHERE user_id = ? AND companion_id = ? AND memory_id = ? AND facet = ?",
        ("u1", "c1", "m1", "warm"),
    )
    row = await cursor.fetchone()
    assert row[0] == 3


@pytest.mark.asyncio
async def test_query_facets_for_memories_respects_floor(store):
    """Memory associations below COOCCURRENCE_FLOOR don't show up."""
    for _ in range(2):  # below floor
        await store.record_memory_associations(
            memory_ids=["m1"], facets=["warm"], user_id="u1", companion_id="c1"
        )
    result = await store.query_facets_for_memories(
        ["m1"], user_id="u1", companion_id="c1"
    )
    assert result == []

    for _ in range(COOCCURRENCE_FLOOR):
        await store.record_memory_associations(
            memory_ids=["m1"], facets=["warm"], user_id="u1", companion_id="c1"
        )
    result = await store.query_facets_for_memories(
        ["m1"], user_id="u1", companion_id="c1"
    )
    assert any(name == "warm" for name, _ in result)


# ----------------------------------------------------------------------
# Decay
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decay_cooccurrence_multiplies_counts(store):
    """Decay multiplies counts by the decay factor (CAST to INTEGER)."""
    for _ in range(100):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0)],
            user_id="u1",
            companion_id="c1",
        )
    cursor = await store._conn.execute(
        "SELECT count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    before = (await cursor.fetchone())[0]
    assert before == 100

    await store.decay_cooccurrence(user_id="u1", companion_id="c1", decay_factor=DECAY_FACTOR)
    cursor = await store._conn.execute(
        "SELECT count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    after = (await cursor.fetchone())[0]
    assert after == 99  # CAST(100 * 0.99 AS INTEGER) = 99


@pytest.mark.asyncio
async def test_decay_floors_count_at_one(store):
    """Decay must never drop a row below count=1 — without this floor,
    pairs that fire only once decay to 0 immediately (CAST(1*0.99) = 0)
    and are pruned before they can ever accumulate past
    COOCCURRENCE_FLOOR. Mirrors MemoryStore.decay_cooccurrence behavior.
    """
    await store.record_activations(
        [("warm", 1.0), ("patient", 1.0)],
        user_id="u1",
        companion_id="c1",
    )
    # First decay tick: count=1 should survive at count=1 thanks to MAX floor.
    await store.decay_cooccurrence(
        user_id="u1", companion_id="c1", decay_factor=DECAY_FACTOR
    )
    cursor = await store._conn.execute(
        "SELECT count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1

    # Repeated decay still floors at 1.
    for _ in range(10):
        await store.decay_cooccurrence(
            user_id="u1", companion_id="c1", decay_factor=DECAY_FACTOR
        )
    cursor = await store._conn.execute(
        "SELECT count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1


@pytest.mark.asyncio
async def test_decay_two_decays_one(store):
    """count=2 with 0.99 decay → CAST(1.98) = 1 (still above the MAX floor)."""
    for _ in range(2):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0)],
            user_id="u1",
            companion_id="c1",
        )
    # count is now 2
    await store.decay_cooccurrence(
        user_id="u1", companion_id="c1", decay_factor=DECAY_FACTOR
    )
    cursor = await store._conn.execute(
        "SELECT count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    row = await cursor.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_record_activations_auto_seeds_vocabulary():
    """If a caller instantiates PersonalityStore and forgets to call
    `seed_vocabulary()`, the first `record_activations` should auto-seed
    rather than silently drop every labeled facet.
    """
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, description TEXT)"
    )
    sql = _MIGRATION.read_text(encoding="utf-8")
    await conn.executescript(sql)
    await conn.commit()

    backend = _StubBackend(conn)
    fresh_store = PersonalityStore(backend)
    # Deliberately NOT calling fresh_store.seed_vocabulary().

    ids = await fresh_store.record_activations(
        [("warm", 0.7), ("patient", 0.5)],
        user_id="u1",
        companion_id="c1",
    )

    # Facets accepted (not silently dropped)
    assert len(ids) == 2
    # And the vocabulary table is now populated
    facets = await fresh_store.list_facets()
    assert len(facets) == len(SEED_FACETS)

    await conn.close()


@pytest.mark.asyncio
async def test_decay_scoped_to_user(store):
    """Decay scoped to one user doesn't affect another user's counts."""
    for _ in range(100):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0)], user_id="u1", companion_id="c1"
        )
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0)], user_id="u2", companion_id="c1"
        )

    await store.decay_cooccurrence(user_id="u1", companion_id="c1", decay_factor=0.5)

    cursor = await store._conn.execute(
        "SELECT count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    u1_count = (await cursor.fetchone())[0]
    cursor = await store._conn.execute(
        "SELECT count FROM personality_facet_cooccurrence "
        "WHERE user_id = ? AND companion_id = ?",
        ("u2", "c1"),
    )
    u2_count = (await cursor.fetchone())[0]
    assert u1_count == 50  # decayed
    assert u2_count == 100  # untouched
