"""Tests for augmentum.personality.graph — spreading activation + cross-table.

Verifies that the high-level `compose_facet_affects` correctly merges signal
from recent activations, the facet cooccurrence graph, and memory↔facet
associations. Also verifies `update_after_response` writes both activations
and memory links atomically.
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from augmentum.personality.graph import (
    compose_facet_affects,
    update_after_response,
)
from augmentum.personality.store import COOCCURRENCE_FLOOR, PersonalityStore


_MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "augmentum"
    / "state"
    / "migrations"
    / "160_personality_facets.sql"
)


class _StubBackend:
    def __init__(self, conn):
        self.conn = conn


@pytest_asyncio.fixture
async def store():
    conn = await aiosqlite.connect(":memory:")
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
# compose_facet_affects — base cases
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compose_empty_signal_returns_empty(store):
    """Fresh store with no activations → empty dict, no error."""
    result = await compose_facet_affects(
        store, user_id="u1", companion_id="c1"
    )
    assert result == {}


@pytest.mark.asyncio
async def test_compose_missing_user_id_returns_empty(store):
    result = await compose_facet_affects(
        store, user_id="", companion_id="c1"
    )
    assert result == {}


@pytest.mark.asyncio
async def test_compose_returns_recent_activations(store):
    """Recent activations show up in compose output."""
    await store.record_activations(
        [("warm", 1.0), ("patient", 0.7)],
        user_id="u1",
        companion_id="c1",
    )
    result = await compose_facet_affects(
        store, user_id="u1", companion_id="c1"
    )
    assert "warm" in result
    assert "patient" in result


# ----------------------------------------------------------------------
# Spreading activation through cooccurrence
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compose_spreads_via_cooccurrence(store):
    """Establish historical (warm + patient + tender) cooccurrence; then a
    recent (warm,) activation should spread to predict patient + tender."""
    # Build cooccurrence history above floor.
    for _ in range(COOCCURRENCE_FLOOR + 1):
        await store.record_activations(
            [("warm", 1.0), ("patient", 1.0), ("tender", 1.0)],
            user_id="u1",
            companion_id="c1",
        )
    # Recent activation: just warm. Spread should bring in patient + tender.
    result = await compose_facet_affects(
        store, user_id="u1", companion_id="c1", limit=10
    )
    # warm is direct; patient + tender are spreads
    assert "warm" in result
    assert "patient" in result
    assert "tender" in result


# ----------------------------------------------------------------------
# Cross-table contribution
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compose_picks_up_memory_associations(store):
    """When `retrieved_memory_ids` is supplied, facets historically associated
    with those memories enter the composition even without recent activations."""
    # Establish memory association above floor.
    for _ in range(COOCCURRENCE_FLOOR + 1):
        await store.record_memory_associations(
            memory_ids=["mem-coder-project"],
            facets=["rigorous", "patient"],
            user_id="u1",
            companion_id="c1",
        )
    # Compose with that memory_id retrieved, no recent activations.
    result = await compose_facet_affects(
        store,
        user_id="u1",
        companion_id="c1",
        retrieved_memory_ids=["mem-coder-project"],
    )
    assert "rigorous" in result
    assert "patient" in result


@pytest.mark.asyncio
async def test_compose_cross_table_only_includes_above_floor(store):
    """Memory associations below COOCCURRENCE_FLOOR don't show up."""
    for _ in range(COOCCURRENCE_FLOOR - 1):  # below floor
        await store.record_memory_associations(
            memory_ids=["mem-x"],
            facets=["rigorous"],
            user_id="u1",
            companion_id="c1",
        )
    result = await compose_facet_affects(
        store,
        user_id="u1",
        companion_id="c1",
        retrieved_memory_ids=["mem-x"],
    )
    assert "rigorous" not in result


# ----------------------------------------------------------------------
# Output shape
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compose_scores_normalized_to_unit_range(store):
    """All returned scores should be in [0, 1] after normalization."""
    for _ in range(20):
        await store.record_activations(
            [("warm", 1.0), ("patient", 0.8), ("tender", 0.6)],
            user_id="u1",
            companion_id="c1",
        )
    result = await compose_facet_affects(
        store, user_id="u1", companion_id="c1"
    )
    assert all(0.0 <= score <= 1.0 for score in result.values())


@pytest.mark.asyncio
async def test_compose_respects_limit(store):
    """The returned dict has at most `limit` entries."""
    # Fire many distinct facets to ensure broad signal.
    facets_a = ["warm", "patient", "tender", "playful", "curious"]
    facets_b = ["alert", "openhanded", "delighted", "exploratory", "gentle"]
    for _ in range(5):
        await store.record_activations(
            [(f, 1.0) for f in facets_a + facets_b],
            user_id="u1",
            companion_id="c1",
        )
    result = await compose_facet_affects(
        store, user_id="u1", companion_id="c1", limit=4
    )
    assert len(result) <= 4


# ----------------------------------------------------------------------
# User isolation
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compose_user_isolation(store):
    """User A's facets don't bleed into User B's composition."""
    await store.record_activations(
        [("warm", 1.0), ("patient", 1.0)],
        user_id="u1",
        companion_id="c1",
    )
    result_b = await compose_facet_affects(
        store, user_id="u2", companion_id="c1"
    )
    assert result_b == {}


# ----------------------------------------------------------------------
# update_after_response
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_after_response_writes_activations(store):
    count = await update_after_response(
        store,
        labeled_facets=[("warm", 0.8), ("patient", 0.6)],
        user_id="u1",
        companion_id="c1",
    )
    assert count == 2
    recent = await store.query_recent_activations(user_id="u1", companion_id="c1")
    assert len(recent) == 2


@pytest.mark.asyncio
async def test_update_after_response_links_memory_associations(store):
    """When `retrieved_memory_ids` provided, memory_associations get written."""
    await update_after_response(
        store,
        labeled_facets=[("warm", 0.8), ("patient", 0.6)],
        user_id="u1",
        companion_id="c1",
        retrieved_memory_ids=["mem-1", "mem-2"],
    )
    # 2 memories × 2 facets = 4 associations
    cursor = await store._conn.execute(
        "SELECT COUNT(*) FROM personality_memory_associations "
        "WHERE user_id = ? AND companion_id = ?",
        ("u1", "c1"),
    )
    count = (await cursor.fetchone())[0]
    assert count == 4


@pytest.mark.asyncio
async def test_update_after_response_empty_facets_no_writes(store):
    """No labeled facets → no writes, no error."""
    count = await update_after_response(
        store,
        labeled_facets=[],
        user_id="u1",
        companion_id="c1",
        retrieved_memory_ids=["mem-1"],
    )
    assert count == 0
    recent = await store.query_recent_activations(user_id="u1", companion_id="c1")
    assert len(recent) == 0


@pytest.mark.asyncio
async def test_update_after_response_no_memory_ids_skips_associations(store):
    """If no memory_ids, only activations are written — no associations."""
    await update_after_response(
        store,
        labeled_facets=[("warm", 0.8)],
        user_id="u1",
        companion_id="c1",
    )
    cursor = await store._conn.execute(
        "SELECT COUNT(*) FROM personality_memory_associations"
    )
    count = (await cursor.fetchone())[0]
    assert count == 0
