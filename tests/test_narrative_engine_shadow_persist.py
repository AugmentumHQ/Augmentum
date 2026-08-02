"""Tests for Phase 3 chunk 1: engine shadow writes to branch-tagged tables.

When NarrativePersistence is attached to a NarrativeEngine, every:
  - apply_state_memory_response → STATE snapshot + LEDGER rows in new tables
  - branch detection → narrative_branches row
  - attach_persistence call → 'main' branch row (idempotent baseline)

The original in-memory state and legacy JSON columns are unchanged. Tests
verify both paths produce identical content during the migration period.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from augmentum.modes.narrative.engine import NarrativeEngine
from augmentum.modes.narrative.memory import (
    CardType,
    MemoryEntry,
    StateSnapshot,
)
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.narrative_persistence import NarrativePersistence


@pytest.fixture
async def backend():
    be = SQLiteBackend(":memory:")
    await be.connect()
    yield be
    await be.close()


@pytest.fixture
async def attached_engine(backend):
    """Engine with persistence attached + sessions row seeded."""
    sid, uid = "ses_shadow", "user_shadow"
    await backend.conn.execute(
        "INSERT OR IGNORE INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES (?, ?, 't', 'narrative', '{}')",
        (sid, uid),
    )
    await backend.conn.execute(
        "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)", (sid, uid),
    )
    await backend.conn.commit()

    engine = NarrativeEngine(session_id=sid)
    persistence = NarrativePersistence(backend.conn)
    engine.attach_persistence(persistence, uid)
    return engine, persistence, sid, uid


async def _drain_shadow_tasks(engine: NarrativeEngine) -> None:
    """Await all in-flight shadow persist tasks. Called in tests after
    triggering a shadow write so we can deterministically verify rows."""
    while engine._shadow_persist_tasks:  # noqa: SLF001
        # Snapshot to avoid mutation-during-iteration issues
        pending = list(engine._shadow_persist_tasks)  # noqa: SLF001
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_attach_persistence_seeds_main_branch(attached_engine):
    """attach_persistence schedules an upsert for the default 'main' branch
    so vanilla sessions have a row before any branch-detection event."""
    engine, persistence, sid, uid = attached_engine
    await _drain_shadow_tasks(engine)

    branches = await persistence.list_branches(sid, user_id=uid)
    assert len(branches) == 1
    assert branches[0].branch_id == "main"
    assert branches[0].parent_branch_id is None


@pytest.mark.asyncio
async def test_apply_state_memory_response_shadow_writes(attached_engine):
    """A STATE+LEDGER refresh shadow-writes a snapshot row + ledger entries
    while still updating in-memory structures (legacy path unchanged)."""
    engine, persistence, sid, uid = attached_engine
    await _drain_shadow_tasks(engine)

    snapshot = StateSnapshot(
        card_type=CardType.CHARACTER,
        fields={"location": "library", "who_present": "Alice"},
    )
    entries = [
        MemoryEntry(round_num=4, category="discovery", content="found map"),
        MemoryEntry(round_num=8, category="commitment", content="pledged help"),
    ]
    engine.apply_state_memory_response(snapshot, entries, batch_end=8)
    await _drain_shadow_tasks(engine)

    # In-memory still works (legacy path)
    assert engine._state_snapshot is snapshot  # noqa: SLF001
    assert len(engine._memory_ledger) == 2  # noqa: SLF001

    # AND new tables now have the rows
    chain = await persistence.get_branch_ancestry(sid, "main", user_id=uid)
    snap_data = await persistence.get_state_snapshot_at(sid, chain, 100, user_id=uid)
    assert snap_data is not None
    assert snap_data["fields"]["location"] == "library"

    ledger = await persistence.list_ledger_entries(sid, chain, user_id=uid)
    contents = [e["content"] for e in ledger]
    assert contents == ["found map", "pledged help"]


@pytest.mark.asyncio
async def test_empty_snapshot_does_not_write_row(attached_engine):
    """An empty StateSnapshot (e.g. from a malformed parse) shouldn't write a
    row to snapshot history — that would clobber the prior good state when
    Chunk 3 flips reads over."""
    engine, persistence, sid, uid = attached_engine
    await _drain_shadow_tasks(engine)

    # First, write a good snapshot
    good = StateSnapshot(
        card_type=CardType.CHARACTER,
        fields={"location": "valid"},
    )
    engine.apply_state_memory_response(good, [], batch_end=4)
    await _drain_shadow_tasks(engine)

    # Then attempt to write an empty snapshot (simulating refresh failure)
    bad = StateSnapshot(card_type=CardType.CHARACTER, fields={})
    engine.apply_state_memory_response(bad, [], batch_end=8)
    await _drain_shadow_tasks(engine)

    # Snapshot history should have ONE row (only the good one)
    cursor = await persistence._conn.execute(  # noqa: SLF001
        "SELECT COUNT(*) FROM narrative_state_snapshots WHERE session_id = ?", (sid,),
    )
    (n,) = await cursor.fetchone()
    assert n == 1


@pytest.mark.asyncio
async def test_unattached_engine_does_not_write(backend):
    """Without attach_persistence, no rows are written to the new tables.
    This protects test/sync contexts and stages the rollout safely."""
    sid, uid = "ses_no_persist", "user_no_persist"
    await backend.conn.execute(
        "INSERT INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES (?, ?, 't', 'narrative', '{}')",
        (sid, uid),
    )
    await backend.conn.execute(
        "INSERT INTO sessions (id, user_id) VALUES (?, ?)", (sid, uid),
    )
    await backend.conn.commit()

    engine = NarrativeEngine(session_id=sid)
    # Note: NO attach_persistence call

    snapshot = StateSnapshot(
        card_type=CardType.CHARACTER, fields={"location": "x"})
    engine.apply_state_memory_response(
        snapshot, [MemoryEntry(round_num=4, category="x", content="y")],
        batch_end=4,
    )

    # New tables stay empty
    persistence = NarrativePersistence(backend.conn)
    branches = await persistence.list_branches(sid, user_id=uid)
    assert branches == []
    cursor = await backend.conn.execute(
        "SELECT COUNT(*) FROM narrative_ledger_entries WHERE session_id = ?", (sid,),
    )
    (n,) = await cursor.fetchone()
    assert n == 0


@pytest.mark.asyncio
async def test_shadow_write_on_branch_with_in_memory_branch_id(attached_engine):
    """When the engine's branch_tracker.current_branch is set to a non-main
    branch, the shadow write tags rows with that branch_id."""
    engine, persistence, sid, uid = attached_engine
    await _drain_shadow_tasks(engine)

    # Manually flip the branch tracker (simulating apply_branch was called)
    engine._branch_tracker._current_branch = "branch_xyz"  # noqa: SLF001

    snapshot = StateSnapshot(
        card_type=CardType.CHARACTER, fields={"location": "alt-world"})
    entries = [MemoryEntry(round_num=22, category="x", content="alt-event")]
    engine.apply_state_memory_response(snapshot, entries, batch_end=22)
    await _drain_shadow_tasks(engine)

    # Rows tagged with branch_xyz, NOT main
    cursor = await persistence._conn.execute(  # noqa: SLF001
        "SELECT branch_id FROM narrative_ledger_entries WHERE session_id = ?",
        (sid,),
    )
    rows = [r[0] for r in await cursor.fetchall()]
    assert rows == ["branch_xyz"]

    cursor = await persistence._conn.execute(  # noqa: SLF001
        "SELECT branch_id FROM narrative_state_snapshots WHERE session_id = ?",
        (sid,),
    )
    rows = [r[0] for r in await cursor.fetchall()]
    assert rows == ["branch_xyz"]


@pytest.mark.asyncio
async def test_rollback_to_restores_snapshot_from_history(attached_engine):
    """Phase 3 chunk 3: when handler pre-fetches a snapshot via
    prepare_branch_snapshot, rollback_to restores STATE instead of wiping
    to None. Closes the empty-STATE-on-first-turn-after-branch hole."""
    engine, persistence, sid, uid = attached_engine
    await _drain_shadow_tasks(engine)

    # Seed a snapshot at message 16 (in real flow this is from a prior refresh)
    snap_data = {
        "card_type": "character",
        "fields": {"location": "library", "who_present": "Alice"},
    }
    await persistence.store_state_snapshot(
        sid, "main", 16, snap_data, user_id=uid,
    )

    # Set engine state as if message_count is 25 with a non-None snapshot
    engine._state_snapshot = StateSnapshot(  # noqa: SLF001
        card_type=CardType.CHARACTER,
        fields={"location": "tower"},
    )
    engine._state.message_count = 25  # noqa: SLF001

    # Simulate handler's pre-fetch (returns the snapshot at index 16)
    chain = await persistence.get_branch_ancestry(sid, "main", user_id=uid)
    fetched = await persistence.get_state_snapshot_at(sid, chain, 20, user_id=uid)
    assert fetched is not None
    engine.prepare_branch_snapshot(fetched)

    # Now rollback to message 20
    engine.rollback_to(20)

    # STATE should be the recovered snapshot from #16, NOT None and NOT the
    # 'tower' state from before rollback
    assert engine._state_snapshot is not None  # noqa: SLF001
    assert engine._state_snapshot.fields["location"] == "library"  # noqa: SLF001
    # Pending slot was consumed
    assert engine._pending_branch_snapshot is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_rollback_to_legacy_wipes_when_no_pending_snapshot(attached_engine):
    """When prepare_branch_snapshot was NOT called (legacy path), rollback_to
    wipes STATE to None — preserving pre-Phase-3 behavior."""
    engine, persistence, sid, uid = attached_engine
    await _drain_shadow_tasks(engine)

    engine._state_snapshot = StateSnapshot(  # noqa: SLF001
        card_type=CardType.CHARACTER,
        fields={"location": "should-be-wiped"},
    )
    engine._state.message_count = 25  # noqa: SLF001
    # No prepare_branch_snapshot call

    engine.rollback_to(20)

    assert engine._state_snapshot is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_rollback_to_recovers_from_main_when_branch_has_no_snapshot(attached_engine):
    """Three-level scenario: branch B from main, no B snapshots yet. Rollback
    on B walks ancestry to main and recovers from main's snapshot history."""
    engine, persistence, sid, uid = attached_engine
    await _drain_shadow_tasks(engine)

    # Main has snapshots at 4 and 16
    await persistence.store_state_snapshot(
        sid, "main", 4, {"card_type": "character", "fields": {"loc": "early"}},
        user_id=uid,
    )
    await persistence.store_state_snapshot(
        sid, "main", 16, {"card_type": "character", "fields": {"loc": "late"}},
        user_id=uid,
    )
    # Register branch B forking at message 20
    await persistence.upsert_branch(sid, "B", "main", 20, user_id=uid)

    # On B, requesting snapshot at 21 should find main's #16 (not #4, not None)
    chain = await persistence.get_branch_ancestry(sid, "B", user_id=uid)
    fetched = await persistence.get_state_snapshot_at(sid, chain, 21, user_id=uid)
    assert fetched is not None
    assert fetched["fields"]["loc"] == "late"

    engine.prepare_branch_snapshot(fetched)
    engine.rollback_to(20)
    assert engine._state_snapshot.fields["loc"] == "late"  # noqa: SLF001


@pytest.mark.asyncio
async def test_shadow_write_failure_does_not_break_main_path(attached_engine):
    """If a shadow write raises (e.g., DB closed), the main path must still
    succeed — apply_state_memory_response keeps in-memory state intact."""
    engine, persistence, sid, uid = attached_engine
    await _drain_shadow_tasks(engine)

    # Sabotage persistence: replace its conn with one that's closed
    bad_persistence = NarrativePersistence(persistence._conn)  # noqa: SLF001
    # Monkey-patch so any execute call raises
    async def boom(*args, **kwargs):
        raise RuntimeError("synthetic db failure")

    bad_persistence.store_state_snapshot = boom
    bad_persistence.store_ledger_entries = boom
    engine._persistence = bad_persistence  # noqa: SLF001

    snapshot = StateSnapshot(
        card_type=CardType.CHARACTER, fields={"location": "x"})
    # Must not raise
    engine.apply_state_memory_response(
        snapshot, [MemoryEntry(round_num=4, category="x", content="y")],
        batch_end=4,
    )
    await _drain_shadow_tasks(engine)

    # In-memory state preserved
    assert engine._state_snapshot is snapshot  # noqa: SLF001
    assert len(engine._memory_ledger) == 1  # noqa: SLF001
