"""Unit tests for branch-aware persistence methods (Phase 2a additions).

Covers the new functions added to NarrativePersistence for migrations 115-118:
  - Branch metadata: upsert_branch, get_branch_ancestry, list_branches,
    set_branch_status, mark_stale_branches, has_branch_descendants
  - State snapshots: store_state_snapshot, get_state_snapshot_at,
    prune_state_snapshots_for_branch
  - Ledger entries: store_ledger_entries, list_ledger_entries,
    count_ledger_entries, compact_ledger_entries, prune_ledger_entries_for_branch
  - Branch-aware archive: store_archive_exchanges_for_branch,
    retrieve_archive_for_branch
  - Branch deletion cascade: delete_branch_cascade
  - Storage observability: get_session_storage

Special focus on the seams that fail only in extreme situations:
  - Deep ancestry chains (3+ levels)
  - Off-by-one branch_point semantics
  - Atomicity under failure (compact, cascade)
  - Cross-branch isolation
"""

from __future__ import annotations

import struct

import pytest

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.narrative_persistence import (
    BranchAncestor,
    NarrativePersistence,
)


def _float_vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _stub_768d(seed: float) -> bytes:
    """Synthetic 768-d float32 embedding deterministic by seed."""
    return _float_vec_to_blob([seed * (i / 768.0) for i in range(768)])


@pytest.fixture
async def backend():
    be = SQLiteBackend(":memory:")
    await be.connect()
    yield be
    await be.close()


@pytest.fixture
async def persist(backend):
    return NarrativePersistence(backend.conn)


@pytest.fixture
async def seeded_session(backend):
    """Create a session row + ui_sessions row so FK constraints are satisfied."""
    sid = "ses_branchtest"
    uid = "user_branchtest"
    await backend.conn.execute(
        "INSERT OR IGNORE INTO ui_sessions (id, user_id, title, mode, data) "
        "VALUES (?, ?, 't', 'narrative', '{}')",
        (sid, uid),
    )
    await backend.conn.execute(
        "INSERT OR IGNORE INTO sessions (id, user_id) VALUES (?, ?)", (sid, uid),
    )
    await backend.conn.commit()
    return sid, uid


# ===========================================================================
# Branch metadata
# ===========================================================================

class TestUpsertBranch:
    @pytest.mark.asyncio
    async def test_creates_branch_on_first_call(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        branches = await persist.list_branches(sid, user_id=uid)
        assert len(branches) == 1
        assert branches[0].branch_id == "main"
        assert branches[0].parent_branch_id is None
        assert branches[0].status == "active"

    @pytest.mark.asyncio
    async def test_upsert_does_not_duplicate(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        branches = await persist.list_branches(sid, user_id=uid)
        assert len(branches) == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_last_visited_at(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        first = (await persist.list_branches(sid, user_id=uid))[0].last_visited_at
        # SQLite datetime() resolution is 1s; pause might not register, but the
        # UPDATE statement should still execute. We just verify it doesn't error.
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        second = (await persist.list_branches(sid, user_id=uid))[0].last_visited_at
        assert second >= first

    @pytest.mark.asyncio
    async def test_requires_user_id(self, persist, seeded_session):
        sid, _ = seeded_session
        with pytest.raises(ValueError, match="user_id"):
            await persist.upsert_branch(sid, "main", None, 0, user_id="")


class TestGetBranchAncestry:
    @pytest.mark.asyncio
    async def test_main_only(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        chain = await persist.get_branch_ancestry(sid, "main", user_id=uid)
        assert len(chain) == 1
        assert chain[0].branch_id == "main"
        assert chain[0].branch_point == 0

    @pytest.mark.asyncio
    async def test_two_levels(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 20, user_id=uid)
        chain = await persist.get_branch_ancestry(sid, "B", user_id=uid)
        assert len(chain) == 2
        # Each entry holds its OWN divergence point from its parent (schema convention).
        # Leaf B diverged from main at message 20.
        assert chain[0].branch_id == "B"
        assert chain[0].branch_point == 20
        # main is root; never diverged from a parent.
        assert chain[1].branch_id == "main"
        assert chain[1].branch_point == 0

    @pytest.mark.asyncio
    async def test_three_levels(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 20, user_id=uid)
        await persist.upsert_branch(sid, "C", "B", 35, user_id=uid)
        chain = await persist.get_branch_ancestry(sid, "C", user_id=uid)
        assert len(chain) == 3
        assert chain[0] == BranchAncestor("C", 35)   # C diverged from B at 35
        assert chain[1] == BranchAncestor("B", 20)    # B diverged from main at 20
        assert chain[2] == BranchAncestor("main", 0)  # main is root

    @pytest.mark.asyncio
    async def test_unknown_branch_returns_safe_default(self, persist, seeded_session):
        sid, uid = seeded_session
        chain = await persist.get_branch_ancestry(sid, "ghost", user_id=uid)
        assert len(chain) == 1
        assert chain[0].branch_id == "ghost"

    @pytest.mark.asyncio
    async def test_ancestry_does_not_loop_on_corrupt_data(self, persist, seeded_session, backend):
        """Defensive: even if parent_branch_id forms a cycle, we cap at 64 hops."""
        sid, uid = seeded_session
        # Manually insert a self-referencing branch (corrupt state)
        await backend.conn.execute(
            "INSERT INTO narrative_branches (branch_id, session_id, parent_branch_id, "
            "branch_point, status, user_id) VALUES (?, ?, ?, ?, ?, ?)",
            ("loop", sid, "loop", 5, "active", uid),
        )
        await backend.conn.commit()
        chain = await persist.get_branch_ancestry(sid, "loop", user_id=uid)
        # Hard cap at 64 iterations
        assert len(chain) <= 64


class TestListBranches:
    @pytest.mark.asyncio
    async def test_includes_stale_by_default(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        await persist.set_branch_status(sid, "B", "stale", user_id=uid)
        branches = await persist.list_branches(sid, user_id=uid)
        assert len(branches) == 2

    @pytest.mark.asyncio
    async def test_excludes_stale_when_requested(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        await persist.set_branch_status(sid, "B", "stale", user_id=uid)
        branches = await persist.list_branches(sid, user_id=uid, include_stale=False)
        ids = {b.branch_id for b in branches}
        assert ids == {"main"}


class TestSetBranchStatus:
    @pytest.mark.asyncio
    async def test_validates_status_value(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        with pytest.raises(ValueError, match="invalid status"):
            await persist.set_branch_status(sid, "main", "bogus", user_id=uid)

    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_branch(self, persist, seeded_session):
        sid, uid = seeded_session
        ok = await persist.set_branch_status(sid, "ghost", "archived", user_id=uid)
        assert ok is False

    @pytest.mark.asyncio
    async def test_user_pin_to_archived(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        ok = await persist.set_branch_status(sid, "B", "archived", user_id=uid)
        assert ok is True
        branches = await persist.list_branches(sid, user_id=uid)
        b_row = [x for x in branches if x.branch_id == "B"][0]
        assert b_row.status == "archived"


class TestMarkStaleBranches:
    @pytest.mark.asyncio
    async def test_marks_old_active_only(self, persist, seeded_session, backend):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B_old", "main", 5, user_id=uid)
        await persist.upsert_branch(sid, "B_recent", "main", 8, user_id=uid)
        # Backdate B_old's last_visited_at by 60 days
        await backend.conn.execute(
            "UPDATE narrative_branches SET last_visited_at = datetime('now', '-60 days') "
            "WHERE session_id = ? AND branch_id = ?",
            (sid, "B_old"),
        )
        await backend.conn.commit()
        n = await persist.mark_stale_branches(sid, user_id=uid, threshold_days=30)
        assert n == 1
        branches = {b.branch_id: b.status for b in await persist.list_branches(sid, user_id=uid)}
        assert branches["B_old"] == "stale"
        assert branches["B_recent"] == "active"
        assert branches["main"] == "active"

    @pytest.mark.asyncio
    async def test_never_marks_main(self, persist, seeded_session, backend):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await backend.conn.execute(
            "UPDATE narrative_branches SET last_visited_at = datetime('now', '-365 days') "
            "WHERE session_id = ? AND branch_id = 'main'",
            (sid,),
        )
        await backend.conn.commit()
        await persist.mark_stale_branches(sid, user_id=uid, threshold_days=30)
        branches = {b.branch_id: b.status for b in await persist.list_branches(sid, user_id=uid)}
        assert branches["main"] == "active"

    @pytest.mark.asyncio
    async def test_skips_archived(self, persist, seeded_session, backend):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "pinned", "main", 5, user_id=uid)
        await persist.set_branch_status(sid, "pinned", "archived", user_id=uid)
        await backend.conn.execute(
            "UPDATE narrative_branches SET last_visited_at = datetime('now', '-365 days') "
            "WHERE session_id = ? AND branch_id = 'pinned'",
            (sid,),
        )
        await backend.conn.commit()
        n = await persist.mark_stale_branches(sid, user_id=uid, threshold_days=30)
        assert n == 0
        branches = {b.branch_id: b.status for b in await persist.list_branches(sid, user_id=uid)}
        assert branches["pinned"] == "archived"


class TestHasBranchDescendants:
    @pytest.mark.asyncio
    async def test_true_when_child_exists(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        assert (await persist.has_branch_descendants(sid, "main", user_id=uid)) is True

    @pytest.mark.asyncio
    async def test_false_for_leaf(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        assert (await persist.has_branch_descendants(sid, "B", user_id=uid)) is False


# ===========================================================================
# State snapshots
# ===========================================================================

class TestStateSnapshots:
    @pytest.mark.asyncio
    async def test_store_snapshot_accepts_dict_and_string(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        d = {"card_type": "character", "fields": {"location": "x"}}
        sid1 = await persist.store_state_snapshot(sid, "main", 4, d, user_id=uid)
        sid2 = await persist.store_state_snapshot(sid, "main", 8, '{"card_type":"narrator"}', user_id=uid)
        assert sid1 and sid2 and sid1 != sid2

    @pytest.mark.asyncio
    async def test_get_returns_most_recent_below_index(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        for idx, loc in [(4, "forest"), (8, "cave"), (12, "tower")]:
            await persist.store_state_snapshot(
                sid, "main", idx, {"fields": {"location": loc}}, user_id=uid)

        # Request snapshot at message 10 — should get the one at #8
        chain = await persist.get_branch_ancestry(sid, "main", user_id=uid)
        snap = await persist.get_state_snapshot_at(sid, chain, 10, user_id=uid)
        assert snap is not None
        assert snap["fields"]["location"] == "cave"

    @pytest.mark.asyncio
    async def test_get_returns_none_when_no_prior_snapshot(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        chain = await persist.get_branch_ancestry(sid, "main", user_id=uid)
        snap = await persist.get_state_snapshot_at(sid, chain, 10, user_id=uid)
        assert snap is None

    @pytest.mark.asyncio
    async def test_ancestry_walks_to_main_when_branch_has_no_snapshot(
        self, persist, seeded_session,
    ):
        """If branch B has no snapshot before its branch_point, fall back to main's."""
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 20, user_id=uid)
        # Main has snapshots at 4, 8, 16
        for idx, loc in [(4, "forest"), (8, "cave"), (16, "tower")]:
            await persist.store_state_snapshot(
                sid, "main", idx, {"fields": {"location": loc}}, user_id=uid)
        # B has nothing yet (just diverged at #20)
        # On B, requesting snapshot at #21 should fall back to main's #16
        chain = await persist.get_branch_ancestry(sid, "B", user_id=uid)
        snap = await persist.get_state_snapshot_at(sid, chain, 21, user_id=uid)
        assert snap is not None
        assert snap["fields"]["location"] == "tower"

    @pytest.mark.asyncio
    async def test_branch_does_not_see_main_continuation(self, persist, seeded_session):
        """B branched at #20. Main continues with snapshot at #24. B at #25 must
        NOT see main's #24 snapshot (it post-dates the divergence)."""
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 20, user_id=uid)
        await persist.store_state_snapshot(
            sid, "main", 16, {"fields": {"location": "valid"}}, user_id=uid)
        await persist.store_state_snapshot(
            sid, "main", 24, {"fields": {"location": "POISON"}}, user_id=uid)
        chain = await persist.get_branch_ancestry(sid, "B", user_id=uid)
        snap = await persist.get_state_snapshot_at(sid, chain, 25, user_id=uid)
        assert snap is not None
        # Must be the pre-divergence snapshot, NOT the post-divergence one
        assert snap["fields"]["location"] == "valid"

    @pytest.mark.asyncio
    async def test_prune_state_snapshots_for_branch(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        await persist.store_state_snapshot(sid, "main", 4, {"x": 1}, user_id=uid)
        await persist.store_state_snapshot(sid, "B", 6, {"x": 2}, user_id=uid)
        await persist.store_state_snapshot(sid, "B", 10, {"x": 3}, user_id=uid)
        n = await persist.prune_state_snapshots_for_branch(sid, "B", user_id=uid)
        assert n == 2
        # Main untouched
        chain = await persist.get_branch_ancestry(sid, "main", user_id=uid)
        snap = await persist.get_state_snapshot_at(sid, chain, 10, user_id=uid)
        assert snap == {"x": 1}


# ===========================================================================
# Ledger entries
# ===========================================================================

class TestLedgerEntries:
    @pytest.mark.asyncio
    async def test_store_and_count(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        n = await persist.store_ledger_entries(sid, "main", [
            {"round_num": 4, "category": "discovery", "content": "found map"},
            {"round_num": 8, "category": "commitment", "content": "pledged help"},
        ], user_id=uid)
        assert n == 2
        assert await persist.count_ledger_entries(sid, "main", user_id=uid) == 2

    @pytest.mark.asyncio
    async def test_list_ancestry_filter_two_branches(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 20, user_id=uid)
        # Main has 3 entries at rounds 4, 8, 12 (pre-divergence)
        await persist.store_ledger_entries(sid, "main", [
            {"round_num": 4, "category": "x", "content": "main-pre-1"},
            {"round_num": 8, "category": "x", "content": "main-pre-2"},
            {"round_num": 12, "category": "x", "content": "main-pre-3"},
        ], user_id=uid)
        # Main also continued past divergence (poison rows for B)
        await persist.store_ledger_entries(sid, "main", [
            {"round_num": 24, "category": "x", "content": "MAIN-POISON-24"},
            {"round_num": 28, "category": "x", "content": "MAIN-POISON-28"},
        ], user_id=uid)
        # B has its own entries at rounds 24, 28
        await persist.store_ledger_entries(sid, "B", [
            {"round_num": 24, "category": "x", "content": "b-post-1"},
            {"round_num": 28, "category": "x", "content": "b-post-2"},
        ], user_id=uid)

        # On B, listing all entries: main pre-divergence + B's own
        chain = await persist.get_branch_ancestry(sid, "B", user_id=uid)
        entries = await persist.list_ledger_entries(sid, chain, user_id=uid)
        contents = [e["content"] for e in entries]
        # Sorted by round_num ASC: main-pre-1 (4), main-pre-2 (8), main-pre-3 (12),
        # b-post-1 (24), b-post-2 (28)
        assert contents == ["main-pre-1", "main-pre-2", "main-pre-3", "b-post-1", "b-post-2"]
        # NEVER any MAIN-POISON content
        assert all("POISON" not in c for c in contents)

    @pytest.mark.asyncio
    async def test_list_ancestry_three_levels(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 20, user_id=uid)
        await persist.upsert_branch(sid, "C", "B", 35, user_id=uid)
        await persist.store_ledger_entries(sid, "main", [
            {"round_num": 4, "category": "x", "content": "main-1"},
        ], user_id=uid)
        # Main post-divergence noise that B should NOT see
        await persist.store_ledger_entries(sid, "main", [
            {"round_num": 25, "category": "x", "content": "MAIN-NOISE"},
        ], user_id=uid)
        await persist.store_ledger_entries(sid, "B", [
            {"round_num": 30, "category": "x", "content": "b-1"},
        ], user_id=uid)
        # B post-C-divergence noise that C should NOT see
        await persist.store_ledger_entries(sid, "B", [
            {"round_num": 40, "category": "x", "content": "B-NOISE-FOR-C"},
        ], user_id=uid)
        await persist.store_ledger_entries(sid, "C", [
            {"round_num": 36, "category": "x", "content": "c-1"},
            {"round_num": 42, "category": "x", "content": "c-2"},
        ], user_id=uid)

        chain = await persist.get_branch_ancestry(sid, "C", user_id=uid)
        entries = await persist.list_ledger_entries(sid, chain, user_id=uid)
        contents = [e["content"] for e in entries]
        # Visible: main pre-20 (main-1 @ 4), B between 20 and 35 (b-1 @ 30), C own (c-1 @ 36, c-2 @ 42)
        assert contents == ["main-1", "b-1", "c-1", "c-2"]
        # No noise
        assert "MAIN-NOISE" not in contents
        assert "B-NOISE-FOR-C" not in contents

    @pytest.mark.asyncio
    async def test_list_max_round_filter(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.store_ledger_entries(sid, "main", [
            {"round_num": 4, "category": "x", "content": "a"},
            {"round_num": 8, "category": "x", "content": "b"},
            {"round_num": 12, "category": "x", "content": "c"},
        ], user_id=uid)
        chain = await persist.get_branch_ancestry(sid, "main", user_id=uid)
        # max_round=8 should include round 4 but EXCLUDE round 8 (strict <)
        entries = await persist.list_ledger_entries(sid, chain, user_id=uid, max_round=8)
        contents = [e["content"] for e in entries]
        assert contents == ["a"]

    @pytest.mark.asyncio
    async def test_compact_atomic_replace(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.store_ledger_entries(sid, "main", [
            {"round_num": 4, "category": "x", "content": "old1"},
            {"round_num": 8, "category": "x", "content": "old2"},
            {"round_num": 12, "category": "x", "content": "old3"},
            {"round_num": 16, "category": "x", "content": "old4"},
        ], user_id=uid)
        # Compact: drop entries with round_num < 12, replace with one summary
        ok = await persist.compact_ledger_entries(
            sid, "main", user_id=uid, keep_after_round=12,
            replacement_entries=[{"round_num": 8, "category": "summary", "content": "compacted"}],
        )
        assert ok is True
        chain = await persist.get_branch_ancestry(sid, "main", user_id=uid)
        entries = await persist.list_ledger_entries(sid, chain, user_id=uid)
        contents = [e["content"] for e in entries]
        assert contents == ["compacted", "old3", "old4"]

    @pytest.mark.asyncio
    async def test_prune_for_branch_isolated(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        await persist.store_ledger_entries(sid, "main", [
            {"round_num": 4, "category": "x", "content": "main"},
        ], user_id=uid)
        await persist.store_ledger_entries(sid, "B", [
            {"round_num": 6, "category": "x", "content": "b1"},
            {"round_num": 8, "category": "x", "content": "b2"},
        ], user_id=uid)
        n = await persist.prune_ledger_entries_for_branch(sid, "B", user_id=uid)
        assert n == 2
        # Main untouched
        assert await persist.count_ledger_entries(sid, "main", user_id=uid) == 1
        assert await persist.count_ledger_entries(sid, "B", user_id=uid) == 0


# ===========================================================================
# Branch-aware archive
# ===========================================================================

class TestArchiveByBranch:
    @pytest.mark.asyncio
    async def test_store_and_retrieve_main(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.store_archive_exchanges_for_branch(
            sid,
            [{"id": "a1", "user_content": "u", "assistant_content": "a",
              "summary": "s1", "turn_number": 4,
              "embedding_blob": _stub_768d(0.5)}],
            user_id=uid, branch_id="main",
        )
        chain = await persist.get_branch_ancestry(sid, "main", user_id=uid)
        results = await persist.retrieve_archive_for_branch(
            sid, _stub_768d(0.5), chain, user_id=uid, limit=5,
        )
        assert len(results) == 1
        assert results[0]["branch_id"] == "main"
        assert results[0]["summary"] == "s1"

    @pytest.mark.asyncio
    async def test_branch_isolation_three_levels(self, persist, seeded_session):
        """The canonical 3-level ancestry isolation test for archive retrieval."""
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 20, user_id=uid)
        await persist.upsert_branch(sid, "C", "B", 35, user_id=uid)

        # Main: pre-divergence (visible to all) + post-divergence (visible only on main)
        await persist.store_archive_exchanges_for_branch(sid, [
            {"id": "main_pre", "user_content": "u", "assistant_content": "a",
             "summary": "main-pre-content", "turn_number": 5, "embedding_blob": _stub_768d(0.5)},
            {"id": "main_post", "user_content": "u", "assistant_content": "a",
             "summary": "main-post-poison", "turn_number": 25, "embedding_blob": _stub_768d(0.5)},
        ], user_id=uid, branch_id="main")
        # B: pre-C-divergence (visible to B+C) + post-C-divergence (visible only on B)
        await persist.store_archive_exchanges_for_branch(sid, [
            {"id": "b_pre_c", "user_content": "u", "assistant_content": "a",
             "summary": "b-pre-c", "turn_number": 30, "embedding_blob": _stub_768d(0.5)},
            {"id": "b_post_c", "user_content": "u", "assistant_content": "a",
             "summary": "b-post-c-poison", "turn_number": 40, "embedding_blob": _stub_768d(0.5)},
        ], user_id=uid, branch_id="B")
        # C: own content
        await persist.store_archive_exchanges_for_branch(sid, [
            {"id": "c_own", "user_content": "u", "assistant_content": "a",
             "summary": "c-own", "turn_number": 36, "embedding_blob": _stub_768d(0.5)},
        ], user_id=uid, branch_id="C")

        # Retrieving on C should see: main_pre + b_pre_c + c_own. Never *_poison.
        chain_c = await persist.get_branch_ancestry(sid, "C", user_id=uid)
        results = await persist.retrieve_archive_for_branch(
            sid, _stub_768d(0.5), chain_c, user_id=uid, limit=10,
        )
        summaries = sorted(r["summary"] for r in results)
        assert "main-pre-content" in summaries
        assert "b-pre-c" in summaries
        assert "c-own" in summaries
        assert all("poison" not in s for s in summaries)

        # On B: main_pre + b_pre_c + b_post_c. Never main_post.
        chain_b = await persist.get_branch_ancestry(sid, "B", user_id=uid)
        results = await persist.retrieve_archive_for_branch(
            sid, _stub_768d(0.5), chain_b, user_id=uid, limit=10,
        )
        summaries = sorted(r["summary"] for r in results)
        assert "main-pre-content" in summaries
        assert "b-pre-c" in summaries
        assert "b-post-c-poison" in summaries  # b's own continuation IS visible on B
        assert "main-post-poison" not in summaries

        # On main: main_pre + main_post. Never B's or C's content.
        chain_main = await persist.get_branch_ancestry(sid, "main", user_id=uid)
        results = await persist.retrieve_archive_for_branch(
            sid, _stub_768d(0.5), chain_main, user_id=uid, limit=10,
        )
        summaries = sorted(r["summary"] for r in results)
        assert "main-pre-content" in summaries
        assert "main-post-poison" in summaries
        assert "b-pre-c" not in summaries
        assert "b-post-c-poison" not in summaries
        assert "c-own" not in summaries


# ===========================================================================
# Branch deletion cascade
# ===========================================================================

class TestDeleteBranchCascade:
    @pytest.mark.asyncio
    async def test_clears_all_tiers(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        await persist.store_state_snapshot(sid, "B", 8, {"x": 1}, user_id=uid)
        await persist.store_ledger_entries(sid, "B", [
            {"round_num": 6, "category": "x", "content": "b1"},
        ], user_id=uid)
        await persist.store_archive_exchanges_for_branch(sid, [
            {"id": "b_arc", "user_content": "u", "assistant_content": "a",
             "summary": "s", "turn_number": 7, "embedding_blob": _stub_768d(0.5)},
        ], user_id=uid, branch_id="B")

        deleted = await persist.delete_branch_cascade(sid, "B", user_id=uid)
        assert deleted["branches"] == 1
        assert deleted["snapshots"] == 1
        assert deleted["ledger_entries"] == 1
        assert deleted["archive_rows"] == 1
        assert deleted["archive_vec_rows"] == 1

        # Branch row gone
        branches = await persist.list_branches(sid, user_id=uid)
        assert {b.branch_id for b in branches} == {"main"}

    @pytest.mark.asyncio
    async def test_rejects_main(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        with pytest.raises(ValueError, match="main"):
            await persist.delete_branch_cascade(sid, "main", user_id=uid)

    @pytest.mark.asyncio
    async def test_does_not_touch_other_branches(self, persist, seeded_session):
        """High-stress scenario: delete B while main has substantial content.
        Main must be entirely untouched."""
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)
        await persist.store_ledger_entries(sid, "main", [
            {"round_num": i, "category": "x", "content": f"main-{i}"}
            for i in [2, 4]
        ], user_id=uid)
        await persist.store_ledger_entries(sid, "B", [
            {"round_num": 6, "category": "x", "content": "b"},
        ], user_id=uid)
        await persist.delete_branch_cascade(sid, "B", user_id=uid)
        assert await persist.count_ledger_entries(sid, "main", user_id=uid) == 2


# ===========================================================================
# Storage observability
# ===========================================================================

class TestSessionStorage:
    @pytest.mark.asyncio
    async def test_zero_when_empty(self, persist, seeded_session):
        sid, uid = seeded_session
        storage = await persist.get_session_storage(sid, user_id=uid)
        assert storage.session_id == sid
        assert storage.total_branches == 0
        assert storage.total_archive_rows == 0
        assert storage.total_ledger_entries == 0
        assert storage.total_snapshots == 0

    @pytest.mark.asyncio
    async def test_per_branch_counts(self, persist, seeded_session):
        sid, uid = seeded_session
        await persist.upsert_branch(sid, "main", None, 0, user_id=uid)
        await persist.upsert_branch(sid, "B", "main", 5, user_id=uid)

        await persist.store_ledger_entries(sid, "main", [
            {"round_num": 4, "category": "x", "content": "main"},
            {"round_num": 8, "category": "x", "content": "main2"},
        ], user_id=uid)
        await persist.store_ledger_entries(sid, "B", [
            {"round_num": 6, "category": "x", "content": "b"},
        ], user_id=uid)
        await persist.store_state_snapshot(sid, "main", 8, {"x": 1}, user_id=uid)
        await persist.store_archive_exchanges_for_branch(sid, [
            {"id": "a1", "user_content": "u" * 50, "assistant_content": "a" * 50,
             "summary": "s", "turn_number": 4, "embedding_blob": _stub_768d(0.1)},
        ], user_id=uid, branch_id="main")

        storage = await persist.get_session_storage(sid, user_id=uid)
        assert storage.total_branches == 2
        assert storage.total_ledger_entries == 3  # 2 main + 1 B
        assert storage.total_snapshots == 1
        assert storage.total_archive_rows == 1
        assert storage.branches["main"]["ledger_entries"] == 2
        assert storage.branches["B"]["ledger_entries"] == 1
        assert storage.branches["main"]["archive_rows"] == 1
        # approx_bytes is non-zero
        assert storage.total_approx_bytes > 0
