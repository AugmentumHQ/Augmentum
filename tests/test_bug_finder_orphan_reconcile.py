"""Tests for read-time reconciliation of orphaned bug_finder runs.

Pins the contract that ``BugFinderRunStore.mark_orphaned`` flips a
stuck ``running`` row to a terminal state idempotently, and that
already-terminal rows are left alone.

The route-side reconcile path (``_reconcile_orphan_run`` in
``bug_finder_routes.py``) is exercised indirectly here — the store
method is the load-bearing primitive, the route just decides when
to call it.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import aiosqlite
import pytest

from augmentum.bug_finder.store import BugFinderRunStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SCHEMA = """
CREATE TABLE IF NOT EXISTS bug_finder_runs (
    run_id              TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    job_id              TEXT,
    workspace_id        TEXT,
    git_url             TEXT,
    started_at          INTEGER,
    completed_at        INTEGER,
    stop_reason         TEXT,
    stop_detail         TEXT,
    containment_warning TEXT,
    findings_total       INTEGER,
    findings_confirmed   INTEGER,
    findings_fixed       INTEGER,
    findings_fix_failed  INTEGER,
    total_tokens_in      INTEGER,
    total_tokens_out     INTEGER,
    total_wallclock_ms   INTEGER,
    report_json          TEXT
);
CREATE TABLE IF NOT EXISTS bug_finder_findings (
    finding_id          TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    workspace_id        TEXT,
    file                TEXT,
    function            TEXT,
    line_start          INTEGER,
    line_end            INTEGER,
    claim               TEXT,
    claim_signature     TEXT,
    severity            TEXT,
    status              TEXT,
    runs_to_confirm     INTEGER,
    total_runs          INTEGER,
    families_to_confirm INTEGER,
    total_families      INTEGER,
    has_repro           INTEGER,
    has_patch           INTEGER,
    fix_attempts        INTEGER,
    detected_at         INTEGER
);
"""


async def _make_store(tmp_path: Path) -> tuple[BugFinderRunStore, aiosqlite.Connection]:
    db = tmp_path / "bf.db"
    conn = await aiosqlite.connect(str(db))
    await conn.executescript(_SCHEMA)
    await conn.commit()
    return BugFinderRunStore(conn), conn


# ---------------------------------------------------------------------------
# mark_orphaned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_orphaned_flips_running_row(tmp_path: Path) -> None:
    store, conn = await _make_store(tmp_path)
    try:
        await store.start_run(
            run_id="bfr_test_1", user_id="usr_a", job_id="job_1",
            git_url=None, workspace_id="ws_1",
        )
        before = await store.get_run("bfr_test_1", user_id="usr_a")
        assert before is not None
        assert before["stop_reason"] == "running"

        updated = await store.mark_orphaned(
            "bfr_test_1", user_id="usr_a",
            stop_reason="error",
            stop_detail="autoheal killed the process",
        )
        assert updated is True

        after = await store.get_run("bfr_test_1", user_id="usr_a")
        assert after is not None
        assert after["stop_reason"] == "error"
        assert after["stop_detail"] == "autoheal killed the process"
        # completed_at landed
        assert after.get("completed_at") is not None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mark_orphaned_is_idempotent_on_terminal_rows(
    tmp_path: Path,
) -> None:
    """An already-terminal row should NOT be re-stamped. The original
    writer (complete_run or a prior reconcile) knows more than this
    reactive heal-on-read does."""
    store, conn = await _make_store(tmp_path)
    try:
        # Hand-write a row that's already in 'complete' state, with a
        # canonical completed_at and stop_detail. The orphan-reconcile
        # path must not overwrite it.
        canonical_completed_at = 1_700_000_000
        canonical_stop_reason = "complete"
        canonical_stop_detail = "wrapped up cleanly"
        await conn.execute(
            """INSERT INTO bug_finder_runs
                (run_id, user_id, job_id, workspace_id,
                 started_at, completed_at,
                 stop_reason, stop_detail)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("bfr_done", "usr_a", "job_X", "ws_1",
             canonical_completed_at - 10, canonical_completed_at,
             canonical_stop_reason, canonical_stop_detail),
        )
        await conn.commit()

        updated = await store.mark_orphaned(
            "bfr_done", user_id="usr_a",
            stop_reason="error",
            stop_detail="we should NOT see this",
        )
        assert updated is False

        row = await store.get_run("bfr_done", user_id="usr_a")
        assert row is not None
        assert row["stop_reason"] == canonical_stop_reason
        assert row["stop_detail"] == canonical_stop_detail
        assert row["completed_at"] == canonical_completed_at
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mark_orphaned_scoped_to_user(tmp_path: Path) -> None:
    """A user can't accidentally heal a different user's stuck row."""
    store, conn = await _make_store(tmp_path)
    try:
        await store.start_run(
            run_id="bfr_alice", user_id="usr_alice", job_id="job_1",
            git_url=None, workspace_id="ws_1",
        )
        updated = await store.mark_orphaned(
            "bfr_alice", user_id="usr_eve",
            stop_reason="error", stop_detail="evil",
        )
        assert updated is False

        row = await store.get_run("bfr_alice", user_id="usr_alice")
        assert row is not None
        assert row["stop_reason"] == "running"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mark_orphaned_preserves_existing_completed_at(
    tmp_path: Path,
) -> None:
    """``COALESCE(completed_at, ?)`` keeps the original completion
    timestamp when one already exists. Edge case: a row could be in
    a weird state where completed_at is set but stop_reason is still
    'running' (e.g. a partial write earlier). The reconcile pass
    should still flip stop_reason but not bash the prior timestamp."""
    store, conn = await _make_store(tmp_path)
    try:
        original_completed_at = 1_750_000_000
        await conn.execute(
            """INSERT INTO bug_finder_runs
                (run_id, user_id, job_id, workspace_id,
                 started_at, completed_at, stop_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("bfr_partial", "usr_a", "job_1", "ws_1",
             original_completed_at - 100, original_completed_at, "running"),
        )
        await conn.commit()

        updated = await store.mark_orphaned(
            "bfr_partial", user_id="usr_a",
            stop_reason="error", stop_detail="reconciled late",
        )
        assert updated is True

        row = await store.get_run("bfr_partial", user_id="usr_a")
        assert row is not None
        assert row["stop_reason"] == "error"
        assert row["stop_detail"] == "reconciled late"
        # Original timestamp preserved by COALESCE
        assert row["completed_at"] == original_completed_at
    finally:
        await conn.close()
