"""Canonical narrative-data cleanup module.

Single source of truth for removing all narrative-tier data for a session.
Called by:
  - chat-delete endpoint (chat_routes.py)
  - account-deletion path (when implemented)
  - test fixtures cleanup
  - delete_branch_cascade in NarrativePersistence (per-branch slice)

Layered against the chat_routes.py inline ``_SESSION_CLEANUP_TABLES`` frozenset
which still owns NON-narrative tables (lorebook_entries, character_cards, etc.).
The two cleanup paths are intentionally independent so that adding a non-narrative
user-scoped table never requires touching this file.

The audit script (.claude/skills/augmentum-dev/scripts/audit.py) verifies that
every table with a ``session_id`` column appears in either this module's
``NARRATIVE_TABLES`` constant or chat_routes.py's frozenset — drift is caught
at smoke-test time, not at delete time.

Belt-and-suspenders deletion:
  1. ``DELETE FROM sessions`` in chat_routes triggers FK CASCADE on tables
     that have CASCADE configured (eventually all of them).
  2. ``purge_narrative_session()`` is the explicit fallback that handles
     the vec virtual table (FKs don't apply) plus any table that hasn't
     been migrated to CASCADE FK yet.
  3. Post-delete verification SELECT runs against every touched table and
     logs ``cleanup_residue`` warnings on non-zero counts.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Canonical list of narrative tables that hold per-session data.
# AUDIT NOTE: keep in sync with the migrations that introduce session_id
# columns; the audit script enforces coverage against migration analysis.
NARRATIVE_TABLES: tuple[str, ...] = (
    "narrative_archive",
    "narrative_state_snapshots",
    "narrative_ledger_entries",
    "narrative_branches",
    "narrative_memory",
)

# Migration markers stored in app_settings as 'narrative_branch_migration:{session_id}'.
_MARKER_PREFIX = "narrative_branch_migration:"


@dataclass
class CleanupReport:
    """Per-tier row counts from a narrative-session cleanup.

    Attributes:
      session_id, user_id: identification
      *_rows: count of rows deleted per tier
      migration_markers: count of app_settings rows deleted
      duration_ms: total cleanup wall-clock time
      residue: {table_name: remaining_count} — populated only if the
        post-delete verification finds rows that should have been removed.
        Empty on healthy delete.
      ok: True iff cleanup completed without exception AND residue is empty.
    """
    session_id: str = ""
    user_id: str = ""
    archive_rows: int = 0
    archive_vec_rows: int = 0
    state_snapshot_rows: int = 0
    ledger_entry_rows: int = 0
    branch_rows: int = 0
    memory_rows: int = 0
    migration_markers: int = 0
    duration_ms: float = 0.0
    residue: dict[str, int] = field(default_factory=dict)
    ok: bool = True

    def to_event_kwargs(self) -> dict:
        """Flatten into structlog-friendly kwargs for ``log.info``."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "archive_rows": self.archive_rows,
            "archive_vec_rows": self.archive_vec_rows,
            "state_snapshot_rows": self.state_snapshot_rows,
            "ledger_entry_rows": self.ledger_entry_rows,
            "branch_rows": self.branch_rows,
            "memory_rows": self.memory_rows,
            "migration_markers": self.migration_markers,
            "duration_ms": round(self.duration_ms, 2),
            "residue_tables": list(self.residue.keys()),
            "ok": self.ok,
        }


async def purge_narrative_session(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    user_id: str,
) -> CleanupReport:
    """Atomically remove all narrative data for one session.

    Wraps everything in BEGIN IMMEDIATE / COMMIT so partial failures roll back.
    Vec rows are deleted FIRST (their JOIN target is the archive table); the
    branches row is deleted LAST (it's the parent for snapshots/ledger/archive
    via branch_id, though no FK is declared — order is defensive against
    future schema migrations adding cascades).

    On exception, returns a report with ``ok=False`` and best-effort row counts.
    The caller logs the report regardless of success so anomalies surface.

    Idempotent: re-running on an already-purged session returns zeros.
    """
    if not user_id:
        raise ValueError("purge_narrative_session requires user_id")

    started = time.perf_counter()
    report = CleanupReport(session_id=session_id, user_id=user_id)

    try:
        # aiosqlite creates an implicit transaction on the first DML; an
        # explicit BEGIN IMMEDIATE is only safe when no caller has already
        # started one. chat_routes.py runs DELETE FROM ui_sessions before
        # invoking us, which puts the connection in deferred-transaction
        # state — an explicit BEGIN there would raise. The implicit
        # transaction provides atomicity via the final commit (or rollback
        # on exception), which is what we actually need.
        # 1. archive_vec rows (joined by id with narrative_archive)
        cursor = await conn.execute(
            "DELETE FROM narrative_archive_vec WHERE id IN ("
            "SELECT id FROM narrative_archive "
            "WHERE session_id = ? AND user_id = ?)",
            (session_id, user_id),
        )
        report.archive_vec_rows = cursor.rowcount

        # 2. archive rows
        cursor = await conn.execute(
            "DELETE FROM narrative_archive WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        report.archive_rows = cursor.rowcount

        # 3. state snapshots
        cursor = await conn.execute(
            "DELETE FROM narrative_state_snapshots "
            "WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        report.state_snapshot_rows = cursor.rowcount

        # 4. ledger entries
        cursor = await conn.execute(
            "DELETE FROM narrative_ledger_entries "
            "WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        report.ledger_entry_rows = cursor.rowcount

        # 5. branches (parent for the rest via branch_id)
        cursor = await conn.execute(
            "DELETE FROM narrative_branches "
            "WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        report.branch_rows = cursor.rowcount

        # 6. memory row (the JSON-blob legacy storage)
        cursor = await conn.execute(
            "DELETE FROM narrative_memory "
            "WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        report.memory_rows = cursor.rowcount

        # 7. migration markers in app_settings
        cursor = await conn.execute(
            "DELETE FROM app_settings WHERE key = ?",
            (f"{_MARKER_PREFIX}{session_id}",),
        )
        report.migration_markers = cursor.rowcount

        await conn.commit()
    except Exception:
        report.ok = False
        log.warning("purge_narrative_session_failed",
                    session_id=session_id, user_id=user_id, exc_info=True)
        try:
            await conn.rollback()
        except Exception as rb_exc:
            log.debug(
                "purge_narrative_session_rollback_failed",
                session_id=session_id,
                error=str(rb_exc),
            )
        report.duration_ms = (time.perf_counter() - started) * 1000.0
        return report

    # Post-delete verification — runs in its own (read-only) implicit txn.
    # Any row remaining is a structural bug: either the WHERE clause was wrong,
    # a CASCADE backstabbed us with stale orphans, or another writer slipped
    # in mid-purge. Log per-table residue counts.
    try:
        residue: dict[str, int] = {}
        for table in NARRATIVE_TABLES:
            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            row = await cursor.fetchone()
            count = int(row[0]) if row else 0
            if count > 0:
                residue[table] = count
        # archive_vec residue: rows whose id is NOT in narrative_archive but were
        # tagged with this session previously. Detected via lack of join target.
        # This catches the pre-existing orphan-vector leak even after our explicit
        # DELETE — anything remaining here is a deeper issue worth surfacing.
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM narrative_archive_vec v "
            "WHERE NOT EXISTS (SELECT 1 FROM narrative_archive na WHERE na.id = v.id)",
        )
        row = await cursor.fetchone()
        # NOTE: this is a global orphan count, not session-scoped (vec rows
        # have no session_id). Only logged when a global orphan exists, since
        # session-scoping it would require joining a deleted parent.
        global_orphans = int(row[0]) if row else 0
        if global_orphans > 0:
            residue["narrative_archive_vec_global_orphans"] = global_orphans

        if residue:
            report.residue = residue
            report.ok = False
            log.warning("cleanup_residue", session_id=session_id,
                        user_id=user_id, residue=residue)
    except Exception:
        log.warning("purge_verification_failed",
                    session_id=session_id, exc_info=True)

    report.duration_ms = (time.perf_counter() - started) * 1000.0
    return report
