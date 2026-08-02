"""Startup sweep finalizing orphaned run-ledger rows.

Server restart kills every in-flight asyncio task by definition (see
``coder/run_broker.py``), but several run-ledger tables only flip out of
``status='running'`` from inside those tasks — so a crash or restart
strands rows that look running forever. The live DB had accumulated 82
zombie ``coder_turn_runs`` plus stragglers in ``claude_runs`` and
``xr_sessions`` when this landed (2026-07-18).

``background_jobs`` is deliberately NOT swept here — it has its own
restart story (``JobsStore.requeue_crashed`` re-queues instead of
finalizing, because jobs are resumable; ledger runs are not).

Runs exactly once at boot, before any new run can be created, so every
``running`` row it sees is guaranteed dead — no age heuristics needed.
Each table keeps its own status vocabulary and timestamp convention
(coder_turn_runs stores epoch REALs; claude/pi/xr store
``datetime('now')`` strings).
"""

from __future__ import annotations

import time

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# (table, WHERE clause for dead rows, SET clause, params factory)
_SWEEPS: tuple[tuple[str, str, str, object], ...] = (
    (
        "coder_turn_runs",
        "status = 'running'",
        "status = 'error', finish_reason = 'interrupted_by_restart', "
        "completed_at = ?, updated_at = ?",
        lambda: (time.time(), time.time()),
    ),
    (
        "claude_runs",
        "status = 'running'",
        "status = 'failed', error = 'interrupted by server restart', "
        "updated_at = datetime('now')",
        lambda: (),
    ),
    (
        "pi_runs",
        "status = 'running'",
        "status = 'failed', error = 'interrupted by server restart', "
        "updated_at = datetime('now')",
        lambda: (),
    ),
    (
        "xr_sessions",
        "status IN ('running', 'preflight')",
        "status = 'ended', updated_at = datetime('now')",
        lambda: (),
    ),
)


async def finalize_orphan_runs(conn: aiosqlite.Connection) -> dict[str, int]:
    """Finalize rows stranded in a non-terminal status by a dead process.

    Returns {table: rows_finalized} for the log line. Per-table failures
    are isolated — one missing table (partial migration) must not stop
    the rest of the sweep.
    """
    finalized: dict[str, int] = {}
    for table, where, sets, params in _SWEEPS:
        try:
            cursor = await conn.execute(
                f"UPDATE {table} SET {sets} WHERE {where}",  # noqa: S608 — constants above
                params(),
            )
            await conn.commit()
            if cursor.rowcount:
                finalized[table] = cursor.rowcount
        except Exception as exc:
            log.warning(
                "run_ledger_sweep_table_failed", table=table, error=str(exc),
            )
    if finalized:
        log.info("run_ledger_orphans_finalized", **finalized)
    return finalized
