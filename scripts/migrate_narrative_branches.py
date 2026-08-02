"""Stage 2 of the narrative-branch migration.

Migrations 115-119 (Stage 1) seed every existing session's 'main' branch and
unpack its current STATE/LEDGER into the new tables. This script (Stage 2)
unpacks ALTERNATE branches stored as JSON in narrative_memory.branch_states.

Idempotent: per-session marker rows in app_settings track completion. Re-running
on already-migrated sessions returns a no-op. Per-session errors are logged
but do not abort the run; failed sessions are eligible for retry.

Usage (from repo root):

    python scripts/migrate_narrative_branches.py [--db /path/to/state.db]

The default DB path follows the augmentum convention: data/augmentum.db.

The script does NOT modify or delete the legacy branch_states JSON column. That
column stays as a deprecated read-only fallback through one release window;
migration 6 (later) drops it.

Legacy branch_states schema (from engine.py:1093 _save_branch_state):
    {
      "<branch_id>": {
        "message_count": int,
        "message_history": list,
        "state_snapshot": dict | None,
        "memory_ledger": list of {round_num, category, content},
        "pre_refresh_ledger_len": int,
        "memory_summary": str,
        "last_summary_at": int,
        "facts": list, "contradictions": list,
        "entities": dict, "relationships": list,
        "archive_history_idx": int
      },
      ...
    }

Note: the legacy blob does NOT store branch_point. This script registers
alternate branches with branch_point=0 + parent='main'. The exact divergence
point is reconstructed on next branch entry by branch_tracker.detect_branch().
This is best-effort migration — STATE and LEDGER content is preserved without loss;
metadata may be slightly imprecise until next user interaction with the branch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import aiosqlite


_MARKER_PREFIX = "narrative_branch_migration:"


def _make_id() -> str:
    return uuid.uuid4().hex


async def marker_exists(conn: aiosqlite.Connection, session_id: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM app_settings WHERE key = ? LIMIT 1",
        (f"{_MARKER_PREFIX}{session_id}",),
    )
    return (await cursor.fetchone()) is not None


async def set_marker(
    conn: aiosqlite.Connection,
    session_id: str,
    payload: dict,
) -> None:
    await conn.execute(
        "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
        (f"{_MARKER_PREFIX}{session_id}", json.dumps(payload)),
    )
    await conn.commit()


async def iterate_sessions_with_branch_states(
    conn: aiosqlite.Connection,
):
    """Yield (session_id, user_id, blob_json) for every session with non-empty
    branch_states. Empty / null / '{}' blobs are skipped."""
    cursor = await conn.execute(
        """SELECT session_id, user_id, branch_states FROM narrative_memory
            WHERE branch_states IS NOT NULL
              AND branch_states != ''
              AND branch_states != 'null'
              AND branch_states != '{}'"""
    )
    rows = await cursor.fetchall()
    for r in rows:
        yield r[0], (r[1] or ""), r[2]


async def migrate_session(
    conn: aiosqlite.Connection,
    session_id: str,
    user_id: str,
    blob_json: str,
) -> dict:
    """Unpack one session's branch_states_data into rows.

    Returns a per-session report dict.
    """
    report = {
        "session_id": session_id,
        "branches_inserted": 0,
        "snapshots_inserted": 0,
        "ledger_entries_inserted": 0,
        "branches_skipped_main": 0,
        "branches_skipped_invalid": 0,
        "branches_skipped_existing": 0,
        "errors": 0,
    }

    if not user_id:
        # Defensive: legacy data without user_id is anomalous but not impossible.
        # We still migrate but log; downstream queries that filter by user_id
        # will simply not return these rows for any user.
        print(f"  [warn] session {session_id}: empty user_id in narrative_memory")

    try:
        blob = json.loads(blob_json)
    except (json.JSONDecodeError, TypeError):
        print(f"  [error] session {session_id}: invalid branch_states JSON, skipping")
        report["errors"] = 1
        return report

    if not isinstance(blob, dict):
        return report

    for branch_id, saved in blob.items():
        if not isinstance(saved, dict):
            report["branches_skipped_invalid"] += 1
            continue
        if branch_id == "main":
            # Migration 119 already seeded main; do NOT double-insert.
            report["branches_skipped_main"] += 1
            continue

        # 1. Branch row. branch_point=0 best-effort (legacy blob doesn't store it).
        # parent='main' is the only safe default (legacy didn't track parent
        # graphs either; everything was effectively a fork off main).
        cursor = await conn.execute(
            """INSERT OR IGNORE INTO narrative_branches
               (branch_id, session_id, parent_branch_id, branch_point, status, user_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (branch_id, session_id, "main", 0, "active", user_id),
        )
        if cursor.rowcount == 0:
            # Branch already exists in new table — skip its content too to
            # avoid duplicate entries (idempotency safety).
            report["branches_skipped_existing"] += 1
            continue
        report["branches_inserted"] += 1

        # 2. State snapshot — saved.state_snapshot might be None or a dict
        snap = saved.get("state_snapshot")
        msg_count = int(saved.get("message_count", 0))
        if snap and isinstance(snap, dict) and snap:
            await conn.execute(
                """INSERT INTO narrative_state_snapshots
                   (id, session_id, branch_id, message_index, snapshot_data, user_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (_make_id(), session_id, branch_id, msg_count,
                 json.dumps(snap), user_id),
            )
            report["snapshots_inserted"] += 1

        # 3. Ledger entries
        ledger = saved.get("memory_ledger")
        if isinstance(ledger, list):
            for entry in ledger:
                if not isinstance(entry, dict):
                    continue
                await conn.execute(
                    """INSERT INTO narrative_ledger_entries
                       (id, session_id, branch_id, round_num, category, content, user_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (_make_id(), session_id, branch_id,
                     int(entry.get("round_num", 0)),
                     str(entry.get("category", "")),
                     str(entry.get("content", "")),
                     user_id),
                )
                report["ledger_entries_inserted"] += 1

    await conn.commit()
    return report


async def run_migration(db_path: Path) -> dict:
    """Top-level: walk every session with branch_states_data, migrate, set markers."""
    started = time.perf_counter()
    summary = {
        "sessions_migrated": 0,
        "sessions_skipped": 0,
        "sessions_failed": 0,
        "branches_inserted": 0,
        "snapshots_inserted": 0,
        "ledger_entries_inserted": 0,
        "duration_s": 0.0,
    }

    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.execute("PRAGMA foreign_keys = ON")

        sessions: list[tuple[str, str, str]] = []
        async for s in iterate_sessions_with_branch_states(conn):
            sessions.append(s)

        print(f"Found {len(sessions)} session(s) with legacy branch_states data")

        for session_id, user_id, blob_json in sessions:
            if await marker_exists(conn, session_id):
                summary["sessions_skipped"] += 1
                continue

            try:
                rep = await migrate_session(conn, session_id, user_id, blob_json)
                if rep.get("errors", 0):
                    summary["sessions_failed"] += 1
                else:
                    summary["sessions_migrated"] += 1
                    summary["branches_inserted"] += rep["branches_inserted"]
                    summary["snapshots_inserted"] += rep["snapshots_inserted"]
                    summary["ledger_entries_inserted"] += rep["ledger_entries_inserted"]
                    await set_marker(conn, session_id, {
                        "branches": rep["branches_inserted"],
                        "snapshots": rep["snapshots_inserted"],
                        "ledger_entries": rep["ledger_entries_inserted"],
                        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
            except Exception as exc:
                summary["sessions_failed"] += 1
                print(f"  [error] session {session_id}: {exc}")

    summary["duration_s"] = round(time.perf_counter() - started, 3)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/augmentum.db"),
        help="Path to SQLite DB (default: data/augmentum.db)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"ERROR: database file not found: {args.db}", file=sys.stderr)
        return 2

    summary = asyncio.run(run_migration(args.db))
    print()
    print("Migration summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if summary["sessions_failed"] > 0:
        print(f"\n{summary['sessions_failed']} session(s) failed. "
              f"Re-run to retry — marker-based idempotency will skip succeeded ones.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
