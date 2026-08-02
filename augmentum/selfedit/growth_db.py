"""The isolated, durable store for the self-improvement system.

The growth system's own bookkeeping — the **never-pruned** ``self_edit_attempts``
archive and its run transcripts (``claude_runs`` / ``claude_run_events``) — lives
in a SEPARATE SQLite file (``<data>/selfedit/growth.db``) on its OWN connection,
so it can never lock, bloat, or corrupt the main ``augmentum.db``.

The framing the operator chose: **isolated AND durable, not disposable.** The
archive is sacred (rollback restores code, never the lesson), so it's backed up,
not wiped — only the verify/boot-smoke scratch DBs (``:memory:``) are throwaway.
This file is the blast-radius wall: a growth-DB problem stays in the growth DB.

Containment guarantees:
* opened **lazily** (first self-edit route hit), so a missing/corrupt growth DB
  never blocks app boot;
* its OWN connection + ``WAL`` + ``busy_timeout`` → no lock contention with main;
* a failure to open returns ``None`` and the route degrades with a clear error —
  the main app never sees it.

The main DB is still **read** for health probes (strain/services/integrity) and
holds the master-switch setting; the growth system only ever *writes* here.
"""

from __future__ import annotations

import os
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Self-contained schema (mirrors migrations 288 + 287, growth-owned copies).
# Plain INTEGER PRIMARY KEY (no AUTOINCREMENT footgun) for the events table.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS self_edit_attempts (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    objective       TEXT NOT NULL DEFAULT '',
    surface         TEXT NOT NULL DEFAULT '',
    target          TEXT NOT NULL DEFAULT '',
    tier            TEXT NOT NULL DEFAULT 'green',
    status          TEXT NOT NULL DEFAULT 'proposed',
    base_ref        TEXT NOT NULL DEFAULT '',
    candidate_ref   TEXT NOT NULL DEFAULT '',
    run_id          TEXT NOT NULL DEFAULT '',
    gate_passed     INTEGER NOT NULL DEFAULT 0,
    gate_verdict    TEXT NOT NULL DEFAULT '{}',
    files_changed   TEXT NOT NULL DEFAULT '[]',
    outcome         TEXT NOT NULL DEFAULT '',
    lesson          TEXT NOT NULL DEFAULT '',
    promoted_commit TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT 'autonomous',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_self_edit_user_created
    ON self_edit_attempts(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_self_edit_status
    ON self_edit_attempts(user_id, status);

CREATE TABLE IF NOT EXISTS claude_runs (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    workspace_id  TEXT NOT NULL,
    session_id    TEXT NOT NULL DEFAULT '',
    task          TEXT NOT NULL DEFAULT '',
    permission    TEXT NOT NULL DEFAULT 'auto',
    status        TEXT NOT NULL DEFAULT 'running',
    outcome       TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    files_changed TEXT NOT NULL DEFAULT '[]',
    raw_jsonl     TEXT NOT NULL DEFAULT '',
    cost_usd      REAL NOT NULL DEFAULT 0,
    num_turns     INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER NOT NULL DEFAULT 0,
    resumed_from  TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_claude_runs_user_ws
    ON claude_runs(user_id, workspace_id, created_at DESC);

CREATE TABLE IF NOT EXISTS claude_run_events (
    id         INTEGER PRIMARY KEY,
    run_id     TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    kind       TEXT NOT NULL DEFAULT '',
    text       TEXT NOT NULL DEFAULT '',
    tool       TEXT NOT NULL DEFAULT '',
    path       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_claude_run_events_run
    ON claude_run_events(run_id, seq);

-- The learning loop: per-user keep/revert tallies per change-shape. A shape earns
-- trust by accumulation across real verdicts (the archive becomes judgment).
CREATE TABLE IF NOT EXISTS self_edit_preferences (
    user_id       TEXT NOT NULL,
    shape         TEXT NOT NULL,
    kept          INTEGER NOT NULL DEFAULT 0,
    reverted      INTEGER NOT NULL DEFAULT 0,
    last_decision TEXT NOT NULL DEFAULT '',
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, shape)
);
"""


# Columns added to self_edit_attempts after migration 288 — applied idempotently
# to an EXISTING growth.db (the `CREATE TABLE IF NOT EXISTS` above is a no-op once
# the table exists, so additive fields must be ALTER-ed in). name → column DDL.
_ADDED_COLUMNS = {
    # the structured debt class the attempt targeted (``scanner.metric``) — gives
    # the verified skill graph a `target:` atom so it speaks the debt pipeline's
    # vocabulary natively, not only file/surface regions. Reversible: drop and the
    # graph falls back to the columns it already folded.
    "target": "ALTER TABLE self_edit_attempts ADD COLUMN target TEXT NOT NULL DEFAULT ''",
    # where the attempt came from: ``autonomous`` (the engine's own loop, the
    # default so every pre-existing row keeps its meaning), ``git`` (an ingested
    # commit from the live repo's history), ``coder`` (an applied coder-mode
    # turn). Ingest-all-work: the archive stops being a diary of the engine's
    # own outings and becomes the memory of ALL work on the system. Consumers
    # (activation fold, retrodiction, Workshop) weight/filter by this tag —
    # see activation._SOURCE_WEIGHT. Reversible: drop it and every row reads as
    # 'autonomous' again, exactly the pre-ingest world.
    "source": ("ALTER TABLE self_edit_attempts ADD COLUMN source "
               "TEXT NOT NULL DEFAULT 'autonomous'"),
}


async def _ensure_columns(conn: Any) -> None:
    """Add any post-288 columns missing from an existing self_edit_attempts table.

    Cheap (a single PRAGMA) and idempotent — a fresh DB already has them from
    ``_SCHEMA`` and this is a no-op; an older DB gains them without losing a row."""
    try:
        cur = await conn.execute("PRAGMA table_info(self_edit_attempts)")
        cols = {r[1] for r in await cur.fetchall()}
        for name, ddl in _ADDED_COLUMNS.items():
            if name not in cols:
                await conn.execute(ddl)
                log.info("growth_db_column_added", column=name)
    except Exception as exc:  # noqa: BLE001 — contained; never blocks open
        log.warning("growth_db_ensure_columns_failed", error=repr(exc))


def growth_db_path(app_state: Any) -> str:
    """``<data_dir>/selfedit/growth.db``, derived from the main DB's directory so
    it lands on the same always-RW ``/data`` volume."""
    backend = getattr(getattr(app_state, "state_manager", None), "backend", None)
    db_path = getattr(backend, "_db_path", None) or "/data/augmentum.db"
    data_dir = os.path.dirname(db_path) or "/data"
    return os.path.join(data_dir, "selfedit", "growth.db")


async def get_growth_conn(app_state: Any) -> Any | None:
    """Lazily open + cache the isolated growth-DB connection. Returns ``None`` if
    it can't be opened — the caller degrades; the main app is never affected."""
    conn = getattr(app_state, "_growth_conn", None)
    if conn is not None:
        return conn
    try:
        import aiosqlite
        path = growth_db_path(app_state)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = await aiosqlite.connect(path)
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.executescript(_SCHEMA)
        await _ensure_columns(conn)
        await conn.commit()
        app_state._growth_conn = conn
        log.info("growth_db_opened", path=path)
        # Make the docstring's promise true: the archive is the labeled
        # benchmark the whole recursion rests on, and it's the ONE store not
        # covered by the main-DB startup backup. Snapshot it now (best-effort,
        # interval-gated to ~1/hour like the main DB) — a VACUUM INTO on the
        # freshly-opened connection, before any writes land this session.
        await backup_growth_db(conn, path)
        return conn
    except Exception as exc:  # noqa: BLE001 — contained: main app unaffected
        log.warning("growth_db_open_failed", error=repr(exc))
        return None


async def backup_growth_db(conn: Any, path: str) -> str | None:
    """VACUUM INTO snapshot of the growth DB into ``<data>/selfedit/backups/``,
    reusing the main DB's proven backup path (online-safe, self-contained, 7-file
    rotation, ~1/hour interval gate). Best-effort: a backup failure is logged and
    swallowed — losing a backup must never break opening the archive.

    This closes the gap where ``growth.db`` — the never-pruned archive, the
    preference store, and the skill-graph substrate — had NO copy anywhere while
    the main DB was snapshotted every startup. A disk fault would have taken the
    entire labeled benchmark with no recovery."""
    try:
        from augmentum.state.backup import (
            backup_database,
            rotate_backups,
            should_skip_startup_backup,
        )
        if should_skip_startup_backup(path):
            log.debug("growth_db_backup_skipped_recent", path=path)
            return None
        result = await backup_database(conn, path)
        if result:
            rotate_backups(os.path.join(os.path.dirname(path), "backups"))
        return result
    except Exception as exc:  # noqa: BLE001 — contained: never blocks the archive
        log.warning("growth_db_backup_failed", error=repr(exc))
        return None


async def close_growth_conn(app_state: Any) -> None:
    """Close the growth connection (shutdown). Best-effort."""
    conn = getattr(app_state, "_growth_conn", None)
    if conn is None:
        return
    try:
        await conn.close()
    except Exception as exc:  # noqa: BLE001
        log.warning("growth_db_close_failed", error=repr(exc))
    app_state._growth_conn = None
