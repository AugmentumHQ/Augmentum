"""SQLite state backend with WAL mode and migration support."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import re
import sqlite3
import subprocess
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Diagnostic timing on the shared aiosqlite connection. When the loop wedges
# we want one log line per slow query saying *exactly* what SQL is running,
# how long it took, and where in augmentum it was called from. Gate via env
# so it can be turned off without a code change.
#
# Threshold: 400ms. The original 100ms was a deliberately-aggressive setting
# while the 2026-05-11 lock-storm was being root-caused (cause: rich
# show_locals traceback rendering blocking the loop — fixed). On a WSL2 /
# Docker Desktop volume even a single-row SELECT routinely takes 100-170ms
# of pure storage latency under any concurrent load, so 100ms drowned the
# log in noise. 400ms still surfaces genuinely slow ops (the json_each
# media-genres scan, multi-second startup VACUUMs, real lock waits) without
# the floor noise. Override via AUGMENTUM_SLOW_QUERY_MS to tighten when
# actively debugging.
_SLOW_QUERY_MS = float(os.environ.get("AUGMENTUM_SLOW_QUERY_MS", "400"))
_SLOW_QUERY_LOG_ENABLED = os.environ.get(
    "AUGMENTUM_SLOW_QUERY_LOG", "1",
).lower() not in ("0", "false", "no", "off", "")
_QUERY_TRACE_LIMIT = 8  # frames in the cheap caller trace


def _augmentum_caller_trace() -> str:
    """Cheap caller trace — ``file:line:func`` for the closest few augmentum
    frames above us. Used so a slow-query log points at WHICH call site is
    holding the connection without dumping a full traceback every time.
    """
    frames: list[str] = []
    for fr in traceback.extract_stack(limit=_QUERY_TRACE_LIMIT + 6)[-_QUERY_TRACE_LIMIT - 6:]:
        # Only keep frames inside augmentum.* — drop aiosqlite/asyncio/uvicorn
        # internals which are noise.
        if "/augmentum/" not in fr.filename and "\\augmentum\\" not in fr.filename:
            continue
        # Strip the long absolute prefix; keep the package-relative path.
        path = fr.filename.replace("\\", "/")
        idx = path.find("/augmentum/")
        rel = path[idx + 1:] if idx >= 0 else path
        frames.append(f"{rel}:{fr.lineno}:{fr.name}")
    if not frames:
        return ""
    # Most-recent frame last, so a reader sees the call chain top-down.
    return " <- ".join(reversed(frames[-_QUERY_TRACE_LIMIT:]))


# Cross-connection registry of currently-held write transactions.
# Keyed by ``id(connection)`` so each aiosqlite handle is distinct.
# Value: ``(caller_trace, started_monotonic, sql_snippet)``.
#
# Populated when a ``BEGIN`` / ``BEGIN IMMEDIATE`` / ``BEGIN EXCLUSIVE``
# successfully resolves; cleared on ``COMMIT`` / ``ROLLBACK``. Lets a
# slow-op log on connection A surface that connection B is the actual
# holder, which is what we couldn't see during the
# ``resource_snapshot_persist_failed`` storm 2026-05-15 (every retry
# logged the waiter's stack but nothing about who owned the writer).
_ACTIVE_BEGINS: dict[int, tuple[str, float, str]] = {}


def _begins_summary(exclude_conn_id: int | None = None) -> str:
    """Return a short ``conn_id=caller(age_ms)`` list of all currently-
    held BEGINs except the caller's own connection. Empty string when
    none are active — keeps the log line tight on the fast path.
    """
    if not _ACTIVE_BEGINS:
        return ""
    now = time.monotonic()
    parts: list[str] = []
    for cid, (caller, started, _sql) in _ACTIVE_BEGINS.items():
        if cid == exclude_conn_id:
            continue
        age_ms = round((now - started) * 1000.0)
        # Last frame only — full chain is in the holder's own slow_db_op
        # line if it ever finishes, and we want this summary one line.
        tail = caller.rsplit(" <- ", 1)[-1] if caller else "?"
        parts.append(f"conn={cid:x}:{tail}(age_ms={age_ms})")
    return " | ".join(parts)


def _install_query_timing(conn: aiosqlite.Connection) -> None:
    """Wrap ``execute`` / ``executemany`` / ``commit`` so any operation that
    takes longer than ``_SLOW_QUERY_MS`` logs a structured warning.

    aiosqlite's ``execute`` returns a dual-protocol object: callers may
    ``await conn.execute(...)`` (resolves to a Cursor) OR use
    ``async with conn.execute(...) as cursor:`` (also yields the Cursor).
    Both forms are in active use across the codebase, so the wrapper has
    to support both protocols without breaking either. We do that with a
    small adapter class that delegates to the inner aiosqlite return value
    on both ``__await__`` and ``__aenter__``/``__aexit__``.

    Logged fields:
      sql        — first 200 chars of the SQL text (truncated)
      elapsed_ms — wall-clock time the operation took
      caller     — ``file:line:func`` chain inside augmentum/ that issued it
      holders    — other connections' currently-held BEGINs (BEGIN waiters only)

    A slow op here means: aiosqlite's worker thread spent that long handling
    one operation. While it's running, every other ``await conn.execute(...)``
    on the same connection waits in queue.
    """
    if not _SLOW_QUERY_LOG_ENABLED:
        return

    real_execute = conn.execute
    real_executemany = conn.executemany
    real_commit = conn.commit
    conn_id = id(conn)

    def _maybe_log(label: str, sql: str | None, started: float, params_count: int = 0) -> None:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms < _SLOW_QUERY_MS:
            return
        snippet = (sql or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        # Whole-DB maintenance pragmas (integrity/quick/foreign-key check) scan
        # every page by design, so they're inherently O(db size) — seconds on a
        # large DB is normal, not a contention symptom. The startup health check
        # (_post_startup_health_check) runs quick_check once per boot and was
        # tripping this as a multi-second slow_db_op warning every restart.
        # Surface it at info so the timing is still visible without crying wolf.
        upper = snippet.lstrip().upper()
        is_maintenance_scan = upper.startswith("PRAGMA") and any(
            k in upper for k in ("INTEGRITY_CHECK", "QUICK_CHECK", "FOREIGN_KEY_CHECK")
        )
        # When the slow op is a BEGIN/COMMIT — i.e. lock contention,
        # not a slow SELECT — name the other connections holding open
        # write transactions so the next investigator doesn't have to
        # add instrumentation again.
        head = upper[:6]
        holders = _begins_summary(exclude_conn_id=conn_id) if head.startswith("BEGIN") or head == "COMMIT" else ""
        _emit = log.info if is_maintenance_scan else log.warning
        _emit(
            "slow_db_op",
            op=label,
            sql=snippet,
            elapsed_ms=round(elapsed_ms, 1),
            param_rows=params_count,
            caller=_augmentum_caller_trace(),
            holders=holders,
        )

    def _track_begin(sql: str | None, started: float) -> None:
        """Record this connection as holding a write transaction.

        Called from the timed wrapper after a ``BEGIN*`` resolves (so
        the writer lock has actually been acquired). Cleared by
        ``_track_end`` on ``COMMIT``/``ROLLBACK``.
        """
        if not sql:
            return
        head = sql.lstrip().upper()[:6]
        if head.startswith("BEGIN"):
            snippet = sql.strip().replace("\n", " ")[:80]
            _ACTIVE_BEGINS[conn_id] = (_augmentum_caller_trace(), started, snippet)

    def _track_end(sql: str | None) -> None:
        if not sql:
            return
        head = sql.lstrip().upper()[:8]
        if head.startswith("COMMIT") or head.startswith("ROLLBACK"):
            _ACTIVE_BEGINS.pop(conn_id, None)

    class _TimedExecute:
        """Dual-protocol wrapper: ``await`` OR ``async with``.

        Mirrors aiosqlite's ``_ContextManagerMixin`` behaviour by delegating
        to the inner awaitable/cm on both protocols. Timing fires on the
        first protocol path that completes (await OR aexit) — never both.
        """
        __slots__ = ("_inner", "_sql", "_started", "_label", "_param_rows", "_logged")

        def __init__(self, inner, sql, started, label, param_rows=0):
            self._inner = inner
            self._sql = sql
            self._started = started
            self._label = label
            self._param_rows = param_rows
            self._logged = False

        def _log_once(self) -> None:
            if not self._logged:
                self._logged = True
                _maybe_log(self._label, self._sql, self._started, self._param_rows)

        def __await__(self):
            try:
                result = yield from self._inner.__await__()
                # BEGIN/COMMIT/ROLLBACK only register their lock-holder
                # state on SUCCESSFUL completion. If the await raises
                # (e.g. database is locked on BEGIN IMMEDIATE), no
                # writer lock was ever acquired so don't register.
                _track_begin(self._sql, self._started)
                _track_end(self._sql)
                return result
            finally:
                self._log_once()

        async def __aenter__(self):
            # Don't log on enter — wait for exit so the cursor's full
            # lifetime (including fetchall) is captured if the caller does
            # work between aenter and aexit.
            result = await self._inner.__aenter__()
            _track_begin(self._sql, self._started)
            return result

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            try:
                return await self._inner.__aexit__(exc_type, exc_val, exc_tb)
            finally:
                _track_end(self._sql)
                self._log_once()

    def execute_timed(sql, parameters=None, *args, **kwargs):
        started = time.monotonic()
        if parameters is None:
            inner = real_execute(sql, *args, **kwargs)
        else:
            inner = real_execute(sql, parameters, *args, **kwargs)
        return _TimedExecute(inner, sql, started, "execute")

    def executemany_timed(sql, parameters, *args, **kwargs):
        started = time.monotonic()
        n = 0
        with contextlib.suppress(TypeError):
            n = len(parameters) if hasattr(parameters, "__len__") else 0
        inner = real_executemany(sql, parameters, *args, **kwargs)
        return _TimedExecute(inner, sql, started, "executemany", param_rows=n)

    async def commit_timed(*args, **kwargs):
        started = time.monotonic()
        try:
            result = await real_commit(*args, **kwargs)
            _ACTIVE_BEGINS.pop(conn_id, None)
            return result
        finally:
            _maybe_log("commit", "COMMIT", started)

    # Reassign on the instance so we don't have to subclass.
    conn.execute = execute_timed  # type: ignore[method-assign]
    conn.executemany = executemany_timed  # type: ignore[method-assign]
    conn.commit = commit_timed  # type: ignore[method-assign]
    log.info(
        "slow_db_op_logging_installed",
        slow_ms=_SLOW_QUERY_MS,
    )


_DML_PREFIXES = ("INSERT", "UPDATE", "DELETE", "REPLACE", "MERGE")


def _is_dml(sql: str | None) -> bool:
    """True iff ``sql`` opens with INSERT/UPDATE/DELETE/REPLACE/MERGE.

    Used by the safe-rollback wrapper to skip read paths — SELECTs don't
    leave Python sqlite3 in a stuck-transaction state on failure, and
    forcing a rollback on them would clobber a legitimate enclosing
    transaction (e.g. one explicitly opened with BEGIN IMMEDIATE).
    """
    if not isinstance(sql, str):
        return False
    head = sql.lstrip().upper()
    return any(head.startswith(kw) for kw in _DML_PREFIXES)


def install_safe_rollback(conn: aiosqlite.Connection) -> None:
    """Wrap ``execute`` / ``executemany`` so a failing DML auto-rollbacks.

    This is the structural fix for the failure mode that surfaced
    2026-05-22 (8-hour WAL pin → cascading ``database is locked``).
    Python's sqlite3 module starts an implicit deferred transaction
    before any DML; if the statement raises (most commonly
    ``OperationalError: database is locked`` after exhausting
    ``busy_timeout``), the connection is left with
    ``in_transaction=True``. Every read path in this codebase follows
    the pattern ``await conn.execute(SELECT)`` with no enclosing
    transaction discipline, so the next read opens a snapshot inside
    that ghost transaction and HOLDS it until commit/rollback runs
    — which, on read-only paths, never happens. The pinned snapshot
    blocks WAL checkpointing; the WAL grows unbounded; eventually
    every writer hits its busy_timeout and 500s.

    REQUIRED on every persistent aiosqlite connection that touches
    augmentum.db: the main backend conn, dream journal _db,
    resource ledger _ledger_conn, auth session manager _db, and any
    other long-lived handle. One-shot connections that are opened
    and closed per call don't need it (sqlite3 implicitly rolls back
    on connection close).

    Idempotent — safe to call multiple times on the same conn.

    Composes correctly with ``_install_query_timing``: install timing
    first, then this. Safe-rollback wraps the timing-wrapped execute,
    so slow-query logs still fire for the rollback path.
    """
    if getattr(conn, "_augmentum_safe_rollback_installed", False):
        return

    real_execute = conn.execute
    real_executemany = conn.executemany
    conn_ref = conn

    async def _rollback_if_dml(sql: str | None) -> None:
        if not _is_dml(sql):
            return
        # Check the underlying sqlite3 connection's tracked state.
        # aiosqlite exposes the sqlite3.Connection as ``_conn``;
        # ``in_transaction`` is the canonical signal.
        underlying = getattr(conn_ref, "_conn", None)
        if underlying is None or not underlying.in_transaction:
            return
        try:
            await conn_ref.rollback()
        except Exception:
            log.warning("safe_rollback_failed", sql_head=(sql or "")[:32], exc_info=True)

    class _SafeExecute:
        """Dual-protocol wrapper around the timing-wrapped awaitable.

        On exception during ``await`` OR ``async with`` exit, fires
        an awaitable rollback if the SQL was a DML — clearing
        Python's stuck in_transaction state before re-raising.
        """
        __slots__ = ("_inner", "_sql")

        def __init__(self, inner, sql):
            self._inner = inner
            self._sql = sql

        async def _await_with_rollback(self):
            try:
                return await self._inner
            except BaseException:
                await _rollback_if_dml(self._sql)
                raise

        def __await__(self):
            return self._await_with_rollback().__await__()

        async def __aenter__(self):
            try:
                return await self._inner.__aenter__()
            except BaseException:
                await _rollback_if_dml(self._sql)
                raise

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            try:
                return await self._inner.__aexit__(exc_type, exc_val, exc_tb)
            finally:
                if exc_type is not None:
                    await _rollback_if_dml(self._sql)

    def execute_safe(sql, *args, **kwargs):
        inner = real_execute(sql, *args, **kwargs)
        return _SafeExecute(inner, sql)

    def executemany_safe(sql, *args, **kwargs):
        inner = real_executemany(sql, *args, **kwargs)
        return _SafeExecute(inner, sql)

    conn.execute = execute_safe  # type: ignore[method-assign]
    conn.executemany = executemany_safe  # type: ignore[method-assign]
    conn._augmentum_safe_rollback_installed = True  # type: ignore[attr-defined]
    log.info("safe_rollback_installed", conn_id=hex(id(conn)))

_VEC_AVAILABLE = False
try:
    import sqlite_vec
    _VEC_AVAILABLE = True
except ImportError:
    sqlite_vec = None  # type: ignore[assignment]

# PRAGMAs applied on every connection. ``AUGMENTUM_DB_PRAGMAS`` is the
# canonical set — every aiosqlite/sqlite3 connection that touches the
# main augmentum.db (state backend + dream journal + ad-hoc dream-route
# helpers + future call sites) MUST apply this exact list. Inconsistent
# pragmas cause the failure mode that surfaced 2026-05-04: dream
# journal's connection had ``journal_mode=WAL`` but no ``busy_timeout``,
# so during heavy-GPU/event-loop-stalled windows it raised
# ``OperationalError: database is locked`` instantly while the main
# backend's connection (with ``busy_timeout=5000ms``) waited 5s, hit the
# ceiling, and propagated to a 500.
#
# busy_timeout=30000 (30s): chosen so a single event-loop stall can
# burn through it without blowing the timeout. Heavy GPU operations
# (voice cloning, model loading) routinely produce 1-3s stalls;
# multiple stalls can stack. 30s leaves headroom for the worst case
# without making non-OOM bugs hang the UI for minutes.
AUGMENTUM_DB_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA busy_timeout=30000",
    "PRAGMA cache_size=-64000",
    # cell_size_check on every read costs a tiny amount of CPU but makes
    # the engine surface page-cell-size corruption immediately instead
    # of silently returning wrong rows. With this OFF, certain page
    # corruptions only surface during integrity_check sweeps. Per-
    # connection pragma so every aiosqlite/sqlite3 opener inherits.
    "PRAGMA cell_size_check=ON",
)

# Backwards-compat alias for the existing internal call site.
_PRAGMAS = list(AUGMENTUM_DB_PRAGMAS)


async def apply_augmentum_pragmas(conn: aiosqlite.Connection) -> None:
    """Apply the standard pragma set to a fresh aiosqlite connection.

    Use this for any aiosqlite connection that touches augmentum.db
    OUTSIDE the SQLiteBackend's main connection. Ensures consistent
    busy_timeout / WAL / synchronous behavior so writers from
    different connections don't trip "database is locked" against
    each other under load.
    """
    for pragma in AUGMENTUM_DB_PRAGMAS:
        await conn.execute(pragma)


def apply_augmentum_pragmas_sync(conn: object) -> None:
    """Sync variant for ``sqlite3.Connection`` callers (e.g. KVSessionManifest,
    coder/indexer, knowledge subsystem helpers).

    Same pragma set as the async version; foreign_keys is a no-op for
    cross-DB connections (dedicated DBs that don't have FKs into
    augmentum.db) but harmless to enable.
    """
    for pragma in AUGMENTUM_DB_PRAGMAS:
        conn.execute(pragma)


# ─────────────────────────────────────────────────────────────────────
# Transaction helpers — MUST be used on every persistent connection.
#
# Python's sqlite3 module starts an implicit deferred transaction
# before any DML statement when ``isolation_level=""`` (the default
# aiosqlite inherits). If that statement raises — most commonly
# ``OperationalError: database is locked`` after busy_timeout — the
# connection is left in ``in_transaction=True`` state. The bare
# ``try: execute + commit; except: log`` pattern that's idiomatic in
# this codebase doesn't call rollback, so Python keeps thinking a txn
# is open. The next SELECT on that connection opens a real read
# snapshot inside the ghost transaction and HOLDS it until commit or
# rollback runs — which, for read-only paths, never happens.
#
# A held snapshot pins the WAL: ``wal_checkpoint(PASSIVE)`` can't
# reclaim frames newer than the oldest reader's view, and
# ``wal_checkpoint(TRUNCATE)`` returns ``busy=1`` and a no-op. The WAL
# grows unbounded; ~30MB in we start seeing ``database is locked``
# storms across every writer. Symptom on 2026-05-22: dream journal
# DELETEs 500ing for a user; root cause traced back to a single
# failed ``_persist_cycle`` INSERT 8 hours earlier whose stuck
# transaction quietly poisoned the connection.
#
# ``transactional_write`` enforces commit-or-rollback discipline on
# every write block so a transient failure can't leak transaction
# state forward. ``savepoint`` isolates best-effort sub-operations
# (e.g. cleaning auxiliary indexes that may legitimately fail) so
# their failure doesn't poison the outer transaction.
# ─────────────────────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def transactional_write(conn: aiosqlite.Connection):
    """Yield ``conn``; commit on clean exit, rollback on any exception.

    REQUIRED for every write block on a persistent aiosqlite connection
    that's reused across calls (dream journal _db, resource ledger
    _ledger_conn, state backend conn, etc.). See the module-level
    block comment above for the failure mode this prevents.

    Drop-in replacement for the existing
    ``async with self._connect() as db: ... await db.commit()``
    pattern — remove the trailing ``commit()`` and pass ``self._db``
    (or whichever persistent handle) into this helper.

    The rollback path is best-effort: if rollback itself raises
    (rare — usually a dead connection), we log and swallow so the
    original exception still surfaces to the caller. We use
    ``BaseException`` so cancellation paths also clear the txn state.
    """
    try:
        yield conn
        await conn.commit()
    except BaseException:
        try:
            await conn.rollback()
        except Exception:
            log.warning("transactional_write_rollback_failed", exc_info=True)
        raise


@contextlib.asynccontextmanager
async def savepoint(conn: aiosqlite.Connection, name: str = "sp"):
    """Nested savepoint context for best-effort sub-operations.

    Use INSIDE a ``transactional_write`` block when one step is
    allowed to fail (e.g. cleaning an auxiliary FTS/vec index whose
    table may not exist) without aborting the surrounding write. The
    sub-operation's failure rolls back to the savepoint, releases it,
    and propagates — the caller can catch and continue, and the outer
    transaction stays clean.

    ``name`` is sanitised to ``[A-Za-z0-9_]`` because savepoint names
    are SQL identifiers (not parameter-bindable). Defaults to ``sp``
    which is fine for non-nested use; pass a stable per-call name if
    you nest savepoints in the same scope.
    """
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)[:32] or "sp"
    await conn.execute(f"SAVEPOINT {safe}")
    try:
        yield
    except BaseException:
        try:
            await conn.execute(f"ROLLBACK TO SAVEPOINT {safe}")
            await conn.execute(f"RELEASE SAVEPOINT {safe}")
        except Exception:
            log.debug("savepoint_rollback_failed", name=safe, exc_info=True)
        raise
    else:
        await conn.execute(f"RELEASE SAVEPOINT {safe}")


# Migration directory
_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"


def _parse_migration_version(path: Path) -> int | None:
    """Pull the leading numeric version out of a migration filename.

    ``001_initial.sql`` → 1, ``234_observation_substrate.sql`` → 234.
    Returns None for filenames that don't follow the convention.
    """
    try:
        return int(path.stem.split("_")[0])
    except (ValueError, IndexError):
        return None


_BEGIN_RE = re.compile(r"\bBEGIN\b", re.IGNORECASE)
_END_RE = re.compile(r"\bEND\s*;", re.IGNORECASE)


def _strip_trailing_line_comment(line: str) -> str:
    """Remove a trailing ``--`` comment from one line, respecting string literals.

    A naive ``re.sub("--.*", "", line)`` would corrupt string literals that
    contain ``--`` (e.g. ``DEFAULT '--dashes--'``). We walk the line, toggle
    string state on single-quotes (with ``''`` handled as an escaped quote),
    and cut at the first ``--`` seen outside a string.

    This exists because migration files occasionally put a ``;`` inside a
    trailing comment (e.g. ``TEXT DEFAULT '', -- bearer; api-key``). Without
    stripping, the split-on-``;`` below would carve the CREATE TABLE in half
    and SQLite would reject the first half as "incomplete input".
    """
    in_string = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == "'":
            if i + 1 < len(line) and line[i + 1] == "'":
                i += 2
                continue
            in_string = not in_string
        elif not in_string and c == "-" and i + 1 < len(line) and line[i + 1] == "-":
            return line[:i].rstrip()
        i += 1
    return line


class MigrationValidationError(Exception):
    """A migration file failed pre-flight validation (e.g. references a table
    that doesn't exist and isn't created within the same file). Raised by
    ``_run_migrations`` BEFORE any of the offending migration's statements
    execute, so partial state never reaches the database.
    """


# Regex for "this statement creates a table". Covers CREATE TABLE, CREATE
# TABLE IF NOT EXISTS, and CREATE VIRTUAL TABLE (sqlite-vec, FTS5, etc.).
_CREATE_TABLE_RE = re.compile(
    r"\bCREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"['\"`]?(\w+)['\"`]?",
    re.IGNORECASE,
)

# Regex for "this statement renames a table" — the NEW name is effectively
# created by the rename, so the pre-flight validator should treat it as a
# create (otherwise migrations that rename a table and then use the new
# name look like they reference a missing table). Migration 200 was the
# trigger case: `ALTER TABLE coder_workspaces RENAME TO project_checkouts`
# followed by `ALTER TABLE project_checkouts ADD COLUMN ...`.
_RENAME_TABLE_RE = re.compile(
    r"\bALTER\s+TABLE\s+['\"`]?\w+['\"`]?\s+RENAME\s+TO\s+['\"`]?(\w+)['\"`]?",
    re.IGNORECASE,
)

# Regexes for statements that REQUIRE a target table to already exist.
# Tuple of (regex, group-name-for-the-table). Conservative — we catch the
# common DML/DDL shapes that crashed migration 243 (DELETE FROM <table>)
# but don't try to fully parse SQL. Anything we miss falls through to the
# old behavior (raise during execute), which is no worse than today.
_REQUIRE_TABLE_REGEXES = (
    re.compile(r"\bDELETE\s+FROM\s+['\"`]?(\w+)['\"`]?", re.IGNORECASE),
    re.compile(r"\bUPDATE\s+(?:OR\s+\w+\s+)?['\"`]?(\w+)['\"`]?\s+SET\b", re.IGNORECASE),
    re.compile(r"\bALTER\s+TABLE\s+['\"`]?(\w+)['\"`]?", re.IGNORECASE),
    re.compile(r"\bINSERT\s+(?:OR\s+\w+\s+)?INTO\s+['\"`]?(\w+)['\"`]?", re.IGNORECASE),
    # CREATE INDEX / CREATE UNIQUE INDEX [IF NOT EXISTS] <name> ON <table>
    re.compile(
        r"\bCREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?\S+\s+ON\s+"
        r"['\"`]?(\w+)['\"`]?",
        re.IGNORECASE,
    ),
    # CREATE TRIGGER [IF NOT EXISTS] <name> {BEFORE|AFTER|INSTEAD OF} <event> ON <table>
    re.compile(
        r"\bCREATE\s+TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?\S+\s+"
        r"(?:BEFORE|AFTER|INSTEAD\s+OF)\s+\w+(?:\s+OF\s+[\w,\s]+?)?\s+ON\s+"
        r"['\"`]?(\w+)['\"`]?",
        re.IGNORECASE,
    ),
    # We intentionally don't list table-deletion verbs (DROP INDEX/TRIGGER
    # don't need the parent table; the table-deletion verb itself is fine
    # to issue on a nonexistent name when the migration uses the
    # IF EXISTS clause, which we trust the author to include).
)


def _migration_required_tables(sql: str) -> set[str]:
    """Tables a migration's statements claim must already exist, minus the
    tables that the same migration creates itself. Used by ``_run_migrations``
    as a pre-flight check before any statement executes.

    Returns a set of lowercased table names. Empty set means the migration
    only touches tables it creates within itself, which is always safe.
    """
    statements = _split_sql_statements(sql)
    created: set[str] = set()
    for stmt in statements:
        for m in _CREATE_TABLE_RE.finditer(stmt):
            created.add(m.group(1).lower())
        # A rename within the same migration "creates" the new name as
        # far as later statements are concerned.
        for m in _RENAME_TABLE_RE.finditer(stmt):
            created.add(m.group(1).lower())
    required: set[str] = set()
    for stmt in statements:
        for pattern in _REQUIRE_TABLE_REGEXES:
            for m in pattern.finditer(stmt):
                required.add(m.group(1).lower())
    return required - created


async def _list_existing_tables(conn) -> set[str]:
    """Snapshot of currently-existing table names (lowercased)."""
    cursor = await conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    return {r[0].lower() for r in await cursor.fetchall()}


def _split_sql_statements(sql: str) -> list[str]:
    """Split SQL text into individual statements, respecting BEGIN...END blocks.

    Naive ``;`` splitting breaks triggers/procedures that contain semicolons
    inside their body.  This splitter tracks nesting so those stay intact.
    It also strips trailing ``--`` comments from each line so a ``;`` inside
    a comment doesn't get treated as a statement terminator.
    """
    statements: list[str] = []
    current: list[str] = []
    in_block = False

    for raw_line in sql.split("\n"):
        line = _strip_trailing_line_comment(raw_line)
        stripped = line.strip()
        # Skip lines that were purely a comment (now empty after stripping)
        if not stripped:
            # Still append so multi-line formatting round-trips cleanly, but
            # only when we're mid-statement — leading blank lines stay out.
            if current:
                current.append(line)
            continue

        current.append(line)

        if not in_block and _BEGIN_RE.search(stripped):
            in_block = True
            continue

        if in_block:
            if _END_RE.search(stripped):
                # End of block — emit the whole block as one statement
                stmt = "\n".join(current).strip().rstrip(";").strip()
                if stmt:
                    statements.append(stmt)
                current = []
                in_block = False
            continue

        # Outside a block — split on semicolons within the line
        joined = "\n".join(current)
        parts = joined.split(";")
        # All but the last fragment are complete statements
        for part in parts[:-1]:
            stmt = part.strip()
            if stmt:
                statements.append(stmt)
        # The last fragment carries over (may be empty or start of next stmt)
        remainder = parts[-1].strip()
        current = [remainder] if remainder else []

    # Flush anything remaining
    if current:
        stmt = "\n".join(current).strip().rstrip(";").strip()
        if stmt:
            statements.append(stmt)

    return statements


class SQLiteBackend:
    """Async SQLite backend with automatic migrations."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self.vec_enabled: bool = False
        self._recovery_attempted_this_boot: bool = False
        # True when connect() skipped the inline quick_check because the
        # DB exceeds the inline size threshold — the lifespan schedules
        # ``run_quick_check(deferred=True)`` as a background task instead.
        self.deferred_quick_check: bool = False
        # Set by run_quick_check: True = last check found corruption.
        self.quick_check_failed: bool = False

    async def connect(self) -> None:
        """Open connection and apply PRAGMAs + migrations."""
        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row

        # Restrict the DB + WAL/SHM files to owner-only. Defense in depth
        # for installs that bind-mount /data to a host directory: anyone
        # who can read the host file reads every password hash, every
        # API key, every memory. Best-effort — Windows host bind-mounts
        # don't honour POSIX modes, debug-log on failure since this isn't
        # a fatal condition.
        try:
            db_file = Path(self._db_path)
            for f in (db_file, Path(f"{self._db_path}-wal"), Path(f"{self._db_path}-shm")):
                if f.exists():
                    f.chmod(0o600)
        except (OSError, PermissionError) as exc:
            log.debug("db_chmod_skipped", path=self._db_path, error=str(exc))

        try:
            for pragma in _PRAGMAS:
                await self._conn.execute(pragma)
        except Exception as exc:
            err_msg = str(exc).lower()
            if "file is not a database" in err_msg or "disk image is malformed" in err_msg:
                log.warning(
                    "sqlite_corruption_detected",
                    path=self._db_path,
                    error=str(exc),
                )
                # Close the broken connection before recovery
                try:
                    await self._conn.close()
                except Exception as close_exc:
                    log.debug("sqlite_corrupt_conn_close_failed", error=str(close_exc))
                self._conn = None
                # Attempt automatic recovery
                await self._recover_corrupt_db()
                return  # _recover_corrupt_db calls connect() recursively on success
            raise

        # Load sqlite-vec extension for vector search (graceful if unavailable)
        if _VEC_AVAILABLE:
            try:
                await self._conn.enable_load_extension(True)
                await self._conn.load_extension(sqlite_vec.loadable_path())
                await self._conn.enable_load_extension(False)
                self.vec_enabled = True
                log.info("sqlite_vec_loaded")
            except Exception:
                log.debug("sqlite_vec_load_failed", exc_info=True)

        # Diagnostic: wrap execute/executemany/commit so any slow op gets a
        # structured log line with SQL text + caller. Installed AFTER pragma
        # + vec setup so those one-time startup ops aren't logged. Cheap on
        # the fast path (one ``time.monotonic`` + threshold compare).
        _install_query_timing(self._conn)
        # Structural safety net: any DML that raises auto-rollbacks so a
        # transient ``database is locked`` can't leave the connection in
        # in_transaction=True and pin a WAL snapshot on the next SELECT.
        # See install_safe_rollback's docstring for the full failure mode.
        install_safe_rollback(self._conn)

        # Repair any phantom FTS virtual tables BEFORE migrations run.
        # See the method docstring for why this can't live in an SQL
        # migration (schema cache + runner's "already exists" swallow).
        await self._repair_phantom_fts_if_needed()

        await self._run_migrations()

        # Re-assert the pragma set AFTER migrations. ``PRAGMA
        # foreign_keys`` is silently IGNORED inside a transaction: a
        # migration that flips it OFF in autocommit and back ON after
        # its own DML opened the implicit transaction (081 does exactly
        # this) leaves enforcement off for the REST of the process.
        # Verified live 2026-07-17 — fresh installs ran their entire
        # first session unenforced. Migrations commit before returning,
        # so this re-apply runs in autocommit and sticks. The commit is
        # belt-and-braces: if any future migration path leaves a txn
        # open, the pragma would silently no-op again.
        await self._conn.commit()
        for pragma in _PRAGMAS:
            await self._conn.execute(pragma)

        # Create vec0 virtual table if sqlite-vec is available (requires extension loaded)
        if self.vec_enabled:
            await self._ensure_vec_table()

        # Structured corruption sweep — runs once per startup. Catches
        # FTS5 internal corruption that PRAGMA integrity_check misses
        # (the actual failure that hit memories_fts on 2026-05-10).
        # Repairs what's repairable, raises on what isn't so the next
        # boot routes to the manual-repair script.
        #
        # Boot-latency gate (2026-07-02): quick_check walks every B-tree
        # page — measured 40.5s on a 1.5GB DB, all of it blocking first
        # paint (the ASGI lifespan serves nothing until startup returns).
        # Above the size threshold we skip the INLINE quick_check and set
        # ``deferred_quick_check`` so the lifespan runs it as a background
        # task seconds after the app is serving. FTS/vec sweeps stay
        # inline — they're fast (measured ~0.1s) and they REPAIR, so
        # deferring them would let the first MATCH query hit a corrupt
        # index. Small DBs (tests, fresh installs) keep the historical
        # inline-and-raise behavior end to end. ``:memory:`` has no file
        # to stat and is trivially fast — always inline.
        _inline_max_mb = float(
            os.environ.get("AUGMENTUM_QUICK_CHECK_INLINE_MAX_MB", "256")
        )
        _db_mb = 0.0
        if self._db_path != ":memory:":
            try:
                _db_mb = Path(self._db_path).stat().st_size / (1024 * 1024)
            except OSError:
                _db_mb = 0.0
        self.deferred_quick_check = _db_mb > _inline_max_mb
        if self.deferred_quick_check:
            log.info(
                "db_quick_check_deferred",
                db_mb=round(_db_mb, 1),
                inline_max_mb=_inline_max_mb,
            )
        await self._post_startup_health_check(
            skip_quick_check=self.deferred_quick_check
        )

        log.info("sqlite_connected", path=self._db_path, vec_enabled=self.vec_enabled)

        # A clean boot (no auto-recovery this run) re-arms the recovery
        # gate by clearing any stamp left from a prior incident. We
        # don't clear it on recovery-success paths — the stamp must
        # survive across recursive connect() calls so a second
        # corruption in the same boot still trips the gate. Skip for
        # ``:memory:`` (no on-disk siblings exist; tests pass this).
        if not self._recovery_attempted_this_boot and self._db_path != ":memory:":
            stamp = Path(self._db_path).parent / ".augmentum_recovery_stamp"
            if stamp.exists():
                try:
                    stamp.unlink()
                    log.info("sqlite_recovery_gate_cleared", stamp=str(stamp))
                except OSError as exc:
                    log.debug("sqlite_recovery_gate_clear_failed", error=str(exc))

    async def _recover_corrupt_db(self) -> None:
        """Attempt to recover a corrupt database file.

        Strategy:
        1. Remove stale WAL/SHM files and retry (fixes most unclean shutdowns)
        2. Use sqlite3 .recover to salvage data into a new DB
        3. Last resort: back up corrupt file and start fresh
        """
        import shutil
        import sqlite3

        db = Path(self._db_path)
        wal = Path(self._db_path + "-wal")
        shm = Path(self._db_path + "-shm")
        stamp = db.parent / ".augmentum_recovery_stamp"

        # Once-per-incident gate. If a recovery already ran and wasn't
        # cleared by a subsequent clean startup, refuse to silently
        # recover again — that's how the 2026-04 → 2026-05 corruption
        # loop ran for weeks, with auto-recovery hiding the underlying
        # problem and the leaked fds from each pass compounding it.
        # The stamp is cleared automatically when the app reaches a
        # clean startup; or operators can delete it manually after
        # running scripts/repair_augmentum_db.ps1.
        if stamp.exists():
            try:
                prior = stamp.read_text(encoding="utf-8").strip()
            except OSError:
                prior = "<unreadable>"
            log.error(
                "sqlite_recovery_refused",
                stamp=str(stamp),
                prior_recovery=prior,
                hint=(
                    "Auto-recovery already ran and the next startup is "
                    "still corrupt. Stop the container and run "
                    "scripts/repair_augmentum_db.ps1 (or .sh) for a full "
                    "offline rebuild from .recover."
                ),
            )
            raise sqlite3.DatabaseError(
                "sqlite_recovery_refused: prior auto-recovery did not "
                "stick. Run scripts/repair_augmentum_db.ps1 to do a full "
                "offline rebuild, or delete "
                f"{stamp} to re-arm auto-recovery."
            )

        # Mark this boot as having attempted recovery. Written BEFORE
        # any destructive action so a crash mid-recovery still trips
        # the gate on the next start.
        self._touch_recovery_stamp("auto_recover_corrupt_db")
        self._recovery_attempted_this_boot = True

        # Force the corrupt aiosqlite Connection's worker thread to
        # release its underlying sqlite3 handle BEFORE we unlink the
        # WAL/SHM files below. The 2026-05-10 forensics found three
        # leaked (deleted)-WAL fds in /proc/1/fd from prior recovery
        # passes — those abandoned fds were writing to unreachable
        # inodes and corrupting checkpoint state. gc.collect() runs
        # the Connection finalizer which signals the worker to exit.
        gc.collect()

        # --- Step 1: Try removing stale WAL/SHM and reopening ---
        if wal.exists() or shm.exists():
            log.info("sqlite_recovery_step1", action="removing stale WAL/SHM files")
            wal_size = wal.stat().st_size if wal.exists() else 0
            shm_size = shm.stat().st_size if shm.exists() else 0
            for f in (wal, shm):
                try:
                    f.unlink(missing_ok=True)
                except Exception as exc:
                    log.debug("sqlite_recovery_wal_shm_unlink_failed", path=str(f), error=str(exc))
            log.info("sqlite_recovery_wal_removed", wal_size=wal_size, shm_size=shm_size)
            try:
                await self.connect()  # Recursive — will succeed or raise
                log.info("sqlite_recovery_success", method="wal_cleanup")
                return
            except Exception:
                log.warning("sqlite_recovery_step1_failed", exc_info=True)

        # --- Step 2: Try sqlite3 .recover ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        corrupt_path = db.parent / f"{db.stem}_corrupt_{ts}{db.suffix}"
        recovered_path = db.parent / f"{db.stem}_recovered_{ts}{db.suffix}"

        log.info("sqlite_recovery_step2", action="attempting .recover")
        try:
            # Use the sqlite3 CLI for .recover — it can read past corruption.
            # Offloaded: this can run up to 120s and previously blocked the
            # event loop for the whole recovery during startup (2026-06-13
            # loop-stall audit).
            result = await asyncio.to_thread(
                subprocess.run,
                ["sqlite3", str(db), ".recover"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Feed recovered SQL into a new database
                # If the recovered SQL contains "lost_and_found", the schema is fragmented
                if "lost_and_found" in result.stdout:
                    log.warning("sqlite_recovery_fragmented", hint="Salvage yielded lost_and_found table; schema may be broken.")

                conn = sqlite3.connect(str(recovered_path))
                conn.executescript(result.stdout)
                conn.close()

                recovered_size = recovered_path.stat().st_size
                original_size = db.stat().st_size
                log.info(
                    "sqlite_recovery_data_salvaged",
                    original_size=original_size,
                    recovered_size=recovered_size,
                )

                # Swap files: corrupt → backup, recovered → main
                shutil.move(str(db), str(corrupt_path))
                shutil.move(str(recovered_path), str(db))
                # Clean up any leftover WAL/SHM from the corrupt copy
                for f in (wal, shm):
                    f.unlink(missing_ok=True)

                log.info(
                    "sqlite_recovery_success",
                    method="recover",
                    corrupt_backup=str(corrupt_path),
                )
                await self.connect()  # Open the recovered DB
                return
            else:
                log.warning(
                    "sqlite_recovery_step2_no_output",
                    stderr=result.stderr[:500] if result.stderr else "",
                )
        except FileNotFoundError:
            log.warning("sqlite_recovery_no_sqlite3_cli", hint="sqlite3 CLI not available for .recover")
        except subprocess.TimeoutExpired:
            log.warning("sqlite_recovery_timeout")
        except Exception:
            log.warning("sqlite_recovery_step2_failed", exc_info=True)
        finally:
            # Clean up partial recovered file on failure
            if recovered_path.exists() and db.exists():
                try:
                    recovered_path.unlink(missing_ok=True)
                except Exception as exc:
                    log.debug("sqlite_recovery_partial_unlink_failed", path=str(recovered_path), error=str(exc))

        # --- Step 2.5: Try restoring from backups/ directory ---
        backup_dir = db.parent / "backups"
        if backup_dir.exists():
            log.info("sqlite_recovery_step2_5", action="checking backups for restoration")
            # Find most recent backup > 1MB (avoid fresh/broken ones)
            backups = sorted(
                [f for f in backup_dir.iterdir() if f.suffix == ".db" and f.stat().st_size > 1024*1024],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if backups:
                latest = backups[0]
                log.info("sqlite_recovery_restoring_backup", latest=str(latest))
                try:
                    # Backup the corrupt one first (if not already backed up by Step 2)
                    if not corrupt_path.exists():
                        shutil.move(str(db), str(corrupt_path))
                    else:
                        db.unlink()
                    # Copy backup to main
                    shutil.copy2(str(latest), str(db))
                    for f in (wal, shm):
                        f.unlink(missing_ok=True)

                    log.info("sqlite_recovery_success", method="backup_restore", latest=str(latest))
                    await self.connect()
                    return
                except Exception:
                    log.warning("sqlite_recovery_step2_5_failed", exc_info=True)

        # --- Step 3: Last resort — back up corrupt file and start fresh ---
        log.warning(
            "sqlite_recovery_fresh_start",
            action="backing up corrupt DB and creating fresh database",
            corrupt_backup=str(corrupt_path),
            hint="Your data has been preserved in the backup file. "
                 "Manual recovery may be possible with 'sqlite3 old.db .dump'.",
        )
        if not corrupt_path.exists():
            try:
                shutil.move(str(db), str(corrupt_path))
            except Exception:
                shutil.copy2(str(db), str(corrupt_path))
                db.unlink()
        else:
            db.unlink(missing_ok=True)

        for f in (wal, shm):
            f.unlink(missing_ok=True)

        await self.connect()  # Open fresh DB
        log.info("sqlite_recovery_success", method="fresh_start", corrupt_backup=str(corrupt_path))

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            log.info("sqlite_closed")

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SQLite not connected. Call connect() first.")
        return self._conn

    @property
    def db_path(self) -> str:
        """Path this backend opened, for callers that need a SECOND handle.

        Sidecar connections (dream journal, resource ledger, bulk index
        writer) open their own aiosqlite handle against the same file so
        their work doesn't queue behind the shared connection's single
        aiosqlite worker thread. They must pass this path — never
        hardcode — and must apply :func:`apply_augmentum_pragmas` plus
        :func:`install_safe_rollback` to the result.

        ``":memory:"`` is a real value here and is NOT sidecar-safe: a
        second connection to ``":memory:"`` opens a different, empty
        database. Callers must check for it and fall back to ``conn``.
        """
        return self._db_path

    async def _repair_phantom_fts_if_needed(self) -> None:
        """Dispatcher: repair every FTS5 virtual table known to go phantom.

        Why this exists: FTS5 virtual tables are backed by a handful of
        shadow tables (``{name}_data``, ``_idx``, ``_content``,
        ``_docsize``, ``_config``). When a DB corruption + partial
        restore happens, ``sqlite_master`` can end up listing the shadow
        tables (and the sync triggers) *without* the virtual table row
        itself — or the reverse. In either state:

        - ``SELECT … FROM name_fts`` fails with "no such table: name_fts"
          because FTS5 can't open its own shadow tables.
        - ``DROP TABLE IF EXISTS name_fts`` silently no-ops because the
          destructor has nothing to find.
        - ``CREATE VIRTUAL TABLE IF NOT EXISTS name_fts`` short-circuits
          against the connection's cached schema, still phantom.

        Pure-SQL migrations can't fix this; see migration 096 commentary
        for the interacting sqlite quirks. So each known FTS gets a
        Python repair pass that hits ``PRAGMA writable_schema`` + RESET
        to invalidate the cache, then rebuilds the virtual table and
        triggers.

        Safe to run on clean installs: each repair is a no-op when the
        FTS is missing entirely (fresh DB, migration hasn't run yet) or
        healthy. Runs BEFORE migrations so they see a sane world.
        """
        await self._repair_one_phantom_fts(
            fts_name="memories_fts",
            base_table="memories",
            triggers=("memories_ai", "memories_ad", "memories_au"),
            create_fts_sql=(
                "CREATE VIRTUAL TABLE memories_fts USING fts5("
                "content, memory_type, "
                "content=memories, content_rowid=rowid"
                ")"
            ),
            create_trigger_sqls=(
                "CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN "
                "INSERT INTO memories_fts(rowid, content, memory_type) "
                "VALUES (new.rowid, new.content, new.memory_type); "
                "END",
                "CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN "
                "INSERT INTO memories_fts(memories_fts, rowid, content, memory_type) "
                "VALUES ('delete', old.rowid, old.content, old.memory_type); "
                "END",
                "CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN "
                "INSERT INTO memories_fts(memories_fts, rowid, content, memory_type) "
                "VALUES ('delete', old.rowid, old.content, old.memory_type); "
                "INSERT INTO memories_fts(rowid, content, memory_type) "
                "VALUES (new.rowid, new.content, new.memory_type); "
                "END",
            ),
        )
        await self._repair_one_phantom_fts(
            fts_name="file_index_fts",
            base_table="file_index",
            triggers=(
                "file_index_fts_insert",
                "file_index_fts_delete",
                "file_index_fts_update",
            ),
            create_fts_sql=(
                "CREATE VIRTUAL TABLE file_index_fts USING fts5("
                "name, description, tags, "
                "content=file_index, content_rowid=rowid"
                ")"
            ),
            create_trigger_sqls=(
                "CREATE TRIGGER file_index_fts_insert AFTER INSERT ON file_index BEGIN "
                "INSERT INTO file_index_fts(rowid, name, description, tags) "
                "VALUES (new.rowid, new.name, new.description, new.tags); "
                "END",
                "CREATE TRIGGER file_index_fts_delete AFTER DELETE ON file_index BEGIN "
                "INSERT INTO file_index_fts(file_index_fts, rowid, name, description, tags) "
                "VALUES ('delete', old.rowid, old.name, old.description, old.tags); "
                "END",
                "CREATE TRIGGER file_index_fts_update AFTER UPDATE ON file_index BEGIN "
                "INSERT INTO file_index_fts(file_index_fts, rowid, name, description, tags) "
                "VALUES ('delete', old.rowid, old.name, old.description, old.tags); "
                "INSERT INTO file_index_fts(rowid, name, description, tags) "
                "VALUES (new.rowid, new.name, new.description, new.tags); "
                "END",
            ),
        )
        await self._repair_one_phantom_fts(
            fts_name="document_chunks_fts",
            base_table="document_chunks",
            triggers=(
                "trg_doc_chunks_ai",
                "trg_doc_chunks_ad",
                "trg_doc_chunks_au",
            ),
            create_fts_sql=(
                "CREATE VIRTUAL TABLE document_chunks_fts USING fts5("
                "content, "
                "content=document_chunks, content_rowid=rowid"
                ")"
            ),
            create_trigger_sqls=(
                "CREATE TRIGGER trg_doc_chunks_ai AFTER INSERT ON document_chunks BEGIN "
                "INSERT INTO document_chunks_fts(rowid, content) "
                "VALUES (new.rowid, new.content); "
                "END",
                "CREATE TRIGGER trg_doc_chunks_ad AFTER DELETE ON document_chunks BEGIN "
                "INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content) "
                "VALUES('delete', old.rowid, old.content); "
                "END",
                "CREATE TRIGGER trg_doc_chunks_au AFTER UPDATE ON document_chunks BEGIN "
                "INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content) "
                "VALUES('delete', old.rowid, old.content); "
                "INSERT INTO document_chunks_fts(rowid, content) "
                "VALUES (new.rowid, new.content); "
                "END",
            ),
        )
        # Same phantom-FTS class as memories_fts / file_index_fts /
        # document_chunks_fts. Schema mirrors migration 058 exactly. The
        # symptom in this case was every dream cycle crashing on first
        # journal write: dream_entries.INSERT fires dream_entries_ai
        # which targets the missing virtual table.
        await self._repair_one_phantom_fts(
            fts_name="dream_entries_fts",
            base_table="dream_entries",
            triggers=(
                "dream_entries_ai",
                "dream_entries_ad",
                "dream_entries_au",
            ),
            create_fts_sql=(
                "CREATE VIRTUAL TABLE dream_entries_fts USING fts5("
                "content, entry_type, "
                "content=dream_entries, content_rowid=rowid"
                ")"
            ),
            create_trigger_sqls=(
                "CREATE TRIGGER dream_entries_ai AFTER INSERT ON dream_entries BEGIN "
                "INSERT INTO dream_entries_fts(rowid, content, entry_type) "
                "VALUES (new.rowid, new.content, new.entry_type); "
                "END",
                "CREATE TRIGGER dream_entries_ad AFTER DELETE ON dream_entries BEGIN "
                "INSERT INTO dream_entries_fts(dream_entries_fts, rowid, content, entry_type) "
                "VALUES ('delete', old.rowid, old.content, old.entry_type); "
                "END",
                "CREATE TRIGGER dream_entries_au AFTER UPDATE ON dream_entries BEGIN "
                "INSERT INTO dream_entries_fts(dream_entries_fts, rowid, content, entry_type) "
                "VALUES ('delete', old.rowid, old.content, old.entry_type); "
                "INSERT INTO dream_entries_fts(rowid, content, entry_type) "
                "VALUES (new.rowid, new.content, new.entry_type); "
                "END",
            ),
        )

    async def run_quick_check(self, *, deferred: bool = False) -> bool:
        """``PRAGMA quick_check`` on the main DB + regular indexes.

        Inline mode (``deferred=False`` — the default, used by
        ``_post_startup_health_check`` on small DBs and by tests):
        raises ``sqlite3.DatabaseError`` on findings after touching the
        recovery stamp, exactly the historical behavior.

        Deferred mode (``deferred=True`` — scheduled by the lifespan as
        a background task when the DB exceeds the inline size
        threshold): NEVER raises. On findings it touches the recovery
        stamp, logs at error, and sets ``self.quick_check_failed`` so
        health surfaces can report a degraded DB. The next boot sees
        the stamp and refuses silent auto-recovery, same as inline.

        Returns True when the check passed.
        """
        if self._conn is None:
            raise RuntimeError("quick_check requires an open connection.")
        t0 = time.monotonic()
        cursor = await self._conn.execute("PRAGMA quick_check")
        rows = await cursor.fetchall()
        findings = [r[0] for r in rows]
        elapsed = round(time.monotonic() - t0, 2)
        if findings == ["ok"]:
            log.info("db_health_quick_check_ok", elapsed_s=elapsed, deferred=deferred)
            self.quick_check_failed = False
            return True
        log.error(
            "db_health_quick_check_failed",
            elapsed_s=elapsed,
            findings_count=len(findings),
            sample=findings[:10],
            deferred=deferred,
        )
        self._touch_recovery_stamp(
            f"quick_check_failed:{findings[0][:80]}"
        )
        self.quick_check_failed = True
        if deferred:
            return False
        raise sqlite3.DatabaseError(
            f"PRAGMA quick_check returned {len(findings)} findings; "
            f"first: {findings[0][:200]}. "
            "Stop the container and run scripts/repair_augmentum_db.ps1 "
            "to rebuild via offline .recover."
        )

    async def _post_startup_health_check(self, *, skip_quick_check: bool = False) -> None:
        """Structured corruption sweep, run once per ``connect()``.

        PRAGMA integrity_check walks B-trees of base tables + their
        regular indexes, but it does NOT inspect the internal binary
        structures of FTS5 or sqlite-vec virtual tables. Those have
        their own integrity protocols. Without a sweep here, a DB can
        pass ``integrity_check: ok`` and still raise "database disk
        image is malformed" on the first MATCH or vector query —
        exactly the 2026-05-10 ``memories_fts`` failure mode.

        Detection ladder, fastest first:

        1. ``PRAGMA quick_check`` — main DB pages + regular indexes.
           No in-place repair possible at the SQLite layer; trip the
           recovery-gate stamp and raise. The next start sees the
           stamp and refuses to silently auto-recover, surfacing the
           failure to the operator with a pointer to the offline
           repair script.

        2. ``INSERT INTO {fts}({fts}) VALUES('integrity-check')`` —
           per FTS5 virtual table. On failure run ``'rebuild'`` and
           re-verify. Rebuild is idempotent + repairs the common case
           where shadow rows survived a ``.recover`` dump but their
           binary B-tree pages are inconsistent. If rebuild doesn't
           stick, the source data is also corrupt — trip the stamp
           and raise.

        3. ``SELECT count(*)`` smoke probe on each ``*_vec`` virtual
           table — sqlite-vec has no native integrity command. Treat
           as a non-fatal warning: vector search is an enhancement,
           not a correctness-critical surface, and the alternative
           (raising) would deny service for a degraded feature.

        The sweep runs after migrations and after vec table setup so
        new tables created during this boot are also checked.
        """
        if self._conn is None:
            raise RuntimeError("Health check requires an open connection.")

        # --- Layer 1: quick_check on main DB + regular indexes -------
        # On large DBs this is the single biggest boot cost (measured
        # 40.5s on a 1.5GB file, 2026-07-02) — ``connect()`` defers it
        # to a post-startup background task above the size threshold so
        # first paint isn't gated on a full B-tree walk. Small DBs keep
        # the inline raise-on-corruption behavior unchanged.
        if not skip_quick_check:
            await self.run_quick_check()

        # --- Layer 2: FTS5 integrity-check per virtual table --------
        fts_tables = await self._list_fts5_tables()
        fts_repaired: list[str] = []
        for fts in fts_tables:
            if await self._fts5_integrity_ok(fts):
                continue

            log.warning("db_health_fts_corrupt", table=fts, action="rebuild")
            try:
                await self._conn.execute(
                    f'INSERT INTO "{fts}"("{fts}") VALUES(\'rebuild\')'
                )
                await self._conn.commit()
            except Exception as exc:
                log.error(
                    "db_health_fts_rebuild_raised",
                    table=fts,
                    error=str(exc),
                    exc_info=True,
                )
                self._touch_recovery_stamp(f"fts_rebuild_failed:{fts}")
                raise sqlite3.DatabaseError(
                    f"FTS5 rebuild raised for {fts}: {exc}. "
                    "Stop the container and run scripts/repair_augmentum_db.ps1."
                ) from exc

            if not await self._fts5_integrity_ok(fts):
                log.error("db_health_fts_still_corrupt", table=fts)
                self._touch_recovery_stamp(f"fts_unrepaired:{fts}")
                raise sqlite3.DatabaseError(
                    f"FTS5 {fts} failed integrity-check even after rebuild. "
                    "Source data may be corrupt. "
                    "Stop the container and run scripts/repair_augmentum_db.ps1."
                )

            fts_repaired.append(fts)
            log.info("db_health_fts_rebuilt", table=fts)

        # --- Layer 3: sqlite-vec smoke read (best-effort) ------------
        vec_warned: list[str] = []
        if self.vec_enabled:
            for vec in await self._list_vec0_tables():
                try:
                    cursor = await self._conn.execute(
                        f'SELECT count(*) FROM "{vec}"'
                    )
                    await cursor.fetchone()
                except Exception as exc:
                    log.warning(
                        "db_health_vec_smoke_failed",
                        table=vec,
                        error=str(exc),
                    )
                    vec_warned.append(vec)

        log.info(
            "db_health_check_passed",
            fts_count=len(fts_tables),
            fts_repaired=fts_repaired,
            vec_warned=vec_warned,
        )

    async def _list_fts5_tables(self) -> list[str]:
        """All user FTS5 virtual tables (excludes shadow tables)."""
        cursor = await self.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND sql LIKE '%USING fts5%' "
            "AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        return [r[0] for r in await cursor.fetchall()]

    async def _list_vec0_tables(self) -> list[str]:
        """All user sqlite-vec virtual tables (excludes shadow tables)."""
        cursor = await self.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND sql LIKE '%USING vec0%' "
            "AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        return [r[0] for r in await cursor.fetchall()]

    async def _fts5_integrity_ok(self, fts_name: str) -> bool:
        """True iff FTS5's native integrity-check passes.

        Issues ``INSERT INTO {fts}({fts}) VALUES('integrity-check')``,
        which the FTS5 module intercepts and runs against its shadow
        B-trees. Returns False on the DatabaseError sub-classes that
        signal index corruption ("malformed", "no such table" on a
        phantom-FTS surface, etc.).

        The statement looks like DML to Python's sqlite3 wrapper, which
        auto-opens an implicit BEGIN. The FTS5 magic INSERT writes
        nothing, but the wrapper doesn't know that — so we must close
        the implicit transaction here. Leaking it stranded ``VACUUM
        INTO`` on the next startup backup (and silently affected any
        other autocommit-sensitive op on the shared connection).
        """
        try:
            await self.conn.execute(
                f'INSERT INTO "{fts_name}"("{fts_name}") VALUES(\'integrity-check\')'
            )
            await self.conn.commit()
            return True
        except sqlite3.DatabaseError as exc:
            try:
                await self.conn.rollback()
            except sqlite3.Error as rb_exc:
                log.debug(
                    "fts5_integrity_check_rollback_failed",
                    table=fts_name,
                    error=str(rb_exc),
                )
            log.debug(
                "fts5_integrity_check_raised",
                table=fts_name,
                error=str(exc),
            )
            return False

    def _touch_recovery_stamp(self, reason: str) -> None:
        """Append-write the recovery-gate stamp.

        Append (not overwrite) so multiple detection signals across the
        same boot all leave traces — useful when the operator later
        inspects the file to understand why auto-recovery was disabled.
        Each call appends an ISO-8601 UTC timestamp + the reason. The
        existence of the file is the gate; the contents are diagnostic.
        Safe to call repeatedly. No-op on ``:memory:``.
        """
        if self._db_path == ":memory:":
            return
        stamp = Path(self._db_path).parent / ".augmentum_recovery_stamp"
        try:
            line = (
                datetime.now(UTC).isoformat(timespec="seconds")
                + f"\t{reason}\n"
            )
            with stamp.open("a", encoding="utf-8") as f:
                f.write(line)
            try:
                stamp.chmod(0o600)
            except OSError:
                pass
        except OSError as exc:
            log.debug(
                "recovery_stamp_touch_failed",
                reason=reason,
                error=str(exc),
            )

    async def _repair_one_phantom_fts(
        self,
        *,
        fts_name: str,
        base_table: str,
        triggers: tuple[str, ...],
        create_fts_sql: str,
        create_trigger_sqls: tuple[str, ...],
    ) -> None:
        """Generic phantom-FTS repair. See dispatcher docstring for the why."""
        conn = self.conn

        # The base table must exist before we can repair — on a fresh
        # install the early migrations haven't landed yet and there's
        # nothing to rebuild. In that case just exit; the migration will
        # set up the FTS cleanly.
        cursor = await conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (base_table,),
        )
        if not await cursor.fetchone():
            return

        # Inspect state. We care about three signals:
        #   fts_row_present: is {fts_name} listed in sqlite_master?
        #   triggers_present: do the sync triggers exist? (if yes, we
        #                     MUST have a functioning FTS or base-table
        #                     writes will fail — which is the file_index
        #                     failure observed during catalog sync)
        #   healthy: does a zero-row probe succeed?
        cursor = await conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name=?",
            (fts_name,),
        )
        fts_row_present = bool(await cursor.fetchone())

        placeholders = ",".join(["?"] * len(triggers))
        cursor = await conn.execute(
            f"SELECT COUNT(*) FROM sqlite_master "
            f"WHERE type='trigger' AND name IN ({placeholders})",
            triggers,
        )
        trigger_count = (await cursor.fetchone())[0]
        triggers_present = trigger_count > 0

        # Pre-FTS-migration state: no row, no triggers. Nothing to do.
        if not fts_row_present and not triggers_present:
            return

        healthy = False
        if fts_row_present:
            try:
                probe = await conn.execute(
                    f"SELECT rowid FROM {fts_name} LIMIT 0"
                )
                await probe.fetchall()
                healthy = True
            except Exception as exc:
                log.warning(
                    "fts_phantom_detected",
                    fts=fts_name, error=str(exc),
                )

        if healthy:
            return

        log.warning(
            "fts_drift_detected_starting_repair",
            fts=fts_name,
            fts_row_present=fts_row_present,
            triggers_present=triggers_present,
            trigger_count=trigger_count,
        )

        # Drop dangling triggers first so the later CREATE TRIGGER
        # statements don't collide.
        for trigger in triggers:
            try:
                await conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except Exception as exc:
                log.warning(
                    "fts_trigger_drop_failed",
                    fts=fts_name, trigger=trigger, error=str(exc),
                )

        # Force-delete phantom rows from sqlite_master. The WHERE clause
        # is scoped to the exact name and its shadow-table prefix; no
        # other tables are touched.
        try:
            await conn.execute("PRAGMA writable_schema = 1")
            await conn.execute(
                "DELETE FROM sqlite_master "
                "WHERE name = ? OR name LIKE ? ESCAPE '\\'",
                (fts_name, f"{fts_name}\\_%"),
            )
            # RESET disables writable_schema AND invalidates the
            # connection's parsed schema cache so subsequent CREATE
            # statements see the post-DELETE state of sqlite_master.
            await conn.execute("PRAGMA writable_schema = RESET")
            await conn.commit()
        except Exception:
            # If the DELETE itself failed (locked DB etc.), bail out
            # with an explicit rollback so we don't strand the
            # transaction. The subsequent migration will surface the
            # actual error.
            try:
                await conn.rollback()
            except sqlite3.Error as rb_exc:
                log.debug(
                    "phantom_fts_repair_rollback_failed",
                    error=str(rb_exc),
                )
            raise

        # Recreate. No IF NOT EXISTS — if anything went wrong we want
        # the error to surface loudly, not silently.
        await conn.execute(create_fts_sql)
        await conn.execute(
            f"INSERT INTO {fts_name}({fts_name}) VALUES('rebuild')"
        )
        for trigger_sql in create_trigger_sqls:
            await conn.execute(trigger_sql)
        await conn.commit()
        log.info("fts_phantom_repaired", fts=fts_name)

    async def _run_migrations(self) -> None:
        """Apply any unapplied SQL migration files.

        Two-tier bookkeeping:

          1. ``schema_version`` — the historical record, populated by
             each migration's own ``INSERT OR IGNORE INTO schema_version``
             statement. Optional (not every migration writes one) and
             keyed by version number, which means two migrations sharing
             a version number collide. Kept for the per-migration
             description trail other tooling reads.

          2. ``migration_files_applied`` — keyed by FILENAME, populated
             by this runner. The authoritative "did we apply this file?"
             record. Resolves the historical footgun where a new file
             with a version <= max(schema_version) would be silently
             skipped (originally the gate was ``version <= current_version``;
             files added out of order would never run).

        First run with the file-tracking table backfills it from the
        current ``schema_version`` MAX so existing installs don't
        re-run every prior migration. Files whose version is <= the
        backfill watermark AND whose filename matches the convention
        get recorded as already-applied without re-running.
        """
        conn = self.conn

        # ── Tier 2: file-tracking table (idempotent create) ───────────
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS migration_files_applied (
                filename TEXT PRIMARY KEY,
                version INTEGER NOT NULL DEFAULT 0,
                applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
            )
            """
        )

        # ── Tier 1: schema_version snapshot for legacy backfill ───────
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        schema_version_exists = bool(await cursor.fetchone())

        watermark = 0
        if schema_version_exists:
            cursor = await conn.execute("SELECT MAX(version) FROM schema_version")
            row = await cursor.fetchone()
            if row and row[0] is not None:
                watermark = int(row[0])

        # Snapshot of files already recorded by FILENAME.
        cursor = await conn.execute(
            "SELECT filename FROM migration_files_applied"
        )
        applied_names = {r[0] for r in await cursor.fetchall()}

        migration_files = sorted(_MIGRATIONS_DIR.glob("*.sql"))

        # Backwards-compat backfill: on first run with the file table
        # (empty applied_names) AND a non-zero schema_version watermark,
        # mark every file at-or-below the watermark as already-applied
        # without re-running. Fresh installs (watermark=0) skip the
        # backfill — they apply every file normally.
        if not applied_names and watermark > 0:
            for migration_path in migration_files:
                version = _parse_migration_version(migration_path)
                if version is None or version > watermark:
                    continue
                await conn.execute(
                    "INSERT OR IGNORE INTO migration_files_applied "
                    "(filename, version) VALUES (?, ?)",
                    (migration_path.name, version),
                )
                applied_names.add(migration_path.name)
            await conn.commit()
            log.info(
                "migration_files_backfilled",
                count=len(applied_names),
                watermark=watermark,
            )

        # ── Apply unrecorded files ────────────────────────────────────
        applied = 0
        for migration_path in migration_files:
            if migration_path.name in applied_names:
                continue

            version = _parse_migration_version(migration_path)
            if version is None:
                log.warning("skipping_invalid_migration", file=migration_path.name)
                continue

            # An out-of-order migration (version <= prior watermark) is
            # usually a numbering mistake — log loudly but apply it.
            # CREATE TABLE IF NOT EXISTS + ALTER ... idempotent patterns
            # make this safe; the warning surfaces the smell.
            if version <= watermark:
                log.warning(
                    "applying_out_of_order_migration",
                    file=migration_path.name,
                    version=version,
                    watermark=watermark,
                    hint=(
                        "version number is <= prior MAX(schema_version). "
                        "Renumber to a value > watermark on next change."
                    ),
                )

            log.info(
                "applying_migration", version=version, file=migration_path.name,
            )
            sql = migration_path.read_text(encoding="utf-8")

            # Pre-flight: catch the typo'd-table-name class of bug BEFORE
            # any statement runs. The trigger case was migration 243
            # which referenced `settings` instead of `app_settings`,
            # crashed mid-bootstrap, and silently fell back to the
            # in-memory backend — presenting the user as if all their
            # data was gone. This validation is conservative (regex over
            # common DML/DDL shapes); anything it can't parse falls
            # through to runtime, where the old behavior applies.
            required = _migration_required_tables(sql)
            existing = await _list_existing_tables(conn)
            missing = sorted(required - existing)
            if missing:
                raise MigrationValidationError(
                    f"Migration {migration_path.name} references "
                    f"nonexistent table(s) {missing}. Either the table "
                    f"was renamed in an earlier migration (and this "
                    f"file is stale) or this is a typo. Aborting "
                    f"bootstrap to avoid corrupting partial state. "
                    f"Currently-existing tables: {len(existing)}; "
                    f"required-but-missing: {missing}."
                )

            # Execute statements individually to handle "already exists"
            # errors gracefully (e.g. ALTER TABLE ADD COLUMN on re-run).
            # Use a block-aware splitter so BEGIN...END triggers don't
            # break.
            for statement in _split_sql_statements(sql):
                if not statement:
                    continue
                try:
                    await conn.execute(statement)
                except Exception as stmt_err:
                    err_msg = str(stmt_err).lower()
                    if "already exists" in err_msg or "duplicate column" in err_msg:
                        log.debug(
                            "migration_statement_skipped",
                            version=version, reason=str(stmt_err),
                        )
                    else:
                        raise
            # Record the file as applied BEFORE the commit so an
            # interrupted run doesn't double-apply on next boot.
            await conn.execute(
                "INSERT OR REPLACE INTO migration_files_applied "
                "(filename, version) VALUES (?, ?)",
                (migration_path.name, version),
            )
            await conn.commit()
            applied += 1

        if applied:
            log.info("migrations_applied", count=applied)
        else:
            log.debug("migrations_up_to_date", watermark=watermark)

    _EXPECTED_VEC_DIM = 768

    async def _ensure_vec_table(self) -> None:
        """Create vec0 virtual tables for vector search if they don't exist.

        Also detects dimension mismatches (e.g. 384 → 768 model upgrade)
        and recreates the tables.  Existing embeddings in the ``memories``
        table are left untouched — they'll be lazily re-embedded on next
        access/store.
        """
        conn = self.conn

        # --- memories_vec ---
        await self._ensure_single_vec_table(
            conn,
            vec_table="memories_vec",
            parent_table="memories",
            pk_col="memory_id",
        )

        # --- doc_chunks_vec ---
        await self._ensure_single_vec_table(
            conn,
            vec_table="doc_chunks_vec",
            parent_table="document_chunks",
            pk_col="chunk_id",
        )

        # --- narrative_archive_vec ---
        await self._ensure_single_vec_table(
            conn,
            vec_table="narrative_archive_vec",
            parent_table="narrative_archive",
            pk_col="id",
        )

        # --- dream_entries_vec ---
        await self._ensure_single_vec_table(
            conn,
            vec_table="dream_entries_vec",
            parent_table="dream_entries",
            pk_col="id",
        )

        # --- interest_clusters_vec ---
        # Custom-schema vec table (column is `centroid_embedding`, not the
        # generic `embedding`). Some production DBs ended up with a
        # mismatched schema; detect and repair in place.
        await self._ensure_interest_clusters_vec(conn)

    async def _ensure_interest_clusters_vec(self, conn) -> None:
        """Ensure interest_clusters_vec has the (cluster_id, centroid_embedding) schema.

        Migration 073 creates this table, but some deployments ended up
        with a table missing the ``centroid_embedding`` column (cause
        unclear — likely a vec-extension load ordering issue during an
        early migration run). ``find_nearest_cluster`` then fails with
        ``no such column: centroid_embedding``.

        Drop and recreate if the schema is wrong or the dimension no
        longer matches ``_EXPECTED_VEC_DIM``. Discovery signals will
        be re-clustered on subsequent requests.
        """
        cursor = await conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            ("interest_clusters_vec",),
        )
        row = await cursor.fetchone()

        create_sql = (row[0] or "") if row else ""
        schema_ok = (
            "centroid_embedding" in create_sql
            and f"float[{self._EXPECTED_VEC_DIM}]" in create_sql
        )

        if row is not None and not schema_ok:
            log.warning(
                "interest_clusters_vec_schema_mismatch",
                sql=create_sql[:200],
                hint="Recreating with correct schema. Existing cluster centroids "
                     "will be re-embedded as signals arrive.",
            )
            try:
                await conn.execute("DROP TABLE interest_clusters_vec")
                await conn.commit()
                row = None
            except Exception:
                log.warning("interest_clusters_vec_drop_failed", exc_info=True)
                return

        if row is None:
            try:
                await conn.execute(
                    f"CREATE VIRTUAL TABLE interest_clusters_vec USING vec0("
                    f"  cluster_id TEXT PRIMARY KEY,"
                    f"  centroid_embedding float[{self._EXPECTED_VEC_DIM}]"
                    f")"
                )
                await conn.commit()
                log.info(
                    "vec_table_created",
                    table="interest_clusters_vec",
                    dim=self._EXPECTED_VEC_DIM,
                )
            except Exception:
                log.warning(
                    "interest_clusters_vec_create_failed",
                    exc_info=True,
                )

    async def _ensure_single_vec_table(
        self,
        conn,
        vec_table: str,
        parent_table: str,
        pk_col: str,
    ) -> None:
        """Create or recreate a single vec0 table with the expected dimension."""
        cursor = await conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (vec_table,),
        )
        row = await cursor.fetchone()

        needs_create = row is None
        if row is not None:
            # Check if dimension matches (look for float[N] in the CREATE sql)
            create_sql = row[0] or ""
            if f"float[{self._EXPECTED_VEC_DIM}]" not in create_sql:
                # Dimension mismatch — drop and recreate
                log.warning(
                    "vec_table_dimension_mismatch",
                    table=vec_table,
                    expected=self._EXPECTED_VEC_DIM,
                    sql=create_sql[:120],
                    hint="Vec table will be recreated. Existing memories need re-embedding "
                         "via POST /v1/memory/rebuild-profile or re-storing to restore vector search.",
                )
                try:
                    await conn.execute(f"DROP TABLE {vec_table}")
                    await conn.commit()
                    needs_create = True
                except Exception:
                    log.warning("vec_table_drop_failed", table=vec_table, exc_info=True)
                    self.vec_enabled = False
                    return

        if needs_create:
            # Ensure parent table exists
            cursor2 = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (parent_table,),
            )
            if not await cursor2.fetchone():
                return

            try:
                await conn.execute(
                    f"CREATE VIRTUAL TABLE {vec_table} USING vec0("
                    f"  {pk_col} TEXT PRIMARY KEY,"
                    f"  embedding float[{self._EXPECTED_VEC_DIM}]"
                    ")"
                )
                await conn.commit()
                log.info("vec_table_created", table=vec_table, dim=self._EXPECTED_VEC_DIM)
            except Exception:
                log.warning("vec_table_create_failed", table=vec_table, exc_info=True)
                self.vec_enabled = False

    # --- Session operations ---

    async def get_session(self, session_id: str, *, user_id: str = "") -> dict | None:
        """Get a session by ID."""
        query = "SELECT * FROM sessions WHERE id = ?"
        params: list = [session_id]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        cursor = await self.conn.execute(query, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def create_session(self, session_id: str, mode: str = "passthrough", *, user_id: str = "") -> dict:
        """Create a new session."""
        if user_id:
            await self.conn.execute(
                "INSERT INTO sessions (id, mode, user_id) VALUES (?, ?, ?)",
                (session_id, mode, user_id),
            )
        else:
            await self.conn.execute(
                "INSERT INTO sessions (id, mode) VALUES (?, ?)",
                (session_id, mode),
            )
        await self.conn.commit()
        return await self.get_session(session_id, user_id=user_id)  # type: ignore[return-value]

    async def update_session(
        self,
        session_id: str,
        *,
        mode: str | None = None,
        increment_messages: bool = False,
        metadata: str | None = None,
        user_id: str = "",
    ) -> None:
        """Update session fields."""
        updates = ["updated_at = datetime('now')"]
        params: list = []

        if mode is not None:
            updates.append("mode = ?")
            params.append(mode)
        if increment_messages:
            updates.append("message_count = message_count + 1")
        if metadata is not None:
            updates.append("metadata = ?")
            params.append(metadata)

        where = "WHERE id = ?"
        params.append(session_id)
        if user_id:
            where += " AND user_id = ?"
            params.append(user_id)
        await self.conn.execute(
            f"UPDATE sessions SET {', '.join(updates)} {where}",
            params,
        )
        await self.conn.commit()
