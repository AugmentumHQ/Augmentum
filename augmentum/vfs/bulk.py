"""Bulk index writes that don't starve interactive traffic.

Why this module exists
----------------------
``SQLiteBackend`` opens exactly ONE ``aiosqlite.Connection`` for the whole
process (``state/backends/sqlite.py``), and aiosqlite serializes every
operation on that connection through a SINGLE background worker thread.
So the cost a long scan imposes on the rest of the app is not lock
contention — it is queue position. While a catalog sync is running, a
voice turn's query sits behind however many operations the scan has
already enqueued.

``PRAGMA journal_mode=WAL`` does not help here. WAL lets readers proceed
without blocking on a writer *across connections*; with one shared
connection everything serializes at the thread boundary before SQLite is
ever reached.

A media sync of 63k items previously issued, per item, a SELECT (playback
metadata preservation) + an INSERT..ON CONFLICT + a ``commit()``, plus the
``file_index_fts`` delete/insert triggers on every update — roughly 190k
queued operations, none of which yielded the event loop. That is the
stutter users hear during a scan.

This module fixes both halves:

1. **A dedicated connection.** The scan's operations go through their own
   aiosqlite worker thread, so interactive reads on the shared connection
   stop waiting behind them. WAL then does its actual job: the two
   connections don't block each other for reads, and ``busy_timeout``
   (30s, already standard) covers the brief write-lock overlap.

2. **Batched commits + an explicit yield.** Rows are buffered into one
   transaction and committed every ``batch_size``, with an
   ``asyncio.sleep`` between batches. This mirrors
   ``jobs/handlers/journal_vec_backfill.py``, which solved the same
   starvation problem for embedding backfill; the sleep is what lets
   ticks, voice, and chat interleave. Without it, a tight loop can
   monopolize its own thread's CPU even on a separate connection.

Both matter. The connection alone still lets a hot loop hog CPU; the
batching alone still queues behind the shared connection.

Usage::

    async with bulk_index_session(backend) as bulk:
        for item in items:
            await bulk.file_index.register(...)
            await bulk.tick()          # commit cadence + yield

The session commits on clean exit and rolls back on exception, so a
failed scan cannot leave a partial transaction pinned on the sidecar
connection (the failure mode documented at length in
``state/backends/sqlite.py``'s transaction-helper block comment).
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Rows buffered into one transaction before committing. 200 keeps the
# transaction short enough that a crash loses very little work and the
# write lock is never held long, while cutting commit count — and thus
# queued operations — by more than two orders of magnitude versus the
# per-row commit this replaces.
DEFAULT_BATCH_SIZE = 200

# Yield between batches so the event loop can service voice, chat, and
# tick traffic. Matches journal_vec_backfill's 50ms. On a 63k-item scan
# this adds roughly 16s of wall-clock sleep to a job that already runs in
# the background — an unconditionally good trade against audible stutter
# in a live call.
DEFAULT_INTER_BATCH_SLEEP_S = 0.05

# Ceiling on concurrent sidecar connections. The job runner is currently
# a single-worker loop (``jobs/runner.py``), so in practice at most one
# bulk session is open at a time; this bounds the blast radius if the
# runner ever grows a worker pool. Past two concurrent bulk writers they
# would mostly serialize on SQLite's write lock anyway, so more
# connections would buy latency, not throughput.
_MAX_CONCURRENT_SESSIONS = 2

_session_slots = asyncio.Semaphore(_MAX_CONCURRENT_SESSIONS)


class BulkIndexSession:
    """Batching writer bound to one connection. Built by the CM below.

    ``tick()`` after each logical item. Everything buffered since the
    last commit is flushed on exit — including a partial final batch.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sleep_s: float = DEFAULT_INTER_BATCH_SLEEP_S,
        owns_conn: bool,
    ) -> None:
        from augmentum.media.comic_series_store import ComicSeriesStore
        from augmentum.vfs.index import FileIndexService

        self._conn = conn
        self._batch_size = max(1, int(batch_size))
        self._sleep_s = max(0.0, float(sleep_s))
        self._owns_conn = owns_conn
        self._pending = 0
        self._committed = 0
        self._batches = 0

        # Both stores share this session's connection and its commit
        # cadence. Keeping comic_series on the SAME connection as
        # file_index is deliberate: a chapter row references the series
        # row written moments earlier, and same-connection reads see
        # uncommitted writes, so the FK resolves inside the batch.
        self.file_index = FileIndexService(conn, autocommit=False)
        self.series_store = ComicSeriesStore(conn, autocommit=False)

    @property
    def uses_dedicated_connection(self) -> bool:
        """False when we fell back to the shared connection.

        Batching still applies in that case; only the queue isolation is
        unavailable (``:memory:`` backends, or a sidecar that failed to
        open). Callers may want to log the degraded mode.
        """
        return self._owns_conn

    @property
    def committed(self) -> int:
        """Rows confirmed committed. Excludes the un-flushed tail."""
        return self._committed

    async def tick(self, n: int = 1) -> None:
        """Account for ``n`` written rows; commit + yield on batch full."""
        self._pending += n
        if self._pending >= self._batch_size:
            await self.flush()
            if self._sleep_s:
                await asyncio.sleep(self._sleep_s)

    async def flush(self) -> None:
        """Commit whatever is buffered. Safe to call with nothing pending."""
        if self._pending <= 0:
            return
        await self._conn.commit()
        self._committed += self._pending
        self._batches += 1
        self._pending = 0


@contextlib.asynccontextmanager
async def bulk_index_session(
    backend: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_s: float = DEFAULT_INTER_BATCH_SLEEP_S,
):
    """Yield a :class:`BulkIndexSession` for a long indexing run.

    Opens a dedicated aiosqlite connection against the backend's database
    file, applying the canonical pragma set and safe-rollback wrapper that
    every persistent handle in this codebase is required to install.

    Degrades to the shared connection — batching intact, queue isolation
    lost — when a sidecar is impossible or fails:

    * ``:memory:`` backends, where a second connection would open a
      DIFFERENT, empty database rather than the same one. This is the
      common case in tests and is not an error.
    * Any failure opening or configuring the sidecar. A scan that still
      runs on the shared connection is strictly better than a scan that
      raises.

    Commits on clean exit, rolls back on exception, and always closes a
    connection it owns.
    """
    from augmentum.state.backends.sqlite import (
        apply_augmentum_pragmas,
        install_safe_rollback,
    )

    db_path = str(getattr(backend, "db_path", "") or "")
    shared = getattr(backend, "conn", None)

    conn: aiosqlite.Connection | None = None
    owns_conn = False

    async with _session_slots:
        if db_path and db_path != ":memory:":
            try:
                conn = await aiosqlite.connect(db_path)
                await apply_augmentum_pragmas(conn)
                install_safe_rollback(conn)
                owns_conn = True
            except Exception:
                # Fall back rather than fail the scan. Close a
                # half-opened handle so we don't leak an fd.
                log.warning(
                    "bulk_index_sidecar_open_failed",
                    db_path=db_path,
                    exc_info=True,
                )
                if conn is not None:
                    with contextlib.suppress(Exception):
                        await conn.close()
                conn = None
                owns_conn = False

        if conn is None:
            if shared is None:
                raise RuntimeError(
                    "bulk_index_session requires a connected SQLite backend"
                )
            conn = shared

        session = BulkIndexSession(
            conn,
            batch_size=batch_size,
            sleep_s=sleep_s,
            owns_conn=owns_conn,
        )
        log.info(
            "bulk_index_session_open",
            dedicated_connection=owns_conn,
            batch_size=session._batch_size,
        )
        try:
            yield session
            await session.flush()
        except BaseException:
            # Never leave a transaction open on the sidecar. On the
            # shared-connection fallback this matters even more — a
            # pinned snapshot there blocks WAL checkpointing process-wide.
            try:
                await conn.rollback()
            except Exception:
                log.warning("bulk_index_rollback_failed", exc_info=True)
            raise
        finally:
            log.info(
                "bulk_index_session_close",
                committed=session.committed,
                batches=session._batches,
                dedicated_connection=owns_conn,
            )
            if owns_conn:
                with contextlib.suppress(Exception):
                    await conn.close()
