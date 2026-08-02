"""Tests for the bulk index session (augmentum/vfs/bulk.py).

These assert the properties that actually matter for the bug this fixes:
commits are batched rather than per-row, the loop yields between batches,
a dedicated connection is used when possible, rows are durable after the
session closes, and a failure can't leave a transaction pinned.
"""

from __future__ import annotations

import asyncio

import aiosqlite
import pytest

from augmentum.vfs.bulk import bulk_index_session

pytestmark = pytest.mark.asyncio


class _FakeBackend:
    """Stands in for SQLiteBackend: just db_path + conn."""

    def __init__(self, db_path: str, conn: aiosqlite.Connection) -> None:
        self.db_path = db_path
        self.conn = conn


async def _create_schema(conn: aiosqlite.Connection) -> None:
    """Minimal schema — enough for FileIndexService.register."""
    await conn.execute("""
        CREATE TABLE users (id TEXT PRIMARY KEY)
    """)
    await conn.execute("""
        CREATE TABLE file_index (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            name TEXT NOT NULL,
            mime_type TEXT NOT NULL DEFAULT '',
            size_bytes INTEGER NOT NULL DEFAULT 0,
            real_path TEXT,
            description TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '[]',
            thumbnail TEXT,
            embedding BLOB,
            is_directory INTEGER NOT NULL DEFAULT 0,
            parent_id TEXT,
            source_metadata TEXT NOT NULL DEFAULT '{}',
            kind TEXT NOT NULL DEFAULT '',
            scan_status TEXT NOT NULL DEFAULT 'pending',
            mtime INTEGER,
            scan_error TEXT,
            metadata_confidence REAL NOT NULL DEFAULT 0.5,
            series_id TEXT,
            is_trashed INTEGER NOT NULL DEFAULT 0,
            last_played_at TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    await conn.execute(
        "CREATE UNIQUE INDEX idx_file_index_source_unique "
        "ON file_index(user_id, source, source_id)"
    )
    await conn.commit()


async def _make_backend(tmp_path) -> tuple[_FakeBackend, aiosqlite.Connection]:
    """Real on-disk SQLite backend double with schema applied."""
    from augmentum.state.backends.sqlite import (
        apply_augmentum_pragmas,
        install_safe_rollback,
    )

    db_path = str(tmp_path / "augmentum.db")
    conn = await aiosqlite.connect(db_path)
    await apply_augmentum_pragmas(conn)
    install_safe_rollback(conn)
    conn.row_factory = aiosqlite.Row
    await _create_schema(conn)
    return _FakeBackend(db_path, conn), conn


async def _make_memory_backend() -> tuple[_FakeBackend, aiosqlite.Connection]:
    """`:memory:` backend — forces the shared-connection fallback path."""
    from augmentum.state.backends.sqlite import (
        apply_augmentum_pragmas,
        install_safe_rollback,
    )

    conn = await aiosqlite.connect(":memory:")
    await apply_augmentum_pragmas(conn)
    install_safe_rollback(conn)
    conn.row_factory = aiosqlite.Row
    await _create_schema(conn)
    return _FakeBackend(":memory:", conn), conn


async def _register(bulk, n: int, *, user_id: str = "u1") -> None:
    for i in range(n):
        await bulk.file_index.register(
            user_id=user_id,
            source="emby",
            source_id=f"srv:{i}",
            name=f"Item {i}",
        )
        await bulk.tick()


async def test_uses_dedicated_connection_for_ondisk_db(tmp_path):
    """The whole point: scan writes must not share the app's connection."""
    backend, conn = await _make_backend(tmp_path)
    try:
        async with bulk_index_session(backend) as bulk:
            assert bulk.uses_dedicated_connection is True
            # Distinct object from the shared handle.
            assert bulk.file_index._db is not conn
    finally:
        await conn.close()


async def test_memory_backend_falls_back_to_shared_connection(tmp_path):
    """A second :memory: connection would open a DIFFERENT empty DB.

    Falling back is correct, not a failure — but batching must survive.
    """
    backend, conn = await _make_memory_backend()
    try:
        async with bulk_index_session(backend) as bulk:
            assert bulk.uses_dedicated_connection is False
            assert bulk.file_index._db is conn
            # Batching is still in effect even on the fallback path.
            assert bulk.file_index._autocommit is False
    finally:
        await conn.close()


async def test_rows_are_durable_after_session_exit(tmp_path):
    """Buffered rows — including a partial final batch — must commit."""
    backend, conn = await _make_backend(tmp_path)
    try:
        # 25 rows with batch_size 10 => two full batches + a 5-row tail.
        async with bulk_index_session(backend, batch_size=10, sleep_s=0) as bulk:
            await _register(bulk, 25)

        cur = await conn.execute("SELECT COUNT(*) FROM file_index")
        assert (await cur.fetchone())[0] == 25, "tail batch was lost"
    finally:
        await conn.close()


async def test_commits_are_batched_not_per_row(tmp_path):
    """The actual regression guard: N rows must not mean N commits."""
    backend, conn = await _make_backend(tmp_path)
    try:
        async with bulk_index_session(backend, batch_size=10, sleep_s=0) as bulk:
            commits = 0
            real_commit = bulk._conn.commit

            async def counting_commit():
                nonlocal commits
                commits += 1
                return await real_commit()

            bulk._conn.commit = counting_commit  # type: ignore[method-assign]
            await _register(bulk, 100)
            # 100 rows / batch 10 == 10 commits. Pre-fix this was 100.
            assert commits == 10, f"expected 10 batched commits, got {commits}"
    finally:
        await conn.close()


async def test_yields_event_loop_between_batches(tmp_path):
    """Without a yield, a tight scan loop starves voice/chat entirely.

    Runs a competing coroutine alongside the scan and asserts it got
    scheduled repeatedly while the scan was in flight.
    """
    backend, conn = await _make_backend(tmp_path)
    try:
        ticks = 0
        stop = False

        async def competitor():
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0)

        task = asyncio.create_task(competitor())
        async with bulk_index_session(backend, batch_size=5, sleep_s=0.001) as bulk:
            await _register(bulk, 50)
        stop = True
        await task

        # 50 rows / batch 5 == 10 sleeps; each lets the competitor run.
        assert ticks > 10, f"competitor only ran {ticks}x — loop was starved"
    finally:
        await conn.close()


async def test_exception_rolls_back_uncommitted_rows(tmp_path):
    """A failed scan must not silently persist a partial batch."""
    backend, conn = await _make_backend(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="scan blew up"):
            async with bulk_index_session(backend, batch_size=1000, sleep_s=0) as bulk:
                await _register(bulk, 5)
                raise RuntimeError("scan blew up")

        cur = await conn.execute("SELECT COUNT(*) FROM file_index")
        assert (await cur.fetchone())[0] == 0
    finally:
        await conn.close()


async def test_exception_does_not_pin_transaction_on_shared_connection():
    """The WAL-pinning failure mode, tested where it actually bites.

    On the sidecar path a failure closes the connection, which implicitly
    clears any transaction. On the SHARED fallback the connection lives
    on — so a leaked open transaction there would pin a read snapshot,
    block WAL checkpointing, and eventually storm 'database is locked'
    across every writer (see the block comment in backends/sqlite.py).
    """
    backend, conn = await _make_memory_backend()
    try:
        with pytest.raises(RuntimeError, match="scan blew up"):
            async with bulk_index_session(backend, batch_size=1000, sleep_s=0) as bulk:
                assert bulk.uses_dedicated_connection is False
                await _register(bulk, 5)
                raise RuntimeError("scan blew up")

        underlying = conn._conn
        assert not underlying.in_transaction, "left a transaction pinned"

        cur = await conn.execute("SELECT COUNT(*) FROM file_index")
        assert (await cur.fetchone())[0] == 0
    finally:
        await conn.close()


async def test_tick_accounting_reports_committed_rows(tmp_path):
    backend, conn = await _make_backend(tmp_path)
    try:
        async with bulk_index_session(backend, batch_size=10, sleep_s=0) as bulk:
            await _register(bulk, 30)
            assert bulk.committed == 30
    finally:
        await conn.close()


async def test_flush_is_idempotent_with_nothing_pending(tmp_path):
    backend, conn = await _make_backend(tmp_path)
    try:
        async with bulk_index_session(backend, sleep_s=0) as bulk:
            await bulk.flush()
            await bulk.flush()
            assert bulk.committed == 0
    finally:
        await conn.close()


async def test_autocommit_default_preserved_for_direct_callers(tmp_path):
    """Non-bulk callers must keep committing per row — no silent change."""
    from augmentum.vfs.index import FileIndexService

    backend, conn = await _make_backend(tmp_path)
    try:
        svc = FileIndexService(conn)
        assert svc._autocommit is True
        await svc.register(
            user_id="u1", source="emby", source_id="srv:solo", name="Solo",
        )
        # Durable immediately, with no explicit commit from the caller.
        other = await aiosqlite.connect(backend.db_path)
        try:
            cur = await other.execute(
                "SELECT COUNT(*) FROM file_index WHERE source_id = 'srv:solo'"
            )
            assert (await cur.fetchone())[0] == 1
        finally:
            await other.close()
    finally:
        await conn.close()


async def test_session_slot_semaphore_bounds_concurrency(tmp_path):
    """Concurrent sessions must not open unbounded sidecar connections."""
    backend, conn = await _make_backend(tmp_path)
    try:
        live = 0
        peak = 0

        async def run():
            nonlocal live, peak
            async with bulk_index_session(backend, sleep_s=0):
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.02)
                live -= 1

        await asyncio.gather(*(run() for _ in range(6)))
        assert peak <= 2, f"{peak} concurrent sessions exceeded the cap"
    finally:
        await conn.close()
