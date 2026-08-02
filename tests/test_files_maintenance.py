"""Tests for the file maintenance loop helpers.

`purge_old_trash` and `sweep_orphan_blobs` are pure-async helpers that
take their dependencies as arguments — no app fixture needed.  The
fakes here mirror the surface area we actually call (FileIndexService
methods, adapter.delete, BlobStore.sweep_orphans).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from augmentum.vfs.maintenance import (
    purge_old_trash,
    run_maintenance,
    sweep_orphan_blobs,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@dataclass
class _Entry:
    id: str
    user_id: str
    source: str
    source_id: str


class _FakeIndex:
    def __init__(self, trashed: list[_Entry], purge_count: int = 0):
        self._trashed = trashed
        self._purge_count = purge_count
        self.purge_calls: list[int] = []

    async def list_trashed_older_than(self, days, *, limit=1000):
        return list(self._trashed)

    async def purge_all_old_trash(self, days):
        self.purge_calls.append(days)
        return self._purge_count


class _FakeAdapter:
    name = "uploads"

    def __init__(self, *, fail_on: set[str] | None = None, missing: set[str] | None = None):
        self.deleted: list[tuple[str, str]] = []
        self._fail = fail_on or set()
        self._missing = missing or set()

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        if source_id in self._fail:
            raise RuntimeError("boom")
        if source_id in self._missing:
            return False
        self.deleted.append((source_id, user_id))
        return True


class _FakeBlobStore:
    def __init__(self, swept: int):
        self._swept = swept
        self.sweep_calls = 0

    async def sweep_orphans(self, *, limit=1000):
        self.sweep_calls += 1
        return self._swept


# --- purge_old_trash ----------------------------------------------------

class TestPurgeOldTrash:
    def test_disabled_when_days_zero(self):
        async def go():
            idx = _FakeIndex(trashed=[_Entry("f1", "u1", "uploads", "ul_1")])
            adapter = _FakeAdapter()
            result = await purge_old_trash(
                idx, days=0, adapter_lookup=lambda _: adapter,
            )
            assert result == {"adapter_deleted": 0, "index_deleted": 0, "errors": 0}
            assert adapter.deleted == []
            assert idx.purge_calls == []
        _run(go())

    def test_dispatches_to_adapter(self):
        async def go():
            idx = _FakeIndex(trashed=[
                _Entry("f1", "u1", "uploads", "ul_1"),
                _Entry("f2", "u2", "uploads", "ul_2"),
            ])
            adapter = _FakeAdapter()
            result = await purge_old_trash(
                idx, days=30, adapter_lookup=lambda _: adapter,
            )
            assert result["adapter_deleted"] == 2
            assert ("ul_1", "u1") in adapter.deleted
            assert ("ul_2", "u2") in adapter.deleted
            # Bare index purge still runs as a safety net for non-adapter rows.
            assert idx.purge_calls == [30]
        _run(go())

    def test_no_adapter_falls_through_to_index(self):
        async def go():
            idx = _FakeIndex(
                trashed=[_Entry("f1", "u1", "images", "img_1")],
                purge_count=1,
            )
            result = await purge_old_trash(
                idx, days=30, adapter_lookup=lambda _: None,
            )
            assert result["adapter_deleted"] == 0
            assert result["index_deleted"] == 1
            assert idx.purge_calls == [30]
        _run(go())

    def test_adapter_failure_counted_not_raised(self):
        async def go():
            idx = _FakeIndex(trashed=[
                _Entry("f1", "u1", "uploads", "ul_ok"),
                _Entry("f2", "u1", "uploads", "ul_bad"),
            ])
            adapter = _FakeAdapter(fail_on={"ul_bad"})
            result = await purge_old_trash(
                idx, days=30, adapter_lookup=lambda _: adapter,
            )
            assert result["adapter_deleted"] == 1
            assert result["errors"] == 1
        _run(go())

    def test_empty_trash_skips_index_purge(self):
        async def go():
            idx = _FakeIndex(trashed=[])
            adapter = _FakeAdapter()
            result = await purge_old_trash(
                idx, days=30, adapter_lookup=lambda _: adapter,
            )
            assert result == {"adapter_deleted": 0, "index_deleted": 0, "errors": 0}
            assert idx.purge_calls == []
        _run(go())


# --- sweep_orphan_blobs -------------------------------------------------

class TestSweepOrphanBlobs:
    def test_returns_count(self):
        async def go():
            store = _FakeBlobStore(swept=7)
            assert await sweep_orphan_blobs(store) == 7
            assert store.sweep_calls == 1
        _run(go())


# --- run_maintenance ----------------------------------------------------

class TestRunMaintenance:
    def test_handles_missing_components(self):
        async def go():
            # Both None — should not raise.
            result = await run_maintenance(
                file_index=None, blob_store=None,
                adapter_lookup=lambda _: None, trash_ttl_days=30,
            )
            assert result == {"trash": None, "orphans": 0}
        _run(go())

    def test_full_cycle(self):
        async def go():
            idx = _FakeIndex(trashed=[_Entry("f1", "u1", "uploads", "ul_1")])
            adapter = _FakeAdapter()
            store = _FakeBlobStore(swept=3)
            result = await run_maintenance(
                file_index=idx, blob_store=store,
                adapter_lookup=lambda _: adapter, trash_ttl_days=30,
            )
            assert result["trash"]["adapter_deleted"] == 1
            assert result["orphans"] == 3
        _run(go())


# --- BlobStore.sweep_orphans (real, in-memory) -------------------------

class TestFTSSearchExcludesTrash:
    """Regression test — FTS5 search path used to ignore is_trashed."""

    def test_trashed_rows_not_returned(self):
        async def go():
            import aiosqlite

            from augmentum.vfs.index import FileIndexService

            conn = await aiosqlite.connect(":memory:")
            # Minimal schema: file_index + FTS5 + sync triggers + soft-delete cols.
            await conn.executescript("""
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
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    is_favorite INTEGER NOT NULL DEFAULT 0,
                    is_trashed INTEGER NOT NULL DEFAULT 0,
                    trashed_at TEXT,
                    kind TEXT NOT NULL DEFAULT ''
                );
                CREATE VIRTUAL TABLE file_index_fts USING fts5(
                    name, description, tags,
                    content=file_index, content_rowid=rowid
                );
                CREATE TRIGGER fts_ins AFTER INSERT ON file_index BEGIN
                    INSERT INTO file_index_fts(rowid, name, description, tags)
                    VALUES (new.rowid, new.name, new.description, new.tags);
                END;
                CREATE TRIGGER fts_del AFTER DELETE ON file_index BEGIN
                    INSERT INTO file_index_fts(file_index_fts, rowid, name, description, tags)
                    VALUES ('delete', old.rowid, old.name, old.description, old.tags);
                END;
            """)

            # One live row + one trashed row, both matching the search term.
            await conn.execute(
                "INSERT INTO file_index (id, user_id, source, source_id, name, "
                "is_trashed) VALUES (?, ?, ?, ?, ?, ?)",
                ("f_live", "u1", "uploads", "ul_a", "report alpha", 0),
            )
            await conn.execute(
                "INSERT INTO file_index (id, user_id, source, source_id, name, "
                "is_trashed, trashed_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                ("f_dead", "u1", "uploads", "ul_b", "report beta", 1),
            )
            await conn.commit()

            idx = FileIndexService(conn)
            hits = await idx.search("report", user_id="u1", limit=10)
            ids = {h.id for h in hits}
            assert ids == {"f_live"}, f"expected only live row, got {ids}"
            await conn.close()
        _run(go())


class TestBlobStoreSweep:
    def test_removes_zero_refcount_rows(self):
        async def go():
            import aiosqlite

            from augmentum.vfs.blobs import BlobStore

            conn = await aiosqlite.connect(":memory:")
            await conn.execute("""
                CREATE TABLE blobs (
                    sha256 TEXT PRIMARY KEY, size_bytes INTEGER, mime_type TEXT,
                    real_path TEXT, refcount INTEGER, created_at TEXT
                )
            """)
            # Row at refcount 0 with no real_path on disk — sweep should
            # still drop the row (the missing-file path is best-effort).
            await conn.execute(
                "INSERT INTO blobs VALUES (?, ?, ?, ?, ?, ?)",
                ("dead", 100, "image/png", "/nope/missing", 0, "2026-01-01"),
            )
            # Row at refcount 5 — must NOT be touched.
            await conn.execute(
                "INSERT INTO blobs VALUES (?, ?, ?, ?, ?, ?)",
                ("alive", 100, "image/png", "/nope/alive", 5, "2026-01-01"),
            )
            await conn.commit()

            # BlobStore wants a base_dir; tmp path is fine, we don't write.
            store = BlobStore(conn, base_dir="/tmp")
            purged = await store.sweep_orphans()
            assert purged == 1

            cursor = await conn.execute("SELECT sha256 FROM blobs")
            remaining = {r[0] for r in await cursor.fetchall()}
            assert remaining == {"alive"}
            await conn.close()
        _run(go())
