"""Tests for file index service."""

from __future__ import annotations

import asyncio
import json

import aiosqlite
import pytest


# ---------------------------------------------------------------------------
# Minimal file_index schema for atomic-enrichment tests. Skips the FTS
# triggers and users(id) FK — neither matters for what we're testing
# and they'd just add setup noise.
# ---------------------------------------------------------------------------
_FILE_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_index (
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
    last_enrichment_attempt TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def _make_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(":memory:")
    await db.executescript(_FILE_INDEX_SCHEMA)
    await db.commit()
    return db


async def _seed_file(db: aiosqlite.Connection, *, file_id: str = "fi_1",
                    user_id: str = "usr_1",
                    source_metadata: str = '{}') -> None:
    await db.execute(
        "INSERT INTO file_index "
        "(id, user_id, source, source_id, name, source_metadata) "
        "VALUES (?, ?, 'artifacts', 'art_1', 'book.epub', ?)",
        (file_id, user_id, source_metadata),
    )
    await db.commit()


class TestEnrichFileAtomic:
    """``enrich_file_atomic`` writes every field in one transaction."""

    def test_no_op_when_nothing_to_change(self):
        async def _run():
            from augmentum.vfs.index import FileIndexService
            db = await _make_db()
            try:
                await _seed_file(db)
                idx = FileIndexService(db)
                # No fields, no stamp → returns False without writing.
                wrote = await idx.enrich_file_atomic(
                    "fi_1", user_id="usr_1",
                )
                assert wrote is False
            finally:
                await db.close()

        asyncio.run(_run())

    def test_writes_all_fields(self):
        async def _run():
            from augmentum.vfs.index import FileIndexService
            db = await _make_db()
            try:
                await _seed_file(db)
                idx = FileIndexService(db)
                wrote = await idx.enrich_file_atomic(
                    "fi_1", user_id="usr_1",
                    description="A book about books",
                    thumbnail="data:image/jpeg;base64,XYZ",
                    embedding=b"\x00\x01\x02\x03",
                    stamp_attempt=True,
                )
                assert wrote is True

                cursor = await db.execute(
                    "SELECT description, thumbnail, embedding, "
                    "       last_enrichment_attempt "
                    "FROM file_index WHERE id = 'fi_1'"
                )
                row = await cursor.fetchone()
                assert row[0] == "A book about books"
                assert row[1] == "data:image/jpeg;base64,XYZ"
                assert row[2] == b"\x00\x01\x02\x03"
                assert row[3] is not None  # stamp set
            finally:
                await db.close()

        asyncio.run(_run())

    def test_source_metadata_merges_with_existing(self):
        async def _run():
            from augmentum.vfs.index import FileIndexService
            db = await _make_db()
            try:
                # Existing row already has cover_url from media_server adapter.
                await _seed_file(
                    db,
                    source_metadata=json.dumps({"cover_url": "/covers/orig.jpg"}),
                )
                idx = FileIndexService(db)
                # EPUB extractor adds author/publisher; cover_url must survive.
                await idx.enrich_file_atomic(
                    "fi_1", user_id="usr_1",
                    source_metadata_merge={"author": "Tolkien", "publisher": "Houghton Mifflin"},
                )
                cursor = await db.execute(
                    "SELECT source_metadata FROM file_index WHERE id = 'fi_1'"
                )
                row = await cursor.fetchone()
                merged = json.loads(row[0])
                assert merged["cover_url"] == "/covers/orig.jpg"
                assert merged["author"] == "Tolkien"
                assert merged["publisher"] == "Houghton Mifflin"
            finally:
                await db.close()

        asyncio.run(_run())

    def test_stamp_only(self):
        """stamp_attempt=True with no fields still writes (just the stamp)."""
        async def _run():
            from augmentum.vfs.index import FileIndexService
            db = await _make_db()
            try:
                await _seed_file(db)
                idx = FileIndexService(db)
                wrote = await idx.enrich_file_atomic(
                    "fi_1", user_id="usr_1", stamp_attempt=True,
                )
                assert wrote is True
                cursor = await db.execute(
                    "SELECT last_enrichment_attempt, description "
                    "FROM file_index WHERE id = 'fi_1'"
                )
                row = await cursor.fetchone()
                assert row[0] is not None
                assert row[1] == ""  # description untouched
            finally:
                await db.close()

        asyncio.run(_run())

    def test_single_transaction_envelope(self):
        """One BEGIN IMMEDIATE … COMMIT regardless of how many fields are set."""
        async def _run():
            from augmentum.vfs.index import FileIndexService
            db = await _make_db()
            try:
                await _seed_file(db)
                idx = FileIndexService(db)

                real_execute = db.execute
                real_commit = db.commit
                ops: list[str] = []

                async def _spy_execute(sql, *a, **kw):
                    first = sql.strip().split()[0].upper() if sql.strip() else ""
                    ops.append(("execute", first))
                    return await real_execute(sql, *a, **kw)

                async def _spy_commit(*a, **kw):
                    ops.append(("commit", ""))
                    return await real_commit(*a, **kw)

                db.execute = _spy_execute
                db.commit = _spy_commit
                try:
                    await idx.enrich_file_atomic(
                        "fi_1", user_id="usr_1",
                        description="x", thumbnail="y",
                        embedding=b"z", stamp_attempt=True,
                        source_metadata_merge={"k": "v"},
                    )
                finally:
                    db.execute = real_execute
                    db.commit = real_commit

                assert sum(1 for o in ops if o == ("commit", "")) == 1
                # First execute is BEGIN; the SELECT for source_metadata
                # merge happens inside the transaction; one final UPDATE.
                first_executes = [o[1] for o in ops if o[0] == "execute"]
                assert first_executes[0] == "BEGIN"
            finally:
                await db.close()

        asyncio.run(_run())


class TestFileIndexModels:
    def test_file_entry_to_dict(self):
        from augmentum.vfs.models import FileEntry
        e = FileEntry(
            id="fi_abc", user_id="usr_1", source="artifacts",
            source_id="art_1", name="report.pdf", mime_type="application/pdf",
            size_bytes=1024, description="A report",
        )
        d = e.to_dict()
        assert d["name"] == "report.pdf"
        assert d["source"] == "artifacts"
        assert "user_id" not in d  # Not in public dict

    def test_file_entry_to_card(self):
        from augmentum.vfs.models import FileEntry
        e = FileEntry(
            id="fi_abc", user_id="usr_1", source="images",
            source_id="img_1", name="sunset.png", mime_type="image/png",
            size_bytes=3_200_000, description="Sunset over ocean",
            tags=["vacation", "beach"], created_at="2026-03-15T12:00:00",
        )
        card = e.to_card()
        assert "sunset.png" in card
        assert "image/png" in card
        assert "3.1MB" in card or "3.2MB" in card
        assert "Sunset over ocean" in card
        assert "vacation" in card

    def test_human_size(self):
        from augmentum.vfs.models import _human_size
        assert _human_size(0) == "0B"
        assert _human_size(1023) == "1023B"
        assert _human_size(1024) == "1.0KB"
        assert _human_size(1_048_576) == "1.0MB"
        assert _human_size(3_200_000) == "3.1MB"

    def test_vfs_node(self):
        from augmentum.vfs.models import VFSNode
        n = VFSNode(path="/Artifacts", name="Artifacts", is_dir=True)
        assert n.is_dir is True
        assert n.path == "/Artifacts"

    def test_search_result(self):
        from augmentum.vfs.models import SearchResult
        r = SearchResult(source="file", item=None, score=0.95)
        assert r.score == 0.95
