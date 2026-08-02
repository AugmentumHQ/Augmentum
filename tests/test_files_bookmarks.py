"""Tests for the BookmarksAdapter — saved external URLs in file_index.

The adapter is the meat of the bookmarks feature; the route is a thin
wrapper. Testing the adapter directly avoids spinning up the full app.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from augmentum.vfs.adapters.bookmarks import BookmarksAdapter, bookmark_id


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _setup_db():
    """In-memory file_index with the columns + FTS triggers the adapter
    relies on. Mirrors the prod schema (074 + 075 + 085 migrations) but
    inlined so the test has no migration runner dependency."""
    import aiosqlite

    conn = await aiosqlite.connect(":memory:")
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
        CREATE UNIQUE INDEX idx_file_index_source_unique
            ON file_index(user_id, source, source_id);
        CREATE VIRTUAL TABLE file_index_fts USING fts5(
            name, description, tags,
            content=file_index, content_rowid=rowid
        );
        CREATE TRIGGER fts_ins AFTER INSERT ON file_index BEGIN
            INSERT INTO file_index_fts(rowid, name, description, tags)
            VALUES (new.rowid, new.name, new.description, new.tags);
        END;
    """)
    return conn


def _bind_index(conn):
    """The adapter calls register_file()/unregister_file() which are
    module-level helpers reading the global FileIndexService. Wire that up."""
    from augmentum.vfs import set_file_index
    from augmentum.vfs.index import FileIndexService
    set_file_index(FileIndexService(conn))


# --- bookmark_id (deterministic) ----------------------------------------

class TestBookmarkId:
    def test_same_url_same_id(self):
        a = bookmark_id("https://www.youtube.com/watch?v=abc123")
        b = bookmark_id("https://www.youtube.com/watch?v=abc123")
        assert a == b

    def test_different_urls_distinct_ids(self):
        a = bookmark_id("https://www.youtube.com/watch?v=abc")
        b = bookmark_id("https://www.youtube.com/watch?v=xyz")
        assert a != b

    def test_id_format(self):
        bid = bookmark_id("https://example.com/x")
        assert bid.startswith("bm_")
        assert len(bid) == 19  # "bm_" + 16 hex


# --- save() ------------------------------------------------------------

class TestBookmarksAdapterSave:
    def test_save_creates_file_index_row(self):
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)

            info = await adapter.save(
                url="https://www.youtube.com/watch?v=foo",
                title="Test Video",
                user_id="u1",
                thumbnail="https://i.ytimg.com/vi/foo/hq.jpg",
                channel="Test Channel",
                duration=300,
                platform="youtube",
                video_id="foo",
            )
            assert info["url"] == "https://www.youtube.com/watch?v=foo"
            assert info["channel"] == "Test Channel"

            # Row should exist with kind=video.
            cursor = await conn.execute(
                "SELECT name, kind, source, mime_type, source_metadata "
                "FROM file_index WHERE source = 'bookmarks' AND user_id = 'u1'",
            )
            rows = await cursor.fetchall()
            assert len(rows) == 1
            name, kind, source, mime, meta_json = rows[0]
            assert name == "Test Video"
            assert kind == "video"
            assert source == "bookmarks"
            assert mime == "application/x-bookmark"
            meta = json.loads(meta_json)
            assert meta["url"] == "https://www.youtube.com/watch?v=foo"
            assert meta["platform"] == "youtube"
            assert meta["video_id"] == "foo"
            assert meta["thumbnail"].endswith("hq.jpg")
            await conn.close()
        _run(go())

    def test_resave_same_url_is_idempotent(self):
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)

            await adapter.save(
                url="https://example.com/v", title="First", user_id="u1",
            )
            await adapter.save(
                url="https://example.com/v", title="Updated", user_id="u1",
                channel="ChannelB",
            )

            cursor = await conn.execute(
                "SELECT COUNT(*), name FROM file_index WHERE source = 'bookmarks'",
            )
            count, name = await cursor.fetchone()
            assert count == 1
            assert name == "Updated"  # second save updated the title
            await conn.close()
        _run(go())

    def test_resave_clears_trashed_state(self):
        # Re-saving a URL that was previously trashed should restore it,
        # not stay in the trash bin.
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)

            await adapter.save(url="https://example.com/x", title="X", user_id="u1")
            # Soft-delete via direct UPDATE (mirrors the index.soft_delete path)
            await conn.execute(
                "UPDATE file_index SET is_trashed = 1, trashed_at = datetime('now') "
                "WHERE source = 'bookmarks'",
            )
            await conn.commit()

            await adapter.save(url="https://example.com/x", title="X again", user_id="u1")

            cursor = await conn.execute(
                "SELECT is_trashed, trashed_at FROM file_index WHERE source = 'bookmarks'",
            )
            trashed, when = await cursor.fetchone()
            assert trashed == 0
            assert when is None
            await conn.close()
        _run(go())

    def test_url_required(self):
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)
            with pytest.raises(ValueError):
                await adapter.save(url="", title="x", user_id="u1")
            await conn.close()
        _run(go())

    def test_user_id_required(self):
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)
            with pytest.raises(ValueError):
                await adapter.save(url="https://x.com/", title="x", user_id="")
            await conn.close()
        _run(go())

    def test_url_must_be_http(self):
        # SSRF guard — file:/// or javascript: shouldn't make it past save.
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)
            for bad in ("file:///etc/passwd", "javascript:alert(1)", "data:,"):
                with pytest.raises(ValueError):
                    await adapter.save(url=bad, title="x", user_id="u1")
            await conn.close()
        _run(go())


# --- delete() / list_source_ids() / resolve() ---------------------------

class TestBookmarksAdapterProtocol:
    def test_delete_removes_row(self):
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)

            info = await adapter.save(
                url="https://example.com/v", title="X", user_id="u1",
            )
            assert await adapter.delete(info["source_id"], user_id="u1")

            cursor = await conn.execute("SELECT COUNT(*) FROM file_index")
            assert (await cursor.fetchone())[0] == 0
            await conn.close()
        _run(go())

    def test_list_source_ids_user_scoped(self):
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)

            await adapter.save(url="https://a.com/", title="A", user_id="u1")
            await adapter.save(url="https://b.com/", title="B", user_id="u1")
            await adapter.save(url="https://c.com/", title="C", user_id="u2")

            u1_ids = await adapter.list_source_ids(user_id="u1")
            u2_ids = await adapter.list_source_ids(user_id="u2")
            assert len(u1_ids) == 2
            assert len(u2_ids) == 1
            assert set(u1_ids).isdisjoint(set(u2_ids))
            await conn.close()
        _run(go())

    def test_resolve_returns_none(self):
        # Bookmarks have no on-disk file — resolve() always returns None
        # so /download won't try to stream them.
        async def go():
            conn = await _setup_db()
            _bind_index(conn)
            adapter = BookmarksAdapter(conn)
            info = await adapter.save(url="https://x.com/", title="x", user_id="u1")
            assert await adapter.resolve(info["source_id"], user_id="u1") is None
            await conn.close()
        _run(go())
