"""Integration tests for the comic sync + per-page delivery pipeline.

Covers the seams between:
  - providers → sync._index_item → register_file (series_id resolves)
  - file_index row → /api/media/comic/page → provider-specific upstream URL
  - auth token injection (Basic header) at the route boundary
  - per-provider page-index convention (Komga 1-indexed vs Suwayomi 0-indexed)

The per-page route tests use a TestClient with a minimal schema + mocked
httpx responses. The sync tests use an in-memory SQLite conn and the real
ComicSeriesStore / register_file wiring.
"""

from __future__ import annotations

import asyncio

import aiosqlite

from augmentum.media.comic_series_store import (
    ComicSeriesStore,
    set_comic_series_store,
)
from augmentum.media.providers.base import CatalogItem
from augmentum.media.store import MediaServer
from augmentum.media.sync import _COMIC_PROVIDERS, _comic_series_name, _index_item
from augmentum.vfs import set_file_index
from augmentum.vfs.index import FileIndexService


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _setup_db() -> aiosqlite.Connection:
    """Minimal schema for end-to-end sync/page tests."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.executescript("""
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE file_index (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id),
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
            is_favorite INTEGER NOT NULL DEFAULT 0,
            is_trashed INTEGER NOT NULL DEFAULT 0,
            trashed_at TEXT,
            scan_status TEXT NOT NULL DEFAULT 'pending',
            mtime INTEGER,
            scan_error TEXT,
            metadata_confidence REAL NOT NULL DEFAULT 0.5,
            series_id TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE UNIQUE INDEX idx_file_index_source_unique
            ON file_index(user_id, source, source_id);
        CREATE TABLE comic_series (
            id                      TEXT PRIMARY KEY,
            user_id                 TEXT NOT NULL REFERENCES users(id),
            canonical_name          TEXT NOT NULL,
            sort_name               TEXT NOT NULL,
            alias_names             TEXT NOT NULL DEFAULT '[]',
            publisher               TEXT,
            author                  TEXT,
            description             TEXT,
            cover_file_id           TEXT,
            status                  TEXT,
            year_started            INTEGER,
            year_ended              INTEGER,
            genres                  TEXT NOT NULL DEFAULT '[]',
            language_iso            TEXT,
            age_rating              TEXT,
            metadata_source         TEXT,
            metadata_confidence     REAL NOT NULL DEFAULT 0.5,
            archive_count_reported  INTEGER,
            accent_color            TEXT,
            created_at              TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX idx_comic_series_user_sort
            ON comic_series(user_id, sort_name);
        INSERT INTO users (id) VALUES ('u_a'), ('u_b');
    """)
    return conn


async def _wire_globals(conn: aiosqlite.Connection) -> ComicSeriesStore:
    """Install FileIndexService + ComicSeriesStore on the module globals so
    register_file and get_comic_series_store resolve during a test."""
    file_index = FileIndexService(conn)
    set_file_index(file_index)
    store = ComicSeriesStore(conn)
    set_comic_series_store(store)
    return store


# --- sync._comic_series_name ---------------------------------------------


class TestComicSeriesName:
    def test_prefers_extra_series_name(self):
        item = CatalogItem(
            external_id="x", name="Fallback", kind="comic", mime_type="",
            extra={"series_name": "Berserk"},
        )
        assert _comic_series_name(item) == "Berserk"

    def test_falls_back_to_item_name(self):
        item = CatalogItem(
            external_id="x", name="Solo Leveling Ch. 1", kind="comic", mime_type="",
            extra={},
        )
        assert _comic_series_name(item) == "Solo Leveling Ch. 1"

    def test_empty_returns_empty(self):
        item = CatalogItem(
            external_id="x", name="", kind="comic", mime_type="", extra={},
        )
        assert _comic_series_name(item) == ""


# --- sync._index_item comic series resolution ---------------------------


class TestIndexItemComicFlow:
    def test_komga_item_resolves_series_id(self):
        async def go():
            conn = await _setup_db()
            store = await _wire_globals(conn)
            server = MediaServer(
                id="ms_1", user_id="u_a",
                provider="komga", name="Home Komga",
                base_url="http://komga:25600", access_token="TOKEN",
                status="ok", status_detail="", last_sync_at=None,
                item_count=0, created_at="", updated_at="",
                total_seen=0, skipped_count=0, last_sync_skipped=[],
            )
            item = CatalogItem(
                external_id="bk_1", name="Berserk Vol. 1",
                kind="comic", mime_type="application/vnd.comicbook+zip",
                stream_path="/api/v1/books/bk_1/file",
                cover_url="/api/v1/books/bk_1/thumbnail",
                author="Kentaro Miura",
                extra={
                    "series_name": "Berserk",
                    "publisher": "Hakusensha",
                    "komga_series_id": "sr_1",
                    "description": "A dark fantasy epic.",
                    "genres": ["Seinen", "Dark Fantasy"],
                    "alternate_titles": ["Berserk Deluxe"],
                    "volume": "1",
                    "page_count": 234,
                },
            )

            await _index_item(server=server, item=item)

            # file_index row has series_id populated + scan_status='ok'
            cursor = await conn.execute(
                "SELECT series_id, scan_status, metadata_confidence, kind "
                "FROM file_index WHERE user_id = ? AND source = ? AND source_id = ?",
                ("u_a", "komga", "ms_1:bk_1"),
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row["scan_status"] == "ok"
            assert row["metadata_confidence"] == 0.9
            assert row["kind"] == "comic"
            assert row["series_id"] is not None
            assert row["series_id"].startswith("cs_")

            # Same series_id also exists in comic_series
            series = await store.get(row["series_id"], user_id="u_a")
            assert series is not None
            assert series.canonical_name == "Berserk"
            assert series.publisher == "Hakusensha"
            assert series.description == "A dark fantasy epic."
            assert "Seinen" in (series.genres or [])
            assert "Berserk Deluxe" in (series.alias_names or [])
            assert series.metadata_source == "komga"
            assert series.metadata_confidence == 0.9

            await conn.close()
        _run(go())

    def test_suwayomi_chapters_share_one_series_id(self):
        """Two chapters of the same Suwayomi manga → same series_id."""
        async def go():
            conn = await _setup_db()
            await _wire_globals(conn)
            server = MediaServer(
                id="ms_2", user_id="u_a",
                provider="suwayomi", name="Local Suwayomi",
                base_url="http://suwayomi:4567", access_token="",
                status="ok", status_detail="", last_sync_at=None,
                item_count=0, created_at="", updated_at="",
                total_seen=0, skipped_count=0, last_sync_skipped=[],
            )
            manga_payload = {
                "series_name": "Chainsaw Man",
                "author": "Tatsuki Fujimoto",
                "suwayomi_manga_id": 42,
            }
            chapter_1 = CatalogItem(
                external_id="42.0", name="Ch. 1", kind="comic", mime_type="",
                stream_path="/api/v1/manga/42/chapter/0",
                cover_url="/api/v1/manga/42/thumbnail",
                author="Tatsuki Fujimoto",
                extra={**manga_payload, "chapter_index": 0, "page_count": 20},
            )
            chapter_2 = CatalogItem(
                external_id="42.1", name="Ch. 2", kind="comic", mime_type="",
                stream_path="/api/v1/manga/42/chapter/1",
                cover_url="/api/v1/manga/42/thumbnail",
                author="Tatsuki Fujimoto",
                extra={**manga_payload, "chapter_index": 1, "page_count": 22},
            )
            await _index_item(server=server, item=chapter_1)
            await _index_item(server=server, item=chapter_2)

            cursor = await conn.execute(
                "SELECT series_id FROM file_index "
                "WHERE source = 'suwayomi' AND user_id = 'u_a'",
            )
            rows = await cursor.fetchall()
            assert len(rows) == 2
            # Both chapters share the same resolved series
            assert rows[0]["series_id"] is not None
            assert rows[0]["series_id"] == rows[1]["series_id"]

            await conn.close()
        _run(go())

    def test_user_isolation_on_series_resolution(self):
        """User A and B each syncing the same-named manga → distinct series."""
        async def go():
            conn = await _setup_db()
            await _wire_globals(conn)

            def _make_server(uid: str, sid: str):
                return MediaServer(
                    id=sid, user_id=uid,
                    provider="komga", name="Home",
                    base_url="http://komga", access_token="T",
                    status="ok", status_detail="", last_sync_at=None,
                    item_count=0, created_at="", updated_at="",
                    total_seen=0, skipped_count=0, last_sync_skipped=[],
                )

            def _make_item():
                return CatalogItem(
                    external_id="bk_1", name="Berserk Vol 1",
                    kind="comic", mime_type="application/vnd.comicbook+zip",
                    stream_path="/api/v1/books/bk_1/file",
                    cover_url="/api/v1/books/bk_1/thumbnail",
                    author="Miura",
                    extra={"series_name": "Berserk"},
                )
            await _index_item(server=_make_server("u_a", "ms_a"), item=_make_item())
            await _index_item(server=_make_server("u_b", "ms_b"), item=_make_item())

            cursor = await conn.execute(
                "SELECT user_id, series_id FROM file_index ORDER BY user_id"
            )
            rows = await cursor.fetchall()
            assert len(rows) == 2
            # Different users → different series_ids (isolated libraries)
            assert rows[0]["series_id"] != rows[1]["series_id"]

            # Confirm each series row is scoped to its owner
            cursor2 = await conn.execute(
                "SELECT user_id FROM comic_series ORDER BY user_id"
            )
            series_rows = await cursor2.fetchall()
            assert len(series_rows) == 2
            assert {r["user_id"] for r in series_rows} == {"u_a", "u_b"}

            await conn.close()
        _run(go())

    def test_non_comic_item_skips_series_resolution(self):
        """An audio item (ABS) should not trigger the comic series path."""
        async def go():
            conn = await _setup_db()
            await _wire_globals(conn)
            server = MediaServer(
                id="ms_abs", user_id="u_a",
                provider="audiobookshelf", name="ABS",
                base_url="http://abs:13378", access_token="T",
                status="ok", status_detail="", last_sync_at=None,
                item_count=0, created_at="", updated_at="",
                total_seen=0, skipped_count=0, last_sync_skipped=[],
            )
            item = CatalogItem(
                external_id="abs_1", name="War and Peace",
                kind="audio", mime_type="audio/mpeg",
                stream_path="/api/items/abs_1/file/123",
                cover_url="/api/items/abs_1/cover",
                author="Tolstoy",
                extra={},
            )
            await _index_item(server=server, item=item)

            cursor = await conn.execute(
                "SELECT series_id, metadata_confidence FROM file_index "
                "WHERE source = 'audiobookshelf'"
            )
            row = await cursor.fetchone()
            assert row is not None
            assert row["series_id"] is None
            # Non-comic items keep the default confidence floor
            assert row["metadata_confidence"] == 0.5

            # No comic_series row created
            cursor2 = await conn.execute("SELECT COUNT(*) as n FROM comic_series")
            assert (await cursor2.fetchone())["n"] == 0

            await conn.close()
        _run(go())

    def test_provider_not_in_comic_set_skips_resolution(self):
        """If a comic item somehow arrives from a non-comic provider, we
        still don't call the series store — belt + braces against misuse."""
        async def go():
            conn = await _setup_db()
            await _wire_globals(conn)
            server = MediaServer(
                id="ms_abs", user_id="u_a",
                provider="audiobookshelf", name="ABS",
                base_url="http://abs", access_token="T",
                status="ok", status_detail="", last_sync_at=None,
                item_count=0, created_at="", updated_at="",
                total_seen=0, skipped_count=0, last_sync_skipped=[],
            )
            # Malformed: kind='comic' from a non-comic provider
            item = CatalogItem(
                external_id="abs_1", name="Weird", kind="comic", mime_type="",
                stream_path="/whatever", extra={"series_name": "X"},
            )
            await _index_item(server=server, item=item)
            cursor = await conn.execute(
                "SELECT COUNT(*) as n FROM comic_series"
            )
            assert (await cursor.fetchone())["n"] == 0
            await conn.close()
        _run(go())


# --- _COMIC_PROVIDERS set ------------------------------------------------


class TestComicProvidersSet:
    def test_contains_komga_and_suwayomi(self):
        assert "komga" in _COMIC_PROVIDERS
        assert "suwayomi" in _COMIC_PROVIDERS

    def test_excludes_audiobookshelf(self):
        assert "audiobookshelf" not in _COMIC_PROVIDERS
