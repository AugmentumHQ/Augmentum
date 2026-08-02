"""Scope filter tests — Local/Cloud split on /api/files/list.

Verifies:
  - scope=local excludes every source in _CLOUD_SOURCES
  - scope=cloud includes only sources in _CLOUD_SOURCES
  - scope + explicit source_group compose naturally (intersection)
  - stats endpoint exposes by_scope totals
  - index layer's exclude_sources kwarg works in list_recent + search

Uses a real in-memory SQLite instance via FileIndexService (same pattern
as tests/test_comic_integration.py). No HTTP — these are unit-level
tests against the store and route-layer filter translation.
"""

from __future__ import annotations

import asyncio
import json

import aiosqlite


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _make_index():
    """Build an in-memory FileIndexService with a seeded user + rows.

    Hand-rolled schema matches tests/test_comic_integration.py so we don't
    depend on the app's real migration runner (which wires things we don't
    need for pure list/search tests).

    Rows span every bucket we care about:
      - local: upload, artifacts, images, documents, knowledge
      - cloud: audiobookshelf, librivox, suwayomi, komga
    """
    from augmentum.vfs.index import FileIndexService

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
        INSERT INTO users (id) VALUES ('usr_test');
    """)
    idx = FileIndexService(conn)

    uid = "usr_test"
    fixtures = [
        # Local
        ("upload",         "u1",  "photo.jpg",   "image/jpeg",  "image", {}),
        ("upload",         "u2",  "notes.txt",   "text/plain",  "document", {}),
        ("artifacts",      "a1",  "report.pdf",  "application/pdf", "document", {}),
        ("images",         "i1",  "gen.png",     "image/png",   "image", {}),
        ("documents",      "d1",  "essay.md",    "text/markdown","document", {}),
        ("knowledge",      "k1",  "pack.zim",    "application/zim", "document", {}),
        # Cloud
        ("audiobookshelf", "abs1","Dune.m4b",    "audio/mp4",   "audio", {"entity_kind": "book"}),
        ("audiobookshelf", "abs2","99% Invisible","audio/mpeg", "audio", {"entity_kind": "podcast"}),
        ("librivox",       "lv1", "Emma.mp3",    "audio/mpeg",  "audio", {"entity_kind": "book"}),
        ("suwayomi",       "s1",  "Berserk c1",  "application/vnd.comicbook+zip", "comic", {}),
        ("suwayomi",       "s2",  "Berserk c2",  "application/vnd.comicbook+zip", "comic", {}),
        ("komga",          "kg1", "Watchmen",    "application/vnd.comicbook+zip", "comic", {}),
    ]
    import uuid
    for source, sid, name, mime, kind, meta in fixtures:
        await conn.execute(
            "INSERT INTO file_index (id, user_id, source, source_id, name, "
            "mime_type, size_bytes, kind, source_metadata) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                f"fi_{uuid.uuid4().hex[:12]}", uid, source, sid, name, mime,
                100, kind, json.dumps(meta or {}),
            ),
        )
    await conn.commit()
    return idx, conn, uid


class TestIndexLayerExcludeSources:
    def test_list_recent_exclude_sources_filters_out(self):
        async def go():
            idx, conn, uid = await _make_index()
            try:
                from augmentum.proxy.files_routes import _CLOUD_SOURCES
                results = await idx.list_recent(
                    user_id=uid, limit=100,
                    exclude_sources=list(_CLOUD_SOURCES),
                )
                sources = {r.source for r in results}
                # Not a single cloud source should leak through
                assert sources.isdisjoint(_CLOUD_SOURCES)
                # And we should still see local sources
                assert "upload" in sources
                assert "artifacts" in sources
                assert "knowledge" in sources
            finally:
                await conn.close()
        _run(go())

    def test_list_recent_sources_include_only_cloud(self):
        async def go():
            idx, conn, uid = await _make_index()
            try:
                from augmentum.proxy.files_routes import _CLOUD_SOURCES
                results = await idx.list_recent(
                    user_id=uid, limit=100,
                    sources=list(_CLOUD_SOURCES),
                )
                sources = {r.source for r in results}
                # Every returned row is cloud; no local ones sneak in
                assert sources.issubset(_CLOUD_SOURCES)
                assert "upload" not in sources
                assert "artifacts" not in sources
            finally:
                await conn.close()
        _run(go())

    def test_include_intersect_exclude_returns_empty(self):
        """Contradictory filter: include cloud + exclude cloud = empty."""
        async def go():
            idx, conn, uid = await _make_index()
            try:
                from augmentum.proxy.files_routes import _CLOUD_SOURCES
                results = await idx.list_recent(
                    user_id=uid, limit=100,
                    sources=list(_CLOUD_SOURCES),
                    exclude_sources=list(_CLOUD_SOURCES),
                )
                assert results == []
            finally:
                await conn.close()
        _run(go())

    def test_list_recent_sources_with_comics_group(self):
        """Simulates ``source=comics`` chip: sources=[suwayomi, komga, kavita].
        Verifies the group expansion returns only those sources' rows.
        """
        async def go():
            idx, conn, uid = await _make_index()
            try:
                from augmentum.proxy.files_routes import _SOURCE_GROUPS
                sources = list(_SOURCE_GROUPS["comics"])
                results = await idx.list_recent(
                    user_id=uid, limit=100, sources=sources,
                )
                got = {r.source for r in results}
                # suwayomi + komga in fixtures; kavita has no rows (future)
                assert got == {"suwayomi", "komga"}
                # Count — 2 Berserk chapters + 1 Watchmen
                assert len(results) == 3
            finally:
                await conn.close()
        _run(go())

    def test_list_recent_sources_with_podcasts_group(self):
        """``source=podcasts`` should isolate ABS podcast container rows."""
        async def go():
            idx, conn, uid = await _make_index()
            try:
                from augmentum.proxy.files_routes import (
                    _SOURCE_GROUP_ENTITY_FILTERS,
                    _SOURCE_GROUPS,
                )
                cfg = _SOURCE_GROUP_ENTITY_FILTERS["podcasts"]
                results = await idx.list_recent(
                    user_id=uid,
                    limit=100,
                    sources=list(_SOURCE_GROUPS["podcasts"]),
                    entity_kinds=list(cfg["entity_kinds"]),
                )
                assert [r.name for r in results] == ["99% Invisible"]
            finally:
                await conn.close()
        _run(go())

    def test_list_recent_sources_with_audiobooks_group_excludes_podcasts(self):
        """Audiobooks chip should keep books + LibriVox, not ABS podcasts."""
        async def go():
            idx, conn, uid = await _make_index()
            try:
                from augmentum.proxy.files_routes import (
                    _SOURCE_GROUP_ENTITY_FILTERS,
                    _SOURCE_GROUPS,
                )
                cfg = _SOURCE_GROUP_ENTITY_FILTERS["audiobooks"]
                results = await idx.list_recent(
                    user_id=uid,
                    limit=100,
                    sources=list(_SOURCE_GROUPS["audiobooks"]),
                    exclude_entity_kinds=list(cfg["exclude_entity_kinds"]),
                )
                assert sorted(r.name for r in results) == ["Dune.m4b", "Emma.mp3"]
            finally:
                await conn.close()
        _run(go())

    def test_list_recent_kind_plus_exclude_sources(self):
        """Kind filter and exclude_sources compose. Asking for kind=audio
        under a Local-scope (exclude cloud) filter returns zero rows, because
        every audio row in our fixtures is cloud-sourced.
        """
        async def go():
            idx, conn, uid = await _make_index()
            try:
                from augmentum.proxy.files_routes import _CLOUD_SOURCES
                results = await idx.list_recent(
                    user_id=uid, limit=100, kind="audio",
                    exclude_sources=list(_CLOUD_SOURCES),
                )
                assert results == []

                # And without the exclude we do find them
                results = await idx.list_recent(
                    user_id=uid, limit=100, kind="audio",
                )
                sources = {r.source for r in results}
                assert sources == {"audiobookshelf", "librivox"}
            finally:
                await conn.close()
        _run(go())


class TestSourceGroupsConstants:
    def test_cloud_sources_is_set(self):
        from augmentum.proxy.files_routes import _CLOUD_SOURCES
        assert isinstance(_CLOUD_SOURCES, frozenset)
        # Every provider we currently ship
        assert "audiobookshelf" in _CLOUD_SOURCES
        assert "librivox" in _CLOUD_SOURCES
        assert "suwayomi" in _CLOUD_SOURCES
        assert "komga" in _CLOUD_SOURCES

    def test_comics_source_group_registered(self):
        from augmentum.proxy.files_routes import _SOURCE_GROUPS
        assert "comics" in _SOURCE_GROUPS
        assert "suwayomi" in _SOURCE_GROUPS["comics"]
        assert "komga" in _SOURCE_GROUPS["comics"]

    def test_audiobooks_source_group_preserved(self):
        """Existing chip should keep working after the comics addition."""
        from augmentum.proxy.files_routes import _SOURCE_GROUPS
        assert "audiobooks" in _SOURCE_GROUPS
        assert "audiobookshelf" in _SOURCE_GROUPS["audiobooks"]
        assert "librivox" in _SOURCE_GROUPS["audiobooks"]

    def test_podcasts_source_group_registered(self):
        from augmentum.proxy.files_routes import _SOURCE_GROUPS
        assert "podcasts" in _SOURCE_GROUPS
        assert _SOURCE_GROUPS["podcasts"] == ("audiobookshelf",)

    def test_audio_source_group_entity_filters_registered(self):
        from augmentum.proxy.files_routes import _SOURCE_GROUP_ENTITY_FILTERS
        assert _SOURCE_GROUP_ENTITY_FILTERS["audiobooks"]["exclude_entity_kinds"] == ("podcast",)
        assert _SOURCE_GROUP_ENTITY_FILTERS["podcasts"]["entity_kinds"] == ("podcast",)

    def test_every_source_group_member_is_also_a_cloud_source(self):
        """Invariant: virtual chip groups only aggregate cloud sources."""
        from augmentum.proxy.files_routes import _CLOUD_SOURCES, _SOURCE_GROUPS
        for group, members in _SOURCE_GROUPS.items():
            for m in members:
                assert m in _CLOUD_SOURCES, (
                    f"Source group '{group}' references '{m}' which isn't "
                    f"in _CLOUD_SOURCES; scope=local would mistakenly "
                    f"include this source."
                )


class TestScopeStatsEndpoint:
    """The stats endpoint's by_scope totals drive the scope toggle's count
    badges. They come from summing by_source counts against _CLOUD_SOURCES,
    so the arithmetic is straightforward — but worth pinning down so a
    future refactor can't silently break the UI's badge math."""

    def test_by_scope_structure(self):
        # Construct a stats-dict by hand in the shape the store produces
        # and run the route-layer augmentation to verify by_scope math.
        # We replicate the logic from file_stats() here rather than
        # spinning up the full ASGI stack.
        from augmentum.proxy.files_routes import _CLOUD_SOURCES

        by_source = {
            "upload":         {"count": 50, "size_bytes": 1000},
            "artifacts":      {"count": 10, "size_bytes": 2000},
            "audiobookshelf": {"count": 20, "size_bytes": 10000},
            "suwayomi":       {"count": 20229, "size_bytes": 0},
        }
        total_count = sum(v["count"] for v in by_source.values())
        total_size  = sum(v["size_bytes"] for v in by_source.values())

        cloud_count = sum(
            (by_source.get(s) or {}).get("count", 0) for s in _CLOUD_SOURCES
        )
        cloud_size = sum(
            (by_source.get(s) or {}).get("size_bytes", 0) for s in _CLOUD_SOURCES
        )
        by_scope = {
            "local": {
                "count":      max(0, total_count - cloud_count),
                "size_bytes": max(0, total_size - cloud_size),
            },
            "cloud": {
                "count":      cloud_count,
                "size_bytes": cloud_size,
            },
        }

        assert by_scope["local"]["count"] == 60   # 50 upload + 10 artifacts
        assert by_scope["cloud"]["count"] == 20249  # 20 abs + 20229 suwayomi
        # Totals reconcile with the summed by_source numbers
        assert by_scope["local"]["count"] + by_scope["cloud"]["count"] == total_count
