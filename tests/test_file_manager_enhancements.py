"""Tests for file manager enhancements: favorites, trash, tags."""

from augmentum.vfs.models import FileEntry


class TestFileEntryEnhancements:
    def test_to_dict_includes_favorite_and_trash(self):
        entry = FileEntry(
            id="fi_abc", user_id="usr_1", source="artifacts",
            source_id="a1", name="test.txt",
            is_favorite=True, is_trashed=False,
        )
        d = entry.to_dict()
        assert d["is_favorite"] is True
        assert d["is_trashed"] is False
        assert "updated_at" in d

    def test_to_dict_excludes_internal_fields(self):
        entry = FileEntry(
            id="fi_abc", user_id="usr_1", source="artifacts",
            source_id="a1", name="test.txt",
            trashed_at="2026-04-07T12:00:00",
        )
        d = entry.to_dict()
        assert "trashed_at" not in d
        assert "user_id" not in d
        assert "real_path" not in d
        assert "embedding" not in d

    def test_default_values(self):
        entry = FileEntry(
            id="fi_abc", user_id="usr_1", source="images",
            source_id="i1", name="photo.png",
        )
        assert entry.is_favorite is False
        assert entry.is_trashed is False
        assert entry.trashed_at is None



import aiosqlite

_SCHEMA = """
CREATE TABLE file_index (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    name TEXT NOT NULL,
    mime_type TEXT DEFAULT '',
    size_bytes INTEGER DEFAULT 0,
    real_path TEXT,
    description TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    thumbnail TEXT,
    embedding BLOB,
    is_directory INTEGER DEFAULT 0,
    parent_id TEXT,
    source_metadata TEXT DEFAULT '{}',
    kind TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    is_favorite INTEGER NOT NULL DEFAULT 0,
    is_trashed INTEGER NOT NULL DEFAULT 0,
    trashed_at TEXT,
    UNIQUE(user_id, source, source_id)
)
"""

async def _make_index():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(_SCHEMA)
    await conn.commit()
    from augmentum.vfs.index import FileIndexService
    return conn, FileIndexService(conn)


class TestFavorites:
    async def test_toggle_favorite_on(self):
        conn, idx = await _make_index()
        fid = await idx.register(user_id="u1", source="artifacts", source_id="a1", name="test.txt")
        result = await idx.toggle_favorite(fid, user_id="u1")
        assert result is True
        entry = await idx.get(fid, user_id="u1")
        assert entry.is_favorite is True
        await conn.close()

    async def test_toggle_favorite_off(self):
        conn, idx = await _make_index()
        fid = await idx.register(user_id="u1", source="artifacts", source_id="a1", name="test.txt")
        await idx.toggle_favorite(fid, user_id="u1")
        await idx.toggle_favorite(fid, user_id="u1")
        entry = await idx.get(fid, user_id="u1")
        assert entry.is_favorite is False
        await conn.close()

    async def test_list_favorites(self):
        conn, idx = await _make_index()
        f1 = await idx.register(user_id="u1", source="artifacts", source_id="a1", name="fav.txt")
        await idx.register(user_id="u1", source="artifacts", source_id="a2", name="nope.txt")
        await idx.toggle_favorite(f1, user_id="u1")
        favs = await idx.list_favorites(user_id="u1")
        assert len(favs) == 1
        assert favs[0].name == "fav.txt"
        await conn.close()


class TestTrash:
    async def test_soft_delete(self):
        conn, idx = await _make_index()
        fid = await idx.register(user_id="u1", source="artifacts", source_id="a1", name="del.txt")
        ok = await idx.soft_delete(fid, user_id="u1")
        assert ok is True
        entry = await idx.get(fid, user_id="u1")
        assert entry.is_trashed is True
        assert entry.trashed_at is not None
        await conn.close()

    async def test_soft_deleted_excluded_from_recent(self):
        conn, idx = await _make_index()
        await idx.register(user_id="u1", source="artifacts", source_id="a1", name="visible.txt")
        fid2 = await idx.register(user_id="u1", source="artifacts", source_id="a2", name="hidden.txt")
        await idx.soft_delete(fid2, user_id="u1")
        recent = await idx.list_recent(user_id="u1")
        names = [f.name for f in recent]
        assert "visible.txt" in names
        assert "hidden.txt" not in names
        await conn.close()

    async def test_restore(self):
        conn, idx = await _make_index()
        fid = await idx.register(user_id="u1", source="artifacts", source_id="a1", name="restore.txt")
        await idx.soft_delete(fid, user_id="u1")
        ok = await idx.restore(fid, user_id="u1")
        assert ok is True
        entry = await idx.get(fid, user_id="u1")
        assert entry.is_trashed is False
        assert entry.trashed_at is None
        await conn.close()

    async def test_list_trash(self):
        conn, idx = await _make_index()
        fid = await idx.register(user_id="u1", source="artifacts", source_id="a1", name="trashed.txt")
        await idx.register(user_id="u1", source="artifacts", source_id="a2", name="alive.txt")
        await idx.soft_delete(fid, user_id="u1")
        trash = await idx.list_trash(user_id="u1")
        assert len(trash) == 1
        assert trash[0].name == "trashed.txt"
        await conn.close()

    async def test_purge(self):
        conn, idx = await _make_index()
        fid = await idx.register(user_id="u1", source="artifacts", source_id="a1", name="gone.txt")
        await idx.soft_delete(fid, user_id="u1")
        deleted = await idx.purge_trash(user_id="u1")
        assert deleted == 1
        entry = await idx.get(fid, user_id="u1")
        assert entry is None
        await conn.close()


class TestUpdateTags:
    async def test_update_tags(self):
        conn, idx = await _make_index()
        fid = await idx.register(user_id="u1", source="artifacts", source_id="a1", name="tag.txt")
        ok = await idx.update_tags(fid, tags=["python", "code"], user_id="u1")
        assert ok is True
        entry = await idx.get(fid, user_id="u1")
        assert entry.tags == ["python", "code"]
        await conn.close()


class TestRoutesExist:
    def test_new_routes_registered(self):
        from augmentum.proxy.files_routes import router
        prefix = router.prefix  # "/api/files"
        paths = [r.path for r in router.routes]
        assert f"{prefix}/favorite/{{file_id}}" in paths
        assert f"{prefix}/trash" in paths
        assert f"{prefix}/restore/{{file_id}}" in paths
        assert f"{prefix}/purge-trash" in paths
        assert f"{prefix}/tags/{{file_id}}" in paths
        assert f"{prefix}/zip" in paths
        assert f"{prefix}/summarize/{{file_id}}" in paths


class TestSourceGroups:
    """The Audiobooks chip is a virtual slug that expands to a set of
    concrete row sources (ABS + LibriVox today). Makes sure the
    `sources=[...]` list filter on list_recent/search includes rows from
    every listed provider and excludes rows from other sources."""

    async def test_list_recent_with_sources_list_unions_rows(self):
        conn, idx = await _make_index()
        await idx.register(user_id="u1", source="audiobookshelf",
                           source_id="abs1", name="Dune.mp3")
        await idx.register(user_id="u1", source="librivox",
                           source_id="lv1", name="Odyssey.mp3")
        await idx.register(user_id="u1", source="artifacts",
                           source_id="a1", name="notes.txt")
        out = await idx.list_recent(
            user_id="u1",
            sources=["audiobookshelf", "librivox"],
        )
        names = sorted(f.name for f in out)
        assert names == ["Dune.mp3", "Odyssey.mp3"]
        await conn.close()

    async def test_sources_list_excludes_non_members(self):
        conn, idx = await _make_index()
        await idx.register(user_id="u1", source="artifacts",
                           source_id="a1", name="notes.txt")
        out = await idx.list_recent(
            user_id="u1",
            sources=["audiobookshelf", "librivox"],
        )
        assert out == []
        await conn.close()

    async def test_sources_wins_over_source_when_both_given(self):
        conn, idx = await _make_index()
        await idx.register(user_id="u1", source="audiobookshelf",
                           source_id="abs1", name="abs.mp3")
        await idx.register(user_id="u1", source="librivox",
                           source_id="lv1", name="lv.mp3")
        # source="artifacts" would exclude both rows if it won; with
        # `sources` winning we should still see both media rows.
        out = await idx.list_recent(
            user_id="u1",
            source="artifacts",
            sources=["audiobookshelf", "librivox"],
        )
        assert len(out) == 2
        await conn.close()

    def test_route_layer_defines_audiobooks_group(self):
        from augmentum.proxy.files_routes import _SOURCE_GROUPS
        assert "audiobooks" in _SOURCE_GROUPS
        members = set(_SOURCE_GROUPS["audiobooks"])
        # Must match the frontend MEDIA_SOURCES set in files/state.js and
        # the backend _MEDIA_SOURCES set in vfs/index.py.
        assert members == {"audiobookshelf", "librivox"}
