"""Smoke + contract tests for comic_series_store.

Covers:
  - derive_sort_name pure function (article stripping, whitespace, case)
  - create_or_resolve_series idempotence (same name → same id)
  - user_id isolation (A's series invisible to B even if name matches)
  - update_series partial fields, name → sort_name sync, other fields preserved
  - list_series pagination + sort
"""

from __future__ import annotations

import asyncio
import json

import aiosqlite
import pytest

from augmentum.media.comic_series_store import (
    ComicSeriesStore,
    derive_sort_name,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _setup_db() -> aiosqlite.Connection:
    """Minimal pre-101 + post-101 schema for comic_series tests."""
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.executescript("""
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
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


# --- derive_sort_name ---------------------------------------------------


class TestDeriveSortName:
    def test_lowercases(self):
        assert derive_sort_name("Berserk") == "berserk"

    def test_strips_leading_the(self):
        assert derive_sort_name("The Walking Dead") == "walking dead"

    def test_strips_leading_a(self):
        assert derive_sort_name("A Silent Voice") == "silent voice"

    def test_strips_leading_an(self):
        assert derive_sort_name("An Adventure Tale") == "adventure tale"

    def test_article_stripping_is_case_insensitive(self):
        assert derive_sort_name("THE Walking Dead") == "walking dead"

    def test_preserves_embedded_the(self):
        # Only leading articles strip; embedded "the" stays
        assert derive_sort_name("Beauty and the Beast") == "beauty and the beast"

    def test_collapses_whitespace(self):
        assert derive_sort_name("Berserk    Volume") == "berserk volume"

    def test_preserves_punctuation(self):
        # Punctuation differences can mean different series
        # (e.g. JoJo's vs JoJos — keep that signal)
        assert derive_sort_name("JoJo's Bizarre Adventure") == "jojo's bizarre adventure"

    def test_empty_string(self):
        assert derive_sort_name("") == ""

    def test_whitespace_only(self):
        assert derive_sort_name("   ") == ""


# --- create_or_resolve_series -------------------------------------------


class TestCreateOrResolve:
    def test_first_call_creates(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            series_id = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            assert series_id.startswith("cs_")
            row = await store.get(series_id, user_id="u_a")
            assert row is not None
            assert row.canonical_name == "Berserk"
            assert row.sort_name == "berserk"
            assert row.metadata_source == "filename"
            assert row.metadata_confidence == 0.5
            await conn.close()
        _run(go())

    def test_second_call_returns_same_id(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            id1 = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            id2 = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            assert id1 == id2
            assert await store.count(user_id="u_a") == 1
            await conn.close()
        _run(go())

    def test_resolves_via_sort_name(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            # "The Walking Dead" and "Walking Dead" should resolve to the
            # same series — the leading article is stripped by sort_name.
            id1 = await store.create_or_resolve_series(
                user_id="u_a", name="The Walking Dead",
            )
            id2 = await store.create_or_resolve_series(
                user_id="u_a", name="Walking Dead",
            )
            assert id1 == id2
            await conn.close()
        _run(go())

    def test_case_insensitive(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            id1 = await store.create_or_resolve_series(
                user_id="u_a", name="berserk",
            )
            id2 = await store.create_or_resolve_series(
                user_id="u_a", name="BERSERK",
            )
            assert id1 == id2
            await conn.close()
        _run(go())

    def test_user_isolation(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            id_a = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            id_b = await store.create_or_resolve_series(
                user_id="u_b", name="Berserk",
            )
            # Same name, different users → different IDs (isolated libraries)
            assert id_a != id_b
            # User B can't read user A's series even with the right ID
            assert await store.get(id_a, user_id="u_b") is None
            assert await store.get(id_b, user_id="u_a") is None
            await conn.close()
        _run(go())

    def test_idempotent_does_not_overwrite_metadata(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            # First call with high confidence + publisher
            series_id = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
                metadata_source="comicinfo",
                metadata_confidence=1.0,
                publisher="Hakusensha",
            )
            # Second call with low confidence + no publisher should NOT
            # downgrade the existing metadata — resolve is pure identity
            await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
                metadata_source="filename",
                metadata_confidence=0.3,
            )
            row = await store.get(series_id, user_id="u_a")
            assert row is not None
            assert row.metadata_source == "comicinfo"
            assert row.metadata_confidence == 1.0
            assert row.publisher == "Hakusensha"
            await conn.close()
        _run(go())

    def test_rejects_empty_name(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            with pytest.raises(ValueError):
                await store.create_or_resolve_series(
                    user_id="u_a", name="",
                )
            with pytest.raises(ValueError):
                await store.create_or_resolve_series(
                    user_id="u_a", name="   ",
                )
            await conn.close()
        _run(go())

    def test_rejects_empty_user(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            with pytest.raises(ValueError):
                await store.create_or_resolve_series(
                    user_id="", name="Berserk",
                )
            await conn.close()
        _run(go())


# --- update_series ------------------------------------------------------


class TestUpdateSeries:
    def test_partial_update_preserves_other_fields(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            series_id = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
                publisher="Hakusensha",
            )
            ok = await store.update_series(
                series_id, user_id="u_a",
                description="Dark fantasy manga.",
            )
            assert ok is True
            row = await store.get(series_id, user_id="u_a")
            assert row.description == "Dark fantasy manga."
            assert row.publisher == "Hakusensha"  # preserved
            assert row.canonical_name == "Berserk"  # preserved
            await conn.close()
        _run(go())

    def test_rename_updates_sort_name(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            series_id = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            await store.update_series(
                series_id, user_id="u_a",
                canonical_name="Berserk (Kodansha)",
            )
            row = await store.get(series_id, user_id="u_a")
            assert row.canonical_name == "Berserk (Kodansha)"
            assert row.sort_name == "berserk (kodansha)"
            # But series_id is stable — the same series, renamed
            resolved = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk (Kodansha)",
            )
            assert resolved == series_id
            await conn.close()
        _run(go())

    def test_genres_stored_as_json(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            series_id = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            await store.update_series(
                series_id, user_id="u_a",
                genres=["Seinen", "Dark Fantasy", "Horror"],
            )
            row = await store.get(series_id, user_id="u_a")
            assert row.genres == ["Seinen", "Dark Fantasy", "Horror"]
            # Verify round-trip through JSON column
            cursor = await conn.execute(
                "SELECT genres FROM comic_series WHERE id = ?", (series_id,),
            )
            raw = await cursor.fetchone()
            assert json.loads(raw["genres"]) == ["Seinen", "Dark Fantasy", "Horror"]
            await conn.close()
        _run(go())

    def test_user_isolation_on_update(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            series_id = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            # User B cannot update user A's series
            ok = await store.update_series(
                series_id, user_id="u_b",
                description="Hack attempt.",
            )
            assert ok is False
            row = await store.get(series_id, user_id="u_a")
            assert row.description is None
            await conn.close()
        _run(go())

    def test_empty_update_returns_false(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            series_id = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            ok = await store.update_series(series_id, user_id="u_a")
            assert ok is False
            await conn.close()
        _run(go())


# --- list_series --------------------------------------------------------


class TestListSeries:
    def test_sort_by_name(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            await store.create_or_resolve_series(user_id="u_a", name="Berserk")
            await store.create_or_resolve_series(user_id="u_a", name="Akira")
            await store.create_or_resolve_series(user_id="u_a", name="The Walking Dead")
            rows = await store.list_series(user_id="u_a", sort="sort_name")
            names = [r.canonical_name for r in rows]
            # 'akira', 'berserk', 'walking dead' after sort_name stripping
            assert names == ["Akira", "Berserk", "The Walking Dead"]
            await conn.close()
        _run(go())

    def test_pagination(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            for i in range(25):
                await store.create_or_resolve_series(
                    user_id="u_a", name=f"Series {i:02d}",
                )
            page1 = await store.list_series(user_id="u_a", limit=10, offset=0)
            page2 = await store.list_series(user_id="u_a", limit=10, offset=10)
            assert len(page1) == 10
            assert len(page2) == 10
            # No overlap
            ids1 = {r.id for r in page1}
            ids2 = {r.id for r in page2}
            assert not (ids1 & ids2)
            assert await store.count(user_id="u_a") == 25
            await conn.close()
        _run(go())

    def test_user_isolation_on_list(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            await store.create_or_resolve_series(user_id="u_a", name="Berserk")
            await store.create_or_resolve_series(user_id="u_b", name="Akira")
            a_list = await store.list_series(user_id="u_a")
            b_list = await store.list_series(user_id="u_b")
            assert len(a_list) == 1
            assert a_list[0].canonical_name == "Berserk"
            assert len(b_list) == 1
            assert b_list[0].canonical_name == "Akira"
            await conn.close()
        _run(go())


# --- delete -------------------------------------------------------------


class TestDelete:
    def test_deletes_scoped_to_user(self):
        async def go():
            conn = await _setup_db()
            store = ComicSeriesStore(conn)
            series_id = await store.create_or_resolve_series(
                user_id="u_a", name="Berserk",
            )
            # User B can't delete user A's series
            assert await store.delete(series_id, user_id="u_b") is False
            assert await store.get(series_id, user_id="u_a") is not None
            # User A can
            assert await store.delete(series_id, user_id="u_a") is True
            assert await store.get(series_id, user_id="u_a") is None
            await conn.close()
        _run(go())
