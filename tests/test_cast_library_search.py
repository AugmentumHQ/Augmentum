"""Media surface gap-fill: tile enrichment + cross-library search +
genre-related.

Pins three contracts the Media console (ui/scripts/media.js +
ui/scripts/consumption/) renders from:

  1. ``_entry_to_tile`` carries the presentation-depth fields
     (year / is_finished / unplayed_count) the tile badges read.
  2. ``GET /api/cast/library/search`` groups FTS hits into kind buckets
     in a stable display order, drops docs, and 401s anonymous callers.
  3. ``_related_by_genre`` ("More like this") ranks by shared-genre
     count, restricts video peers to movie/series, and stays user-scoped
     via the query it issues.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from augmentum.proxy.cast_routes import _entry_to_tile, cast_library_search
from augmentum.proxy.media_routes import _related_by_genre


def _entry(**over):
    base = {
        "id": "f1",
        "name": "Arrival",
        "kind": "video",
        "source": "jellyfin",
        "source_metadata": {
            "server_id": "srv1",
            "entity_kind": "movie",
            "year": 2016,
            "is_finished": True,
            "unplayed_count": 0,
            "progress_pct": 100.0,
            "duration_s": 7020.0,
        },
    }
    meta_over = over.pop("meta", {})
    base.update(over)
    base["source_metadata"] = {**base["source_metadata"], **meta_over}
    return SimpleNamespace(**base)


class TestEntryToTilePresentationFields:
    def test_movie_tile_carries_state_fields(self):
        tile = _entry_to_tile(_entry())
        assert tile is not None
        assert tile["year"] == 2016
        assert tile["is_finished"] is True
        assert tile["unplayed_count"] == 0
        # Existing contract fields don't regress.
        assert tile["file_id"] == "f1"
        assert tile["play"]["action"] == "cast"

    def test_series_tile_carries_unplayed_count(self):
        tile = _entry_to_tile(_entry(
            name="Severance",
            meta={
                "entity_kind": "series",
                "is_finished": False,
                "unplayed_count": 7,
                "progress_pct": 0.0,
            },
        ))
        assert tile is not None
        assert tile["play"]["action"] == "browse_series"
        assert tile["unplayed_count"] == 7
        assert tile["is_finished"] is False

    def test_missing_state_fields_default_safely(self):
        tile = _entry_to_tile(_entry(meta={
            "year": None, "is_finished": None, "unplayed_count": None,
        }))
        assert tile["year"] == 0
        assert tile["is_finished"] is False
        assert tile["unplayed_count"] == 0


class _FakeIndex:
    def __init__(self, entries):
        self._entries = entries
        self.calls = []

    async def search(self, q, **kwargs):
        self.calls.append((q, kwargs))
        return self._entries


def _fake_request(*, user, file_index):
    state = SimpleNamespace(file_index=file_index)
    app = SimpleNamespace(state=state)
    return SimpleNamespace(scope={"user": user}, app=app)


class TestCastLibrarySearch:
    @pytest.mark.asyncio
    async def test_anonymous_caller_rejected(self):
        req = _fake_request(user=None, file_index=_FakeIndex([]))
        with pytest.raises(HTTPException) as exc:
            await cast_library_search(req, q="arrival")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_query_short_circuits(self):
        idx = _FakeIndex([_entry()])
        req = _fake_request(user=SimpleNamespace(id="u1"), file_index=idx)
        body = await cast_library_search(req, q="   ")
        assert body == {"sections": [], "query": ""}
        assert idx.calls == []  # no FTS work for an empty box

    @pytest.mark.asyncio
    async def test_results_grouped_in_display_order(self):
        entries = [
            _entry(id="a1", name="Dune audiobook", kind="audio",
                   meta={"entity_kind": ""}),
            _entry(id="m1", name="Dune", meta={"entity_kind": "movie"}),
            _entry(id="d1", name="Dune notes", kind="doc",
                   meta={"entity_kind": ""}),
            _entry(id="s1", name="Dune: Prophecy",
                   meta={"entity_kind": "series", "unplayed_count": 3}),
        ]
        idx = _FakeIndex(entries)
        req = _fake_request(user=SimpleNamespace(id="u1"), file_index=idx)
        body = await cast_library_search(req, q="dune")
        ids = [s["id"] for s in body["sections"]]
        # Movies before shows before audiobooks; docs never surface.
        assert ids == ["movies", "shows", "audiobooks"]
        assert body["sections"][0]["items"][0]["file_id"] == "m1"
        assert body["query"] == "dune"
        # Seasons are excluded at the query layer, not post-filtered.
        _, kwargs = idx.calls[0]
        assert kwargs["exclude_entity_kinds"] == ["season"]
        assert kwargs["user_id"] == "u1"


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class _FakeDb:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def execute(self, sql, params):
        self.queries.append((sql, params))
        return _FakeCursor(self.rows)


class TestRelatedByGenre:
    @pytest.mark.asyncio
    async def test_ranks_by_shared_genres_and_filters_entity_kinds(self):
        # rows: (id, name, genres_json, entity_kind, has_cover, progress)
        rows = [
            ("one_shared", "Interstellar", json.dumps(["Sci-Fi"]), "movie", 1, 0),
            ("two_shared", "Blade Runner", json.dumps(["Sci-Fi", "Drama"]), "movie", 1, 42.0),
            ("episode_skip", "Some Episode", json.dumps(["Sci-Fi", "Drama"]), "episode", 1, 0),
            ("no_overlap", "Paddington", json.dumps(["Family"]), "movie", 0, 0),
        ]
        db = _FakeDb(rows)
        idx = SimpleNamespace(_db=db)
        entry = SimpleNamespace(id="seed", kind="video")
        meta = {"genres": ["Sci-Fi", "Drama"], "entity_kind": "movie"}

        resp = await _related_by_genre(idx, entry, meta, uid="u1", limit=10)
        body = json.loads(resp.body)
        ids = [item["id"] for item in body["items"]]
        assert ids == ["two_shared", "one_shared"]
        assert body["match_key"] == "genre"
        assert body["display_name"] == "Sci-Fi"
        # Cover routing matches the author-axis contract.
        assert body["items"][0]["cover_url"] == "/api/media/cover/two_shared"
        assert body["items"][1]["progress_pct"] == 0
        # The scan is user-scoped + kind-scoped at the SQL layer.
        sql, params = db.queries[0]
        assert "user_id = ?" in sql
        assert params[0] == "u1"
        assert params[2] == "video"

    @pytest.mark.asyncio
    async def test_seed_without_genres_returns_empty(self):
        idx = SimpleNamespace(_db=_FakeDb([]))
        entry = SimpleNamespace(id="seed", kind="video")
        resp = await _related_by_genre(idx, entry, {"genres": []}, uid="u1", limit=10)
        body = json.loads(resp.body)
        assert body["items"] == []
        assert idx._db.queries == []  # no scan when there's nothing to match


class TestComicChapterDrillOrder:
    """The drill-in chapter list and the comic reader's prev/next
    siblings must agree on reading order. Siblings come from
    /api/files/comics/series/{id}/chapters which orders by
    extra.chapter_source_order — the cast drill endpoint used to order
    by chapter_number alone, which disagrees for decimal/missing
    numbering. Pins the chapter_source_order-first sort.
    """

    @pytest.mark.asyncio
    async def test_chapters_sorted_by_source_order_first(self, monkeypatch):
        import augmentum.proxy.cast_routes as cr

        def comic(eid, *, src_order, chap_num, name):
            return SimpleNamespace(
                id=eid,
                name=name,
                kind="comic",
                source="suwayomi",
                series_id="series_1",
                created_at="2026-06-01T00:00:00Z",
                source_metadata={
                    "server_id": "srv1",
                    "provider": "suwayomi",
                    "entity_kind": "",
                    "is_finished": False,
                    "unplayed_count": 0,
                    "progress_pct": 0.0,
                    "duration_s": 0.0,
                    "year": 0,
                    "extra": {
                        "suwayomi_manga_id": "42",
                        "chapter_source_order": src_order,
                        "chapter_number": chap_num,
                    },
                },
            )

        # chapter_number order would yield ch2 < ch10.5 < ch11 — but the
        # provider's source order says the 10.5 special reads LAST.
        rows = [
            comic("b", src_order=2, chap_num=11.0, name="Ch 11"),
            comic("c", src_order=3, chap_num=10.5, name="Special"),
            comic("a", src_order=1, chap_num=2.0, name="Ch 2"),
        ]

        class _Idx:
            async def get(self, file_id, *, user_id):
                return rows[0]

        async def _fake_fetch(request, idx, uid):
            return rows

        monkeypatch.setattr(cr, "_fetch_user_comic_chapters", _fake_fetch)
        req = SimpleNamespace(
            scope={"user": SimpleNamespace(id="u1")},
            app=SimpleNamespace(state=SimpleNamespace(file_index=_Idx())),
        )
        body = await cr.cast_library_chapters("b", req)
        assert [c["file_id"] for c in body["chapters"]] == ["a", "b", "c"]
        # Series block carries the comic_series linkage for rich headers.
        assert body["series"]["series_id"] == "series_1"


class TestSeriesIdSurfacedToClient:
    """series_id must reach the client through every FileEntry path.

    The comic reader keys per-series prefs AND sibling-chapter
    resolution (prev/next, auto-advance) on ``file.series_id``. Before
    this landed, ``to_dict`` dropped the column, so every chapter
    opened outside the comics drill-in read as a one-chapter series
    ("End of series" after chapter 1 from Media tiles).
    """

    def test_to_dict_includes_series_id(self):
        from augmentum.vfs.models import FileEntry
        entry = FileEntry(
            id="ch1", user_id="u1", source="suwayomi", source_id="s:1",
            name="Chapter 1", kind="comic", series_id="series_42",
        )
        assert entry.to_dict()["series_id"] == "series_42"

    def test_to_dict_empty_series_id_serializes_as_empty_string(self):
        from augmentum.vfs.models import FileEntry
        entry = FileEntry(
            id="ch1", user_id="u1", source="uploads", source_id="u:1",
            name="loose.cbz", kind="comic",
        )
        assert entry.to_dict()["series_id"] == ""

    def test_row_to_entry_maps_series_id_from_entry_columns(self):
        """Positional mapping: series_id rides at index 21, after
        last_played_at. A drift here silently nulls the field for every
        list/search query."""
        from augmentum.vfs.index import _ENTRY_COLUMNS, FileIndexService
        cols = [c.strip() for c in _ENTRY_COLUMNS.split(",")]
        assert cols[-1] == "series_id"
        assert cols[20] == "last_played_at"
        row = [None] * len(cols)
        row[0] = "ch1"
        row[1] = "u1"
        row[19] = "comic"
        row[21] = "series_42"
        entry = FileIndexService._row_to_entry(
            SimpleNamespace(), row,  # method only touches `row`
        )
        assert entry.series_id == "series_42"
        assert entry.kind == "comic"

    def test_row_to_entry_tolerates_short_rows(self):
        """Defensive g(i) contract: a row from an older 21-column SELECT
        (no series_id) maps to None instead of raising."""
        from augmentum.vfs.index import FileIndexService
        row = ["ch1", "u1"] + [None] * 19  # 21 columns, pre-series_id shape
        entry = FileIndexService._row_to_entry(SimpleNamespace(), row)
        assert entry.series_id is None
