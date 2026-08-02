"""Smoke + contract tests for the LibriVox built-in provider.

Live-API test (pytest.mark.live) lives in tests/live/test_live_librivox.py.
This file runs offline with mocked httpx and exercises:
  - Provider construction + Protocol conformance
  - browse() URL + param shape
  - Feed → BrowseResult normalisation edge cases
  - archive.org identifier extraction from url_iarchive / url_zip
  - build_stream_url / build_cover_url absolute URL construction
  - normalise_details_to_catalog chapter stitching + filter rules
  - login() raises (LibriVox has no auth)
  - fetch_catalog returns [] (browse-only by design)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.media.providers.base import (
    BrowseResult,
    CatalogItem,
    MediaProvider,
    provider_supports_browse,
)
from augmentum.media.providers.librivox import (
    ARCHIVE_COLLECTION,
    ARCHIVE_COVER,
    ARCHIVE_DOWNLOAD,
    ARCHIVE_SEARCH,
    LibrivoxProvider,
    _archive_identifier,
    _browse_result_from_archive_doc,
    _browse_result_from_feed,
    _clean_html_text,
    _filename_from_listen_url,
    _parse_length,
    _parse_runtime,
    _safe_int,
    _strip_librivox_preamble,
    normalise_details_to_catalog,
    normalise_librivox_sections,
)


def _mock_response(status: int = 200, body: object = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=body if body is not None else {})
    return r


def _mock_http(get_return: MagicMock) -> AsyncMock:
    """Build an AsyncMock httpx.AsyncClient whose .get returns the given response."""
    m = AsyncMock()
    m.get = AsyncMock(return_value=get_return)
    return m


# --- Smoke ---------------------------------------------------------------


class TestSmoke:
    def test_name(self):
        http = AsyncMock()
        p = LibrivoxProvider(http)
        assert p.name == "librivox"

    def test_protocol_conformance(self):
        """runtime_checkable Protocol — exercise with isinstance()."""
        http = AsyncMock()
        p = LibrivoxProvider(http)
        assert isinstance(p, MediaProvider)

    def test_supports_browse(self):
        """Feature detector must see browse()."""
        http = AsyncMock()
        p = LibrivoxProvider(http)
        assert provider_supports_browse(p) is True


# --- URL construction ----------------------------------------------------


class TestUrls:
    def test_build_stream_url_prepends_archive_download(self):
        p = LibrivoxProvider(AsyncMock())
        url = p.build_stream_url("", "pride_and_prejudice_0711_librivox/chapter_01.mp3", "")
        assert url == f"{ARCHIVE_DOWNLOAD}/pride_and_prejudice_0711_librivox/chapter_01.mp3"

    def test_build_stream_url_strips_leading_slash(self):
        p = LibrivoxProvider(AsyncMock())
        url = p.build_stream_url("", "/identifier/file.mp3", "")
        assert url == f"{ARCHIVE_DOWNLOAD}/identifier/file.mp3"

    def test_build_stream_url_returns_empty_for_empty_path(self):
        p = LibrivoxProvider(AsyncMock())
        assert p.build_stream_url("", "", "") == ""

    def test_build_cover_url(self):
        p = LibrivoxProvider(AsyncMock())
        url = p.build_cover_url("", "my_book_librivox", "")
        assert url == f"{ARCHIVE_COVER}/my_book_librivox"

    def test_build_cover_url_empty_external_id(self):
        p = LibrivoxProvider(AsyncMock())
        assert p.build_cover_url("", "", "") == ""


# --- No-op methods (login, verify, progress, catalog) --------------------


class TestNoOps:
    async def test_login_raises(self):
        p = LibrivoxProvider(AsyncMock())
        with pytest.raises(ValueError):
            await p.login("", "user", "pass")

    async def test_verify_token_always_true(self):
        p = LibrivoxProvider(AsyncMock())
        assert await p.verify_token("", "") is True

    async def test_fetch_catalog_empty(self):
        p = LibrivoxProvider(AsyncMock())
        # fetch_catalog never hits the network — LibriVox is browse-only.
        assert await p.fetch_catalog("", "") == []

    async def test_fetch_progress_empty(self):
        p = LibrivoxProvider(AsyncMock())
        assert await p.fetch_progress("", "") == {}

    async def test_push_progress_returns_true(self):
        p = LibrivoxProvider(AsyncMock())
        ok = await p.push_progress(
            "", "", external_id="x",
            current_time_s=10.0, duration_s=100.0,
        )
        assert ok is True

    async def test_ping_returns_builtin_info(self):
        p = LibrivoxProvider(AsyncMock())
        info = await p.ping("")
        assert info is not None
        assert info.provider == "librivox"
        assert info.is_initialized is True


# --- browse() contract ---------------------------------------------------


def _archive_search_response(docs: list[dict]) -> MagicMock:
    """Helper: shape an archive.org Advanced Search mock response."""
    return _mock_response(200, {"response": {"docs": docs, "numFound": len(docs)}})


class TestBrowse:
    async def test_browse_targets_archive_search(self):
        """Browse must query archive.org's advanced search, not LibriVox's
        feed API — the feed's search=/genre= params are inert upstream."""
        http = _mock_http(_archive_search_response([]))
        p = LibrivoxProvider(http)
        await p.browse(query="austen", category="Fiction", page=2, page_size=10)
        http.get.assert_called_once()
        args, kwargs = http.get.call_args
        url = args[0]
        params = kwargs["params"]
        assert url == ARCHIVE_SEARCH
        # Assert on the flat list of (k, v) tuples since fl[] repeats.
        kv = {k: v for k, v in params}
        assert "q" in kv
        assert f"collection:{ARCHIVE_COLLECTION}" in kv["q"]
        assert 'subject:"fiction"' in kv["q"]
        assert "(austen)" in kv["q"]
        assert kv["rows"] == "10"
        assert kv["page"] == "2"
        assert kv["output"] == "json"

    async def test_browse_no_query_omits_subject_and_freetext(self):
        http = _mock_http(_archive_search_response([]))
        p = LibrivoxProvider(http)
        await p.browse(query="", category="", page=1, page_size=24)
        _, kwargs = http.get.call_args
        kv = {k: v for k, v in kwargs["params"]}
        assert kv["q"] == f"collection:{ARCHIVE_COLLECTION}"
        assert kv["page"] == "1"

    async def test_browse_only_category_no_freetext(self):
        http = _mock_http(_archive_search_response([]))
        p = LibrivoxProvider(http)
        await p.browse(category="Horror")
        _, kwargs = http.get.call_args
        kv = {k: v for k, v in kwargs["params"]}
        assert 'subject:"horror"' in kv["q"]
        # Free-text clause is absent when query is empty.
        assert "(" not in kv["q"].split("AND")[-1].strip()[len('subject:"horror"'):]

    async def test_browse_parses_archive_doc(self):
        docs = [{
            "identifier": "pride_and_prejudice_0711_librivox",
            "title": "Pride and Prejudice",
            "creator": "Jane Austen",
            "description": "LibriVox recording of Pride and Prejudice by Jane Austen."
                           " Read in English by Various. A classic Regency-era romance.",
            "subject": ["librivox", "audiobooks", "romance", "fiction"],
            "runtime": "11:12:00",
            "language": "eng",
        }]
        http = _mock_http(_archive_search_response(docs))
        p = LibrivoxProvider(http)
        results = await p.browse()
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, BrowseResult)
        assert r.external_id == "pride_and_prejudice_0711_librivox"
        assert r.name == "Pride and Prejudice"
        assert r.author == "Jane Austen"
        assert r.duration_ms == (11 * 3600 + 12 * 60) * 1000
        assert r.cover_url == f"{ARCHIVE_COVER}/pride_and_prejudice_0711_librivox"
        # Noise tags are filtered out of the displayed subject list.
        assert "romance" in r.extra["genres"]
        assert "librivox" not in r.extra["genres"]
        assert "audiobooks" not in r.extra["genres"]
        # Language code gets expanded for human display.
        assert r.extra["language"] == "English"
        # Archive-browsed rows don't carry librivox_id / external enrichment.
        assert r.extra["librivox_id"] == ""
        assert r.extra["url_text_source"] == ""
        # Preamble stripped so the actual blurb leads.
        assert "A classic Regency-era romance" in r.description
        assert "LibriVox recording of" not in r.description

    async def test_browse_http_error_returns_empty(self):
        http = _mock_http(_mock_response(500, {}))
        p = LibrivoxProvider(http)
        assert await p.browse() == []

    async def test_browse_exception_returns_empty(self):
        http = AsyncMock()
        http.get = AsyncMock(side_effect=RuntimeError("connection refused"))
        p = LibrivoxProvider(http)
        assert await p.browse() == []

    async def test_browse_invalid_json_returns_empty(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(side_effect=ValueError("not json"))
        http = _mock_http(resp)
        p = LibrivoxProvider(http)
        assert await p.browse() == []

    async def test_browse_skips_rows_without_identifier(self):
        """Archive.org occasionally returns partial docs (missing identifier)
        for deleted/embargoed items — skip them rather than render a
        broken card."""
        docs = [
            {"title": "No identifier", "creator": "Anon"},
            {"identifier": "", "title": "Empty identifier"},
            {"identifier": "valid_item", "title": "Has Identifier",
             "creator": "Someone", "runtime": "1:00:00"},
        ]
        http = _mock_http(_archive_search_response(docs))
        p = LibrivoxProvider(http)
        results = await p.browse()
        assert len(results) == 1
        assert results[0].external_id == "valid_item"

    async def test_creator_as_list_is_joined(self):
        """Some archive.org docs return `creator` as a list for co-authored
        works — normalise to a comma-joined string."""
        docs = [{
            "identifier": "coauth",
            "title": "Something",
            "creator": ["Jane Doe", "John Smith"],
        }]
        http = _mock_http(_archive_search_response(docs))
        p = LibrivoxProvider(http)
        results = await p.browse()
        assert results[0].author == "Jane Doe, John Smith"


# --- Archive identifier extraction ---------------------------------------


class TestArchiveIdentifier:
    def test_from_url_iarchive(self):
        raw = {"url_iarchive": "http://archive.org/details/moby_dick_librivox"}
        assert _archive_identifier(raw) == "moby_dick_librivox"

    def test_from_url_zip_file(self):
        """LibriVox's real key name — verified live, url_zip doesn't exist."""
        raw = {"url_zip_file":
               "https://archive.org/compress/p_and_p_librivox/formats=64KBPS MP3&file=/p_and_p.zip"}
        assert _archive_identifier(raw) == "p_and_p_librivox"

    def test_from_url_zip_legacy(self):
        """Defensive: if LibriVox ever restores the short-form key, we still cope."""
        raw = {"url_zip": "http://archive.org/download/item_id/item_id_64kb_mp3.zip"}
        assert _archive_identifier(raw) == "item_id"

    def test_prefers_iarchive_over_zip_file(self):
        raw = {
            "url_iarchive": "http://archive.org/details/canonical_id",
            "url_zip_file": "http://archive.org/compress/other_id",
        }
        assert _archive_identifier(raw) == "canonical_id"

    def test_missing_urls_returns_empty(self):
        assert _archive_identifier({}) == ""

    def test_empty_urls_returns_empty(self):
        assert _archive_identifier({"url_iarchive": "", "url_zip_file": ""}) == ""

    def test_malformed_url_returns_empty(self):
        assert _archive_identifier({"url_iarchive": "not a url"}) == ""


# --- Feed normalisation edge cases ---------------------------------------


class TestBrowseResultFromFeed:
    def test_string_authors(self):
        raw = {
            "id": "1",
            "title": "Test",
            "totaltimesecs": 100,
            "url_iarchive": "http://archive.org/details/test_id",
            "authors": ["Homer", "Virgil"],
        }
        r = _browse_result_from_feed(raw)
        assert r.author == "Homer, Virgil"

    def test_malformed_duration_defaults_to_zero(self):
        raw = {
            "id": "1",
            "title": "Test",
            "totaltimesecs": "not a number",
            "url_iarchive": "http://archive.org/details/test_id",
        }
        r = _browse_result_from_feed(raw)
        assert r.duration_ms == 0

    def test_none_raw_returns_none(self):
        assert _browse_result_from_feed(None) is None

    def test_string_genres(self):
        raw = {
            "id": "1",
            "title": "Test",
            "url_iarchive": "http://archive.org/details/test_id",
            "genres": ["Fiction", " Romance ", ""],
        }
        r = _browse_result_from_feed(raw)
        assert r.extra["genres"] == ["Fiction", "Romance"]


# --- Length parsing ------------------------------------------------------


class TestParseLength:
    def test_numeric_seconds(self):
        assert _parse_length("123.5") == 123.5

    def test_mmss(self):
        assert _parse_length("5:30") == 330.0

    def test_hhmmss(self):
        assert _parse_length("1:02:03") == 3723.0

    def test_none(self):
        assert _parse_length(None) == 0.0

    def test_empty(self):
        assert _parse_length("") == 0.0

    def test_gibberish(self):
        assert _parse_length("not a time") == 0.0


# --- Safe int ------------------------------------------------------------


class TestSafeInt:
    def test_plain(self):
        assert _safe_int(5) == 5
        assert _safe_int("5") == 5

    def test_slash_fraction(self):
        """archive.org tracks come as 'N/TOTAL'."""
        assert _safe_int("3/12") == 3

    def test_none(self):
        assert _safe_int(None) is None

    def test_invalid(self):
        assert _safe_int("abc") is None


# --- Detail normalisation (pin-time) -------------------------------------


def _fake_browse_result(external_id: str = "test_id") -> BrowseResult:
    return BrowseResult(
        external_id=external_id,
        name="Test Book",
        author="Jane Doe",
        duration_ms=0,
        cover_url=f"{ARCHIVE_COVER}/{external_id}",
        description="A test.",
        license="public-domain",
        extra={
            "language":        "English",
            "librivox_url":    "http://lv/x",
            "librivox_id":     "99",
            "copyright_year":  "1900",
            "totaltime":       "3:00:00",
            "url_text_source": "http://gutenberg/123",
            "url_project":     "http://wiki/test",
            "url_rss":         "http://lv/rss/99",
            "authors":         [{"name": "Jane Doe", "dob": "1850", "dod": "1920"}],
        },
    )


class TestNormaliseDetails:
    def test_builds_chapters_with_contiguous_offsets(self):
        archive_meta = {
            "metadata": {"creator": "Various Narrators"},
            "files": [
                {"name": "chapter_01.mp3", "format": "VBR MP3", "length": "120.5",
                 "size": "100", "title": "Chapter 1", "track": "1/3"},
                {"name": "chapter_02.mp3", "format": "VBR MP3", "length": "90.0",
                 "size": "90",  "title": "Chapter 2", "track": "2/3"},
                {"name": "chapter_03.mp3", "format": "VBR MP3", "length": "60.0",
                 "size": "60",  "title": "Chapter 3", "track": "3/3"},
            ],
        }
        # Note: test file uses VBR MP3 intentionally — normaliser filters
        # OUT 'vbr' as a 128kbps reencode marker. Using plain MP3 here.
        archive_meta["files"] = [
            {**f, "format": "MP3"} for f in archive_meta["files"]
        ]
        out = normalise_details_to_catalog(
            archive_meta=archive_meta,
            browse_result=_fake_browse_result(),
        )
        assert len(out["chapters"]) == 3
        assert out["chapters"][0]["start"] == 0
        assert out["chapters"][0]["end"] == 120.5
        assert out["chapters"][1]["start"] == 120.5
        assert out["chapters"][1]["end"] == 210.5
        assert out["chapters"][2]["file_index"] == 2
        assert out["duration_ms"] == 270500   # 270.5 seconds in ms
        assert out["narrator"] == "Various Narrators"

    def test_filters_non_mp3_files(self):
        archive_meta = {
            "metadata": {},
            "files": [
                {"name": "book.zip",       "format": "Archive",         "length": "0"},
                {"name": "chapter_01.mp3", "format": "MP3",             "length": "60"},
                {"name": "book.txt",       "format": "Text",            "length": "0"},
                {"name": "book.jpg",       "format": "JPEG",            "length": "0"},
            ],
        }
        out = normalise_details_to_catalog(
            archive_meta=archive_meta,
            browse_result=_fake_browse_result(),
        )
        assert len(out["audio_files"]) == 1
        assert out["audio_files"][0]["name"] == "chapter_01.mp3"

    def test_filters_128kbps_reencodes(self):
        archive_meta = {
            "metadata": {},
            "files": [
                {"name": "chapter_01_64kb.mp3", "format": "MP3",         "length": "60"},
                {"name": "chapter_01_128.mp3",  "format": "128 MP3",     "length": "60"},
                {"name": "chapter_01.mp3",      "format": "VBR MP3",     "length": "60"},
            ],
        }
        out = normalise_details_to_catalog(
            archive_meta=archive_meta,
            browse_result=_fake_browse_result(),
        )
        # Only chapter_01_64kb.mp3 should survive: plain MP3 format, no
        # '128' in the format string, no 'vbr'.
        names = [a["name"] for a in out["audio_files"]]
        assert names == ["chapter_01_64kb.mp3"]

    def test_sorts_by_track_number(self):
        archive_meta = {
            "metadata": {},
            "files": [
                {"name": "z.mp3", "format": "MP3", "length": "10", "track": "3/3"},
                {"name": "a.mp3", "format": "MP3", "length": "10", "track": "1/3"},
                {"name": "m.mp3", "format": "MP3", "length": "10", "track": "2/3"},
            ],
        }
        out = normalise_details_to_catalog(
            archive_meta=archive_meta,
            browse_result=_fake_browse_result(),
        )
        names = [a["name"] for a in out["audio_files"]]
        assert names == ["a.mp3", "m.mp3", "z.mp3"]

    def test_falls_back_to_browse_duration_when_files_missing_durations(self):
        archive_meta = {"metadata": {}, "files": []}
        br = _fake_browse_result()
        br.duration_ms = 999_000
        out = normalise_details_to_catalog(
            archive_meta=archive_meta, browse_result=br,
        )
        assert out["duration_ms"] == 999_000

    def test_no_mp3_files_produces_no_chapters(self):
        """Edge case: some LibriVox items don't have archive.org mirrors yet."""
        archive_meta = {"metadata": {}, "files": [
            {"name": "source.txt", "format": "Text", "length": "0"},
        ]}
        out = normalise_details_to_catalog(
            archive_meta=archive_meta,
            browse_result=_fake_browse_result(),
        )
        assert out["audio_files"] == []
        assert out["chapters"] == []

    def test_carries_license_and_language(self):
        archive_meta = {"metadata": {}, "files": [
            {"name": "a.mp3", "format": "MP3", "length": "10"},
        ]}
        out = normalise_details_to_catalog(
            archive_meta=archive_meta,
            browse_result=_fake_browse_result(),
        )
        assert out["language"] == "English"
        assert out["librivox_url"] == "http://lv/x"

    def test_fallback_preserves_enrichment_fields(self):
        """Archive.org fallback path must also carry enrichment (via browse_result.extra)."""
        archive_meta = {"metadata": {}, "files": [
            {"name": "a.mp3", "format": "MP3", "length": "10"},
        ]}
        out = normalise_details_to_catalog(
            archive_meta=archive_meta,
            browse_result=_fake_browse_result(),
        )
        assert out["copyright_year"] == "1900"
        assert out["totaltime"] == "3:00:00"
        assert out["url_text_source"] == "http://gutenberg/123"
        assert out["authors_detailed"] == [
            {"name": "Jane Doe", "dob": "1850", "dod": "1920"},
        ]


# --- fetch_item_details --------------------------------------------------


class TestFetchItemDetails:
    async def test_returns_archive_metadata_on_success(self):
        meta = {"metadata": {"title": "X"}, "files": []}
        http = _mock_http(_mock_response(200, meta))
        p = LibrivoxProvider(http)
        result = await p.fetch_item_details("", "", external_id="some_id")
        assert result == meta
        url = http.get.call_args[0][0]
        assert url.endswith("/metadata/some_id")

    async def test_returns_none_on_http_error(self):
        http = _mock_http(_mock_response(404, {}))
        p = LibrivoxProvider(http)
        assert await p.fetch_item_details("", "", external_id="x") is None

    async def test_returns_none_on_empty_external_id(self):
        http = _mock_http(_mock_response(200, {}))
        p = LibrivoxProvider(http)
        assert await p.fetch_item_details("", "", external_id="") is None
        # Must NOT have fired a request for an empty id.
        http.get.assert_not_called()

    async def test_returns_none_on_exception(self):
        http = AsyncMock()
        http.get = AsyncMock(side_effect=RuntimeError("boom"))
        p = LibrivoxProvider(http)
        assert await p.fetch_item_details("", "", external_id="x") is None


# --- fetch_book_by_id (LibriVox feed, pin-time primary path) -------------


class TestFetchBookById:
    async def test_returns_single_book(self):
        body = {"books": [{"id": "253", "title": "P&P", "sections": []}]}
        http = _mock_http(_mock_response(200, body))
        p = LibrivoxProvider(http)
        book = await p.fetch_book_by_id("253")
        assert book is not None
        assert book["title"] == "P&P"
        params = http.get.call_args[1]["params"]
        assert params["id"] == "253"
        assert params["extended"] == 1
        # coverart=1 unlocks coverart_jpg / coverart_thumbnail in the
        # response. Verified live 2026-04-20: adds ~200 bytes to the
        # payload for a noticeable visual upgrade on the detail panel.
        assert params["coverart"] == 1

    async def test_none_on_empty_id(self):
        http = _mock_http(_mock_response(200, {}))
        p = LibrivoxProvider(http)
        assert await p.fetch_book_by_id("") is None
        http.get.assert_not_called()

    async def test_none_on_404(self):
        http = _mock_http(_mock_response(404, {"error": "not found"}))
        p = LibrivoxProvider(http)
        assert await p.fetch_book_by_id("999999") is None

    async def test_none_on_empty_books_array(self):
        http = _mock_http(_mock_response(200, {"books": []}))
        p = LibrivoxProvider(http)
        assert await p.fetch_book_by_id("1") is None


class TestRecentlyAdded:
    """``since=`` is the only sort-by-freshness path LibriVox exposes — this
    is what powers the Catalog overlay's landing state."""

    async def test_sends_since_and_extended_params(self):
        http = _mock_http(_mock_response(200, {"books": []}))
        p = LibrivoxProvider(http)
        out = await p.recently_added(days=30, limit=24)
        assert out == []
        params = http.get.call_args[1]["params"]
        assert params["format"] == "json"
        assert params["extended"] == 1
        assert params["coverart"] == 1
        # since is a unix timestamp ≈ 30 days before now. Can't assert
        # exact value (wall clock) but a sane window works.
        import time as _time
        now = int(_time.time())
        assert now - 30 * 86400 - 120 <= params["since"] <= now - 30 * 86400 + 120
        assert params["limit"] == 24

    async def test_clamps_limit_to_upstream_max(self):
        """LibriVox caps limit at 100 — passing 500 must clamp silently."""
        http = _mock_http(_mock_response(200, {"books": []}))
        p = LibrivoxProvider(http)
        await p.recently_added(days=7, limit=500)
        params = http.get.call_args[1]["params"]
        assert params["limit"] == 100

    async def test_parses_books_into_browse_results(self):
        body = {
            "books": [
                {
                    "id":            "99",
                    "title":         "Brand New Book",
                    "url_iarchive":  "https://archive.org/details/new_librivox",
                    "authors":       [{"first_name": "A", "last_name": "B"}],
                    "totaltimesecs": 3600,
                    "coverart_jpg":  "https://archive.org/download/c/N.jpg",
                },
            ],
        }
        http = _mock_http(_mock_response(200, body))
        p = LibrivoxProvider(http)
        out = await p.recently_added(days=30, limit=24)
        assert len(out) == 1
        assert out[0].name == "Brand New Book"
        assert out[0].external_id == "new_librivox"
        # Coverart from the feed should ride through into the browse result.
        assert out[0].extra["coverart_jpg"] == "https://archive.org/download/c/N.jpg"

    async def test_non_200_returns_empty(self):
        http = _mock_http(_mock_response(500, {}))
        p = LibrivoxProvider(http)
        assert await p.recently_added() == []

    async def test_exception_returns_empty(self):
        http = AsyncMock()
        http.get = AsyncMock(side_effect=RuntimeError("network out"))
        p = LibrivoxProvider(http)
        assert await p.recently_added() == []

    async def test_invalid_json_returns_empty(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.json = MagicMock(side_effect=ValueError("not json"))
        http = _mock_http(resp)
        p = LibrivoxProvider(http)
        assert await p.recently_added() == []

    async def test_clamps_days_to_minimum_of_one(self):
        """days<1 would compute a future or same-day since=, yielding an
        empty feed. Clamp silently to 1."""
        http = _mock_http(_mock_response(200, {"books": []}))
        p = LibrivoxProvider(http)
        await p.recently_added(days=0)
        params = http.get.call_args[1]["params"]
        import time as _time
        now = int(_time.time())
        # Should be ≈ now - 86400, not ≈ now.
        assert params["since"] <= now - 86400 + 120


# --- listen_url → filename extraction ------------------------------------


class TestFilenameFromListenUrl:
    def test_standard_archive_url(self):
        url = "https://www.archive.org/download/pride_librivox/chapter_01_64kb.mp3"
        assert _filename_from_listen_url(url) == "chapter_01_64kb.mp3"

    def test_https_no_www(self):
        url = "https://archive.org/download/some_id/file.mp3"
        assert _filename_from_listen_url(url) == "file.mp3"

    def test_strips_query_string(self):
        url = "https://archive.org/download/some_id/file.mp3?foo=bar"
        assert _filename_from_listen_url(url) == "file.mp3"

    def test_empty_returns_empty(self):
        assert _filename_from_listen_url("") == ""

    def test_no_download_segment(self):
        assert _filename_from_listen_url("https://archive.org/details/whatever") == ""

    def test_download_but_no_filename(self):
        assert _filename_from_listen_url("https://archive.org/download/some_id") == ""


# --- normalise_librivox_sections (primary pin path) ----------------------


def _fake_lv_book() -> dict:
    return {
        "id":    "253",
        "title": "Pride and Prejudice",
        "sections": [
            {
                "section_number": "1",
                "title":     "Chapters 1-3",
                "playtime":  "1132",
                "listen_url":
                    "https://www.archive.org/download/pride_librivox/pp_01-03_64kb.mp3",
                "readers":   [{"reader_id": "168", "display_name": "Chris Goringe"}],
            },
            {
                "section_number": "2",
                "title":     "Chapters 4-5",
                "playtime":  "865",
                "listen_url":
                    "https://www.archive.org/download/pride_librivox/pp_04-05_64kb.mp3",
                "readers":   [{"reader_id": "19", "display_name": "Kara Shallenberg"}],
            },
            {
                "section_number": "3",
                "title":     "Chapter 6",
                "playtime":  "777",
                "listen_url":
                    "https://www.archive.org/download/pride_librivox/pp_06_64kb.mp3",
                "readers":   [{"reader_id": "89", "display_name": "Kristen McQuillin"}],
            },
        ],
    }


class TestNormaliseSections:
    def test_builds_chapters_from_sections(self):
        out = normalise_librivox_sections(
            librivox_book=_fake_lv_book(),
            browse_result=_fake_browse_result("pride_librivox"),
        )
        assert len(out["audio_files"]) == 3
        assert len(out["chapters"]) == 3
        assert out["chapters"][0]["start"] == 0
        assert out["chapters"][0]["end"] == 1132
        assert out["chapters"][1]["start"] == 1132
        assert out["chapters"][2]["end"] == 1132 + 865 + 777
        assert out["duration_ms"] == (1132 + 865 + 777) * 1000

    def test_captures_per_section_readers(self):
        out = normalise_librivox_sections(
            librivox_book=_fake_lv_book(),
            browse_result=_fake_browse_result(),
        )
        assert out["chapters"][0]["narrators"] == ["Chris Goringe"]
        assert out["chapters"][1]["narrators"] == ["Kara Shallenberg"]

    def test_book_narrator_dedupes_across_sections(self):
        book = _fake_lv_book()
        # Make the second reader repeat the first.
        book["sections"][1]["readers"] = [
            {"reader_id": "168", "display_name": "Chris Goringe"},
        ]
        out = normalise_librivox_sections(
            librivox_book=book, browse_result=_fake_browse_result(),
        )
        # Two distinct readers across 3 sections → comma-joined.
        assert out["narrator"] == "Chris Goringe, Kristen McQuillin"
        assert out["narrators"] == ["Chris Goringe", "Kristen McQuillin"]

    def test_book_narrator_compresses_many_readers(self):
        book = _fake_lv_book()
        # Force >3 distinct readers.
        book["sections"] = [
            {"section_number": str(i), "title": f"S{i}", "playtime": "60",
             "listen_url": f"https://archive.org/download/x/x_{i:02}.mp3",
             "readers": [{"reader_id": str(i), "display_name": f"Reader {i}"}]}
            for i in range(1, 6)
        ]
        out = normalise_librivox_sections(
            librivox_book=book, browse_result=_fake_browse_result(),
        )
        # Formatted as "First + N others" when we have 5 distinct readers.
        assert out["narrator"].startswith("Reader 1 + ")
        assert "others" in out["narrator"]
        assert len(out["narrators"]) == 5

    def test_skips_section_with_no_listen_url(self):
        book = _fake_lv_book()
        book["sections"][1]["listen_url"] = ""
        out = normalise_librivox_sections(
            librivox_book=book, browse_result=_fake_browse_result(),
        )
        assert len(out["audio_files"]) == 2   # 3 sections → 2 playable

    def test_sorts_by_section_number(self):
        book = _fake_lv_book()
        # Reverse the sections — normaliser should re-sort.
        book["sections"] = list(reversed(book["sections"]))
        out = normalise_librivox_sections(
            librivox_book=book, browse_result=_fake_browse_result(),
        )
        assert out["audio_files"][0]["name"] == "pp_01-03_64kb.mp3"
        assert out["audio_files"][2]["name"] == "pp_06_64kb.mp3"

    def test_empty_sections_returns_empty_audio(self):
        out = normalise_librivox_sections(
            librivox_book={"sections": []},
            browse_result=_fake_browse_result(),
        )
        assert out["audio_files"] == []
        assert out["chapters"] == []

    def test_carries_language_and_librivox_url(self):
        out = normalise_librivox_sections(
            librivox_book=_fake_lv_book(),
            browse_result=_fake_browse_result(),
        )
        assert out["language"] == "English"
        assert out["librivox_url"] == "http://lv/x"
        assert out["librivox_id"] == "99"

    def test_carries_enrichment_fields(self):
        """Author dates, year, text source, RSS must survive the normaliser
        so the pin route can persist them into source_metadata."""
        out = normalise_librivox_sections(
            librivox_book=_fake_lv_book(),
            browse_result=_fake_browse_result(),
        )
        assert out["copyright_year"] == "1900"
        assert out["totaltime"] == "3:00:00"
        assert out["url_text_source"] == "http://gutenberg/123"
        assert out["url_project"] == "http://wiki/test"
        assert out["url_rss"] == "http://lv/rss/99"
        assert out["authors_detailed"] == [
            {"name": "Jane Doe", "dob": "1850", "dod": "1920"},
        ]
        # Translators + url_other default to empty when the browse result
        # didn't carry them — confirms the keys are always present so
        # downstream consumers can rely on `get("translators", [])` shape.
        assert out["translators"] == []

    def test_carries_coverart_and_zip_fields(self):
        """coverart_jpg / coverart_thumbnail / url_zip_file must reach
        source_metadata so the detail panel can render the LibriVox
        cover + ZIP link without a refetch."""
        br = _fake_browse_result()
        br.extra["coverart_jpg"]       = "https://archive.org/download/c/P.jpg"
        br.extra["coverart_thumbnail"] = "https://archive.org/download/c/P_thumb.jpg"
        br.extra["url_zip_file"]       = "https://archive.org/compress/pride_librivox"
        out = normalise_librivox_sections(
            librivox_book=_fake_lv_book(), browse_result=br,
        )
        assert out["coverart_jpg"] == "https://archive.org/download/c/P.jpg"
        assert out["coverart_thumbnail"] == "https://archive.org/download/c/P_thumb.jpg"
        assert out["url_zip_file"] == "https://archive.org/compress/pride_librivox"

    def test_coverart_and_zip_default_empty_when_missing(self):
        """Keys must always be present (empty string when absent) so the
        route layer's dict-indexing into source_metadata never KeyErrors."""
        out = normalise_librivox_sections(
            librivox_book=_fake_lv_book(),
            browse_result=_fake_browse_result(),
        )
        assert out["coverart_jpg"] == ""
        assert out["coverart_thumbnail"] == ""
        assert out["url_zip_file"] == ""
        assert out["url_other"] == ""


class TestParseRuntime:
    """archive.org ships runtime as 'H:MM:SS' / 'MM:SS' / seconds-as-string."""

    def test_h_mm_ss(self):
        assert _parse_runtime("4:30:15") == (4 * 3600 + 30 * 60 + 15) * 1000

    def test_mm_ss(self):
        assert _parse_runtime("12:45") == (12 * 60 + 45) * 1000

    def test_seconds_only(self):
        assert _parse_runtime("90") == 90 * 1000

    def test_unparseable_returns_zero(self):
        assert _parse_runtime("not-a-time") == 0

    def test_none_and_empty_return_zero(self):
        assert _parse_runtime(None) == 0
        assert _parse_runtime("") == 0
        assert _parse_runtime("   ") == 0

    def test_period_separator_normalized_to_colon(self):
        """Archive.org emits 'HH:MM.SS' for some rows (e.g. dracula_librivox
        has runtime='16:31.09' meaning 16h 31m 9s, not 16m 31.09s).
        Without this fix the browse card would show '16m' for a 16-hour book.
        Allow ±1s tolerance for float-rounding drift in the normaliser."""
        got = _parse_runtime("16:31.09")
        expected = (16 * 3600 + 31 * 60 + 9) * 1000
        assert abs(got - expected) < 1000

    def test_period_with_more_colons_left_alone(self):
        """Sub-second precision on a full H:MM:SS is real (rare but valid).
        Promoting the period here would break a well-formed input, so
        the normaliser only touches cases with exactly one of each."""
        # 1:30:45.5 → 1h 30m 45.5s
        got = _parse_runtime("1:30:45.5")
        expected = (1 * 3600 + 30 * 60 + 45.5) * 1000
        assert abs(got - expected) < 10

    def test_decimal_seconds_without_colon_untouched(self):
        """'90.5' means 90.5 seconds — the period normalization must not
        fire when there's no colon, or we'd turn it into a broken '90:5'."""
        assert _parse_runtime("90.5") == 90500


class TestStripLibrivoxPreamble:
    """The card view shows description prose; stripping the stock
    'LibriVox recording of X by Y' lead-in keeps the card density useful
    without losing meaningful text."""

    def test_strips_standard_preamble(self):
        raw = ("LibriVox recording of Dracula by Bram Stoker. "
               "Read in English by David Clarke. "
               "Count Dracula attempts to move from Transylvania to England.")
        out = _strip_librivox_preamble(raw)
        assert out.startswith("Count Dracula")
        assert "LibriVox recording" not in out

    def test_strips_short_preamble_no_reader(self):
        raw = "LibriVox recording of Emma by Jane Austen. A comedy of manners."
        out = _strip_librivox_preamble(raw)
        assert out == "A comedy of manners."

    def test_leaves_unknown_shape_alone(self):
        raw = "A standalone description with no LibriVox preamble."
        assert _strip_librivox_preamble(raw) == raw

    def test_empty_returns_empty(self):
        assert _strip_librivox_preamble("") == ""
        assert _strip_librivox_preamble(None) == ""  # type: ignore[arg-type]


class TestCleanHtmlText:
    """LibriVox descriptions ship with inline HTML — strip to plain text
    so the detail panel can render them through escapeHtml without visible
    tags, while still preserving entity conversion and prose flow."""

    def test_strips_inline_tags(self):
        assert _clean_html_text("<i>Hamlet</i> is a tragedy.") == "Hamlet is a tragedy."

    def test_unescapes_entities(self):
        assert _clean_html_text("A &amp; B &mdash; pair") == "A & B — pair"

    def test_collapses_whitespace_after_tag_strip(self):
        raw = "<p>Line one.</p><br/><br/><p>  Line two.  </p>"
        assert _clean_html_text(raw) == "Line one. Line two."

    def test_empty_and_none_are_empty(self):
        assert _clean_html_text("") == ""
        assert _clean_html_text(None) == ""  # type: ignore[arg-type]

    def test_plain_text_untouched(self):
        assert _clean_html_text("Already clean prose.") == "Already clean prose."


class TestBrowseResultEnrichment:
    """Translator records + url_other are propagated from the feed into
    BrowseResult.extra so per-book normalisers can store them in
    source_metadata for the detail panel."""

    def test_translators_parsed_from_feed(self):
        raw = {
            "id":            "12",
            "title":         "Crime and Punishment",
            "url_iarchive":  "https://archive.org/details/crime_and_punishment_rt_librivox",
            "authors":       [{"first_name": "Fyodor", "last_name": "Dostoyevsky"}],
            "translators":   [
                {"first_name": "Constance", "last_name": "Garnett",
                 "dob": "1861", "dod": "1946"},
            ],
            "url_other":     "https://en.wikipedia.org/wiki/Crime_and_Punishment",
        }
        br = _browse_result_from_feed(raw)
        assert br is not None
        assert br.extra["translators"] == [
            {"name": "Constance Garnett", "dob": "1861", "dod": "1946"},
        ]
        assert br.extra["url_other"] == "https://en.wikipedia.org/wiki/Crime_and_Punishment"

    def test_description_stripped_of_html(self):
        raw = {
            "id":           "99",
            "title":        "X",
            "url_iarchive": "https://archive.org/details/x",
            "description":  "<i>An epic</i> &mdash; about things.",
        }
        br = _browse_result_from_feed(raw)
        assert br is not None
        assert br.description == "An epic — about things."

    def test_missing_translators_yields_empty_list(self):
        raw = {
            "id":           "1",
            "title":        "Y",
            "url_iarchive": "https://archive.org/details/y",
        }
        br = _browse_result_from_feed(raw)
        assert br is not None
        assert br.extra["translators"] == []
        assert br.extra["url_other"] == ""

    def test_coverart_fields_extracted_when_present(self):
        """coverart=1 responses populate coverart_jpg + coverart_thumbnail.
        Prefer the thumbnail for the grid (lighter); keep the full JPG
        for the detail panel."""
        raw = {
            "id":                 "52",
            "title":              "Letters of Two Brides",
            "url_iarchive":       "https://archive.org/details/letters",
            "coverart_jpg":       "https://archive.org/download/cdcover/Letters.jpg",
            "coverart_thumbnail": "https://archive.org/download/cdcover/Letters_thumb.jpg",
        }
        br = _browse_result_from_feed(raw)
        assert br is not None
        assert br.extra["coverart_jpg"] == "https://archive.org/download/cdcover/Letters.jpg"
        assert br.extra["coverart_thumbnail"] == "https://archive.org/download/cdcover/Letters_thumb.jpg"
        # Grid cover should be the lighter thumbnail when present.
        assert br.cover_url == "https://archive.org/download/cdcover/Letters_thumb.jpg"

    def test_coverart_falls_back_to_archive_services_img(self):
        """When LibriVox didn't produce covers for a book, the grid cover
        falls back to archive.org's generic /services/img endpoint so
        the card still shows something."""
        raw = {
            "id":           "7",
            "title":        "No Cover Book",
            "url_iarchive": "https://archive.org/details/no_cover_librivox",
        }
        br = _browse_result_from_feed(raw)
        assert br is not None
        assert br.extra["coverart_jpg"] == ""
        assert br.extra["coverart_thumbnail"] == ""
        assert br.cover_url == f"{ARCHIVE_COVER}/no_cover_librivox"

    def test_coverart_full_jpg_when_thumbnail_missing(self):
        """If only the full JPG is returned (old catalog entries), the
        grid still upgrades off archive.org/services/img."""
        raw = {
            "id":           "3",
            "title":        "Half Covered",
            "url_iarchive": "https://archive.org/details/half",
            "coverart_jpg": "https://archive.org/download/cdcover/Half.jpg",
        }
        br = _browse_result_from_feed(raw)
        assert br is not None
        assert br.cover_url == "https://archive.org/download/cdcover/Half.jpg"

    def test_url_zip_file_extracted(self):
        """Whole-book ZIP is surfaced as a 'Download MP3s' link in the
        detail panel — store it on the browse result so the pin
        normaliser can forward it into source_metadata."""
        raw = {
            "id":            "4",
            "title":         "Z",
            "url_iarchive":  "https://archive.org/details/z",
            "url_zip_file":  "https://archive.org/compress/z",
        }
        br = _browse_result_from_feed(raw)
        assert br is not None
        assert br.extra["url_zip_file"] == "https://archive.org/compress/z"
