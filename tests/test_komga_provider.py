"""Contract tests for KomgaProvider.

Mocked httpx. Verifies request shapes we send to Komga and response
parsing for the endpoints we depend on. Live tests against a real Komga
instance live under tests/live/.

Layers covered:
  - _encode_basic / _basic_header helpers
  - ping (actuator + claim fallback)
  - login success + 401 + bad response
  - verify_token
  - fetch_catalog traversal (series → books, pagination, empty-page break)
  - _book_to_catalog_item mapping
  - build_stream_url / build_cover_url URL shapes
  - fetch_progress pagination + aud-shaped translation
  - push_progress page vs completed payload variants
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.media.providers.base import CatalogItem, MediaProvider, ProviderInfo
from augmentum.media.providers.komga import (
    KomgaProvider,
    _basic_header,
    _book_to_catalog_item,
    _encode_basic,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_response(status_code: int, json_body: Any = None) -> MagicMock:
    """Build a MagicMock httpx.Response with given status + body."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body if json_body is not None else {})
    return resp


# --- Helpers -------------------------------------------------------------


class TestAuthHelpers:
    def test_encode_basic_matches_rfc(self):
        # "alice:s3cret" → base64 "YWxpY2U6czNjcmV0"
        assert _encode_basic("alice", "s3cret") == "YWxpY2U6czNjcmV0"

    def test_encode_basic_unicode(self):
        # Unicode creds should UTF-8 encode cleanly
        enc = _encode_basic("ユーザー", "パスワード")
        decoded = base64.b64decode(enc).decode("utf-8")
        assert decoded == "ユーザー:パスワード"

    def test_basic_header_shape(self):
        h = _basic_header("YWxpY2U6czNjcmV0")
        assert h == {"Authorization": "Basic YWxpY2U6czNjcmV0"}


# --- Protocol conformance ------------------------------------------------


class TestProtocolConformance:
    def test_duck_types_as_media_provider(self):
        http = MagicMock()
        p = KomgaProvider(http)
        # MediaProvider is a runtime_checkable Protocol — any object with
        # the full method set matches. This is the cheapest guard against
        # a method rename silently breaking the factory wiring.
        assert isinstance(p, MediaProvider)
        assert p.name == "komga"


# --- Ping ----------------------------------------------------------------


class TestPing:
    def test_ping_returns_info_on_actuator_success(self):
        async def go():
            http = MagicMock()
            http.get = AsyncMock(return_value=_mock_response(200, {
                "komga": {"version": "1.12.0"},
                "build": {"version": "1.12.0", "name": "komga"},
            }))
            p = KomgaProvider(http)
            info = await p.ping("http://localhost:25600/")
            assert info is not None
            assert isinstance(info, ProviderInfo)
            assert info.provider == "komga"
            assert info.base_url == "http://localhost:25600"
            assert info.version == "1.12.0"
        _run(go())

    def test_ping_falls_back_to_claim_on_actuator_auth(self):
        async def go():
            http = MagicMock()
            responses = [
                _mock_response(401),              # actuator locked
                _mock_response(200, {"isClaimed": True}),
            ]
            http.get = AsyncMock(side_effect=responses)
            p = KomgaProvider(http)
            info = await p.ping("http://localhost:25600")
            assert info is not None
            assert info.provider == "komga"
            assert info.is_initialized is True
        _run(go())

    def test_ping_returns_none_on_unknown_server(self):
        async def go():
            http = MagicMock()
            # Actuator returns 404, claim returns 404 — not a Komga server
            http.get = AsyncMock(side_effect=[
                _mock_response(404),
                _mock_response(404),
            ])
            p = KomgaProvider(http)
            info = await p.ping("http://localhost:8080")
            assert info is None
        _run(go())

    def test_ping_returns_none_on_actuator_missing_komga_marker(self):
        async def go():
            http = MagicMock()
            # Actuator returns 200 but body doesn't mention komga → fallback
            # to /claim, which also fails → no fingerprint match
            http.get = AsyncMock(side_effect=[
                _mock_response(200, {"build": {"name": "other-spring-boot"}}),
                _mock_response(404),
            ])
            p = KomgaProvider(http)
            info = await p.ping("http://localhost:8080")
            assert info is None
        _run(go())

    def test_ping_swallows_transport_errors(self):
        async def go():
            http = MagicMock()
            http.get = AsyncMock(side_effect=Exception("connection refused"))
            p = KomgaProvider(http)
            assert await p.ping("http://nothinghere") is None
        _run(go())


# --- Login / verify_token ------------------------------------------------


class TestLogin:
    def test_login_returns_base64_token_on_200(self):
        async def go():
            http = MagicMock()
            http.get = AsyncMock(return_value=_mock_response(200, {
                "id": "u_1", "email": "alice@example.com",
            }))
            p = KomgaProvider(http)
            token = await p.login("http://localhost", "alice", "s3cret")
            assert token == "YWxpY2U6czNjcmV0"
            # Verify Auth header was sent correctly
            http.get.assert_called_once()
            call_kwargs = http.get.call_args.kwargs
            assert call_kwargs["headers"]["Authorization"] == "Basic YWxpY2U6czNjcmV0"
        _run(go())

    def test_login_raises_valueerror_on_401(self):
        async def go():
            http = MagicMock()
            http.get = AsyncMock(return_value=_mock_response(401))
            p = KomgaProvider(http)
            with pytest.raises(ValueError, match="Invalid"):
                await p.login("http://localhost", "alice", "wrong")
        _run(go())

    def test_login_raises_runtimeerror_on_500(self):
        async def go():
            http = MagicMock()
            http.get = AsyncMock(return_value=_mock_response(500))
            p = KomgaProvider(http)
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await p.login("http://localhost", "alice", "s3cret")
        _run(go())


class TestVerifyToken:
    def test_verify_token_true_on_200(self):
        async def go():
            http = MagicMock()
            http.get = AsyncMock(return_value=_mock_response(200, {}))
            p = KomgaProvider(http)
            assert await p.verify_token("http://localhost", "TOKEN") is True
        _run(go())

    def test_verify_token_false_on_401(self):
        async def go():
            http = MagicMock()
            http.get = AsyncMock(return_value=_mock_response(401))
            p = KomgaProvider(http)
            assert await p.verify_token("http://localhost", "BADTOKEN") is False
        _run(go())

    def test_verify_token_false_on_exception(self):
        async def go():
            http = MagicMock()
            http.get = AsyncMock(side_effect=Exception("net down"))
            p = KomgaProvider(http)
            assert await p.verify_token("http://localhost", "TOKEN") is False
        _run(go())


# --- Book → CatalogItem mapping ------------------------------------------


class TestBookMapping:
    @staticmethod
    def _series_payload(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "komga_series_id": "sr_1",
            "series_name":     "Berserk",
            "publisher":       "Hakusensha",
            "language":        "ja",
            "status":          "ongoing",
            "genres":          ["Seinen", "Dark Fantasy"],
            "tags":            [],
            "age_rating":      "mature",
            "total_book_count": 41,
            "authors":         [],
        }
        base.update(overrides)
        return base

    @staticmethod
    def _book(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "id": "bk_1",
            "name": "Berserk Vol. 1",
            "seriesId": "sr_1",
            "sizeBytes": 250_000_000,
            "metadata": {
                "title": "Berserk Vol. 1",
                "number": "1",
                "numberSort": 1.0,
                "authors": [{"name": "Kentaro Miura", "role": "writer"}],
                "summary": "Dark fantasy manga.",
                "releaseDate": "1989-08-25",
            },
            "media": {
                "pagesCount": 234,
                "mediaType": "application/vnd.comicbook+zip",
            },
            "readProgress": None,
            "created": "2023-01-01T00:00:00Z",
            "lastModified": "2023-01-02T00:00:00Z",
        }
        base.update(overrides)
        return base

    def test_basic_fields(self):
        item = _book_to_catalog_item(self._book(), self._series_payload())
        assert item is not None
        assert isinstance(item, CatalogItem)
        assert item.external_id == "bk_1"
        assert item.name == "Berserk Vol. 1"
        assert item.kind == "comic"
        assert item.mime_type == "application/vnd.comicbook+zip"
        assert item.size_bytes == 250_000_000
        assert item.duration_ms == 0

    def test_stream_path_format(self):
        item = _book_to_catalog_item(self._book(), self._series_payload())
        assert item.stream_path == "/api/v1/books/bk_1/file"
        # Series-first UI prefers the series poster; book poster is kept in extra.
        assert item.cover_url == "/api/v1/series/sr_1/thumbnail"
        assert item.extra["book_cover_url"] == "/api/v1/books/bk_1/thumbnail"

    def test_volume_extraction(self):
        item = _book_to_catalog_item(self._book(), self._series_payload())
        assert item.extra["volume"] == "1"
        assert item.extra["volume_sort"] == 1.0

    def test_decimal_volume_sort(self):
        book = self._book()
        book["metadata"]["numberSort"] = 1.5
        item = _book_to_catalog_item(book, self._series_payload())
        assert item.extra["volume_sort"] == 1.5

    def test_series_payload_preserved(self):
        item = _book_to_catalog_item(self._book(), self._series_payload())
        assert item.extra["komga_series_id"] == "sr_1"
        assert item.extra["series_name"] == "Berserk"
        assert item.extra["publisher"] == "Hakusensha"
        assert item.extra["status"] == "ongoing"
        assert "Seinen" in item.extra["genres"]

    def test_progress_without_read_record(self):
        item = _book_to_catalog_item(self._book(), self._series_payload())
        assert item.progress_pct == 0.0
        assert item.extra["current_page"] == 0
        assert item.extra["is_finished"] is False

    def test_progress_mid_read(self):
        book = self._book(readProgress={"page": 100, "completed": False})
        item = _book_to_catalog_item(book, self._series_payload())
        assert item.progress_pct == pytest.approx(100 / 234, abs=0.001)
        assert item.extra["current_page"] == 100
        assert item.extra["is_finished"] is False

    def test_progress_finished(self):
        book = self._book(readProgress={"page": 234, "completed": True})
        item = _book_to_catalog_item(book, self._series_payload())
        assert item.extra["is_finished"] is True

    def test_author_extraction(self):
        item = _book_to_catalog_item(self._book(), self._series_payload())
        assert item.author == "Kentaro Miura"

    def test_no_id_returns_none(self):
        book = self._book(id="")
        item = _book_to_catalog_item(book, self._series_payload())
        assert item is None

    def test_missing_media_defaults_to_zero_pages(self):
        book = self._book()
        book["media"] = {}
        item = _book_to_catalog_item(book, self._series_payload())
        assert item is not None
        assert item.extra["page_count"] == 0
        assert item.progress_pct == 0.0


# --- fetch_catalog traversal ---------------------------------------------


class TestFetchCatalog:
    def test_traverses_series_and_books(self):
        async def go():
            http = MagicMock()
            series_book_payload = {
                "content": [{
                    "id": "bk_1", "name": "Vol 1", "seriesId": "sr_1",
                    "sizeBytes": 100,
                    "metadata": {"title": "Vol 1", "number": "1", "numberSort": 1.0,
                                 "authors": []},
                    "media": {"pagesCount": 200, "mediaType": "application/vnd.comicbook+zip"},
                    "readProgress": None,
                    "created": "", "lastModified": "",
                }],
                "last": True,
            }
            responses = [
                # Series page 0: one series, last=True
                _mock_response(200, {
                    "content": [{
                        "id": "sr_1", "name": "Berserk",
                        "booksCount": 1,
                        "metadata": {"title": "Berserk", "publisher": "Hakusensha",
                                     "language": "ja", "status": "ongoing",
                                     "genres": [], "tags": [], "ageRating": ""},
                        "booksMetadata": {"authors": []},
                    }],
                    "last": True,
                }),
                # Books under sr_1
                _mock_response(200, series_book_payload),
            ]
            http.post = AsyncMock(side_effect=responses)
            p = KomgaProvider(http)

            items = await p.fetch_catalog("http://localhost", "TOKEN")
            assert len(items) == 1
            assert items[0].external_id == "bk_1"
            assert items[0].extra["komga_series_id"] == "sr_1"
            assert items[0].extra["series_name"] == "Berserk"
            first = http.post.call_args_list[0]
            assert first.args[0] == "http://localhost/api/v1/series/list"
            assert first.kwargs["json"] == {}
            second = http.post.call_args_list[1]
            assert second.args[0] == "http://localhost/api/v1/books/list"
            assert second.kwargs["json"] == {
                "condition": {
                    "seriesId": {"operator": "is", "value": "sr_1"},
                },
            }
        _run(go())

    def test_empty_library_returns_empty(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(200, {
                "content": [], "last": True,
            }))
            p = KomgaProvider(http)
            items = await p.fetch_catalog("http://localhost", "TOKEN")
            assert items == []
        _run(go())

    def test_paginates_across_multiple_series_pages(self):
        async def go():
            http = MagicMock()
            # Build three series pages: first has 2 series, second has 1,
            # third is empty-but-last (Komga sometimes returns empty with last=True).
            def _series(id_: str):
                return {
                    "id": id_, "name": f"Series {id_}", "booksCount": 1,
                    "metadata": {"title": f"Series {id_}", "genres": [], "tags": []},
                    "booksMetadata": {},
                }

            def _book(id_: str, sid: str):
                return {
                    "id": id_, "name": f"Book {id_}", "seriesId": sid,
                    "sizeBytes": 0,
                    "metadata": {"title": f"Book {id_}", "number": "1", "numberSort": 1.0, "authors": []},
                    "media": {"pagesCount": 1, "mediaType": "application/vnd.comicbook+zip"},
                    "readProgress": None, "created": "", "lastModified": "",
                }

            responses = [
                # Series page 0: 2 series, not last
                _mock_response(200, {
                    "content": [_series("sr_a"), _series("sr_b")],
                    "last": False,
                }),
                # Books for sr_a
                _mock_response(200, {"content": [_book("bk_a", "sr_a")], "last": True}),
                # Books for sr_b
                _mock_response(200, {"content": [_book("bk_b", "sr_b")], "last": True}),
                # Series page 1: 1 more series, last=True
                _mock_response(200, {
                    "content": [_series("sr_c")],
                    "last": True,
                }),
                # Books for sr_c
                _mock_response(200, {"content": [_book("bk_c", "sr_c")], "last": True}),
            ]
            http.post = AsyncMock(side_effect=responses)
            p = KomgaProvider(http)

            items = await p.fetch_catalog("http://localhost", "TOKEN")
            assert len(items) == 3
            assert {it.external_id for it in items} == {"bk_a", "bk_b", "bk_c"}
        _run(go())

    def test_failed_series_page_breaks_gracefully(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(500))
            p = KomgaProvider(http)
            items = await p.fetch_catalog("http://localhost", "TOKEN")
            assert items == []
        _run(go())


# --- URL builders --------------------------------------------------------


class TestUrlBuilders:
    def test_build_stream_url_preserves_leading_slash(self):
        p = KomgaProvider(MagicMock())
        url = p.build_stream_url(
            "http://localhost:25600/",
            "/api/v1/books/bk_1/file",
            "TOKEN",
        )
        assert url == "http://localhost:25600/api/v1/books/bk_1/file"
        # Note: no token in URL — Komga requires Authorization header

    def test_build_stream_url_adds_leading_slash(self):
        p = KomgaProvider(MagicMock())
        url = p.build_stream_url(
            "http://localhost",
            "api/v1/books/bk_1/file",
            "TOKEN",
        )
        assert url == "http://localhost/api/v1/books/bk_1/file"

    def test_build_cover_url(self):
        p = KomgaProvider(MagicMock())
        url = p.build_cover_url("http://localhost:25600", "bk_1", "TOKEN")
        assert url == "http://localhost:25600/api/v1/books/bk_1/thumbnail"


# --- fetch_progress ------------------------------------------------------


class TestFetchProgress:
    def test_returns_page_shaped_records(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(200, {
                "content": [
                    {
                        "id": "bk_1",
                        "media": {"pagesCount": 200},
                        "readProgress": {"page": 50, "completed": False},
                    },
                    {
                        "id": "bk_2",
                        "media": {"pagesCount": 180},
                        "readProgress": {"page": 180, "completed": True},
                    },
                ],
                "last": True,
            }))
            p = KomgaProvider(http)
            progress = await p.fetch_progress("http://localhost", "TOKEN")
            assert set(progress.keys()) == {"bk_1", "bk_2"}

            bk1 = progress["bk_1"]
            assert bk1["current_time_s"] == 50.0
            assert bk1["duration_s"] == 200.0
            assert bk1["progress"] == pytest.approx(50 / 200)
            assert bk1["is_finished"] is False

            bk2 = progress["bk_2"]
            assert bk2["is_finished"] is True
            call = http.post.call_args
            assert call.args[0] == "http://localhost/api/v1/books/list"
            assert call.kwargs["json"] == {
                "condition": {
                    "anyOf": [
                        {"readStatus": {"operator": "is", "value": "IN_PROGRESS"}},
                        {"readStatus": {"operator": "is", "value": "READ"}},
                    ],
                },
            }
        _run(go())

    def test_skips_books_without_progress(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(200, {
                "content": [
                    {"id": "bk_1", "media": {"pagesCount": 200}, "readProgress": None},
                    {"id": "bk_2", "media": {"pagesCount": 180},
                     "readProgress": {"page": 5, "completed": False}},
                ],
                "last": True,
            }))
            p = KomgaProvider(http)
            progress = await p.fetch_progress("http://localhost", "TOKEN")
            # bk_1 has no readProgress → not included
            assert set(progress.keys()) == {"bk_2"}
        _run(go())


# --- push_progress -------------------------------------------------------


class TestPushProgress:
    def test_finished_sends_completed_true(self):
        async def go():
            http = MagicMock()
            http.patch = AsyncMock(return_value=_mock_response(204))
            p = KomgaProvider(http)
            ok = await p.push_progress(
                "http://localhost", "TOKEN",
                external_id="bk_1",
                current_time_s=200, duration_s=200, is_finished=True,
            )
            assert ok is True
            call = http.patch.call_args
            assert call.kwargs["json"] == {"completed": True}

        _run(go())

    def test_partial_sends_page_int(self):
        async def go():
            http = MagicMock()
            http.patch = AsyncMock(return_value=_mock_response(200))
            p = KomgaProvider(http)
            ok = await p.push_progress(
                "http://localhost", "TOKEN",
                external_id="bk_1",
                current_time_s=47, duration_s=200, is_finished=False,
            )
            assert ok is True
            call = http.patch.call_args
            assert call.kwargs["json"] == {"page": 47}

        _run(go())

    def test_partial_floors_to_minimum_page_one(self):
        """Komga page numbers are 1-indexed; protect against 0 leakage."""
        async def go():
            http = MagicMock()
            http.patch = AsyncMock(return_value=_mock_response(200))
            p = KomgaProvider(http)
            await p.push_progress(
                "http://localhost", "TOKEN",
                external_id="bk_1",
                current_time_s=0, duration_s=200, is_finished=False,
            )
            assert http.patch.call_args.kwargs["json"] == {"page": 1}

        _run(go())

    def test_returns_false_on_http_error(self):
        async def go():
            http = MagicMock()
            http.patch = AsyncMock(return_value=_mock_response(500))
            p = KomgaProvider(http)
            ok = await p.push_progress(
                "http://localhost", "TOKEN",
                external_id="bk_1",
                current_time_s=47, duration_s=200, is_finished=False,
            )
            assert ok is False
        _run(go())

    def test_returns_false_on_transport_exception(self):
        async def go():
            http = MagicMock()
            http.patch = AsyncMock(side_effect=Exception("timeout"))
            p = KomgaProvider(http)
            ok = await p.push_progress(
                "http://localhost", "TOKEN",
                external_id="bk_1",
                current_time_s=47, duration_s=200, is_finished=False,
            )
            assert ok is False
        _run(go())
