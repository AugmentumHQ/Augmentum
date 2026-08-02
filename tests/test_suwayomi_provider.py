"""Contract tests for SuwayomiProvider (GraphQL transport).

Mocked httpx. Verifies:
  - Every method POSTs to /api/graphql with well-formed query bodies
  - The _gql helper triages response failure modes correctly (401/403,
    non-200, HTML body, GraphQL errors array)
  - Translation helpers (_build_manga_payload, _chapter_to_catalog_item)
    produce the CatalogItem shape the sync layer expects
  - external_id is the 3-part "{manga_id}.{source_order}.{chapter_db_id}"
    form needed to drive the updateChapter mutation
  - Legacy 2-part external_ids fail push_progress cleanly

Live tests against a real Suwayomi 2.x live under tests/live/.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.media.providers.base import MediaProvider
from augmentum.media.providers.suwayomi import (
    SuwayomiProvider,
    _auth_headers,
    _build_manga_payload,
    _chapter_to_catalog_item,
    _encode_basic,
)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_response(
    status_code: int,
    json_body: Any = None,
    *,
    content_type: str = "application/json",
    text: str | None = None,
) -> MagicMock:
    """Build a MagicMock response. ``content_type`` defaults to JSON; pass
    ``text`` to simulate a non-JSON 200 (the HTML-SPA-catch-all case).
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    if text is not None:
        resp.text = text
        resp.json = MagicMock(side_effect=ValueError("no JSON"))
    else:
        resp.text = json.dumps(json_body) if json_body is not None else ""
        resp.json = MagicMock(return_value=json_body if json_body is not None else {})
    return resp


def _gql_ok(data: dict) -> MagicMock:
    return _mock_response(200, {"data": data})


def _gql_errors(messages: list[str]) -> MagicMock:
    return _mock_response(
        200, {"errors": [{"message": m} for m in messages], "data": None},
    )


# --- Auth helpers ----------------------------------------------------------


class TestAuthHelpers:
    def test_encode_basic(self):
        assert _encode_basic("alice", "s3cret") == "YWxpY2U6czNjcmV0"

    def test_encode_basic_unicode(self):
        enc = _encode_basic("ユーザー", "パスワード")
        assert base64.b64decode(enc).decode("utf-8") == "ユーザー:パスワード"

    def test_auth_headers_empty_token(self):
        assert _auth_headers("") == {}

    def test_auth_headers_with_token(self):
        h = _auth_headers("YWxpY2U6czNjcmV0")
        assert h == {"Authorization": "Basic YWxpY2U6czNjcmV0"}


# --- Protocol conformance --------------------------------------------------


class TestProtocolConformance:
    def test_duck_types_as_media_provider(self):
        p = SuwayomiProvider(MagicMock())
        assert isinstance(p, MediaProvider)
        assert p.name == "suwayomi"


# --- _gql transport helper --------------------------------------------------


class TestGqlTransport:
    def test_posts_to_api_graphql_path(self):
        """Every call lands on /api/graphql with JSON body."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"aboutServer": {"name": "Suwayomi-Server"}}))
            p = SuwayomiProvider(http)
            await p._gql("http://host:4567", "", "{ aboutServer { name } }")

            http.post.assert_called_once()
            args, kwargs = http.post.call_args
            assert args[0] == "http://host:4567/api/graphql"
            assert kwargs["json"]["query"] == "{ aboutServer { name } }"
            assert kwargs["headers"]["Content-Type"] == "application/json"
        _run(go())

    def test_strips_trailing_slash(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"x": 1}))
            p = SuwayomiProvider(http)
            await p._gql("http://host:4567/", "", "{ x }")
            assert http.post.call_args.args[0] == "http://host:4567/api/graphql"
        _run(go())

    def test_sends_auth_header_when_token_present(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"x": 1}))
            p = SuwayomiProvider(http)
            await p._gql("http://host", "tok123", "{ x }")
            headers = http.post.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Basic tok123"
        _run(go())

    def test_omits_auth_header_when_token_empty(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"x": 1}))
            p = SuwayomiProvider(http)
            await p._gql("http://host", "", "{ x }")
            headers = http.post.call_args.kwargs["headers"]
            assert "Authorization" not in headers
        _run(go())

    def test_passes_variables(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"x": 1}))
            p = SuwayomiProvider(http)
            await p._gql("http://host", "", "query Q($a: Int) { x(a:$a) }", variables={"a": 5})
            body = http.post.call_args.kwargs["json"]
            assert body["variables"] == {"a": 5}
        _run(go())

    def test_omits_variables_key_when_none(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"x": 1}))
            p = SuwayomiProvider(http)
            await p._gql("http://host", "", "{ x }")
            body = http.post.call_args.kwargs["json"]
            assert "variables" not in body
        _run(go())

    def test_401_raises_value_error(self):
        """_gql distinguishes auth failure (401/403) from other HTTP errors."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(401, text="Unauthorized"))
            p = SuwayomiProvider(http)
            with pytest.raises(ValueError, match="Authentication rejected"):
                await p._gql("http://host", "bad", "{ x }")
        _run(go())

    def test_403_raises_value_error(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(403, text="Forbidden"))
            p = SuwayomiProvider(http)
            with pytest.raises(ValueError):
                await p._gql("http://host", "tok", "{ x }")
        _run(go())

    def test_500_raises_runtime_error(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(500, text="boom"))
            p = SuwayomiProvider(http)
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await p._gql("http://host", "", "{ x }")
        _run(go())

    def test_html_body_on_200_raises_diagnostic_error(self):
        """The smoking-gun case: SPA catch-all serves index.html for an API path
        that doesn't exist. Must surface loudly, not as opaque JSON errors."""
        async def go():
            html = "<!DOCTYPE html><html><head><title>Suwayomi</title>..."
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(
                200, text=html, content_type="text/html",
            ))
            p = SuwayomiProvider(http)
            with pytest.raises(RuntimeError) as exc:
                await p._gql("http://host", "", "{ x }")
            msg = str(exc.value)
            assert "non-JSON" in msg
            assert "text/html" in msg
            # Body snippet is included so users can see what came back.
            assert "Suwayomi" in msg
        _run(go())

    def test_graphql_errors_array_surfaces_first_message(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_errors([
                "Cannot query field 'nope' on type 'Manga'",
                "Field 'also' deprecated",
            ]))
            p = SuwayomiProvider(http)
            with pytest.raises(RuntimeError) as exc:
                await p._gql("http://host", "", "{ manga { nope } }")
            assert "Cannot query field 'nope'" in str(exc.value)
        _run(go())

    def test_missing_data_field_raises(self):
        async def go():
            http = MagicMock()
            # Well-formed JSON but no `data` key
            http.post = AsyncMock(return_value=_mock_response(200, {"something": "else"}))
            p = SuwayomiProvider(http)
            with pytest.raises(RuntimeError, match="missing `data`"):
                await p._gql("http://host", "", "{ x }")
        _run(go())

    def test_success_returns_data_dict(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"aboutServer": {"name": "Suwayomi-Server", "version": "v2.1"}}))
            p = SuwayomiProvider(http)
            data = await p._gql("http://host", "", "{ aboutServer { name version } }")
            assert data == {"aboutServer": {"name": "Suwayomi-Server", "version": "v2.1"}}
        _run(go())


# --- Ping ------------------------------------------------------------------


class TestPing:
    def test_ping_success(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "aboutServer": {"name": "Suwayomi-Server", "version": "v2.1.1867"},
            }))
            p = SuwayomiProvider(http)
            info = await p.ping("http://host:4567")
            assert info is not None
            assert info.provider == "suwayomi"
            assert info.base_url == "http://host:4567"
            assert info.version == "v2.1.1867"
            assert info.server_name == "Suwayomi-Server"
        _run(go())

    def test_ping_accepts_legacy_tachidesk_name(self):
        """Older builds identify as Tachidesk-Server; still a valid match."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "aboutServer": {"name": "Tachidesk-Server", "version": "v1.x"},
            }))
            p = SuwayomiProvider(http)
            info = await p.ping("http://host:4567")
            assert info is not None
            assert info.version == "v1.x"
        _run(go())

    def test_ping_rejects_arbitrary_graphql_servers(self):
        """If aboutServer returns some unrelated name, don't claim it."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "aboutServer": {"name": "SomeOtherGQL", "version": "1.0"},
            }))
            p = SuwayomiProvider(http)
            info = await p.ping("http://host:4567")
            assert info is None
        _run(go())

    def test_ping_returns_none_on_401(self):
        """Unusual but possible: some Suwayomi configs gate even aboutServer."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(401, text="U"))
            p = SuwayomiProvider(http)
            info = await p.ping("http://host:4567")
            assert info is None
        _run(go())

    def test_ping_returns_none_on_connection_error(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(side_effect=Exception("connection refused"))
            p = SuwayomiProvider(http)
            info = await p.ping("http://host:4567")
            assert info is None
        _run(go())

    def test_ping_sends_no_auth_header(self):
        """aboutServer is public; ping must work even with no credentials."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "aboutServer": {"name": "Suwayomi-Server", "version": "v2"},
            }))
            p = SuwayomiProvider(http)
            await p.ping("http://host")
            assert "Authorization" not in http.post.call_args.kwargs["headers"]
        _run(go())


# --- Login -----------------------------------------------------------------


class TestLogin:
    def test_login_no_auth_returns_empty_token(self):
        """User leaves creds blank → ping validates, return empty token."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "aboutServer": {"name": "Suwayomi-Server", "version": "v2"},
            }))
            p = SuwayomiProvider(http)
            token = await p.login("http://host", "", "")
            assert token == ""
        _run(go())

    def test_login_no_auth_raises_if_not_suwayomi(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "aboutServer": {"name": "Kavita", "version": "1"},
            }))
            p = SuwayomiProvider(http)
            with pytest.raises(RuntimeError, match="unreachable or not recognized"):
                await p.login("http://host", "", "")
        _run(go())

    def test_login_with_creds_probes_auth_gated_query(self):
        """With creds, login hits the auth-gated mangas query to validate."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "mangas": {"nodes": []},
            }))
            p = SuwayomiProvider(http)
            token = await p.login("http://host", "alice", "s3cret")
            assert token == _encode_basic("alice", "s3cret")
            # Sent the Basic header
            assert http.post.call_args.kwargs["headers"]["Authorization"] == f"Basic {token}"
            # Sent the auth-probe query (not aboutServer)
            sent_query = http.post.call_args.kwargs["json"]["query"]
            assert "mangas" in sent_query
        _run(go())

    def test_login_wrong_password_raises_value_error(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(401, text="Unauthorized"))
            p = SuwayomiProvider(http)
            with pytest.raises(ValueError, match="Invalid username or password"):
                await p.login("http://host", "alice", "wrong")
        _run(go())


# --- verify_token ----------------------------------------------------------


class TestVerifyToken:
    def test_verify_success(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"mangas": {"nodes": []}}))
            p = SuwayomiProvider(http)
            assert (await p.verify_token("http://host", "tok")) is True
        _run(go())

    def test_verify_401(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(401, text="U"))
            p = SuwayomiProvider(http)
            assert (await p.verify_token("http://host", "badtok")) is False
        _run(go())

    def test_verify_connection_error(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(side_effect=Exception("timeout"))
            p = SuwayomiProvider(http)
            assert (await p.verify_token("http://host", "tok")) is False
        _run(go())


# --- Translation helpers ---------------------------------------------------


class TestBuildMangaPayload:
    def test_all_fields_populated(self):
        manga = {
            "id": 42,
            "title": "Berserk",
            "author": "Kentaro Miura",
            "artist": "Kentaro Miura",
            "description": "Dark fantasy epic",
            "status": "ONGOING",
            "genre": ["Dark Fantasy", "Action"],
            "sourceId": "mangadex",
            "realUrl": "https://mangadex.org/title/abc",
            "thumbnailUrl": "/api/v1/manga/42/thumbnail",
            "lastFetchedAt": 1700000000,
            "inLibraryAt": 1699000000,
        }
        payload = _build_manga_payload(manga)
        assert payload["suwayomi_manga_id"] == 42
        assert payload["series_name"] == "Berserk"
        assert payload["author"] == "Kentaro Miura"
        assert payload["status"] == "ongoing"
        assert payload["genres"] == ["Dark Fantasy", "Action"]

    def test_missing_fields_become_empty(self):
        """Manga with only id/title; every optional field falls back cleanly."""
        payload = _build_manga_payload({"id": 1, "title": "x"})
        assert payload["author"] == ""
        assert payload["genres"] == []
        assert payload["status"] is None
        assert payload["thumbnail_url"] == ""


class TestChapterToCatalogItem:
    def _manga(self):
        return _build_manga_payload({"id": 42, "title": "Berserk", "author": "Miura"})

    def test_happy_path(self):
        item = _chapter_to_catalog_item({
            "id": 999,
            "mangaId": 42,
            "sourceOrder": 5,
            "name": "Chapter 5",
            "chapterNumber": 5.0,
            "pageCount": 20,
            "lastPageRead": 10,
            "isRead": False,
            "isBookmarked": False,
            "isDownloaded": False,
            "scanlator": "SomeScan",
            "realUrl": "https://source.example/c/5",
            "fetchedAt": 1700000000,
            "uploadDate": 1699500000,
        }, self._manga())
        assert item is not None
        # external_id is the 3-part form needed for push_progress
        assert item.external_id == "42.5.999"
        assert item.kind == "comic"
        assert item.name == "Chapter 5"
        assert item.mime_type == "application/vnd.comicbook+zip"
        assert item.progress_pct == 0.5  # 10/20
        assert item.cover_url == "/api/v1/manga/42/thumbnail"
        assert item.stream_path == "/api/v1/manga/42/chapter/5"
        assert item.author == "Miura"
        # extra carries chapter_db_id for the mutation
        assert item.extra["chapter_db_id"] == 999
        assert item.extra["chapter_source_order"] == 5
        assert item.extra["page_count"] == 20
        assert item.extra["scanlator"] == "SomeScan"
        assert item.extra["series_name"] == "Berserk"

    def test_returns_none_without_source_order(self):
        item = _chapter_to_catalog_item({
            "id": 1, "mangaId": 1, "pageCount": 20,
        }, self._manga())
        assert item is None

    def test_returns_none_without_chapter_id(self):
        item = _chapter_to_catalog_item({
            "mangaId": 1, "sourceOrder": 5, "pageCount": 20,
        }, self._manga())
        assert item is None

    def test_returns_none_without_manga_id(self):
        # Manga payload missing the suwayomi_manga_id
        bad_manga = {"series_name": "x"}
        item = _chapter_to_catalog_item({
            "id": 1, "sourceOrder": 5, "pageCount": 20,
        }, bad_manga)
        assert item is None

    def test_returns_none_for_non_dict(self):
        assert _chapter_to_catalog_item("not a dict", self._manga()) is None
        assert _chapter_to_catalog_item(None, self._manga()) is None

    def test_synthesized_name_when_missing(self):
        item = _chapter_to_catalog_item({
            "id": 1, "mangaId": 42, "sourceOrder": 3,
            "chapterNumber": 3.0, "pageCount": 0,
            "name": "",
        }, self._manga())
        assert item is not None
        assert "Berserk" in item.name
        assert "3" in item.name

    def test_zero_page_count_progress_is_zero_not_div_by_zero(self):
        item = _chapter_to_catalog_item({
            "id": 1, "mangaId": 42, "sourceOrder": 1,
            "pageCount": 0, "lastPageRead": 0,
            "name": "x",
        }, self._manga())
        assert item is not None
        assert item.progress_pct == 0.0


# --- Catalog ---------------------------------------------------------------


class TestFetchCatalog:
    def test_fetches_library_and_flattens_to_chapters(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "mangas": {"nodes": [
                    {
                        "id": 1, "title": "Berserk", "author": "Miura",
                        "artist": "", "description": "", "status": "ONGOING",
                        "genre": [], "sourceId": "", "realUrl": "",
                        "thumbnailUrl": "", "lastFetchedAt": 0, "inLibraryAt": 0,
                        "chapters": {"nodes": [
                            {"id": 100, "mangaId": 1, "name": "Ch 1",
                             "sourceOrder": 1, "chapterNumber": 1.0,
                             "scanlator": "", "pageCount": 20,
                             "isRead": False, "isBookmarked": False, "isDownloaded": False,
                             "lastPageRead": 0, "uploadDate": 0,
                             "realUrl": "", "fetchedAt": 0},
                            {"id": 101, "mangaId": 1, "name": "Ch 2",
                             "sourceOrder": 2, "chapterNumber": 2.0,
                             "scanlator": "", "pageCount": 22,
                             "isRead": True, "isBookmarked": False, "isDownloaded": False,
                             "lastPageRead": 22, "uploadDate": 0,
                             "realUrl": "", "fetchedAt": 0},
                        ]},
                    },
                    {
                        "id": 2, "title": "Vinland Saga", "author": "Yukimura",
                        "artist": "", "description": "", "status": "ONGOING",
                        "genre": [], "sourceId": "", "realUrl": "",
                        "thumbnailUrl": "", "lastFetchedAt": 0, "inLibraryAt": 0,
                        "chapters": {"nodes": [
                            {"id": 200, "mangaId": 2, "name": "Ch 1",
                             "sourceOrder": 1, "chapterNumber": 1.0,
                             "scanlator": "", "pageCount": 30,
                             "isRead": False, "isBookmarked": False, "isDownloaded": False,
                             "lastPageRead": 5, "uploadDate": 0,
                             "realUrl": "", "fetchedAt": 0},
                        ]},
                    },
                ]},
            }))
            p = SuwayomiProvider(http)
            items = await p.fetch_catalog("http://host", "")
            assert len(items) == 3  # 2 Berserk chapters + 1 Vinland
            # external_ids are {manga_id}.{source_order}.{chapter_db_id}
            ids = {i.external_id for i in items}
            assert ids == {"1.1.100", "1.2.101", "2.1.200"}
            # Per-item authorship traced back to the parent manga
            berserk = [i for i in items if i.extra["series_name"] == "Berserk"]
            assert len(berserk) == 2
            assert all(i.author == "Miura" for i in berserk)
        _run(go())

    def test_library_query_passes_first_variable(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"mangas": {"nodes": []}}))
            p = SuwayomiProvider(http)
            await p.fetch_catalog("http://host", "tok")
            body = http.post.call_args.kwargs["json"]
            assert "Library" in body["query"]
            assert body["variables"]["first"] > 0

        _run(go())

    def test_empty_library_returns_empty_list(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"mangas": {"nodes": []}}))
            p = SuwayomiProvider(http)
            items = await p.fetch_catalog("http://host", "tok")
            assert items == []
        _run(go())

    def test_manga_with_no_chapters_contributes_nothing(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "mangas": {"nodes": [
                    {"id": 1, "title": "Empty", "chapters": {"nodes": []}},
                ]},
            }))
            p = SuwayomiProvider(http)
            items = await p.fetch_catalog("http://host", "tok")
            assert items == []
        _run(go())

    def test_unexpected_shape_returns_empty_and_warns(self):
        async def go():
            http = MagicMock()
            # mangas field is not a dict-with-nodes; still shouldn't crash
            http.post = AsyncMock(return_value=_gql_ok({"mangas": {"nodes": "not a list"}}))
            p = SuwayomiProvider(http)
            items = await p.fetch_catalog("http://host", "tok")
            assert items == []
        _run(go())

    def test_raises_on_transport_failure(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(500, text="boom"))
            p = SuwayomiProvider(http)
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await p.fetch_catalog("http://host", "tok")
        _run(go())


# --- fetch_progress --------------------------------------------------------


class TestFetchProgress:
    def test_returns_external_id_keyed_dict(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "chapters": {"nodes": [
                    {"id": 100, "mangaId": 1, "sourceOrder": 5,
                     "isRead": False, "lastPageRead": 10, "pageCount": 20},
                    {"id": 101, "mangaId": 2, "sourceOrder": 1,
                     "isRead": True, "lastPageRead": 30, "pageCount": 30},
                ]},
            }))
            p = SuwayomiProvider(http)
            prog = await p.fetch_progress("http://host", "tok")
            assert "1.5.100" in prog
            assert prog["1.5.100"]["current_time_s"] == 10.0
            assert prog["1.5.100"]["duration_s"] == 20.0
            assert prog["1.5.100"]["progress"] == 0.5
            assert prog["1.5.100"]["is_finished"] is False
            # Finished chapter
            assert prog["2.1.101"]["is_finished"] is True
        _run(go())

    def test_returns_empty_dict_on_transport_failure(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_mock_response(500, text="boom"))
            p = SuwayomiProvider(http)
            # fetch_progress swallows errors (not worth blocking the UI over)
            prog = await p.fetch_progress("http://host", "tok")
            assert prog == {}
        _run(go())

    def test_zero_page_count_produces_zero_progress(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "chapters": {"nodes": [
                    {"id": 1, "mangaId": 1, "sourceOrder": 1,
                     "isRead": True, "lastPageRead": 0, "pageCount": 0},
                ]},
            }))
            p = SuwayomiProvider(http)
            prog = await p.fetch_progress("http://host", "tok")
            assert prog["1.1.1"]["progress"] == 0.0

        _run(go())


# --- fetch_item_details ----------------------------------------------------


class TestFetchItemDetails:
    def test_returns_manga_and_specific_chapter(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "manga": {
                    "id": 42, "title": "Berserk",
                    "chapters": {"nodes": [
                        {"id": 100, "sourceOrder": 1, "name": "Ch 1"},
                        {"id": 101, "sourceOrder": 2, "name": "Ch 2"},
                    ]},
                },
            }))
            p = SuwayomiProvider(http)
            details = await p.fetch_item_details("http://host", "tok", external_id="42.2.101")
            assert details is not None
            assert details["manga"]["title"] == "Berserk"
            assert details["chapter"]["name"] == "Ch 2"
        _run(go())

    def test_returns_none_for_malformed_external_id(self):
        async def go():
            http = MagicMock()
            p = SuwayomiProvider(http)
            assert (await p.fetch_item_details("http://host", "tok", external_id="bad")) is None
            assert (await p.fetch_item_details("http://host", "tok", external_id="a.b.c")) is None
            # http.post was never called — parsing fails before the request
            http.post.assert_not_called() if hasattr(http, "post") else None
        _run(go())

    def test_returns_none_when_chapter_not_in_list(self):
        """Chapter was deleted upstream since last sync → fetch returns None."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "manga": {
                    "id": 42, "title": "Berserk",
                    "chapters": {"nodes": [
                        {"id": 100, "sourceOrder": 1, "name": "Ch 1"},
                    ]},
                },
            }))
            p = SuwayomiProvider(http)
            # Asking for sourceOrder=99, doesn't exist
            details = await p.fetch_item_details("http://host", "tok", external_id="42.99.9999")
            assert details is None
        _run(go())


# --- push_progress ---------------------------------------------------------


class TestPushProgress:
    def test_sends_updateChapter_mutation_with_page_and_read(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "updateChapter": {"chapter": {"id": 100, "lastPageRead": 10, "isRead": False}},
            }))
            p = SuwayomiProvider(http)
            ok = await p.push_progress(
                "http://host", "tok",
                external_id="1.5.100",
                current_time_s=10.0, duration_s=20.0,
                is_finished=False,
            )
            assert ok is True

            body = http.post.call_args.kwargs["json"]
            assert "updateChapter" in body["query"]
            assert body["variables"] == {
                "input": {"id": 100, "patch": {"lastPageRead": 10}},
            }
        _run(go())

    def test_finished_sets_isRead_true_and_lastPageRead_max(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({
                "updateChapter": {"chapter": {"id": 100, "lastPageRead": 19, "isRead": True}},
            }))
            p = SuwayomiProvider(http)
            ok = await p.push_progress(
                "http://host", "tok",
                external_id="1.5.100",
                current_time_s=0.0, duration_s=20.0,
                is_finished=True,
            )
            assert ok is True
            body = http.post.call_args.kwargs["json"]
            assert body["variables"]["input"]["patch"] == {
                "isRead": True, "lastPageRead": 19,
            }
        _run(go())

    def test_legacy_two_part_external_id_skips_cleanly(self):
        """Legacy 2-part external_ids can't be updated; return False, no raise."""
        async def go():
            http = MagicMock()
            http.post = AsyncMock()
            p = SuwayomiProvider(http)
            ok = await p.push_progress(
                "http://host", "tok",
                external_id="1.5",  # 2-part legacy
                current_time_s=10.0, duration_s=20.0,
            )
            assert ok is False
            http.post.assert_not_called()
        _run(go())

    def test_non_int_chapter_id_returns_false(self):
        async def go():
            http = MagicMock()
            p = SuwayomiProvider(http)
            ok = await p.push_progress(
                "http://host", "tok",
                external_id="1.5.nothex",
                current_time_s=10.0, duration_s=20.0,
            )
            assert ok is False
        _run(go())

    def test_returns_false_on_graphql_error(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_errors(["Chapter not found"]))
            p = SuwayomiProvider(http)
            ok = await p.push_progress(
                "http://host", "tok",
                external_id="1.5.100",
                current_time_s=10.0, duration_s=20.0,
            )
            assert ok is False
        _run(go())

    def test_current_time_rounded_to_int_page(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"updateChapter": {"chapter": {"id": 100}}}))
            p = SuwayomiProvider(http)
            await p.push_progress(
                "http://host", "tok",
                external_id="1.5.100",
                current_time_s=7.7, duration_s=20.0,
            )
            body = http.post.call_args.kwargs["json"]
            # Rounded to nearest int page
            assert body["variables"]["input"]["patch"]["lastPageRead"] == 8
        _run(go())

    def test_negative_current_time_clamped_to_zero(self):
        async def go():
            http = MagicMock()
            http.post = AsyncMock(return_value=_gql_ok({"updateChapter": {"chapter": {"id": 100}}}))
            p = SuwayomiProvider(http)
            await p.push_progress(
                "http://host", "tok",
                external_id="1.5.100",
                current_time_s=-3.0, duration_s=20.0,
            )
            body = http.post.call_args.kwargs["json"]
            assert body["variables"]["input"]["patch"]["lastPageRead"] == 0
        _run(go())


# --- Streaming URL helpers -------------------------------------------------


class TestStreamingUrls:
    def test_build_stream_url_prepends_base(self):
        p = SuwayomiProvider(MagicMock())
        url = p.build_stream_url("http://host:4567", "/api/v1/manga/42/chapter/5", "tok")
        assert url == "http://host:4567/api/v1/manga/42/chapter/5"

    def test_build_stream_url_normalizes_path(self):
        p = SuwayomiProvider(MagicMock())
        # No leading slash on stream_path → still works
        url = p.build_stream_url("http://host/", "api/v1/manga/42/chapter/5", "tok")
        assert url == "http://host/api/v1/manga/42/chapter/5"

    def test_build_cover_url_extracts_manga_id_from_3_part_external(self):
        p = SuwayomiProvider(MagicMock())
        url = p.build_cover_url("http://host:4567", "42.5.100", "tok")
        assert url == "http://host:4567/api/v1/manga/42/thumbnail"

    def test_build_cover_url_handles_legacy_2_part_external(self):
        p = SuwayomiProvider(MagicMock())
        url = p.build_cover_url("http://host", "42.5", "tok")
        assert url == "http://host/api/v1/manga/42/thumbnail"

    def test_build_cover_url_no_dot(self):
        # external_id was somehow flat — use the whole thing as manga_id
        p = SuwayomiProvider(MagicMock())
        url = p.build_cover_url("http://host", "42", "tok")
        assert url == "http://host/api/v1/manga/42/thumbnail"
