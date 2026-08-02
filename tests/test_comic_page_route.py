"""Contract tests for the comic per-page delivery route.

Covers the `/api/media/comic/page/{file_id}?page=N` endpoint:
  - kind='comic' gate
  - per-provider upstream URL shape (Komga 1-indexed, Suwayomi 0-indexed)
  - Basic-auth header injected only when access_token present
  - upstream 404 → 404 passthrough, 5xx → 502
  - upstream Content-Type preserved
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@dataclass
class _FakeEntry:
    """Minimal stand-in for vfs.index.FileEntry."""
    id: str
    user_id: str
    source: str
    kind: str
    source_metadata: dict = field(default_factory=dict)


def _make_stream_ctx(status: int, body: bytes, content_type: str = "image/jpeg"):
    """Build an async-context-manager mock that yields a streamed response."""
    response = MagicMock()
    response.status_code = status
    response.headers = {"content-type": content_type}
    response.aread = AsyncMock(return_value=body)

    class _Ctx:
        async def __aenter__(self):
            return response
        async def __aexit__(self, exc_type, exc, tb):
            return None

    return _Ctx()


# --- Direct handler invocation (bypasses FastAPI routing) ----------------
#
# The handler is ``comic_page`` and it takes (file_id, request). We build
# a minimal Request-like object with app.state + query_params + scope so
# the handler's _user_id / _get_index / _get_store / _http accessors work.


class _FakeRequest:
    def __init__(
        self,
        *,
        user_id: str,
        entry: _FakeEntry | None,
        server,
        page: str = "1",
        extra_query: dict | None = None,
        http_stream_factory=None,
    ):
        self.scope = {"user": MagicMock(id=user_id)}
        self.query_params = {"page": page, **(extra_query or {})}

        # file_index mock
        file_index = MagicMock()
        file_index.get = AsyncMock(return_value=entry)

        # store mock
        store = MagicMock()
        store.get = AsyncMock(return_value=server)

        # state_manager → backend → conn (used by _get_store); we bypass
        # that chain and inject the store directly via _get_store patching
        # in the tests.
        self.app = MagicMock()
        self.app.state.file_index = file_index
        self.app.state.state_manager = MagicMock()
        self.app.state.state_manager.backend = MagicMock()

        # http client mock with optional stream factory
        http = MagicMock()
        if http_stream_factory is not None:
            http.stream = MagicMock(side_effect=lambda *a, **kw: http_stream_factory(*a, **kw))
        self.app.state.http_client = http
        self._store = store
        self.headers = {}


class _FakeServer:
    def __init__(self, *, base_url: str, access_token: str = ""):
        self.base_url = base_url
        self.access_token = access_token


# --- Tests ---------------------------------------------------------------


def test_rejects_non_comic_kind():
    async def go():
        from augmentum.proxy.media_routes import comic_page
        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="audiobookshelf",
            kind="audio",
            source_metadata={"server_id": "ms_1", "provider": "audiobookshelf",
                             "external_id": "abs_1"},
        )
        req = _FakeRequest(user_id="u_a", entry=entry,
                           server=_FakeServer(base_url="http://abs"))
        # Patch _get_store to return our mock (it normally reads from a
        # SQLiteBackend on the real app state)
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)
        assert resp.status_code == 400
        body = json.loads(bytes(resp.body))
        assert "Not a comic" in body["error"]
    _run(go())


def test_returns_404_when_entry_missing():
    async def go():
        from augmentum.proxy.media_routes import comic_page
        req = _FakeRequest(user_id="u_a", entry=None,
                           server=_FakeServer(base_url=""))
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_nope", req)
        assert resp.status_code == 404
    _run(go())


def test_rejects_invalid_page_number():
    async def go():
        from augmentum.proxy.media_routes import comic_page
        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "komga",
                             "external_id": "bk_1"},
        )
        req = _FakeRequest(user_id="u_a", entry=entry,
                           server=_FakeServer(base_url="http://komga"),
                           page="not-a-number")
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)
        assert resp.status_code == 400
    _run(go())


def test_rejects_page_zero_and_below():
    async def go():
        from augmentum.proxy.media_routes import comic_page
        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "komga",
                             "external_id": "bk_1"},
        )
        req = _FakeRequest(user_id="u_a", entry=entry,
                           server=_FakeServer(base_url="http://komga"),
                           page="0")
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)
        assert resp.status_code == 400
    _run(go())


def test_komga_1_indexed_url_shape():
    """Komga pages are 1-indexed on the upstream; our API is also 1-indexed
    so the index passes through unchanged."""
    async def go():
        from augmentum.proxy.media_routes import comic_page
        captured_url = []

        def stream_factory(method, url, **kw):
            captured_url.append((method, url, kw))
            return _make_stream_ctx(200, b"\xff\xd8image-bytes")

        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "komga",
                             "external_id": "bk_book_id"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://komga:25600/",
                               access_token="YWxpY2U6czNjcmV0"),
            page="3",
            http_stream_factory=stream_factory,
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)

        assert resp.status_code == 200
        method, url, kw = captured_url[0]
        assert method == "GET"
        assert url == "http://komga:25600/api/v1/books/bk_book_id/pages/3"
        # Basic header attached server-side
        assert kw["headers"]["Authorization"] == "Basic YWxpY2U6czNjcmV0"
    _run(go())


def test_suwayomi_0_indexed_url_shape():
    """Suwayomi pages are 0-indexed upstream — page 1 → /page/0, page 5 → /page/4."""
    async def go():
        from augmentum.proxy.media_routes import comic_page
        captured_url = []

        def stream_factory(method, url, **kw):
            captured_url.append((method, url, kw))
            return _make_stream_ctx(200, b"image-bytes")

        entry = _FakeEntry(
            id="fi_2", user_id="u_a", source="suwayomi", kind="comic",
            source_metadata={"server_id": "ms_s", "provider": "suwayomi",
                             "external_id": "42.5"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://suwayomi:4567",
                               access_token=""),
            page="5",
            http_stream_factory=stream_factory,
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_2", req)

        assert resp.status_code == 200
        method, url, kw = captured_url[0]
        assert url == "http://suwayomi:4567/api/v1/manga/42/chapter/5/page/4"
        # No-auth Suwayomi → no Authorization header
        assert "Authorization" not in kw["headers"]
    _run(go())


def test_suwayomi_with_auth_sends_basic_header():
    async def go():
        from augmentum.proxy.media_routes import comic_page
        captured = []

        def stream_factory(method, url, **kw):
            captured.append(kw)
            return _make_stream_ctx(200, b"img")

        entry = _FakeEntry(
            id="fi_2", user_id="u_a", source="suwayomi", kind="comic",
            source_metadata={"server_id": "ms_s", "provider": "suwayomi",
                             "external_id": "1.0"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://suwayomi",
                               access_token="TOKEN"),
            http_stream_factory=stream_factory,
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            await comic_page("fi_2", req)
        assert captured[0]["headers"]["Authorization"] == "Basic TOKEN"
    _run(go())


def test_komga_raw_page_url_shape():
    """``quality=raw`` should use Komga's raw page endpoint for the reader."""
    async def go():
        from augmentum.proxy.media_routes import comic_page
        captured_url = []

        def stream_factory(method, url, **kw):
            captured_url.append((method, url, kw))
            return _make_stream_ctx(200, b"img")

        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "komga",
                             "external_id": "bk_book_id"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://komga:25600", access_token="T"),
            page="3",
            extra_query={"quality": "raw"},
            http_stream_factory=stream_factory,
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)

        assert resp.status_code == 200
        _, url, _ = captured_url[0]
        assert url == "http://komga:25600/api/v1/books/bk_book_id/pages/3/raw"
    _run(go())


def test_komga_thumbnail_page_url_shape():
    """``thumb=1`` should use Komga's per-page thumbnail endpoint."""
    async def go():
        from augmentum.proxy.media_routes import comic_page
        captured_url = []

        def stream_factory(method, url, **kw):
            captured_url.append((method, url, kw))
            return _make_stream_ctx(200, b"img")

        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "komga",
                             "external_id": "bk_book_id"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://komga:25600", access_token="T"),
            page="3",
            extra_query={"thumb": "1"},
            http_stream_factory=stream_factory,
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)

        assert resp.status_code == 200
        _, url, _ = captured_url[0]
        assert url == "http://komga:25600/api/v1/books/bk_book_id/pages/3/thumbnail"
    _run(go())


def test_upstream_404_passes_through():
    async def go():
        from augmentum.proxy.media_routes import comic_page

        def stream_factory(method, url, **kw):
            return _make_stream_ctx(404, b"", "application/json")

        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "komga",
                             "external_id": "bk_1"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://komga",
                               access_token="T"),
            page="9999",
            http_stream_factory=stream_factory,
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)
        assert resp.status_code == 404
    _run(go())


def test_upstream_500_returns_502():
    """Non-standard upstream errors get wrapped as 502 (bad gateway)."""
    async def go():
        from augmentum.proxy.media_routes import comic_page

        def stream_factory(method, url, **kw):
            return _make_stream_ctx(500, b"", "text/html")

        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "komga",
                             "external_id": "bk_1"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://komga",
                               access_token="T"),
            http_stream_factory=stream_factory,
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)
        # 500 isn't in the passthrough list → wrapped as 502
        assert resp.status_code == 502
    _run(go())


def test_unknown_provider_returns_400():
    """Defensive: if a source=komga file somehow arrives with provider='other'
    in source_metadata, we reject with 400 rather than serving garbage."""
    async def go():
        from augmentum.proxy.media_routes import comic_page
        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "unknown",
                             "external_id": "bk_1"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://x", access_token="T"),
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)
        assert resp.status_code == 400
    _run(go())


def test_upstream_content_type_preserved():
    """If the upstream returns image/webp, our response should too."""
    async def go():
        from augmentum.proxy.media_routes import comic_page

        def stream_factory(method, url, **kw):
            return _make_stream_ctx(200, b"WEBP-bytes", "image/webp")

        entry = _FakeEntry(
            id="fi_1", user_id="u_a", source="komga", kind="comic",
            source_metadata={"server_id": "ms_1", "provider": "komga",
                             "external_id": "bk_1"},
        )
        req = _FakeRequest(
            user_id="u_a", entry=entry,
            server=_FakeServer(base_url="http://komga", access_token="T"),
            http_stream_factory=stream_factory,
        )
        from unittest.mock import patch
        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            resp = await comic_page("fi_1", req)
        assert resp.status_code == 200
        assert resp.media_type == "image/webp"
    _run(go())
