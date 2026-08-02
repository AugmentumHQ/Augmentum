"""Contract tests for the media cover proxy route."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@dataclass
class _FakeEntry:
    id: str
    user_id: str
    source_metadata: dict = field(default_factory=dict)


class _FakeRequest:
    def __init__(self, *, user_id: str, entry: _FakeEntry | None, server):
        self.scope = {"user": MagicMock(id=user_id)}
        self.query_params = {}

        file_index = MagicMock()
        file_index.get = AsyncMock(return_value=entry)

        store = MagicMock()
        store.get = AsyncMock(return_value=server)

        self.app = MagicMock()
        self.app.state.file_index = file_index
        self.app.state.state_manager = MagicMock()
        self.app.state.state_manager.backend = MagicMock()
        self.app.state.http_client = MagicMock()
        self._store = store


class _FakeServer:
    def __init__(self, *, provider: str, base_url: str, access_token: str = ""):
        self.provider = provider
        self.base_url = base_url
        self.access_token = access_token


def test_audiobookshelf_cover_uses_tokenized_url_not_basic_auth():
    async def go():
        from augmentum.proxy.media_routes import media_cover

        captured = {}

        async def _fake_proxy_cover(http_client, url: str, auth_header: str = ""):
            captured["url"] = url
            captured["auth_header"] = auth_header
            if False:
                yield b""

        entry = _FakeEntry(
            id="fi_abs_1",
            user_id="u_abs",
            source_metadata={
                "server_id": "ms_abs",
                "external_id": "li_1",
                "has_cover": True,
                "cover_url": "/api/items/li_1/cover",
            },
        )
        req = _FakeRequest(
            user_id="u_abs",
            entry=entry,
            server=_FakeServer(
                provider="audiobookshelf",
                base_url="http://abs:13378",
                access_token="t_xyz",
            ),
        )

        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            with patch("augmentum.proxy.media_routes._proxy_cover", side_effect=_fake_proxy_cover):
                resp = await media_cover("fi_abs_1", req)

        assert resp.status_code == 200
        assert captured["url"] == "http://abs:13378/api/items/li_1/cover?token=t_xyz"
        assert captured["auth_header"] == ""

    _run(go())


def test_komga_cover_hint_keeps_basic_auth():
    async def go():
        from augmentum.proxy.media_routes import media_cover

        captured = {}

        async def _fake_proxy_cover(http_client, url: str, auth_header: str = ""):
            captured["url"] = url
            captured["auth_header"] = auth_header
            if False:
                yield b""

        entry = _FakeEntry(
            id="fi_komga_1",
            user_id="u_kg",
            source_metadata={
                "server_id": "ms_komga",
                "external_id": "bk_1",
                "has_cover": True,
                "cover_url": "/api/v1/series/se_1/thumbnail",
            },
        )
        req = _FakeRequest(
            user_id="u_kg",
            entry=entry,
            server=_FakeServer(
                provider="komga",
                base_url="http://komga:25600",
                access_token="YWxpY2U6czNjcmV0",
            ),
        )

        with patch("augmentum.proxy.media_routes._get_store", return_value=req._store):
            with patch("augmentum.proxy.media_routes._proxy_cover", side_effect=_fake_proxy_cover):
                resp = await media_cover("fi_komga_1", req)

        assert resp.status_code == 200
        assert captured["url"] == "http://komga:25600/api/v1/series/se_1/thumbnail"
        assert captured["auth_header"] == "Basic YWxpY2U6czNjcmV0"

    _run(go())
