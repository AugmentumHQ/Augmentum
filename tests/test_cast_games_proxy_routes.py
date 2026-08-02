"""Route-level tests for the origin proxy.

Pins:
  - POST /api/cast/games/{id}/proxy/start: mint token + 400 on unsafe URL
  - GET /api/cast/game-proxy/{token}/{path}: 404 on unknown token
  - GET /api/cast/game-proxy/{token}/{path}: 200 with rewritten HTML
  - GET /api/cast/game-proxy/{token}/{path}: adapter loader injected
  - GET /api/cast/game-proxy/{token}/{path}: blocked headers stripped
  - GET works without auth header (the token IS the credential)
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from augmentum.cast.games.proxy.fetcher import ProxyFetcher
from augmentum.cast.games.proxy.session_store import ProxySessionStore


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _StubHTTPClient:
    """Minimal httpx-compatible async client returning canned responses
    keyed by URL prefix."""

    def __init__(self, responses: dict[str, httpx.Response]) -> None:
        self._responses = responses

    async def get(self, url: str, headers: dict[str, Any] | None = None) -> httpx.Response:
        for prefix, resp in self._responses.items():
            if url.startswith(prefix):
                return resp
        return httpx.Response(404, request=httpx.Request("GET", url))

    async def aclose(self) -> None:
        pass


@pytest.fixture
def proxy_app(app):
    """Wires the proxy substrate onto the test app with a stubbed
    HTTP client so we don't hit the real network."""
    store = ProxySessionStore()
    stub_client = _StubHTTPClient({
        "https://example.com/game/index.html": httpx.Response(
            200,
            content=(
                b'<html><head><title>Game</title>'
                b'<script src="/game.js"></script></head>'
                b'<body><img src="/sprites/foo.png"></body></html>'
            ),
            headers={
                "Content-Type": "text/html",
                "Set-Cookie": "evil=1",
                "X-Frame-Options": "DENY",
                "Content-Security-Policy": "default-src 'self'",
            },
            request=httpx.Request("GET", "https://example.com/game/index.html"),
        ),
        "https://example.com/game.js": httpx.Response(
            200,
            content=b'console.log("hi");',
            headers={"Content-Type": "application/javascript"},
            request=httpx.Request("GET", "https://example.com/game.js"),
        ),
    })
    fetcher = ProxyFetcher(client=stub_client, allow_private=True)
    app.state.cast_proxy_session_store = store
    app.state.cast_proxy_fetcher = fetcher
    yield app


@pytest.fixture
def proxy_client(proxy_app):
    tc = TestClient(proxy_app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    return tc


# ── /proxy/start ─────────────────────────────────────────────────


def test_start_mints_token_and_returns_surface_url(proxy_client):
    r = proxy_client.post(
        "/api/cast/games/t1/proxy/start",
        json={
            "receiver_id": "recv-1",
            "source_url": "https://example.com/game/index.html",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token"].startswith("cgp_")
    assert body["source_origin"] == "https://example.com"
    assert body["surface_url"].startswith("/api/cast/game-proxy/")
    assert body["surface_url"].endswith("/game/index.html")


def test_start_rejects_unsafe_source_url(proxy_client):
    r = proxy_client.post(
        "/api/cast/games/t1/proxy/start",
        json={"source_url": "http://127.0.0.1/"},
    )
    assert r.status_code == 400


def test_start_requires_auth(proxy_app):
    tc = TestClient(proxy_app)
    r = tc.post(
        "/api/cast/games/t1/proxy/start",
        json={"source_url": "https://example.com/"},
    )
    assert r.status_code in (401, 403)


# ── /game-proxy/{token}/{path} ───────────────────────────────────


def test_proxy_serves_rewritten_html(proxy_client, proxy_app):
    # Mint a session first
    store = proxy_app.state.cast_proxy_session_store
    sess = store.mint(
        user_id="usr_test",  # matches conftest's test_user
        receiver_id="recv-1",
        title_id="t1",
        source_url="https://example.com/game/index.html",
    )
    r = proxy_client.get(f"/api/cast/game-proxy/{sess.token}/game/index.html")
    assert r.status_code == 200
    body = r.text
    # Same-origin URLs rewritten to ride the proxy
    assert f"/api/cast/game-proxy/{sess.token}/game.js" in body
    assert f"/api/cast/game-proxy/{sess.token}/sprites/foo.png" in body
    # Adapter loader injected
    assert "__augmentum_cast_loader_marker__" in body
    assert "universal-input-adapter.js" in body


def test_proxy_strips_blocked_response_headers(proxy_client, proxy_app):
    store = proxy_app.state.cast_proxy_session_store
    sess = store.mint(
        user_id="usr_test", receiver_id="", title_id="t1",
        source_url="https://example.com/game/index.html",
    )
    r = proxy_client.get(f"/api/cast/game-proxy/{sess.token}/game/index.html")
    # The source's hostile Set-Cookie / X-Frame-Options DENY must NOT
    # leak through. The global SecurityHeadersMiddleware re-stamps
    # X-Frame-Options to SAMEORIGIN (the FRAMEABLE_PREFIXES list
    # includes our path) so the receiver can iframe us, but the
    # source's DENY is gone.
    assert "set-cookie" not in {k.lower() for k in r.headers.keys()}
    xfo = r.headers.get("x-frame-options", "")
    assert xfo.upper() != "DENY", (
        "source's X-Frame-Options: DENY leaked through the proxy"
    )


def test_proxy_rewrites_csp_to_allow_frame_ancestors(proxy_client, proxy_app):
    store = proxy_app.state.cast_proxy_session_store
    sess = store.mint(
        user_id="usr_test", receiver_id="", title_id="t1",
        source_url="https://example.com/game/index.html",
    )
    r = proxy_client.get(f"/api/cast/game-proxy/{sess.token}/game/index.html")
    csp = r.headers.get("content-security-policy", "")
    assert "frame-ancestors 'self'" in csp


def test_proxy_unknown_token_404(proxy_client):
    r = proxy_client.get("/api/cast/game-proxy/cgp_doesnotexist/index.html")
    assert r.status_code == 404


def test_proxy_works_without_auth_header(proxy_app):
    """The token is the credential. Cross-origin iframes can't carry
    our session cookie, so the proxy endpoint must be reachable from
    a bare WebView."""
    store = proxy_app.state.cast_proxy_session_store
    sess = store.mint(
        user_id="usr_test", receiver_id="", title_id="t1",
        source_url="https://example.com/game/index.html",
    )
    tc = TestClient(proxy_app)  # no Authorization header
    r = tc.get(f"/api/cast/game-proxy/{sess.token}/game/index.html")
    assert r.status_code == 200


def test_proxy_503_without_store(app):
    app.state.cast_proxy_session_store = None
    app.state.cast_proxy_fetcher = None
    tc = TestClient(app)
    r = tc.get("/api/cast/game-proxy/cgp_x/index.html")
    assert r.status_code == 503
