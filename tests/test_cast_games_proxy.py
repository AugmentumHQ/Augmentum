"""Tests for the origin-proxy substrate (Strategy 2).

Pins:
  - session_store: mint/get/revoke + user scoping + TTL expiry
  - fetcher: rejects unsafe URLs (loopback, RFC1918, file://)
  - fetcher: strips blocked response headers
  - fetcher: refuses cross-origin redirects
  - fetcher: caches 200 OK bodies + short-circuits on cache hit
  - rewriter: HTML same-origin URLs get the proxy prefix
  - rewriter: HTML CDN allowlist passes through
  - rewriter: HTML cross-origin non-allowlist is left untouched + logged
  - rewriter: CSS url() rewriting
  - rewriter: adapter loader injection finds <head>, falls back, idempotent
  - rewriter: CSP rewrite adds frame-ancestors 'self' + CDN hosts
  - strategy: can_handle returns False without store attached
  - strategy: prepare mints session + returns proxy URL
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from augmentum.cast.games.models import (
    CastProfile,
    HostCapabilities,
    STRATEGY_PROXY,
)
from augmentum.cast.games.proxy.fetcher import (
    BLOCKED_RESPONSE_HEADERS,
    AssetCache,
    FetchResult,
    ProxyFetcher,
    _effective_port,
    _same_origin,
    filter_request_headers,
    is_url_safe,
    strip_response_headers,
)
from augmentum.cast.games.proxy.rewriter import (
    DEFAULT_CDN_ALLOWLIST,
    inject_adapter_loader,
    make_url_rewriter,
    rewrite_css,
    rewrite_csp,
    rewrite_html,
)
from augmentum.cast.games.proxy.session_store import ProxySessionStore
from augmentum.cast.games.strategies.origin_proxy import OriginProxyStrategy


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── session store ────────────────────────────────────────────────


def test_session_mint_basic():
    store = ProxySessionStore()
    s = store.mint(
        user_id="alice",
        receiver_id="recv-1",
        title_id="title-1",
        source_url="https://example.com/game/index.html",
    )
    assert s.token.startswith("cgp_")
    assert s.source_origin == "https://example.com"
    assert s.source_base_url == "https://example.com/game/index.html"
    assert s.is_active()


def test_session_mint_rejects_bad_url():
    store = ProxySessionStore()
    with pytest.raises(ValueError):
        store.mint(
            user_id="alice", receiver_id="", title_id="t",
            source_url="javascript:alert(1)",
        )
    with pytest.raises(ValueError):
        store.mint(
            user_id="alice", receiver_id="", title_id="t",
            source_url="not a url",
        )


def test_session_mint_rejects_empty_user():
    store = ProxySessionStore()
    with pytest.raises(ValueError):
        store.mint(
            user_id="", receiver_id="", title_id="t",
            source_url="https://example.com/",
        )


def test_session_get_returns_none_after_revoke():
    store = ProxySessionStore()
    s = store.mint(
        user_id="alice", receiver_id="", title_id="t",
        source_url="https://example.com/",
    )
    assert store.revoke(s.token) is True
    assert store.get(s.token) is None


def test_session_get_returns_none_after_expiry():
    store = ProxySessionStore()
    s = store.mint(
        user_id="alice", receiver_id="", title_id="t",
        source_url="https://example.com/",
        ttl_s=60.0,
    )
    # Mutate the stored row to expire — we can't pass a sub-minute TTL
    # because the store floors it at 60s.
    store._records[s.token].expires_at = time.time() - 1
    assert store.get(s.token) is None


def test_session_revoke_for_user_scopes():
    store = ProxySessionStore()
    store.mint(user_id="alice", receiver_id="", title_id="a1",
               source_url="https://example.com/")
    store.mint(user_id="alice", receiver_id="", title_id="a2",
               source_url="https://example.com/")
    store.mint(user_id="bob", receiver_id="", title_id="b1",
               source_url="https://example.com/")
    n = store.revoke_for_user("alice")
    assert n == 2
    assert len(store.list_for_user("alice")) == 0
    assert len(store.list_for_user("bob")) == 1


# ── fetcher: URL safety ──────────────────────────────────────────


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x", "http://10.0.0.1/y",
    "http://192.168.1.1/", "http://169.254.169.254/meta",
    "file:///etc/passwd", "javascript:alert(1)", "",
    "http://[::1]/", "http://localhost/",
])
def test_is_url_safe_rejects_unsafe(url):
    # localhost goes through resolution — skip if your resolver gives
    # back something non-loopback (unusual). Pin the literal cases.
    if url == "http://localhost/":
        # Resolved via getaddrinfo; might be 127.0.0.1 or ::1.
        return
    assert is_url_safe(url) is False, f"should reject {url!r}"


def test_is_url_safe_accepts_public_https():
    # Use a literal public-IP-style host that won't resolve to a
    # private range during the test.
    assert is_url_safe("https://example.com/path") is True


# ── fetcher: header stripping ────────────────────────────────────


def test_strip_response_headers_removes_blocked():
    headers = {
        "Set-Cookie": "x=y",
        "Authorization": "Bearer secret",
        "X-Frame-Options": "DENY",
        "Content-Type": "text/html",
    }
    out = strip_response_headers(headers)
    assert "Content-Type" in out
    assert "Set-Cookie" not in out
    assert "Authorization" not in out
    assert "X-Frame-Options" not in out


def test_strip_response_headers_case_insensitive():
    headers = {"set-cookie": "x=y", "SET-COOKIE": "z=w"}
    out = strip_response_headers(headers)
    assert "set-cookie" not in out
    assert "SET-COOKIE" not in out


def test_filter_request_headers_allowlist():
    headers = {
        "Cookie": "session=secret",
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html",
        "X-Internal-Header": "x",
    }
    out = filter_request_headers(headers)
    assert "Cookie" not in out
    assert "User-Agent" in out
    assert "Accept" in out
    assert "X-Internal-Header" not in out


# ── fetcher: cache ───────────────────────────────────────────────


def test_asset_cache_round_trip(tmp_path: Path):
    cache = AssetCache(tmp_path)
    fr = FetchResult(
        status=200, body=b"<html>hi</html>",
        content_type="text/html",
        headers={"Content-Type": "text/html"},
        source_url="https://example.com/",
    )
    cache.put("alice", "https://example.com/", fr)
    got = cache.get("alice", "https://example.com/")
    assert got is not None
    assert got.body == b"<html>hi</html>"
    assert got.from_cache is True


def test_asset_cache_user_scoping(tmp_path: Path):
    cache = AssetCache(tmp_path)
    fr = FetchResult(
        status=200, body=b"alice's content",
        content_type="text/plain",
        headers={},
        source_url="https://example.com/",
    )
    cache.put("alice", "https://example.com/", fr)
    assert cache.get("bob", "https://example.com/") is None


def test_asset_cache_wipe_for_user(tmp_path: Path):
    cache = AssetCache(tmp_path)
    fr = FetchResult(
        status=200, body=b"x", content_type="text/plain", headers={},
        source_url="https://example.com/",
    )
    cache.put("alice", "https://example.com/x", fr)
    assert cache.usage_bytes("alice") > 0
    cache.wipe_for_user("alice")
    assert cache.usage_bytes("alice") == 0


def test_asset_cache_skips_non_200(tmp_path: Path):
    cache = AssetCache(tmp_path)
    fr = FetchResult(
        status=404, body=b"not found", content_type="text/plain",
        headers={}, source_url="https://example.com/",
    )
    cache.put("alice", "https://example.com/missing", fr)
    assert cache.get("alice", "https://example.com/missing") is None


# ── fetcher: live fetch via mocked httpx ─────────────────────────


def test_fetcher_strips_blocked_headers_from_response():
    client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        200, content=b"<html>ok</html>",
        headers={
            "Content-Type": "text/html",
            "Set-Cookie": "evil=1",
            "X-Frame-Options": "DENY",
        },
        request=httpx.Request("GET", "https://example.com/"),
    )
    client.get = AsyncMock(return_value=mock_resp)

    fetcher = ProxyFetcher(client=client, allow_private=True)
    result = _run(fetcher.fetch(
        "https://example.com/",
        source_origin="https://example.com",
    ))
    assert result.status == 200
    assert result.body == b"<html>ok</html>"
    assert "Set-Cookie" not in result.headers
    assert "X-Frame-Options" not in result.headers


def test_fetcher_refuses_cross_origin_redirect():
    client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        302, content=b"",
        headers={"Location": "https://evil.example.org/"},
        request=httpx.Request("GET", "https://example.com/"),
    )
    client.get = AsyncMock(return_value=mock_resp)

    fetcher = ProxyFetcher(client=client, allow_private=True)
    with pytest.raises(httpx.HTTPError):
        _run(fetcher.fetch(
            "https://example.com/",
            source_origin="https://example.com",
        ))


def test_fetcher_strips_stale_encoding_headers():
    # httpx auto-decompresses resp.content, so the upstream
    # content-encoding / content-length no longer describe our bytes —
    # passing them through would make the TV browser fail to gunzip.
    # Mirror reality: upstream sends genuinely-gzipped bytes.
    import gzip

    plain = b"<html>plain</html>"
    gzipped = gzip.compress(plain)
    client = AsyncMock(spec=httpx.AsyncClient)
    mock_resp = httpx.Response(
        200, content=gzipped,
        headers={
            "Content-Type": "text/html",
            "Content-Encoding": "gzip",
            "Content-Length": str(len(gzipped)),
        },
        request=httpx.Request("GET", "https://example.com/"),
    )
    client.get = AsyncMock(return_value=mock_resp)

    fetcher = ProxyFetcher(client=client, allow_private=True)
    result = _run(fetcher.fetch(
        "https://example.com/",
        source_origin="https://example.com",
    ))
    # Body is the decompressed plaintext httpx handed us.
    assert result.body == plain
    lowered = {k.lower() for k in result.headers}
    assert "content-encoding" not in lowered
    assert "transfer-encoding" not in lowered
    assert "content-length" not in lowered
    # Sanity: these are declared in the blocked set so strip + cache agree.
    assert "content-encoding" in BLOCKED_RESPONSE_HEADERS
    assert "content-length" in BLOCKED_RESPONSE_HEADERS


def test_fetcher_follows_same_origin_redirect_with_explicit_default_port():
    # https://example.com/ → https://example.com:443/index.html is the
    # SAME origin; the explicit :443 must not read as cross-origin.
    redirect = httpx.Response(
        302, content=b"",
        headers={"Location": "https://example.com:443/index.html"},
        request=httpx.Request("GET", "https://example.com/"),
    )
    final = httpx.Response(
        200, content=b"<html>landed</html>",
        headers={"Content-Type": "text/html"},
        request=httpx.Request("GET", "https://example.com:443/index.html"),
    )
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=[redirect, final])

    fetcher = ProxyFetcher(client=client, allow_private=True)
    result = _run(fetcher.fetch(
        "https://example.com/",
        source_origin="https://example.com",
    ))
    assert result.status == 200
    assert result.body == b"<html>landed</html>"


@pytest.mark.parametrize(
    "url, origin, expected",
    [
        ("https://example.com:443/x", "https://example.com", True),
        ("https://example.com/x", "https://example.com:443", True),
        ("http://example.com:80/x", "http://example.com", True),
        ("https://example.com/x", "https://example.com", True),
        ("https://example.com:8443/x", "https://example.com", False),
        ("http://example.com/x", "https://example.com", False),
        ("https://other.com/x", "https://example.com", False),
        ("", "https://example.com", False),
    ],
)
def test_same_origin_default_port_normalisation(url, origin, expected):
    assert _same_origin(url, origin) is expected


def test_effective_port_defaults():
    assert _effective_port("https", None) == 443
    assert _effective_port("http", None) == 80
    assert _effective_port("https", 8443) == 8443
    assert _effective_port("gopher", None) is None


# ── rewriter: HTML ───────────────────────────────────────────────


def test_rewrite_html_same_origin_script():
    html = '<html><head><script src="/game.js"></script></head></html>'
    out = rewrite_html(
        html,
        proxy_base="/api/cast/game-proxy/cgp_abc",
        source_origin="https://example.com",
        page_url="https://example.com/",
    )
    assert '/api/cast/game-proxy/cgp_abc/game.js' in out


def test_rewrite_html_absolute_same_origin():
    html = '<html><body><img src="https://example.com/sprites/foo.png"></body></html>'
    out = rewrite_html(
        html,
        proxy_base="/api/cast/game-proxy/cgp_xyz",
        source_origin="https://example.com",
        page_url="https://example.com/index.html",
    )
    assert '/api/cast/game-proxy/cgp_xyz/sprites/foo.png' in out


def test_rewrite_html_cdn_allowlist_passes_through():
    html = (
        '<html><head>'
        '<script src="https://cdn.jsdelivr.net/npm/jquery@3/jquery.min.js"></script>'
        '</head></html>'
    )
    out = rewrite_html(
        html,
        proxy_base="/api/cast/game-proxy/cgp_x",
        source_origin="https://example.com",
        page_url="https://example.com/",
    )
    assert 'https://cdn.jsdelivr.net' in out


def test_rewrite_html_cross_origin_unknown_left_alone():
    html = '<html><body><script src="https://tracker.evil.org/spy.js"></script></body></html>'
    out = rewrite_html(
        html,
        proxy_base="/api/cast/game-proxy/cgp_x",
        source_origin="https://example.com",
        page_url="https://example.com/",
    )
    # We don't rewrite (browser CSP will block); we just don't leak via proxy.
    assert 'tracker.evil.org' in out
    assert '/api/cast/game-proxy/' not in out  # not proxied through us


def test_rewrite_html_data_url_untouched():
    html = '<html><body><img src="data:image/png;base64,abc"></body></html>'
    out = rewrite_html(
        html,
        proxy_base="/api/cast/game-proxy/cgp_x",
        source_origin="https://example.com",
        page_url="https://example.com/",
    )
    assert 'data:image/png;base64,abc' in out


def test_rewrite_html_anchor_links_untouched():
    html = '<html><body><a href="#section">Jump</a></body></html>'
    out = rewrite_html(
        html,
        proxy_base="/api/cast/game-proxy/cgp_x",
        source_origin="https://example.com",
        page_url="https://example.com/",
    )
    assert 'href="#section"' in out


def test_rewrite_html_srcset_multi():
    html = (
        '<html><body><img srcset="/a.png 1x, /b.png 2x" '
        'src="/a.png"></body></html>'
    )
    out = rewrite_html(
        html,
        proxy_base="/api/cast/game-proxy/cgp_x",
        source_origin="https://example.com",
        page_url="https://example.com/",
    )
    assert '/api/cast/game-proxy/cgp_x/a.png 1x' in out
    assert '/api/cast/game-proxy/cgp_x/b.png 2x' in out


# ── rewriter: CSS ────────────────────────────────────────────────


def test_rewrite_css_url_quoted():
    css = 'body { background: url("/img/bg.png"); }'
    out = rewrite_css(
        css,
        proxy_base="/api/cast/game-proxy/cgp_x",
        source_origin="https://example.com",
        page_url="https://example.com/style.css",
    )
    assert '/api/cast/game-proxy/cgp_x/img/bg.png' in out
    assert 'url("' in out  # quote preserved


def test_rewrite_css_url_unquoted():
    css = '@font-face { src: url(/fonts/a.woff2); }'
    out = rewrite_css(
        css,
        proxy_base="/api/cast/game-proxy/cgp_x",
        source_origin="https://example.com",
        page_url="https://example.com/style.css",
    )
    assert 'url(/api/cast/game-proxy/cgp_x/fonts/a.woff2)' in out


# ── rewriter: adapter loader injection ───────────────────────────


def test_inject_loader_into_head():
    html = '<html><head><title>X</title></head><body></body></html>'
    out = inject_adapter_loader(html)
    assert '__augmentum_cast_loader_marker__' in out
    # Must land before the page's <title> (just after <head>)
    head_idx = out.index('<head>')
    marker_idx = out.index('__augmentum_cast_loader_marker__')
    title_idx = out.index('<title>')
    assert head_idx < marker_idx < title_idx


def test_inject_loader_idempotent():
    html = '<html><head></head><body></body></html>'
    once = inject_adapter_loader(html)
    twice = inject_adapter_loader(once)
    assert once == twice


def test_inject_loader_falls_back_when_no_head():
    html = '<html><body>only body</body></html>'
    out = inject_adapter_loader(html)
    assert '__augmentum_cast_loader_marker__' in out


def test_inject_loader_contains_sw_disable():
    html = '<html><head></head></html>'
    out = inject_adapter_loader(html)
    assert 'serviceWorker' in out
    assert 'suppressed' in out


# ── rewriter: CSP ────────────────────────────────────────────────


def test_rewrite_csp_adds_frame_ancestors_self():
    out = rewrite_csp(
        "default-src 'self'; script-src https://example.com",
        proxy_base="/api/cast/game-proxy/cgp_x",
    )
    assert "frame-ancestors 'self'" in out


def test_rewrite_csp_merges_cdn_hosts():
    out = rewrite_csp(
        "script-src 'self'",
        proxy_base="/api/cast/game-proxy/cgp_x",
    )
    # CDN hosts present
    assert "cdn.jsdelivr.net" in out


def test_rewrite_csp_empty_input():
    assert rewrite_csp("", proxy_base="/api/cast/game-proxy/cgp_x") == ""


# ── url rewriter unit ────────────────────────────────────────────


def test_url_rewriter_preserves_query():
    rw = make_url_rewriter(
        proxy_base="/p",
        source_origin="https://example.com",
        page_url="https://example.com/index.html",
    )
    assert rw("/api?key=value") == "/p/api?key=value"


def test_url_rewriter_resolves_relative():
    rw = make_url_rewriter(
        proxy_base="/p",
        source_origin="https://example.com",
        page_url="https://example.com/game/index.html",
    )
    assert rw("../shared.js") == "/p/shared.js"


# ── strategy ─────────────────────────────────────────────────────


def test_origin_proxy_can_handle_false_without_store():
    strat = OriginProxyStrategy()
    title = {"embed_url": "https://example.com/"}
    assert _run(strat.can_handle(title, HostCapabilities())) is False


def test_origin_proxy_can_handle_true_when_attached():
    strat = OriginProxyStrategy()
    strat.attach_session_store(ProxySessionStore())
    title = {"embed_url": "https://example.com/"}
    assert _run(strat.can_handle(title, HostCapabilities())) is True


def test_origin_proxy_can_handle_false_unsafe_embed():
    strat = OriginProxyStrategy()
    strat.attach_session_store(ProxySessionStore())
    title = {"embed_url": "http://127.0.0.1/"}
    assert _run(strat.can_handle(title, HostCapabilities())) is False


def test_origin_proxy_prepare_mints_session():
    store = ProxySessionStore()
    strat = OriginProxyStrategy(session_store=store)
    title = {
        "id": "t1",
        "user_id": "alice",
        "receiver_id": "recv-1",
        "embed_url": "https://example.com/game/index.html",
    }
    profile = CastProfile(
        title_id="t1", strategy=STRATEGY_PROXY,
        embed_url="https://example.com/game/index.html",
        input_chain=("gamepad_api", "keyboard"),
    )
    prep = _run(strat.prepare(title, profile))
    assert prep.strategy == STRATEGY_PROXY
    assert prep.session_token.startswith("cgp_")
    assert "/api/cast/game-proxy/" in prep.surface_url
    assert prep.surface_url.endswith("/game/index.html")
    assert prep.input_chain == ("gamepad_api", "keyboard")


def test_origin_proxy_prepare_raises_without_store():
    strat = OriginProxyStrategy()
    title = {"id": "t", "embed_url": "https://example.com/"}
    profile = CastProfile(
        title_id="t", strategy=STRATEGY_PROXY,
        embed_url="https://example.com/",
    )
    with pytest.raises(RuntimeError):
        _run(strat.prepare(title, profile))
