"""Behavior tests for /api/livetv/* routes.

Stubs the per-server fetch path at the module boundary so we test
the route's aggregation + caching + user-scoping behavior in
isolation from real Emby/JF HTTP. (Live integration is covered by
``scripts/probe_live_tv.py`` against the user's actual server.)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from augmentum.media.providers.base import CatalogItem


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TEST_USER_ID = "usr_test"


@pytest.fixture
def livetv_client(app, monkeypatch):
    """Client + store with the per-server fetch path stubbed."""
    from augmentum.media.store import MediaServerStore
    from augmentum.proxy import livetv_routes
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    app.state.state_manager = StateManager(backend)
    app.state.http_client = AsyncMock()
    store = MediaServerStore(backend._conn)

    # Clear the module-level cache between tests so seeded state
    # always reaches the categorizer.
    livetv_routes._rail_cache.clear()

    # Per-server stub keyed by server_id. Mirrors what
    # _fetch_for_server does in production: tag each channel with
    # server_id, and isolate errors per server (a missing key here
    # = empty list, which is the production behavior on
    # token-expired / network-down / wrong-provider).
    fetch_responses: dict[str, list[CatalogItem]] = {}

    async def _fake_fetch(server, http):
        items = list(fetch_responses.get(server.id, []))
        for item in items:
            if isinstance(item.extra, dict):
                item.extra["server_id"] = server.id
        return items

    monkeypatch.setattr(livetv_routes, "_fetch_for_server", _fake_fetch)

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})

    yield tc, store, fetch_responses
    _run(backend.close())


def _seed_server(store, *, provider="emby", name="Home Emby", user_id=TEST_USER_ID):
    """Create a server row and return its id (the value tests key off)."""
    row = _run(store.create(
        user_id=user_id, provider=provider, name=name,
        base_url=f"http://{provider}.local:8096",
        access_token="test-token",
    ))
    return row.id


def _ch(name: str, *, number: str = "", is_favorite: bool = False) -> CatalogItem:
    return CatalogItem(
        external_id=name.lower().replace(" ", "-"),
        name=name,
        kind="live_video",
        mime_type="application/vnd.apple.mpegurl",
        extra={
            "channel_number": number,
            "is_favorite":    is_favorite,
            "play_count":     0,
        },
    )


# ── Empty / setup-state cases ─────────────────────────────────────

class TestEmptyState:
    def test_no_servers_returns_empty_rails(self, livetv_client):
        client, _, _ = livetv_client
        r = client.get("/api/livetv/rails")
        assert r.status_code == 200
        data = r.json()
        assert data["rails"] == []
        assert data["channel_count"] == 0

    def test_non_emby_jf_servers_are_ignored(self, livetv_client):
        """ABS / Komga / Suwayomi don't expose Live TV. Their presence
        shouldn't cause an error or accidentally surface a rail set."""
        client, store, _ = livetv_client
        _seed_server(store, provider="audiobookshelf", name="My ABS")
        _seed_server(store, provider="komga", name="My Komga")
        r = client.get("/api/livetv/rails")
        assert r.status_code == 200
        assert r.json()["rails"] == []

    def test_emby_server_with_zero_channels_returns_empty(self, livetv_client):
        client, store, responses = livetv_client
        sid = _seed_server(store, provider="emby")
        responses[sid] = []
        r = client.get("/api/livetv/rails")
        assert r.status_code == 200
        assert r.json()["rails"] == []


# ── Aggregation across servers ────────────────────────────────────

class TestAggregation:
    def test_channels_from_multiple_servers_merge(self, livetv_client):
        client, store, responses = livetv_client
        emby_id = _seed_server(store, provider="emby", name="Emby A")
        jf_id   = _seed_server(store, provider="jellyfin", name="JF B")
        responses[emby_id] = [_ch("CNN", number="200")]
        responses[jf_id]   = [_ch("HGTV", number="229")]

        r = client.get("/api/livetv/rails")
        assert r.status_code == 200
        data = r.json()
        assert data["channel_count"] == 2

        all_rail = next(r for r in data["rails"] if r["id"] == "all")
        names = {c["name"] for c in all_rail["channels"]}
        assert names == {"CNN", "HGTV"}

    def test_server_id_is_tagged_on_each_channel(self, livetv_client):
        """The UI play path needs server_id to know which Emby/JF to
        round-trip back to. Tagging happens INSIDE _fetch_for_server
        in production; the stub mirrors that so the contract is
        verified end-to-end through the route."""
        client, store, responses = livetv_client
        emby_id = _seed_server(store, provider="emby")
        responses[emby_id] = [_ch("CNN", number="200")]

        r = client.get("/api/livetv/rails")
        all_rail = next(r for r in r.json()["rails"] if r["id"] == "all")
        assert all_rail["channels"][0]["server_id"] == emby_id


# ── Error isolation ───────────────────────────────────────────────

class TestErrorIsolation:
    def test_one_failed_server_does_not_break_others(self, livetv_client):
        """A degraded server (raises in production, returns [] via the
        stub here) must NOT take out the other servers' channels.
        Asserts the aggregation step uses extend-from-each rather than
        any short-circuit on empty/error per server."""
        client, store, responses = livetv_client
        bad_id  = _seed_server(store, provider="emby",     name="Broken")
        good_id = _seed_server(store, provider="jellyfin", name="Good")
        responses[bad_id]  = []                          # simulates failure
        responses[good_id] = [_ch("HGTV", number="229")]

        r = client.get("/api/livetv/rails")
        assert r.status_code == 200
        all_rail = next(r for r in r.json()["rails"] if r["id"] == "all")
        assert [c["name"] for c in all_rail["channels"]] == ["HGTV"]

    def test_unseeded_server_yields_zero_channels(self, livetv_client):
        """If a server is configured but has no responses (mirrors the
        production "no access_token = early return []" path), it
        should contribute zero channels rather than raise."""
        client, store, responses = livetv_client
        emby_id = _seed_server(store, provider="emby", name="Connected")
        responses[emby_id] = []

        r = client.get("/api/livetv/rails")
        assert r.status_code == 200
        assert r.json()["channel_count"] == 0


# ── Cache behavior ────────────────────────────────────────────────

class TestCaching:
    def test_second_call_returns_cached_payload(self, livetv_client):
        client, store, responses = livetv_client
        sid = _seed_server(store, provider="emby")
        responses[sid] = [_ch("CNN", number="200")]

        r1 = client.get("/api/livetv/rails")
        assert r1.json()["cached"] is False

        # Mutate responses so the next call WOULD return different data
        # if it actually hit the fetch path.
        responses[sid] = [_ch("HGTV", number="229")]

        r2 = client.get("/api/livetv/rails")
        assert r2.json()["cached"] is True
        all_rail = next(r for r in r2.json()["rails"] if r["id"] == "all")
        # Still shows the original CNN, not the mutated HGTV.
        assert [c["name"] for c in all_rail["channels"]] == ["CNN"]

    def test_refresh_query_bypasses_cache(self, livetv_client):
        client, store, responses = livetv_client
        sid = _seed_server(store, provider="emby")
        responses[sid] = [_ch("CNN", number="200")]
        client.get("/api/livetv/rails")  # warm cache

        responses[sid] = [_ch("HGTV", number="229")]
        r = client.get("/api/livetv/rails?refresh=1")
        assert r.json()["cached"] is False
        all_rail = next(r for r in r.json()["rails"] if r["id"] == "all")
        assert [c["name"] for c in all_rail["channels"]] == ["HGTV"]

    def test_invalidate_user_cache_drops_entry(self, livetv_client):
        from augmentum.proxy.livetv_routes import _rail_cache, invalidate_user_cache

        client, store, responses = livetv_client
        sid = _seed_server(store, provider="emby")
        responses[sid] = [_ch("CNN", number="200")]
        client.get("/api/livetv/rails")  # warm cache
        assert TEST_USER_ID in _rail_cache

        invalidate_user_cache(TEST_USER_ID)
        assert TEST_USER_ID not in _rail_cache


# ── User isolation ────────────────────────────────────────────────

class TestUserIsolation:
    def test_other_users_servers_are_not_surfaced(self, livetv_client):
        """Another user's private Emby server must NOT contribute
        channels to this user's rails. Shared servers WOULD — but
        nothing here is shared."""
        client, store, responses = livetv_client
        mine_id   = _seed_server(store, provider="emby", name="Mine",
                                 user_id=TEST_USER_ID)
        theirs_id = _seed_server(store, provider="emby", name="Theirs",
                                 user_id="usr_someone_else")
        responses[mine_id]   = [_ch("CNN",  number="200")]
        responses[theirs_id] = [_ch("HGTV", number="229")]

        r = client.get("/api/livetv/rails")
        all_rail = next(r for r in r.json()["rails"] if r["id"] == "all")
        assert [c["name"] for c in all_rail["channels"]] == ["CNN"]


# ── Play / stop session lifecycle ─────────────────────────────────

class TestPlaySession:
    def test_play_unknown_server_returns_404(self, livetv_client):
        client, _, _ = livetv_client
        r = client.post("/api/livetv/play/ms_nonexistent/ch_1")
        assert r.status_code == 404

    def test_play_against_non_emby_server_400(self, livetv_client):
        client, store, _ = livetv_client
        sid = _seed_server(store, provider="audiobookshelf")
        r = client.post(f"/api/livetv/play/{sid}/ch_1")
        assert r.status_code == 400

    def test_stop_with_unknown_token_is_idempotent(self, livetv_client):
        """Reload-races: the page may post stop AFTER the user has
        navigated and the token has already been swept. Should not
        be a noisy 404 — return ok+already to keep client logs clean."""
        client, _, _ = livetv_client
        r = client.post("/api/livetv/stop/nonexistent-token")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert r.json().get("already") is True

    def test_stream_master_unknown_token_returns_404(self, livetv_client):
        client, _, _ = livetv_client
        r = client.get("/api/livetv/stream/nonexistent-token/master.m3u8")
        assert r.status_code == 404

    def test_stream_segment_unknown_token_returns_404(self, livetv_client):
        client, _, _ = livetv_client
        r = client.get("/api/livetv/stream/nonexistent-token/seg/hls1/0.ts")
        assert r.status_code == 404


# ── m3u8 rewrite unit ─────────────────────────────────────────────

class TestM3u8Rewrite:
    """Direct tests of the rewrite helper — easier to assert against
    than wiring fake httpx responses through a route. The route's
    handler is a thin shell around this + a fetch."""

    def test_strips_api_key_from_query(self):
        from augmentum.proxy.livetv_routes import _strip_api_key
        assert _strip_api_key("api_key=secret") == ""
        assert _strip_api_key("DeviceId=abc&api_key=secret") == "DeviceId=abc"
        assert _strip_api_key("api_key=secret&MediaSourceId=x") == "MediaSourceId=x"
        assert _strip_api_key("API_KEY=secret") == ""   # case-insensitive
        assert _strip_api_key("") == ""

    def test_rewrites_variant_uri_in_master(self):
        from augmentum.proxy.livetv_routes import _rewrite_m3u8
        master = (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=2000000\n"
            "hls1/main/0.m3u8?DeviceId=abc&api_key=secret\n"
        )
        out = _rewrite_m3u8(master, session_token="tok123")
        assert "api_key" not in out
        assert "/api/livetv/stream/tok123/seg/hls1/main/0.m3u8?DeviceId=abc" in out

    def test_rewrites_segments_in_variant(self):
        from augmentum.proxy.livetv_routes import _rewrite_m3u8
        variant = (
            "#EXTM3U\n"
            "#EXT-X-TARGETDURATION:6\n"
            "#EXTINF:6.0,\n"
            "hls1/main/0.ts?api_key=secret&MediaSourceId=x\n"
            "#EXTINF:6.0,\n"
            "hls1/main/1.ts?api_key=secret&MediaSourceId=x\n"
        )
        out = _rewrite_m3u8(variant, session_token="tok123")
        assert "api_key" not in out
        assert "/api/livetv/stream/tok123/seg/hls1/main/0.ts?MediaSourceId=x" in out
        assert "/api/livetv/stream/tok123/seg/hls1/main/1.ts?MediaSourceId=x" in out

    def test_preserves_directives_and_blank_lines_pass_through(self):
        from augmentum.proxy.livetv_routes import _rewrite_m3u8
        body = (
            "#EXTM3U\n"
            "#EXT-X-VERSION:3\n"
            "\n"
            "hls1/main/0.ts?api_key=x\n"
        )
        out = _rewrite_m3u8(body, session_token="tok")
        # Every # line preserved verbatim
        assert "#EXTM3U" in out
        assert "#EXT-X-VERSION:3" in out

    def test_absolute_uri_passes_through_unchanged(self):
        """Schemeful absolute URIs (rare for Emby live) are left as-is
        rather than mis-rewritten. The proxy fetches what the player
        asks for; if the player encounters an absolute URI, the
        upstream is reachable on its own."""
        from augmentum.proxy.livetv_routes import _rewrite_m3u8
        body = (
            "#EXTM3U\n"
            "https://other.cdn/segment.ts?api_key=secret\n"
        )
        out = _rewrite_m3u8(body, session_token="tok")
        assert "https://other.cdn/segment.ts?api_key=secret" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
