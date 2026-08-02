"""Smoke + contract tests for the media-server feature (Phase 1: Audiobookshelf).

Covers:
  - MediaServerStore CRUD with user_id isolation
  - AudiobookshelfProvider fingerprint + login + catalog parsing (mocked HTTP)
  - MediaServerAdapter protocol shape + user scoping
  - Detector fingerprints real responses only

Live-server tests live separately under tests/live/ when we wire them.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from augmentum.media.providers.audiobookshelf import (
    AudiobookshelfProvider, _item_from_abs,
)
from augmentum.media.providers.base import CatalogItem, DEFAULT_PORTS, ProviderInfo
from augmentum.media.sync import sync_server
from augmentum.media.providers.emby import EmbyProvider
from augmentum.media.providers.jellyfin import JellyfinProvider
from augmentum.media.store import MediaServerStore, _normalize_base_url
from augmentum.vfs.adapters.media_server import MediaServerAdapter


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _setup_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    # Minimal schema: users (FK target) + user_media_servers (feature under
    # test) + file_index (required by the adapter's delete path).
    await conn.executescript("""
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE user_media_servers (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            access_token TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'untested',
            status_detail TEXT NOT NULL DEFAULT '',
            last_sync_at TEXT,
            item_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            total_seen INTEGER NOT NULL DEFAULT 0,
            skipped_count INTEGER NOT NULL DEFAULT 0,
            scope TEXT NOT NULL DEFAULT 'private',
            last_sync_skipped TEXT NOT NULL DEFAULT '[]'
        );
        CREATE UNIQUE INDEX idx_user_media_servers_unique
            ON user_media_servers(user_id, provider, base_url);
        CREATE INDEX idx_user_media_servers_scope
            ON user_media_servers(scope);
        CREATE TABLE file_index (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            source TEXT NOT NULL,
            source_id TEXT NOT NULL,
            name TEXT NOT NULL,
            source_metadata TEXT NOT NULL DEFAULT '{}'
        );
        CREATE UNIQUE INDEX idx_file_index_source_unique
            ON file_index(user_id, source, source_id);
        INSERT INTO users (id) VALUES ('u_a'), ('u_b');
    """)
    return conn


# --- store.MediaServerStore -------------------------------------------


class TestNormalize:
    def test_strips_trailing_slashes(self):
        assert _normalize_base_url("http://x:80/") == "http://x:80"
        assert _normalize_base_url("http://x:80//") == "http://x:80"

    def test_empty_stays_empty(self):
        assert _normalize_base_url("") == ""


class TestMediaServerStore:
    def test_create_and_get(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            server = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="Home", base_url="http://abs:13378/",
            )
            assert server.user_id == "u_a"
            assert server.base_url == "http://abs:13378"  # trailing slash normalized
            assert server.status == "untested"
            fetched = await store.get(server.id, user_id="u_a")
            assert fetched is not None
            assert fetched.id == server.id
        _run(go())

    def test_user_isolation(self):
        """User A's server invisible to user B — the core multi-tenant invariant."""
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            a = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            assert await store.get(a.id, user_id="u_b") is None
            b_list = await store.list_for_user(user_id="u_b")
            assert b_list == []
        _run(go())

    def test_update_merges_fields(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            s = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            updated = await store.update(
                s.id, user_id="u_a", status="ok", item_count=42,
            )
            assert updated is not None
            assert updated.status == "ok"
            assert updated.item_count == 42
            assert updated.name == "A"  # untouched
        _run(go())

    def test_get_visible_caches_and_invalidates_on_write(self):
        """get_visible is cached for read bursts (cast images) but any write
        THROUGH the store clears it, so nothing stale is ever served."""
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            s = await store.create(
                user_id="u_a", provider="jellyfin",
                name="J", base_url="http://jf:8096",
            )
            # First call populates the cache.
            v1 = await store.get_visible(s.id, user_id="u_a")
            assert v1 is not None and v1.name == "J"

            # An OUT-OF-BAND raw write (bypassing the store) is NOT reflected
            # while cached — proves the cache is genuinely serving.
            await conn.execute(
                "UPDATE user_media_servers SET name = ? WHERE id = ?",
                ("RAW", s.id),
            )
            await conn.commit()
            assert (await store.get_visible(s.id, user_id="u_a")).name == "J"

            # A write THROUGH the store invalidates → next read is fresh
            # (and now also reflects the out-of-band change, proving a clear).
            await store.update(s.id, user_id="u_a", status="ok")
            v2 = await store.get_visible(s.id, user_id="u_a")
            assert v2.status == "ok"
            assert v2.name == "RAW"

            # Delete also invalidates → None, not a stale hit.
            await store.delete(s.id, user_id="u_a")
            assert await store.get_visible(s.id, user_id="u_a") is None
        _run(go())

    def test_find_match_by_url(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            hit = await store.find_match(
                user_id="u_a", provider="audiobookshelf",
                base_url="http://abs:13378/",   # trailing slash still matches
            )
            assert hit is not None
            miss_user = await store.find_match(
                user_id="u_b", provider="audiobookshelf",
                base_url="http://abs:13378",
            )
            assert miss_user is None
        _run(go())

    def test_diagnostics_fields_persist(self):
        """total_seen / skipped_count / last_sync_skipped round-trip via
        the store, including the JSON-encoded skipped-titles list."""
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            s = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            skipped_sample = [
                {"title": "Folder Book", "author": "X", "reason": "folder_needs_detail_fetch"},
                {"title": "Meta Only",   "author": "",  "reason": "no_audio_files"},
            ]
            updated = await store.update(
                s.id, user_id="u_a",
                total_seen=453, skipped_count=141,
                last_sync_skipped=skipped_sample,
            )
            assert updated is not None
            assert updated.total_seen == 453
            assert updated.skipped_count == 141
            assert updated.last_sync_skipped == skipped_sample
            # Round-trip through get() to confirm JSON decode happens
            # on both paths (not just the fresh-write object).
            fetched = await store.get(s.id, user_id="u_a")
            assert fetched is not None
            assert fetched.skipped_count == 141
            assert fetched.last_sync_skipped[0]["reason"] == "folder_needs_detail_fetch"
        _run(go())

    def test_malformed_skipped_json_falls_back_to_empty(self):
        """Hand-edited or partially-written rows must not crash reads."""
        async def go():
            conn = await _setup_db()
            # Skip the store.create path and poke the row directly with
            # an invalid JSON payload to simulate corruption.
            await conn.execute(
                "INSERT INTO user_media_servers "
                "(id, user_id, provider, name, base_url, last_sync_skipped) "
                "VALUES ('ms_junk', 'u_a', 'audiobookshelf', 'A', "
                "'http://abs:13378', '{not json')",
            )
            await conn.commit()
            store = MediaServerStore(conn)
            got = await store.get("ms_junk", user_id="u_a")
            assert got is not None
            assert got.last_sync_skipped == []
        _run(go())

    def test_delete_scoped_to_owner(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            s = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            # user_b deletes: should report no-op
            assert (await store.delete(s.id, user_id="u_b")) is False
            # user_a deletes: succeeds
            assert (await store.delete(s.id, user_id="u_a")) is True
        _run(go())


class TestMediaServerSharing:
    """Admin-shared scope (migration 172).

    The credential row is published read-only; per-user state (file_index
    progress, library overrides) stays scoped to the caller. These tests
    cover the store-level invariants only — route-level admin gating is
    enforced separately by media_routes.py + auth.guards.require_admin.
    """

    def test_default_scope_is_private(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            s = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            assert s.scope == "private"
            # to_dict surfaces the new fields with the right defaults.
            d = s.to_dict(viewer_user_id="u_a")
            assert d["scope"] == "private"
            assert d["is_shared"] is False
            assert d["is_owned_by_viewer"] is True
        _run(go())

    def test_list_visible_returns_own_union_shared(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            admin_srv = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="Admin ABS", base_url="http://abs:13378",
            )
            own_srv = await store.create(
                user_id="u_b", provider="jellyfin",
                name="My JF", base_url="http://jf:8096",
            )
            # u_b can't see u_a's row while it's private.
            visible = await store.list_visible(user_id="u_b")
            assert [v.id for v in visible] == [own_srv.id]
            # Flip admin's row to shared.
            await store.set_scope(
                admin_srv.id, scope="shared", owner_user_id="u_a",
            )
            visible = await store.list_visible(user_id="u_b")
            ids = {v.id for v in visible}
            assert ids == {own_srv.id, admin_srv.id}
            # Order invariant: own rows surface BEFORE shared ones so the
            # UI lays them out predictably.
            assert visible[0].id == own_srv.id
            assert visible[1].id == admin_srv.id
            assert visible[1].scope == "shared"
        _run(go())

    def test_get_visible_resolves_shared_for_non_owner(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            shared = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="Shared", base_url="http://abs:13378",
            )
            # Private — u_b can't see it.
            assert await store.get_visible(shared.id, user_id="u_b") is None
            # Shared — u_b can see it; ownership flag is False.
            await store.set_scope(
                shared.id, scope="shared", owner_user_id="u_a",
            )
            got = await store.get_visible(shared.id, user_id="u_b")
            assert got is not None
            assert got.scope == "shared"
            d = got.to_dict(viewer_user_id="u_b")
            assert d["is_shared"] is True
            assert d["is_owned_by_viewer"] is False
        _run(go())

    def test_set_scope_enforces_ownership(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            srv = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            # Non-owner can't flip — UPDATE matches 0 rows, scope stays.
            await store.set_scope(srv.id, scope="shared", owner_user_id="u_b")
            check = await store.get(srv.id, user_id="u_a")
            assert check is not None
            assert check.scope == "private"
            # Owner CAN flip.
            await store.set_scope(srv.id, scope="shared", owner_user_id="u_a")
            check = await store.get(srv.id, user_id="u_a")
            assert check is not None
            assert check.scope == "shared"
            # And flip back.
            await store.set_scope(srv.id, scope="private", owner_user_id="u_a")
            check = await store.get(srv.id, user_id="u_a")
            assert check is not None
            assert check.scope == "private"
        _run(go())

    def test_set_scope_rejects_invalid_value(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            srv = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            with pytest.raises(ValueError):
                await store.set_scope(
                    srv.id, scope="public", owner_user_id="u_a",
                )
            # And rejects the empty-owner case so we don't accidentally
            # bypass ownership by leaving the caller blank.
            with pytest.raises(ValueError):
                await store.set_scope(
                    srv.id, scope="shared", owner_user_id="",
                )
        _run(go())

    def test_strict_get_unchanged_for_non_owner(self):
        """``store.get(..., user_id=X)`` MUST still return None for users
        who don't own the row, even when scope='shared'. Write-path
        ownership checks rely on this distinction (vs ``get_visible``)."""
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            srv = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            await store.set_scope(srv.id, scope="shared", owner_user_id="u_a")
            # u_b can SEE it via get_visible, can NOT via get().
            assert await store.get_visible(srv.id, user_id="u_b") is not None
            assert await store.get(srv.id, user_id="u_b") is None
            # Same row still returned via strict get() to its owner.
            assert await store.get(srv.id, user_id="u_a") is not None
        _run(go())

    def test_find_match_includes_shared(self):
        """Auto-detect mustn't re-offer a shared server to a non-admin —
        a 'Connect' button on something the admin has already published
        is misleading. find_match needs to honour scope."""
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            srv = await store.create(
                user_id="u_a", provider="audiobookshelf",
                name="A", base_url="http://abs:13378",
            )
            # Private: u_b sees no match.
            miss = await store.find_match(
                user_id="u_b", provider="audiobookshelf",
                base_url="http://abs:13378",
            )
            assert miss is None
            # Shared: u_b finds the same row.
            await store.set_scope(srv.id, scope="shared", owner_user_id="u_a")
            hit = await store.find_match(
                user_id="u_b", provider="audiobookshelf",
                base_url="http://abs:13378",
            )
            assert hit is not None
            assert hit.id == srv.id
        _run(go())


# --- providers.audiobookshelf -----------------------------------------


def _mock_http():
    client = MagicMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    return client


def _resp(status: int, body: dict):
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=body)
    return r


class TestAudiobookshelfPing:
    def test_requires_both_ping_and_status(self):
        """A trivial 200 on /ping isn't enough — fingerprint needs both."""
        async def go():
            http = _mock_http()
            http.get.side_effect = [
                _resp(200, {"success": True}),
                _resp(200, {"isInit": True, "language": "en-us"}),
            ]
            info = await AudiobookshelfProvider(http).ping("http://abs:13378")
            assert isinstance(info, ProviderInfo)
            assert info.provider == "audiobookshelf"
            assert info.is_initialized is True
        _run(go())

    def test_ping_wrong_shape_rejected(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {"hello": "world"})
            info = await AudiobookshelfProvider(http).ping("http://abs:13378")
            assert info is None
        _run(go())

    def test_status_missing_isinit_rejected(self):
        async def go():
            http = _mock_http()
            http.get.side_effect = [
                _resp(200, {"success": True}),
                _resp(200, {"language": "en-us"}),  # no isInit
            ]
            info = await AudiobookshelfProvider(http).ping("http://abs:13378")
            assert info is None
        _run(go())


class TestAudiobookshelfLogin:
    def test_login_returns_token(self):
        async def go():
            http = _mock_http()
            http.post.return_value = _resp(200, {"user": {"token": "t_abc"}})
            token = await AudiobookshelfProvider(http).login(
                "http://abs:13378", "me", "pw",
            )
            assert token == "t_abc"
        _run(go())

    def test_login_passes_follow_redirects(self):
        """Regression: the dogfood ABS (behind Caddy) sent HTTP 308 on POST
        /login. Without follow_redirects the client sees 308 and bails
        with a misleading 'Login failed: HTTP 308' error. Every upstream
        call must pass follow_redirects=True."""
        async def go():
            http = _mock_http()
            http.post.return_value = _resp(200, {"user": {"token": "t_ok"}})
            await AudiobookshelfProvider(http).login(
                "http://abs:13378", "me", "pw",
            )
            kwargs = http.post.call_args.kwargs
            assert kwargs.get("follow_redirects") is True
        _run(go())

    def test_login_bad_creds_raises_valueerror(self):
        async def go():
            http = _mock_http()
            http.post.return_value = _resp(401, {})
            with pytest.raises(ValueError):
                await AudiobookshelfProvider(http).login(
                    "http://abs:13378", "x", "y",
                )
        _run(go())


class TestAudiobookshelfProgress:
    def test_fetch_progress_batches_via_me(self):
        """/api/me returns the full user row including mediaProgress — we
        flatten it to a per-item map keyed by external_id."""
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {
                "mediaProgress": [
                    {
                        "libraryItemId": "li_1",
                        "currentTime": 120.5,
                        "duration": 3600,
                        "progress": 0.033,
                        "isFinished": False,
                    },
                    {
                        "libraryItemId": "li_2",
                        "currentTime": 3600,
                        "duration": 3600,
                        "progress": 1.0,
                        "isFinished": True,
                    },
                ],
            })
            prog = await AudiobookshelfProvider(http).fetch_progress(
                "http://abs:13378", "tok",
            )
            assert prog["li_1"]["current_time_s"] == 120.5
            assert prog["li_1"]["is_finished"] is False
            assert prog["li_2"]["is_finished"] is True
        _run(go())

    def test_fetch_progress_tolerates_missing_fields(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {
                "mediaProgress": [
                    {"libraryItemId": "li_a"},  # all fields absent
                    {"currentTime": 10},        # no id — skipped
                ],
            })
            prog = await AudiobookshelfProvider(http).fetch_progress(
                "http://abs:13378", "tok",
            )
            assert "li_a" in prog
            assert prog["li_a"]["current_time_s"] == 0.0
            assert prog["li_a"]["is_finished"] is False
            # The entry without an id is silently dropped, not crashed on.
            assert len(prog) == 1
        _run(go())

    def test_fetch_progress_keys_podcast_entries_by_item_and_episode(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {
                "mediaProgress": [
                    {
                        "libraryItemId": "li_podcast",
                        "episodeId": "ep_42",
                        "currentTime": 33,
                        "duration": 300,
                        "progress": 0.11,
                        "isFinished": False,
                    },
                ],
            })
            prog = await AudiobookshelfProvider(http).fetch_progress(
                "http://abs:13378", "tok",
            )
            assert "li_podcast:ep_42" in prog
            assert prog["li_podcast:ep_42"]["current_time_s"] == 33
        _run(go())

    def test_fetch_progress_non_200_returns_empty(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(401, {})
            prog = await AudiobookshelfProvider(http).fetch_progress(
                "http://abs:13378", "tok",
            )
            assert prog == {}
        _run(go())

    def test_push_progress_computes_progress_from_current_time(self):
        """Caller passes currentTime + duration; we compute 0-1 progress
        so the upstream server doesn't need to trust the client's math."""
        async def go():
            http = _mock_http()
            http.patch = AsyncMock(return_value=_resp(200, {}))
            ok = await AudiobookshelfProvider(http).push_progress(
                "http://abs:13378", "tok",
                external_id="li_1",
                current_time_s=900,
                duration_s=3600,
            )
            assert ok is True
            http.patch.assert_awaited_once()
            call = http.patch.call_args
            body = call.kwargs.get("json") or call.args[1]
            assert body["currentTime"] == 900
            assert body["duration"] == 3600
            assert 0.2 < body["progress"] < 0.3  # ~0.25

    def test_push_progress_clamps_when_over_duration(self):
        """If the client reports a position past the known duration (rare
        but possible with live-updated audiobooks), progress clamps to 1.0
        instead of spewing a greater-than-one value upstream."""
        async def go():
            http = _mock_http()
            http.patch = AsyncMock(return_value=_resp(200, {}))
            await AudiobookshelfProvider(http).push_progress(
                "http://abs:13378", "tok",
                external_id="li_1",
                current_time_s=5000,
                duration_s=3600,
            )
            body = http.patch.call_args.kwargs["json"]
            assert body["progress"] == 1.0

    def test_push_progress_targets_episode_specific_endpoint(self):
        async def go():
            http = _mock_http()
            http.patch = AsyncMock(return_value=_resp(200, {}))
            ok = await AudiobookshelfProvider(http).push_progress(
                "http://abs:13378",
                "tok",
                external_id="li_podcast",
                episode_id="ep_42",
                current_time_s=45,
                duration_s=300,
            )
            assert ok is True
            assert http.patch.call_args.args[0] == "http://abs:13378/api/me/progress/li_podcast/ep_42"
        _run(go())

    def test_fetch_item_details_can_request_podcast_episode_progress(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {"id": "li_podcast"})
            item = await AudiobookshelfProvider(http).fetch_item_details(
                "http://abs:13378",
                "tok",
                external_id="li_podcast",
                episode_id="ep_42",
            )
            assert item == {"id": "li_podcast"}
            assert http.get.call_args.kwargs["params"] == {
                "expanded": 1,
                "include": "progress",
                "episode": "ep_42",
            }
        _run(go())


class TestAudiobookshelfCoverUrl:
    def test_cover_url_includes_token(self):
        http = _mock_http()
        url = AudiobookshelfProvider(http).build_cover_url(
            "http://abs:13378/", "li_1", "t_xyz",
        )
        assert url == "http://abs:13378/api/items/li_1/cover?token=t_xyz"


class TestEmbyCompatSubtitleUrl:
    def test_emby_subtitle_url_uses_emby_prefix(self):
        http = _mock_http()
        url = EmbyProvider(http).build_subtitle_url(
            "http://localhost:8096",
            external_id="vid_1",
            media_source_id="src_2",
            subtitle_stream_index=3,
            token="tok",
        )
        assert url == (
            "http://localhost:8096/emby/Videos/vid_1/src_2/"
            "Subtitles/3/Stream.vtt?api_key=tok"
        )

    def test_jellyfin_subtitle_url_has_no_emby_prefix(self):
        http = _mock_http()
        url = JellyfinProvider(http).build_subtitle_url(
            "http://localhost:8096/",
            external_id="vid_1",
            media_source_id="src_2",
            subtitle_stream_index=3,
            token="tok",
        )
        assert url == (
            "http://localhost:8096/Videos/vid_1/src_2/"
            "Subtitles/3/Stream.vtt?api_key=tok"
        )

    def test_subtitle_url_rejects_invalid_selection(self):
        http = _mock_http()
        provider = EmbyProvider(http)
        assert provider.build_subtitle_url(
            "http://localhost:8096",
            external_id="vid_1",
            media_source_id="",
            subtitle_stream_index=2,
            token="tok",
        ) == ""
        assert provider.build_subtitle_url(
            "http://localhost:8096",
            external_id="vid_1",
            media_source_id="src_2",
            subtitle_stream_index=-1,
            token="tok",
        ) == ""


class TestEmbyCompatBrowserVideoUrl:
    def test_emby_browser_video_stream_url_uses_emby_prefix(self):
        http = _mock_http()
        url = EmbyProvider(http).build_browser_video_stream_url(
            "http://localhost:8096",
            external_id="vid_1",
            media_source_id="src_main",
            play_session_id="play_123",
            token="tok",
            audio_stream_index=2,
        )
        assert url == (
            "http://localhost:8096/emby/Videos/vid_1/stream.mp4"
            "?MediaSourceId=src_main&PlaySessionId=play_123"
            "&AudioCodec=aac&AudioStreamIndex=2&MaxAudioChannels=2&api_key=tok"
        )

    def test_jellyfin_browser_video_stream_url_has_no_emby_prefix(self):
        http = _mock_http()
        url = JellyfinProvider(http).build_browser_video_stream_url(
            "http://localhost:8096/",
            external_id="vid_1",
            media_source_id="src_main",
            play_session_id="play_123",
            token="tok",
            audio_stream_index=2,
        )
        assert url == (
            "http://localhost:8096/Videos/vid_1/stream.mp4"
            "?MediaSourceId=src_main&PlaySessionId=play_123"
            "&AudioCodec=aac&AudioStreamIndex=2&MaxAudioChannels=2&api_key=tok"
        )

    def test_browser_video_stream_url_rejects_missing_required_fields(self):
        http = _mock_http()
        provider = EmbyProvider(http)
        assert provider.build_browser_video_stream_url(
            "http://localhost:8096",
            external_id="",
            media_source_id="src_main",
            play_session_id="play_123",
            token="tok",
        ) == ""
        assert provider.build_browser_video_stream_url(
            "http://localhost:8096",
            external_id="vid_1",
            media_source_id="",
            play_session_id="play_123",
            token="tok",
        ) == ""
        assert provider.build_browser_video_stream_url(
            "http://localhost:8096",
            external_id="vid_1",
            media_source_id="src_main",
            play_session_id="",
            token="tok",
        ) == ""

    def test_browser_video_stream_url_accepts_start_offset(self):
        http = _mock_http()
        url = EmbyProvider(http).build_browser_video_stream_url(
            "http://localhost:8096",
            external_id="vid_1",
            media_source_id="src_main",
            play_session_id="play_123",
            token="tok",
            audio_stream_index=2,
            start_time_ticks=450_000_000,
        )
        assert url == (
            "http://localhost:8096/emby/Videos/vid_1/stream.mp4"
            "?MediaSourceId=src_main&PlaySessionId=play_123"
            "&AudioCodec=aac&AudioStreamIndex=2&MaxAudioChannels=2"
            "&StartTimeTicks=450000000&api_key=tok"
        )


class TestEmbyCompatBrowserStreamFallback:
    def test_helper_transcodes_ac3_audio_for_browser(self):
        async def go():
            from augmentum.proxy.media_routes import _emby_compat_browser_stream_url

            http = _mock_http()
            http.get.side_effect = [
                _resp(200, {"Id": "user_1"}),
                _resp(200, {
                    "PlaySessionId": "play_123",
                    "MediaSources": [{
                        "Id": "src_main",
                        "Container": "mp4",
                        "VideoCodec": "h264",
                        "DefaultAudioStreamIndex": 1,
                        "MediaStreams": [
                            {"Type": "Video", "Index": 0, "Codec": "h264"},
                            {"Type": "Audio", "Index": 1, "Codec": "ac3", "Channels": 6},
                        ],
                    }],
                }),
            ]
            url = await _emby_compat_browser_stream_url(
                client=EmbyProvider(http),
                server=SimpleNamespace(base_url="http://localhost:8096", access_token="tok"),
                external_id="vid_1",
                cached_meta={
                    "preferred_media_source_id": "src_main",
                    "preferred_audio_stream_index": 1,
                },
                stream_choice={
                    "MediaSourceId": "src_main",
                    "AudioStreamIndex": "1",
                },
            )
            assert url == (
                "http://localhost:8096/emby/Videos/vid_1/stream.mp4"
                "?MediaSourceId=src_main&PlaySessionId=play_123"
                "&AudioCodec=aac&AudioStreamIndex=1&MaxAudioChannels=2&api_key=tok"
            )

        _run(go())

    def test_helper_keeps_direct_stream_for_browser_safe_audio(self):
        async def go():
            from augmentum.proxy.media_routes import _emby_compat_browser_stream_url

            http = _mock_http()
            http.get.side_effect = [
                _resp(200, {"Id": "user_1"}),
                _resp(200, {
                    "PlaySessionId": "play_123",
                    "MediaSources": [{
                        "Id": "src_main",
                        "Container": "mp4",
                        "VideoCodec": "h264",
                        "DefaultAudioStreamIndex": 1,
                        "MediaStreams": [
                            {"Type": "Video", "Index": 0, "Codec": "h264"},
                            {"Type": "Audio", "Index": 1, "Codec": "aac", "Channels": 2},
                        ],
                    }],
                }),
            ]
            url = await _emby_compat_browser_stream_url(
                client=EmbyProvider(http),
                server=SimpleNamespace(base_url="http://localhost:8096", access_token="tok"),
                external_id="vid_1",
                cached_meta={
                    "preferred_media_source_id": "src_main",
                    "preferred_audio_stream_index": 1,
                },
                stream_choice={
                    "MediaSourceId": "src_main",
                    "AudioStreamIndex": "1",
                },
            )
            assert url == ""

        _run(go())

    def test_helper_passes_start_offset_for_browser_transcode(self):
        async def go():
            from augmentum.proxy.media_routes import _emby_compat_browser_stream_url

            http = _mock_http()
            http.get.side_effect = [
                _resp(200, {"Id": "user_1"}),
                _resp(200, {
                    "PlaySessionId": "play_123",
                    "MediaSources": [{
                        "Id": "src_main",
                        "Container": "mp4",
                        "VideoCodec": "h264",
                        "DefaultAudioStreamIndex": 1,
                        "MediaStreams": [
                            {"Type": "Video", "Index": 0, "Codec": "h264"},
                            {"Type": "Audio", "Index": 1, "Codec": "ac3", "Channels": 6},
                        ],
                    }],
                }),
            ]
            url = await _emby_compat_browser_stream_url(
                client=EmbyProvider(http),
                server=SimpleNamespace(base_url="http://localhost:8096", access_token="tok"),
                external_id="vid_1",
                cached_meta={
                    "preferred_media_source_id": "src_main",
                    "preferred_audio_stream_index": 1,
                },
                stream_choice={
                    "MediaSourceId": "src_main",
                    "AudioStreamIndex": "1",
                },
                start_time_s=45.0,
            )
            assert url == (
                "http://localhost:8096/emby/Videos/vid_1/stream.mp4"
                "?MediaSourceId=src_main&PlaySessionId=play_123"
                "&AudioCodec=aac&AudioStreamIndex=1&MaxAudioChannels=2"
                "&StartTimeTicks=450000000&api_key=tok"
            )

        _run(go())


class TestShiftWebVtt:
    def test_shifts_cues_by_offset(self):
        from augmentum.proxy.media_routes import _shift_webvtt

        raw = (
            "WEBVTT\n\n"
            "1\n"
            "00:45:00.000 --> 00:45:03.500\n"
            "First line\n\n"
            "2\n"
            "00:45:05.000 --> 00:45:07.000\n"
            "Second line\n"
        )
        shifted = _shift_webvtt(raw, 2700.0)
        assert "00:00:00.000 --> 00:00:03.500" in shifted
        assert "00:00:05.000 --> 00:00:07.000" in shifted
        assert "First line" in shifted
        assert "Second line" in shifted


class TestEmbyCompatRemoteSessions:
    def test_list_remote_sessions_filters_to_controllable_video_clients(self):
        async def go():
            http = _mock_http()
            http.get.side_effect = [
                _resp(200, {"Id": "user_1"}),
                _resp(200, [
                    {
                        "Id": "sess_tv",
                        "DeviceId": "lg-webos",
                        "DeviceName": "Living Room TV",
                        "Client": "Jellyfin Web",
                        "UserName": "alex",
                        "SupportsMediaControl": True,
                        "SupportsRemoteControl": True,
                        "PlayableMediaTypes": ["Video", "Audio"],
                        "NowPlayingItem": {"Name": "Episode 1"},
                    },
                    {
                        "Id": "sess_audio",
                        "DeviceId": "sonos-1",
                        "DeviceName": "Kitchen Speaker",
                        "Client": "Audio Client",
                        "SupportsMediaControl": True,
                        "PlayableMediaTypes": ["Audio"],
                    },
                    {
                        "Id": "sess_aug",
                        "DeviceId": "augmentum-media",
                        "DeviceName": "Augmentum Server",
                        "Client": "Augmentum",
                        "SupportsMediaControl": True,
                        "PlayableMediaTypes": ["Video"],
                    },
                ]),
            ]

            sessions = await JellyfinProvider(http).list_remote_sessions(
                "http://jf:8096", "tok", media_type="Video",
            )

            assert len(sessions) == 1
            assert sessions[0].session_id == "sess_tv"
            assert sessions[0].device_name == "Living Room TV"
            assert sessions[0].now_playing_title == "Episode 1"
            assert sessions[0].now_playing_item_id == ""
            assert sessions[0].supported_commands == []
            # Jellyfin asks the server to pre-filter to controllable sessions.
            assert http.get.call_args_list[1].kwargs["params"]["ControllableByUserId"] == "user_1"

        _run(go())

    def test_list_remote_sessions_captures_playstate_details(self):
        async def go():
            http = _mock_http()
            http.get.side_effect = [
                _resp(200, {"Id": "user_1"}),
                _resp(200, [
                    {
                        "Id": "sess_tv",
                        "DeviceId": "living-room",
                        "DeviceName": "Living Room",
                        "Client": "Emby Theater",
                        "UserName": "alex",
                        "SupportsMediaControl": True,
                        "SupportsRemoteControl": True,
                        "PlayableMediaTypes": ["Video"],
                        "SupportedCommands": ["VolumeUp", "SetVolume", "ToggleMute"],
                        "NowPlayingItem": {
                            "Id": "item_77",
                            "Name": "Pilot",
                            "SeriesName": "Show",
                            "Type": "Episode",
                            "RunTimeTicks": 600000000,
                        },
                        "PlayState": {
                            "PositionTicks": 120000000,
                            "IsPaused": True,
                            "IsMuted": True,
                            "CanSeek": True,
                            "VolumeLevel": 35,
                            "AudioStreamIndex": 2,
                            "SubtitleStreamIndex": 4,
                        },
                    },
                ]),
            ]

            sessions = await EmbyProvider(http).list_remote_sessions(
                "http://emby:8096", "tok", media_type="Video",
            )

            assert len(sessions) == 1
            session = sessions[0]
            assert session.now_playing_item_id == "item_77"
            assert session.supported_commands == ["VolumeUp", "SetVolume", "ToggleMute"]
            assert session.current_time_s == 12.0
            assert session.duration_s == 60.0
            assert session.is_paused is True
            assert session.is_muted is True
            assert session.can_seek is True
            assert session.volume_level == 35
            assert session.audio_stream_index == 2
            assert session.subtitle_stream_index == 4

        _run(go())

    def test_remote_play_posts_session_command_shape(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {"Id": "user_1"})
            http.post.return_value = _resp(200, {})

            ok = await EmbyProvider(http).remote_play(
                "http://emby:8096",
                "tok",
                session_id="sess_1",
                external_id="item_9",
                start_time_s=12.5,
                play_command="PlayNow",
                media_source_id="src_main",
                audio_stream_index=2,
                subtitle_stream_index=5,
            )

            assert ok is True
            http.post.assert_awaited_once()
            call = http.post.call_args
            assert call.args[0] == "http://emby:8096/emby/Sessions/sess_1/Playing"
            assert call.kwargs["params"]["ItemIds"] == "item_9"
            assert call.kwargs["params"]["PlayCommand"] == "PlayNow"
            assert call.kwargs["params"]["ControllingUserId"] == "user_1"
            assert call.kwargs["params"]["MediaSourceId"] == "src_main"
            assert call.kwargs["params"]["AudioStreamIndex"] == 2
            assert call.kwargs["params"]["SubtitleStreamIndex"] == 5
            body = call.kwargs["json"]
            assert body["ItemIds"] == ["item_9"]
            assert body["ControllingUserId"] == "user_1"
            assert body["StartPositionTicks"] == 125000000

        _run(go())

    def test_remote_command_posts_playstate_request(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {"Id": "user_1"})
            http.post.return_value = _resp(200, {})

            ok = await JellyfinProvider(http).remote_command(
                "http://jf:8096",
                "tok",
                session_id="sess_2",
                command="Seek",
                seek_position_s=30.0,
            )

            assert ok is True
            http.post.assert_awaited_once()
            call = http.post.call_args
            assert call.args[0] == "http://jf:8096/Sessions/sess_2/Playing/Seek"
            assert call.kwargs["params"] == {
                "ControllingUserId": "user_1",
                "SeekPositionTicks": 300000000,
            }
            assert call.kwargs["json"] == {
                "Command": "Seek",
                "ControllingUserId": "user_1",
                "SeekPositionTicks": 300000000,
            }

        _run(go())

    def test_remote_general_command_posts_emby_argument_body(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {"Id": "user_1"})
            http.post.return_value = _resp(200, {})

            ok = await EmbyProvider(http).remote_general_command(
                "http://emby:8096",
                "tok",
                session_id="sess_5",
                command="SetVolume",
                arguments={"Volume": 55},
            )

            assert ok is True
            http.post.assert_awaited_once()
            call = http.post.call_args
            assert call.args[0] == "http://emby:8096/emby/Sessions/sess_5/Command/SetVolume"
            assert call.kwargs["json"] == {
                "ControllingUserId": "user_1",
                "Arguments": {
                    "Volume": "55",
                },
            }

        _run(go())

    def test_remote_general_command_posts_jellyfin_full_general_command_when_needed(self):
        async def go():
            http = _mock_http()
            http.get.return_value = _resp(200, {"Id": "user_1"})
            http.post.return_value = _resp(200, {})

            ok = await JellyfinProvider(http).remote_general_command(
                "http://jf:8096",
                "tok",
                session_id="sess_6",
                command="SetSubtitleStreamIndex",
                arguments={"Index": -1},
            )

            assert ok is True
            http.post.assert_awaited_once()
            call = http.post.call_args
            assert call.args[0] == "http://jf:8096/Sessions/sess_6/Command"
            assert call.kwargs["json"] == {
                "ControllingUserId": "user_1",
                "Name": "SetSubtitleStreamIndex",
                "Arguments": {
                    "Index": "-1",
                },
            }

        _run(go())


class TestAudiobookshelfCatalog:
    def test_fetch_catalog_paginates_past_first_1000_items(self):
        """Large ABS libraries must walk `page` instead of silently truncating at 1000."""
        async def go():
            http = _mock_http()

            def _row(n: int) -> dict:
                return {
                    "id": f"li_{n}",
                    "ino": f"ino_{n}",
                    "isFile": True,
                    "media": {
                        "metadata": {"title": f"Book {n}"},
                        "numAudioFiles": 1,
                        "duration": 60,
                    },
                }

            http.get.side_effect = [
                _resp(200, {
                    "libraries": [{"id": "lib_books", "mediaType": "book"}],
                }),
                _resp(200, {
                    "results": [_row(i) for i in range(1000)],
                    "total": 1001,
                }),
                _resp(200, {
                    "results": [_row(1000)],
                    "total": 1001,
                }),
            ]

            items = await AudiobookshelfProvider(http).fetch_catalog(
                "http://abs:13378", "tok",
            )

            assert len(items) == 1001
            assert items[0].name == "Book 0"
            assert items[-1].name == "Book 1000"
            assert http.get.call_args_list[1].kwargs["params"] == {"limit": 1000, "page": 0}
            assert http.get.call_args_list[2].kwargs["params"] == {"limit": 1000, "page": 1}

        _run(go())

    def test_fetch_catalog_recovers_folder_books_via_item_details(self):
        """Folder-based books missing file inos in the listing should be
        rehydrated via ``GET /api/items/{id}`` instead of surfacing as skipped."""
        async def go():
            http = _mock_http()
            http.get.side_effect = [
                _resp(200, {
                    "libraries": [{"id": "lib_books", "mediaType": "book"}],
                }),
                _resp(200, {
                    "results": [{
                        "id": "li_folder",
                        "isFile": False,
                        "media": {
                            "metadata": {
                                "title": "Folder Book",
                                "authors": ["Robin Author"],
                            },
                            "numAudioFiles": 2,
                        },
                    }],
                }),
                _resp(200, {
                    "id": "li_folder",
                    "isFile": False,
                    "media": {
                        "metadata": {
                            "title": "Folder Book",
                            "authors": [{"id": "au_1", "name": "Robin Author"}],
                            "narrators": [{"id": "na_1", "name": "Casey Reader"}],
                        },
                        "numAudioFiles": 2,
                        "duration": 321,
                        "audioFiles": [{
                            "ino": "af_123",
                            "duration": 321,
                            "metadata": {"size": 777, "format": "mp3"},
                        }],
                        "chapters": [{"title": "One", "start": 0, "end": 321}],
                    },
                }),
            ]

            items = await AudiobookshelfProvider(http).fetch_catalog(
                "http://abs:13378", "tok",
            )

            assert len(items) == 1
            item = items[0]
            assert item.name == "Folder Book"
            assert item.author == "Robin Author"
            assert item.narrator == "Casey Reader"
            assert item.stream_path == "/api/items/li_folder/file/af_123"
            assert item.duration_ms == 321_000
            assert item.extra["chapters"] == [{"title": "One", "start": 0.0, "end": 321.0}]
            assert "skip_reason" not in item.extra

            detail_call = http.get.call_args_list[2]
            assert detail_call.args[0] == "http://abs:13378/api/items/li_folder"
            assert detail_call.kwargs["params"] == {"expanded": 1, "include": "progress"}
            assert detail_call.kwargs["headers"]["Authorization"] == "Bearer tok"

        _run(go())


class TestMediaSync:
    def test_sync_server_persists_summary_with_skip_and_recovery_counts(self):
        async def go():
            conn = await _setup_db()
            store = MediaServerStore(conn)
            server = await store.create(
                user_id="u_a",
                provider="audiobookshelf",
                name="Home ABS",
                base_url="http://abs:13378",
                access_token="tok",
            )

            playable = CatalogItem(
                external_id="li_ok",
                name="Recovered Book",
                kind="audio",
                mime_type="audio/mpeg",
                size_bytes=123,
                duration_ms=120_000,
                progress_pct=0,
                cover_url="",
                author="Robin Author",
                narrator="Casey Reader",
                stream_path="/api/items/li_ok/file/af_1",
                extra={"recovered_via_detail": True},
            )
            skipped = CatalogItem(
                external_id="li_skip",
                name="Metadata Shell",
                kind="audio",
                mime_type="audio/mpeg",
                size_bytes=0,
                duration_ms=0,
                progress_pct=0,
                cover_url="",
                author="",
                narrator="",
                stream_path="",
                extra={"skip_reason": "no_audio_files"},
            )

            provider = MagicMock()
            provider.fetch_catalog = AsyncMock(return_value=[playable, skipped])
            provider.fetch_progress = AsyncMock(return_value={
                "li_ok": {
                    "current_time_s": 12.0,
                    "duration_s": 120.0,
                    "progress": 0.1,
                    "is_finished": False,
                },
            })

            progress_updates: list[tuple[float, str]] = []

            async def _progress(progress: float, stage: str) -> None:
                progress_updates.append((progress, stage))

            with patch("augmentum.media.sync._build_provider", return_value=provider):
                with patch("augmentum.media.sync._index_item", new=AsyncMock()):
                    with patch(
                        "augmentum.media.sync.provider_supports_library_discovery",
                        return_value=False,
                    ):
                        indexed, err = await sync_server(
                            server,
                            store=store,
                            http_client=MagicMock(),
                            progress_callback=_progress,
                        )

            assert indexed == 1
            assert err == ""
            assert progress_updates[0][1] == "Fetching catalog"
            assert progress_updates[-1][1] == (
                "Indexed 1 of 2 · 1 skipped · 1 item recovered via detail fetch"
            )

            refreshed = await store.get(server.id, user_id="u_a")
            assert refreshed is not None
            assert refreshed.status == "ok"
            assert refreshed.item_count == 1
            assert refreshed.total_seen == 2
            assert refreshed.skipped_count == 1
            assert refreshed.status_detail == (
                "Indexed 1 of 2 · 1 skipped · 1 item recovered via detail fetch"
            )
            assert refreshed.last_sync_skipped == [{
                "title": "Metadata Shell",
                "author": "",
                "reason": "no_audio_files",
            }]

        _run(go())


class TestItemFromAbs:
    def test_detail_payload_people_objects_are_flattened(self):
        raw = {
            "id": "li_people",
            "media": {
                "metadata": {
                    "title": "People Book",
                    "authors": [{"id": "au_1", "name": "Robin Author"}],
                    "narrators": [{"id": "na_1", "name": "Casey Reader"}],
                },
                "audioFiles": [{"ino": "123", "metadata": {"size": 500, "format": "mp3"}}],
            },
        }
        item = _item_from_abs(raw, lib_kind="book")
        assert item is not None
        assert item.author == "Robin Author"
        assert item.narrator == "Casey Reader"

    def test_minimal_audiobook_parsed(self):
        raw = {
            "id": "li_x",
            "media": {
                "metadata": {
                    "title": "The Final Empire",
                    "authors": ["Brandon Sanderson"],
                },
                "duration": 123.45,
                "audioFiles": [{"ino": "123", "metadata": {"size": 500, "format": "mp3"}}],
                "chapters": [
                    {"title": "Prologue", "start": 0, "end": 42},
                ],
            },
        }
        item = _item_from_abs(raw, lib_kind="book")
        assert item is not None
        assert item.external_id == "li_x"
        assert item.name == "The Final Empire"
        assert item.author == "Brandon Sanderson"
        assert item.stream_path == "/api/items/li_x/file/123"
        assert item.extra["chapters"] == [{"title": "Prologue", "start": 0.0, "end": 42.0}]

    def test_library_files_fallback_when_audio_files_empty(self):
        """ABS sometimes returns audioFiles=[] but carries ino via libraryFiles.
        We must still derive a stream_path in that case — otherwise all 453 of
        the dogfood books silently disappear from the sync, which is how this
        regression was caught in the first place."""
        raw = {
            "id": "li_fallback",
            "media": {"metadata": {"title": "T"}, "audioFiles": []},
            "libraryFiles": [
                {"fileType": "image", "ino": "999"},
                {"fileType": "audio", "ino": "456"},
            ],
        }
        item = _item_from_abs(raw, lib_kind="book")
        assert item is not None
        assert item.stream_path == "/api/items/li_fallback/file/456"

    def test_single_file_book_uses_top_level_ino(self):
        """Current ABS listings ship neither audioFiles nor libraryFiles for
        single-file books — just top-level `ino` + `isFile: true` + a
        `numAudioFiles` counter. The LibraryItem's ino IS the audio file's
        ino in that case, so the stream path resolves against it directly.
        Real shape from the dogfood server on 2026-04-20.
        """
        raw = {
            "id": "li_single",
            "ino": "8675309",
            "isFile": True,
            "media": {
                "metadata": {"title": "Single-File Audiobook"},
                "numAudioFiles": 1,
                "duration": 600,
                "size": 12345,
            },
        }
        item = _item_from_abs(raw, lib_kind="book")
        assert item is not None
        assert item.stream_path == "/api/items/li_single/file/8675309"
        assert item.size_bytes == 12345
        assert item.duration_ms == 600_000

    def test_podcast_library_item_indexes_as_container(self):
        raw = {
            "id": "li_podcast",
            "media": {
                "metadata": {
                    "title": "Self-Hosted",
                    "author": "Jupiter Broadcasting",
                },
                "numEpisodes": 120,
            },
        }
        item = _item_from_abs(raw, lib_kind="podcast")
        assert item is not None
        assert item.stream_path == ""
        assert item.author == "Jupiter Broadcasting"
        assert item.extra["entity_kind"] == "podcast"
        assert item.extra["index_without_stream"] is True
        assert item.extra["episode_count"] == 120

    def test_folder_book_without_audio_files_still_skipped(self):
        """Folder-based books without a usable ino genuinely can't stream
        from the listing alone. We must return stream_path='' so the sync
        skips cleanly — otherwise we'd generate broken /file/ URLs."""
        raw = {
            "id": "li_folder",
            "isFile": False,
            "media": {"metadata": {"title": "Multi-File"}, "numAudioFiles": 12},
        }
        item = _item_from_abs(raw, lib_kind="book")
        assert item is not None
        assert item.stream_path == ""

    def test_no_audio_files_no_stream_path(self):
        raw = {"id": "li_empty", "media": {"metadata": {"title": "X"}}}
        item = _item_from_abs(raw, lib_kind="book")
        assert item is not None
        assert item.stream_path == ""

    def test_missing_id_skipped(self):
        assert _item_from_abs({}, lib_kind="book") is None


# --- adapters.media_server --------------------------------------------


class TestMediaServerAdapter:
    def test_resolve_always_none(self):
        """Media rows never resolve to bytes/paths — playback goes through
        the streaming proxy, not the generic download endpoint."""
        async def go():
            conn = await _setup_db()
            adapter = MediaServerAdapter("audiobookshelf", conn)
            assert await adapter.resolve("any", user_id="u_a") is None
        _run(go())

    def test_list_source_ids_scoped_to_user(self):
        async def go():
            conn = await _setup_db()
            # Seed rows for both users with the same source.
            await conn.execute(
                "INSERT INTO file_index (id, user_id, source, source_id, name) "
                "VALUES ('f1', 'u_a', 'audiobookshelf', 'x', 'A')"
            )
            await conn.execute(
                "INSERT INTO file_index (id, user_id, source, source_id, name) "
                "VALUES ('f2', 'u_b', 'audiobookshelf', 'y', 'B')"
            )
            await conn.commit()
            adapter = MediaServerAdapter("audiobookshelf", conn)
            ids_a = await adapter.list_source_ids(user_id="u_a")
            ids_b = await adapter.list_source_ids(user_id="u_b")
            assert ids_a == ["x"]
            assert ids_b == ["y"]
        _run(go())


# --- base.DEFAULT_PORTS -----------------------------------------------


class TestDefaults:
    def test_audiobookshelf_default_port(self):
        assert DEFAULT_PORTS["audiobookshelf"] == 13378

    def test_emby_and_jellyfin_share_default(self):
        # Known-good default — both projects ship 8096 out of the box.
        assert DEFAULT_PORTS["emby"] == 8096
        assert DEFAULT_PORTS["jellyfin"] == 8096


# --- media.normalize -------------------------------------------------


class TestNormalizeName:
    """Real-shape coverage: every variant pair that should collapse to
    the same canonical form has a test. When a library mixes these in
    the wild, the "Also by X" query needs equality to work."""

    def test_empty_inputs(self):
        from augmentum.media.normalize import normalize_name
        assert normalize_name("") == ""
        assert normalize_name(None) == ""  # type: ignore[arg-type]

    def test_case_insensitive(self):
        from augmentum.media.normalize import normalize_name
        assert normalize_name("Brandon Sanderson") == normalize_name("BRANDON sanderson")

    def test_comma_reversed_collapses(self):
        """'Sanderson, Brandon' vs 'Brandon Sanderson' — same person,
        different library-import convention. Token-sort makes equality
        work for both."""
        from augmentum.media.normalize import normalize_name
        assert normalize_name("Sanderson, Brandon") == normalize_name("Brandon Sanderson")

    def test_apostrophe_forms_collapse(self):
        from augmentum.media.normalize import normalize_name
        # ASCII + all curly quote variants map to the same form.
        a = normalize_name("O'Brien")
        b = normalize_name("OBrien")
        c = normalize_name("O\u2019Brien")
        d = normalize_name("O\u2018Brien")
        assert a == b == c == d

    def test_periods_and_single_letters_drop(self):
        """'J.F. Brink' vs 'JF Brink': the former produces single-letter
        tokens which we drop; the latter keeps 'jf' as a real token. We
        drop single letters so BOTH reduce to just 'brink' — agreed-upon
        consequence of manual tagging."""
        from augmentum.media.normalize import normalize_name
        # "J.F." and "JF" both reduce to "brink" (single letters drop).
        assert normalize_name("J.F. Brink") == "brink"
        # "JF Brink" keeps 'jf' as a token since it's not single-letter.
        assert "brink" in normalize_name("JF Brink").split()

    def test_stopwords_dropped(self):
        from augmentum.media.normalize import normalize_name
        assert normalize_name("The Beatles") == normalize_name("Beatles")

    def test_token_sort_is_stable(self):
        from augmentum.media.normalize import normalize_name
        # Same tokens → same result regardless of input order.
        assert normalize_name("alpha beta gamma") == normalize_name("gamma alpha beta")


class TestTokensMatchAsRelated:
    """Matching predicate behind /api/media/related (``Also by X``).

    Subset semantics, not equality — real libraries have uploader-
    concatenated author fields ("JF Brink TheFirstDefier") and co-author
    joins ("Jane Austen, Charles Dickens") that strict equality would
    never link back to their solo counterparts.
    """

    def test_exact_match_matches(self):
        from augmentum.media.normalize import (
            normalize_name, tokens_match_as_related,
        )
        a = normalize_name("Brandon Sanderson")
        b = normalize_name("Brandon Sanderson")
        assert tokens_match_as_related(a, b) is True

    def test_extra_token_on_one_side_matches(self):
        """Seed has a junk token appended (common uploader error)."""
        from augmentum.media.normalize import (
            normalize_name, tokens_match_as_related,
        )
        seed = normalize_name("JF Brink TheFirstDefier")
        other = normalize_name("JF Brink")
        assert tokens_match_as_related(seed, other) is True
        # Symmetric: swapping which side has the extra token still matches.
        assert tokens_match_as_related(other, seed) is True

    def test_coauthor_join_matches_solo_title(self):
        """A co-authored book should surface the solo titles of each
        author, not show "no other books"."""
        from augmentum.media.normalize import (
            normalize_name, tokens_match_as_related,
        )
        coauth = normalize_name("Jane Austen, Charles Dickens")
        solo = normalize_name("Jane Austen")
        assert tokens_match_as_related(coauth, solo) is True

    def test_shared_surname_does_not_match(self):
        """Precision guard: 'Jane Smith' and 'John Smith' share only the
        surname — neither is a subset of the other. They're different
        people and must not appear as "Also by"."""
        from augmentum.media.normalize import (
            normalize_name, tokens_match_as_related,
        )
        a = normalize_name("Jane Smith")
        b = normalize_name("John Smith")
        assert tokens_match_as_related(a, b) is False

    def test_fully_disjoint_does_not_match(self):
        from augmentum.media.normalize import tokens_match_as_related
        assert tokens_match_as_related("brandon sanderson", "neil gaiman") is False

    def test_empty_inputs_do_not_match(self):
        from augmentum.media.normalize import tokens_match_as_related
        assert tokens_match_as_related("", "brandon") is False
        assert tokens_match_as_related("brandon", "") is False
        assert tokens_match_as_related("", "") is False


class TestFuzzyMatchScore:
    def test_identical_normalised_is_one(self):
        from augmentum.media.normalize import fuzzy_match_score
        assert fuzzy_match_score("brandon sanderson", "brandon sanderson") == 1.0

    def test_no_overlap_is_zero(self):
        from augmentum.media.normalize import fuzzy_match_score
        assert fuzzy_match_score("brandon sanderson", "neil gaiman") == 0.0

    def test_partial_overlap_is_ratio(self):
        from augmentum.media.normalize import fuzzy_match_score
        # {sanderson} ∩ {brandon, sanderson} = 1, union = 2 → 0.5
        score = fuzzy_match_score("sanderson", "brandon sanderson")
        assert abs(score - 0.5) < 0.001

    def test_empty_inputs(self):
        from augmentum.media.normalize import fuzzy_match_score
        assert fuzzy_match_score("", "brandon") == 0.0
        assert fuzzy_match_score("brandon", "") == 0.0


# --- file_index prefix search --------------------------------------


class TestMediaStatusFilter:
    """list_recent should honour media_status, matching rows by their
    source_metadata.is_finished + progress_pct fields."""

    def test_filters_in_progress(self):
        async def go():
            import aiosqlite, json
            conn = await aiosqlite.connect(":memory:")
            await conn.executescript("""
                CREATE TABLE users (id TEXT PRIMARY KEY);
                CREATE TABLE file_index (
                    id TEXT PRIMARY KEY, user_id TEXT, source TEXT,
                    source_id TEXT, name TEXT, mime_type TEXT DEFAULT '',
                    size_bytes INTEGER DEFAULT 0, real_path TEXT,
                    description TEXT DEFAULT '', tags TEXT DEFAULT '[]',
                    thumbnail TEXT, embedding BLOB,
                    is_directory INTEGER DEFAULT 0, parent_id TEXT,
                    source_metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    is_favorite INTEGER DEFAULT 0, is_trashed INTEGER DEFAULT 0,
                    trashed_at TEXT, kind TEXT DEFAULT ''
                );
                INSERT INTO users VALUES ('u');
            """)
            # Three rows: one in-progress, one finished, one not started.
            rows = [
                ("a", {"progress_pct": 35.0, "is_finished": False}),
                ("b", {"progress_pct": 100.0, "is_finished": True}),
                ("c", {"progress_pct": 0.0, "is_finished": False}),
            ]
            for rid, meta in rows:
                await conn.execute(
                    "INSERT INTO file_index (id, user_id, source, source_id, name, source_metadata) "
                    "VALUES (?, 'u', 'audiobookshelf', ?, ?, ?)",
                    (f"fi_{rid}", rid, f"Book {rid}", json.dumps(meta)),
                )
            await conn.commit()

            from augmentum.vfs.index import FileIndexService
            idx = FileIndexService(conn)
            got = await idx.list_recent(user_id="u", source="audiobookshelf",
                                         media_status="in_progress", limit=10)
            names = sorted(e.name for e in got)
            assert names == ["Book a"]

            got = await idx.list_recent(user_id="u", source="audiobookshelf",
                                         media_status="finished", limit=10)
            assert [e.name for e in got] == ["Book b"]

            got = await idx.list_recent(user_id="u", source="audiobookshelf",
                                         media_status="not_started", limit=10)
            assert [e.name for e in got] == ["Book c"]

            got = await idx.list_recent(user_id="u", source="audiobookshelf",
                                         media_status="all", limit=10)
            assert len(got) == 3  # "all" falls through the whitelist, no predicate added
            await conn.close()
        import asyncio
        asyncio.get_event_loop().run_until_complete(go())

    def test_unknown_status_falls_through(self):
        """Unknown values are silently ignored (not injected)."""
        async def go():
            import aiosqlite, json
            conn = await aiosqlite.connect(":memory:")
            await conn.executescript("""
                CREATE TABLE users (id TEXT PRIMARY KEY);
                CREATE TABLE file_index (
                    id TEXT PRIMARY KEY, user_id TEXT, source TEXT,
                    source_id TEXT, name TEXT, mime_type TEXT DEFAULT '',
                    size_bytes INTEGER DEFAULT 0, real_path TEXT,
                    description TEXT DEFAULT '', tags TEXT DEFAULT '[]',
                    thumbnail TEXT, embedding BLOB,
                    is_directory INTEGER DEFAULT 0, parent_id TEXT,
                    source_metadata TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now')),
                    is_favorite INTEGER DEFAULT 0, is_trashed INTEGER DEFAULT 0,
                    trashed_at TEXT, kind TEXT DEFAULT ''
                );
                INSERT INTO users VALUES ('u');
                INSERT INTO file_index (id, user_id, source, source_id, name) VALUES
                  ('fi_x', 'u', 'audiobookshelf', 'x', 'Book x');
            """)
            await conn.commit()
            from augmentum.vfs.index import FileIndexService
            idx = FileIndexService(conn)
            got = await idx.list_recent(
                user_id="u", source="audiobookshelf",
                media_status="DROP TABLE file_index;", limit=10,
            )
            assert [e.name for e in got] == ["Book x"]
            await conn.close()
        import asyncio
        asyncio.get_event_loop().run_until_complete(go())


class TestFTSPrefixQuery:
    """Search-as-you-type correctness: every keystroke must yield
    results, not wait for a complete token. FTS5 MATCH is full-token
    by default, so we rewrite user input into prefix-star syntax."""

    def test_single_short_token_becomes_prefix(self):
        from augmentum.vfs.index import _build_fts_query
        # Typing 'ra' should match 'Ranger's Apprentice' via the 'ranger'
        # FTS token — without prefix search, it returns zero rows.
        assert _build_fts_query("ra") == "ra*"

    def test_multiword_each_token_prefixed(self):
        from augmentum.vfs.index import _build_fts_query
        # Spaces become implicit AND in FTS5; each part gets its own *.
        assert _build_fts_query("ranger app") == "ranger* app*"

    def test_apostrophe_becomes_space(self):
        from augmentum.vfs.index import _build_fts_query
        # "ranger's" with the default tokenizer indexes ["ranger", "s"] —
        # we strip the apostrophe so the user's literal string matches the
        # tokenizer's view of the indexed text.
        assert _build_fts_query("ranger's") == "ranger* s*"

    def test_empty_returns_empty(self):
        from augmentum.vfs.index import _build_fts_query
        assert _build_fts_query("") == ""
        assert _build_fts_query("   ") == ""

    def test_fts5_operators_stripped(self):
        """Typing ", -, or * must not syntax-error the whole query —
        FTS5 treats them as boolean operators. We strip rather than
        escape so the user's intent (a literal character) matches the
        indexed tokens without surprise operator behaviour."""
        from augmentum.vfs.index import _build_fts_query
        assert _build_fts_query('sanderson"') == "sanderson*"
        assert _build_fts_query("brandon -sanderson") == "brandon* sanderson*"
        assert _build_fts_query("ed(war") == "ed* war*"


# --- media_context.inject_media_context -------------------------------


class TestMediaContextInjection:
    """The media-context header → system-prefix injection. Keeps the
    "what am I listening to?" grounding small-model-safe (pure string
    formatting, no LLM calls)."""

    def _call(self, header_value: str | None, mode: str = "becca_direct"):
        from augmentum.proxy.media_context import inject_media_context

        class _Req:
            headers = {}
            def __init__(self, h):
                self.headers = {} if h is None else {"X-Augmentum-Media-Context": h}

        class _InternalReq:
            def __init__(self):
                self.messages = [{"role": "user", "content": "hi"}]

        ireq = _InternalReq()
        inject_media_context(ireq, _Req(header_value), mode=mode)
        # The injector prepends a pydantic Message; normalize to dicts so
        # assertions read uniformly whether a message was injected or not.
        return [
            m if isinstance(m, dict) else {"role": m.role, "content": m.content}
            for m in ireq.messages
        ]

    def test_non_companion_modes_never_inject(self):
        """Companion-scoped: narrative/passthrough/etc. must NOT get the
        media prefix — it polluted RP reasoning and broke the KV prefix
        (system index 0 changing every turn)."""
        import json
        payload = json.dumps({"title": "Mistborn", "isPlaying": True, "currentTimeS": 60})
        for mode in ("narrative", "passthrough", "analytical", "agentic", "coder", ""):
            msgs = self._call(payload, mode=mode)
            assert msgs == [{"role": "user", "content": "hi"}], f"leaked into {mode!r}"

    def test_no_header_is_noop(self):
        msgs = self._call(None)
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_malformed_json_no_crash(self):
        msgs = self._call("{not json")
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_missing_title_no_prepend(self):
        msgs = self._call('{"author": "x"}')
        assert len(msgs) == 1

    def test_happy_path_prepends_system(self):
        import json
        msgs = self._call(json.dumps({
            "fileId":        "fi_1",
            "title":         "Mistborn",
            "author":        "Brandon Sanderson",
            "chapterIdx":    4,
            "chapterTitle":  "A Heist Gone Wrong",
            "currentTimeS":  3725,  # 1h 2m 5s
            "isPlaying":     True,
        }))
        assert len(msgs) == 2
        sys = msgs[0]
        assert sys["role"] == "system"
        # Key facts must all be present in the one sentence.
        assert "Mistborn" in sys["content"]
        assert "Brandon Sanderson" in sys["content"]
        assert "chapter 5" in sys["content"]             # 0-indexed → 1-indexed
        assert "A Heist Gone Wrong" in sys["content"]
        assert "1:02:05" in sys["content"]
        assert "playing" in sys["content"]

    def test_paused_state_surfaces(self):
        import json
        msgs = self._call(json.dumps({
            "title": "Project Hail Mary", "isPlaying": False, "currentTimeS": 120,
        }))
        assert "paused" in msgs[0]["content"]
        assert "2:00" in msgs[0]["content"]
