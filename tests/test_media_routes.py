"""Behavior tests for media routes (/api/media/*).

Scope note: media_routes is 2957 lines spanning 27 endpoints, many of
which are thin proxies over external media servers (Emby/Jellyfin/ABS/
Komga/Suwayomi/LibriVox). Those paths require HTTP mocking against
each provider's API shape — appropriate for a dedicated integration
suite, out of scope for the route-contract layer tested here.

This file covers the subset of endpoints whose correctness doesn't
depend on upstream provider behavior:

* Server CRUD metadata  (list/update/delete)
* Track-selection persistence (/selection)
* User isolation on every path

Deferred (need provider HTTP mocking):
* POST /servers            — provider login round-trip
* POST /servers/{id}/test  — provider verify_token
* POST /servers/{id}/sync  — jobs + provider sync
* GET  /stream             — streaming proxy
* GET  /subtitle, /comic/*, /cover, /details, /related
* GET  /browse/librivox
* POST /outputs/{id}/remote-play, /remote-command
* POST /progress           — provider progress push (LibriVox path is stubbed
                              through the same code; testing just the local-
                              cache write would be misleading partial coverage)
* POST /pin, DELETE /pin   — gutenberg fetch job + vfs register
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TEST_USER_ID = "usr_test"


@pytest.fixture
def media_client(app):
    """Client with real MediaServerStore + FileIndexService."""
    from augmentum.media.store import MediaServerStore
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager
    from augmentum.vfs.index import FileIndexService

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    app.state.state_manager = StateManager(backend)
    app.state.file_index = FileIndexService(backend._conn)
    app.state.media_server_store = MediaServerStore(backend._conn)

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, app.state.media_server_store, app.state.file_index
    _run(backend.close())


def _seed_server(store, *, user_id=TEST_USER_ID, provider="audiobookshelf",
                 name="Home ABS", base_url="http://abs.local:13378"):
    return _run(store.create(
        user_id=user_id, provider=provider, name=name,
        base_url=base_url, access_token="test-token",
    ))


def _seed_file(idx, *, user_id=TEST_USER_ID, source="audiobookshelf",
               name="book.m4b"):
    import secrets
    return _run(idx.register(
        user_id=user_id, source=source,
        source_id=f"ext-{secrets.token_hex(4)}",
        name=name, mime_type="audio/mp4", size_bytes=1024,
    ))


# ===========================================================================
# GET /api/media/servers
# ===========================================================================

class TestListServers:
    def test_empty_for_fresh_user(self, media_client):
        client, _, _ = media_client
        r = client.get("/api/media/servers")
        assert r.status_code == 200
        data = r.json()
        assert data["servers"] == []
        # Defaults dict is always returned (UI needs port suggestions)
        assert "defaults" in data

    def test_lists_own_servers(self, media_client):
        client, store, _ = media_client
        _seed_server(store, name="Server A", base_url="http://a:13378")
        _seed_server(store, name="Server B", base_url="http://b:13378")

        r = client.get("/api/media/servers")
        servers = r.json()["servers"]
        assert len(servers) == 2
        names = {s["name"] for s in servers}
        assert {"Server A", "Server B"} == names

    def test_isolates_other_users(self, media_client):
        """Cross-tenant leak here exposes server URLs + base_urls for
        another user's personal media libraries."""
        client, store, _ = media_client
        _seed_server(store, user_id=TEST_USER_ID, name="Mine",
                     base_url="http://mine:13378")
        _seed_server(store, user_id="usr_other", name="Theirs",
                     base_url="http://theirs:13378")

        r = client.get("/api/media/servers")
        servers = r.json()["servers"]
        assert len(servers) == 1
        assert servers[0]["name"] == "Mine"


# ===========================================================================
# PUT /api/media/servers/{server_id}  — name/url update without cred swap
# ===========================================================================

class TestUpdateServer:
    def test_updates_name(self, media_client):
        client, store, _ = media_client
        server = _seed_server(store, name="Old Name")
        r = client.put(
            f"/api/media/servers/{server.id}",
            json={"name": "New Name"},
        )
        assert r.status_code == 200
        assert r.json()["server"]["name"] == "New Name"
        reloaded = _run(store.get(server.id, user_id=TEST_USER_ID))
        assert reloaded.name == "New Name"

    def test_update_missing_returns_404(self, media_client):
        client, _, _ = media_client
        r = client.put("/api/media/servers/ms_ghost", json={"name": "x"})
        assert r.status_code == 404

    def test_cannot_update_other_users_server(self, media_client):
        client, store, _ = media_client
        other = _seed_server(store, user_id="usr_other", name="Protected")
        r = client.put(
            f"/api/media/servers/{other.id}",
            json={"name": "Hijacked"},
        )
        assert r.status_code == 404
        # Untouched from the owner's perspective
        reloaded = _run(store.get(other.id, user_id="usr_other"))
        assert reloaded.name == "Protected"


# ===========================================================================
# DELETE /api/media/servers/{server_id}
# ===========================================================================

class TestDeleteServer:
    def test_deletes_own_server(self, media_client):
        client, store, _ = media_client
        server = _seed_server(store, name="Drop me")

        r = client.delete(f"/api/media/servers/{server.id}")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"
        assert _run(store.get(server.id, user_id=TEST_USER_ID)) is None

    def test_missing_server_returns_404(self, media_client):
        client, _, _ = media_client
        r = client.delete("/api/media/servers/ms_ghost")
        assert r.status_code == 404

    def test_cannot_delete_other_users_server(self, media_client):
        client, store, _ = media_client
        other = _seed_server(store, user_id="usr_other", name="Protected")
        r = client.delete(f"/api/media/servers/{other.id}")
        assert r.status_code == 404
        # Still alive from the owner's perspective
        assert _run(store.get(other.id, user_id="usr_other")) is not None


# ===========================================================================
# POST /api/media/selection/{file_id}
# ===========================================================================

class TestPlaybackSelection:
    def test_persists_audio_and_subtitle_choice(self, media_client):
        client, _, idx = media_client
        file_id = _seed_file(idx)
        body = {
            "media_source_id": "stream-hq",
            "audio_stream_index": 2,
            "subtitle_stream_index": 5,
        }
        r = client.post(f"/api/media/selection/{file_id}", json=body)
        assert r.status_code == 200
        sel = r.json()["selection"]
        assert sel["media_source_id"] == "stream-hq"
        assert sel["audio_stream_index"] == 2
        assert sel["subtitle_stream_index"] == 5

        # Source-of-truth: FileEntry.source_metadata was updated
        entry = _run(idx.get(file_id, user_id=TEST_USER_ID))
        assert entry.source_metadata.get("preferred_media_source_id") == "stream-hq"
        assert entry.source_metadata.get("preferred_audio_stream_index") == 2
        assert entry.source_metadata.get("preferred_subtitle_stream_index") == 5

    def test_null_subtitle_clears_prior_selection(self, media_client):
        client, _, idx = media_client
        file_id = _seed_file(idx)
        client.post(f"/api/media/selection/{file_id}", json={
            "media_source_id": "s1", "audio_stream_index": 1,
            "subtitle_stream_index": 7,
        })
        # Re-post with subtitle cleared
        r = client.post(f"/api/media/selection/{file_id}", json={
            "media_source_id": "s1", "audio_stream_index": 1,
            "subtitle_stream_index": None,
        })
        assert r.status_code == 200
        entry = _run(idx.get(file_id, user_id=TEST_USER_ID))
        assert "preferred_subtitle_stream_index" not in entry.source_metadata

    def test_missing_file_returns_404(self, media_client):
        client, _, _ = media_client
        r = client.post("/api/media/selection/fi_ghost", json={
            "media_source_id": "s", "audio_stream_index": 0,
            "subtitle_stream_index": 0,
        })
        assert r.status_code == 404

    def test_cannot_update_other_users_file(self, media_client):
        """ID-guessing attack: another user's file ID should 404, not
        accept the write into their source_metadata."""
        client, _, idx = media_client
        other_id = _seed_file(idx, user_id="usr_other", name="theirs.m4b")
        r = client.post(f"/api/media/selection/{other_id}", json={
            "media_source_id": "hijack", "audio_stream_index": 99,
            "subtitle_stream_index": 99,
        })
        assert r.status_code == 404
        # Untouched from the owner's perspective
        entry = _run(idx.get(other_id, user_id="usr_other"))
        assert "preferred_media_source_id" not in entry.source_metadata


# ===========================================================================
# GET /api/media/receiver-profiles + GET /api/media/outputs/{id}/launch-plan
# ===========================================================================

class TestReceiverPlanning:
    def test_receiver_profiles_lists_known_profiles(self, media_client):
        client, _, _ = media_client

        r = client.get("/api/media/receiver-profiles")

        assert r.status_code == 200
        data = r.json()
        profile_ids = {row["id"] for row in data["profiles"]}
        assert {"cast_video", "dlna_generic_video"} <= profile_ids

    def test_launch_plan_delegates_to_receiver_planner(self, media_client, monkeypatch):
        client, store, idx = media_client
        server = _seed_server(
            store,
            provider="emby",
            name="Living Room Emby",
            base_url="http://emby.local:8096",
        )
        file_id = _run(idx.register(
            user_id=TEST_USER_ID,
            source="emby",
            source_id="msrc:movie_1",
            name="Movie Night",
            mime_type="video/mp4",
            source_metadata={
                "server_id": server.id,
                "external_id": "movie_1",
                "stream_path": "/Videos/movie_1/stream.mp4",
                "entity_kind": "movie",
                "current_time_s": 42.5,
            },
        ))

        class _FakeProvider:
            name = "emby"

        async def _fake_build_receiver_launch_plan(**kwargs):
            assert kwargs["server"].id == server.id
            assert kwargs["file_id"] == file_id
            assert kwargs["entry_name"] == "Movie Night"
            assert kwargs["cached_meta"]["external_id"] == "movie_1"
            assert kwargs["receiver_profile_id"] == "dlna_generic_video"
            return type("Plan", (), {
                "to_dict": lambda self: {
                    "supported": True,
                    "receiver_profile": "dlna_generic_video",
                    "receiver_kind": "dlna",
                    "control_plane": "dlna_avtransport",
                    "title": "Movie Night",
                },
            })()

        monkeypatch.setattr(
            "augmentum.proxy.media_routes._provider_client",
            lambda provider, http_client: _FakeProvider(),
        )
        monkeypatch.setattr(
            "augmentum.proxy.media_routes.build_receiver_launch_plan",
            _fake_build_receiver_launch_plan,
        )

        r = client.get(
            f"/api/media/outputs/{file_id}/launch-plan?receiver_profile=dlna_generic_video",
        )

        assert r.status_code == 200
        data = r.json()
        assert data["supported"] is True
        assert data["receiver_profile"] == "dlna_generic_video"
        assert data["receiver_kind"] == "dlna"

    def test_cast_load_uses_cast_receiver_profile(self, media_client, monkeypatch):
        client, store, idx = media_client
        server = _seed_server(
            store,
            provider="jellyfin",
            name="Jellyfin",
            base_url="http://jf.local:8096",
        )
        file_id = _run(idx.register(
            user_id=TEST_USER_ID,
            source="jellyfin",
            source_id="msrc:episode_9",
            name="Episode 9",
            mime_type="video/mp4",
            source_metadata={
                "server_id": server.id,
                "external_id": "episode_9",
                "stream_path": "/Videos/episode_9/stream",
                "entity_kind": "episode",
            },
        ))

        class _FakeProvider:
            name = "jellyfin"

        async def _fake_build_receiver_launch_plan(**kwargs):
            assert kwargs["receiver_profile_id"] == "cast_video"
            return type("Plan", (), {
                "to_dict": lambda self: {
                    "supported": True,
                    "receiver_profile": "cast_video",
                    "content_url": "http://jf.local:8096/Videos/episode_9/stream",
                    "content_type": "video/mp4",
                    "title": "Episode 9",
                },
            })()

        monkeypatch.setattr(
            "augmentum.proxy.media_routes._provider_client",
            lambda provider, http_client: _FakeProvider(),
        )
        monkeypatch.setattr(
            "augmentum.proxy.media_routes.build_receiver_launch_plan",
            _fake_build_receiver_launch_plan,
        )

        r = client.get(f"/api/media/outputs/{file_id}/cast-load")

        assert r.status_code == 200
        data = r.json()
        assert data["supported"] is True
        assert data["receiver_profile"] == "cast_video"
        assert data["content_type"] == "video/mp4"

    def test_transport_play_starts_dlna_session(self, media_client, monkeypatch):
        client, store, idx = media_client
        server = _seed_server(
            store,
            provider="emby",
            name="Emby",
            base_url="http://emby.local:8096",
        )
        file_id = _run(idx.register(
            user_id=TEST_USER_ID,
            source="emby",
            source_id="msrc:movie_dlna",
            name="Movie DLNA",
            mime_type="video/mp4",
            source_metadata={
                "server_id": server.id,
                "external_id": "movie_dlna",
                "stream_path": "/Videos/movie_dlna/stream",
                "entity_kind": "movie",
            },
        ))

        class _FakeProvider:
            name = "emby"

        receiver = type("Receiver", (), {
            "receiver_id": "dlna_1",
            "label": "Living Room TV",
            "receiver_profile": "dlna_generic_video",
            "supported_commands": ["PlayPause", "Pause", "Unpause", "Stop", "Seek", "SetVolume"],
        })()

        async def _fake_discover(http_client):
            return [receiver]

        async def _fake_plan(**kwargs):
            return type("Plan", (), {
                "supported": True,
                "title": "Movie DLNA",
                "poster_url": "/img/poster.jpg",
                "to_dict": lambda self: {
                    "supported": True,
                    "receiver_profile": "dlna_generic_video",
                    "title": "Movie DLNA",
                },
            })()

        async def _fake_launch(http_client, found_receiver, plan):
            assert found_receiver.receiver_id == "dlna_1"
            assert plan.title == "Movie DLNA"
            return True

        async def _fake_snapshot(http_client, found_receiver):
            assert found_receiver.receiver_id == "dlna_1"
            return {
                "current_time_s": 12.0,
                "duration_s": 120.0,
                "is_paused": False,
                "is_muted": False,
                "can_seek": True,
                "volume_level": 30,
                "supported_commands": ["PlayPause", "Pause", "Unpause", "Stop", "Seek", "SetVolume"],
                "receiver_state": "PLAYING",
            }

        monkeypatch.setattr(
            "augmentum.proxy.media_routes._provider_client",
            lambda provider, http_client: _FakeProvider(),
        )
        monkeypatch.setattr("augmentum.proxy.media_routes.discover_dlna_receivers", _fake_discover)
        monkeypatch.setattr("augmentum.proxy.media_routes.build_receiver_launch_plan", _fake_plan)
        monkeypatch.setattr("augmentum.proxy.media_routes.launch_dlna_receiver", _fake_launch)
        monkeypatch.setattr("augmentum.proxy.media_routes.snapshot_dlna_receiver", _fake_snapshot)

        r = client.post(
            f"/api/media/outputs/{file_id}/transport-play",
            json={
                "transport": "dlna",
                "receiver_id": "dlna_1",
                "receiver_profile": "dlna_generic_video",
            },
        )

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["transport"] == "dlna"
        assert data["session"]["transport_kind"] == "dlna"
        assert data["session"]["receiver_label"] == "Living Room TV"


# ===========================================================================
# Router sanity
# ===========================================================================

class TestRouterShape:
    def test_prefix(self):
        from augmentum.proxy.media_routes import router
        assert router.prefix == "/api/media"

    def test_expected_endpoints_registered(self):
        from augmentum.proxy.media_routes import router
        paths = {r.path for r in router.routes}
        expected = {
            "/api/media/servers",
            "/api/media/servers/{server_id}",
            "/api/media/servers/{server_id}/test",
            "/api/media/servers/{server_id}/sync",
            "/api/media/receiver-profiles",
            "/api/media/outputs/{file_id}",
            "/api/media/outputs/{file_id}/launch-plan",
            "/api/media/outputs/{file_id}/cast-load",
            "/api/media/outputs/{file_id}/transport-play",
            "/api/media/transport-sessions/{session_id}",
            "/api/media/transport-sessions/{session_id}/playstate",
            "/api/media/transport-sessions/{session_id}/general",
            "/api/media/progress/{file_id}",
            "/api/media/selection/{file_id}",
            "/api/media/stream/{file_id}",
            "/api/media/detect",
        }
        assert expected.issubset(paths)


class TestProgressPushSupersession:
    """Per-key supersession contract for ``_schedule_progress_push``.

    The endpoint fires upstream pushes (Audiobookshelf/Emby/Jellyfin) as
    background tasks so a slow upstream can't block the player's 5s
    polling. Per-key supersession is the load-bearing guarantee: if a
    new push arrives while one is still mid-HTTP-roundtrip, the prior
    one is cancelled because the newer position is what should reach
    upstream anyway. Without this, a wedged upstream would let tasks
    accumulate one-per-poll until the request timeout fired.
    """

    def setup_method(self):
        from augmentum.proxy.media_routes import _inflight_progress_pushes
        _inflight_progress_pushes.clear()

    @pytest.mark.asyncio
    async def test_supersedes_in_flight_for_same_key(self):
        """A second push for the same (user, file) cancels the first."""
        from augmentum.proxy.media_routes import (
            _inflight_progress_pushes,
            _schedule_progress_push,
        )

        slow_started = asyncio.Event()
        slow_cancelled = False

        async def _slow():
            nonlocal slow_cancelled
            slow_started.set()
            try:
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                slow_cancelled = True
                raise

        async def _fast():
            await asyncio.sleep(0)

        _schedule_progress_push("usr_a", "fi_1", _slow())
        await slow_started.wait()
        _schedule_progress_push("usr_a", "fi_1", _fast())
        # Yield enough to let the new task run + cancellation propagate.
        for _ in range(5):
            await asyncio.sleep(0)
        assert slow_cancelled
        # Only the latest task is left in the dict; it completes cleanly.
        await asyncio.sleep(0)
        assert ("usr_a", "fi_1") not in _inflight_progress_pushes

    @pytest.mark.asyncio
    async def test_different_keys_run_in_parallel(self):
        """Pushes for different (user, file) keys do NOT cancel each other."""
        from augmentum.proxy.media_routes import _schedule_progress_push

        a_done = asyncio.Event()
        b_done = asyncio.Event()

        async def _a():
            await asyncio.sleep(0)
            a_done.set()

        async def _b():
            await asyncio.sleep(0)
            b_done.set()

        _schedule_progress_push("usr_a", "fi_1", _a())
        _schedule_progress_push("usr_a", "fi_2", _b())
        await asyncio.wait_for(a_done.wait(), timeout=1.0)
        await asyncio.wait_for(b_done.wait(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_exception_in_push_does_not_propagate(self):
        """A failed upstream push must not propagate to the calling request.
        The endpoint already returned 200 from the local write — an HTTP
        failure to upstream is a background concern. Without the
        done-callback, asyncio prints an unhandled-exception warning at
        GC time and the contract relies on luck.
        """
        from augmentum.proxy.media_routes import (
            _inflight_progress_pushes,
            _schedule_progress_push,
        )

        async def _boom():
            raise RuntimeError("upstream 503")

        _schedule_progress_push("usr_a", "fi_1", _boom())
        # Drain to completion. The fact that this line is reached at all
        # — rather than the exception bubbling out of the scheduler —
        # is the contract under test.
        for _ in range(5):
            await asyncio.sleep(0)
        assert ("usr_a", "fi_1") not in _inflight_progress_pushes
