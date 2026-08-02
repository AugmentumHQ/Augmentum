"""Phase 3 — Subsonic client and provider_bridge hook tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.media.subsonic_client import (
    SearchResult,
    SubsonicAlbum,
    SubsonicClient,
    SubsonicError,
    SubsonicSong,
)


class TestSubsonicClient:
    """Unit tests for the Subsonic API client."""

    def test_auth_params_include_all_fields(self):
        client = SubsonicClient("http://localhost:4533", "admin", "secret")
        params = client._auth_params()
        assert "u" in params and params["u"] == "admin"
        assert "t" in params  # salt
        assert "s" in params  # token = md5(password + salt)
        assert params["v"] == "1.16.1"
        assert params["c"] == "Augmentum"
        assert params["f"] == "json"

    def test_rest_url_builds_correctly(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        url = client._rest_url("ping")
        assert url.startswith("http://nav:4533/rest/ping?")
        assert "u=u" in url
        assert "f=json" in url
        assert "v=1.16.1" in url

    def test_stream_url(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        url = client.stream_url("song-123")
        assert "rest/stream" in url
        assert "id=song-123" in url

    def test_cover_art_url(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        url = client.cover_art_url("cover-456", size=200)
        assert "rest/getCoverArt" in url
        assert "id=cover-456" in url
        assert "size=200" in url

    @pytest.mark.asyncio
    async def test_ping_success(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"status": "ok"}
            result = await client.ping()
            assert result is True
            mock_get.assert_awaited_once_with("ping")

    @pytest.mark.asyncio
    async def test_ping_failure_returns_false(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("connection refused")
            result = await client.ping()
            assert result is False

    @pytest.mark.asyncio
    async def test_search_parses_results(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        mock_resp = {
            "searchResult3": {
                "album": [
                    {"id": "a1", "name": "Kind of Blue", "artist": "Miles Davis",
                     "coverArt": "cv1", "songCount": 5, "year": 1959},
                ],
                "song": [
                    {"id": "s1", "title": "So What", "artist": "Miles Davis",
                     "album": "Kind of Blue", "albumId": "a1", "coverArt": "cv1",
                     "duration": 561, "track": 1, "year": 1959, "contentType": "audio/flac"},
                ],
            },
        }
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            result = await client.search("kind of blue")
            assert len(result.albums) == 1
            assert result.albums[0].name == "Kind of Blue"
            assert len(result.songs) == 1
            assert result.songs[0].title == "So What"
            assert result.songs[0].artist == "Miles Davis"

    @pytest.mark.asyncio
    async def test_search_empty_result(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"searchResult3": {}}
            result = await client.search("nonexistent")
            assert result.albums == []
            assert result.songs == []

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("timeout")
            result = await client.search("test")
            assert result.albums == []
            assert result.songs == []

    @pytest.mark.asyncio
    async def test_get_random_songs(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        mock_resp = {
            "randomSongs": {
                "song": [
                    {"id": "s1", "title": "Track 1", "artist": "Artist",
                     "album": "Album", "albumId": "a1", "coverArt": "cv1",
                     "duration": 200, "track": 1, "year": 2020, "contentType": "audio/mpeg"},
                ],
            },
        }
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            songs = await client.get_random_songs(size=5)
            assert len(songs) == 1
            assert songs[0].title == "Track 1"

    @pytest.mark.asyncio
    async def test_get_starred(self):
        client = SubsonicClient("http://nav:4533", "u", "p")
        mock_resp = {
            "starred2": {
                "song": [
                    {"id": "s1", "title": "Fave", "artist": "Artist",
                     "album": "Album", "albumId": "a1", "coverArt": "cv1",
                     "duration": 180, "track": 1, "year": 2020, "contentType": "audio/mpeg"},
                ],
            },
        }
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_resp
            songs = await client.get_starred()
            assert len(songs) == 1
            assert songs[0].title == "Fave"

    @pytest.mark.asyncio
    async def test_ping_returns_false_on_error_status(self):
        """A non-ok status from the server returns False from ping."""
        client = SubsonicClient("http://nav:4533", "u", "p")
        with patch.object(client, "_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = SubsonicError("Wrong username or password")
            result = await client.ping()
            assert result is False


class TestProviderBridgeHook:
    """The provider_bridge hook is no longer a stub — it handles subsonic."""

    def test_hook_is_registered(self):
        from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
        assert "provider_bridge" in KNOWN_INTEGRATION_HOOKS
        install_fn = KNOWN_INTEGRATION_HOOKS["provider_bridge"][0]
        uninstall_fn = KNOWN_INTEGRATION_HOOKS["provider_bridge"][1]
        assert callable(install_fn)
        assert callable(uninstall_fn)

    @pytest.mark.asyncio
    async def test_unknown_protocol_logs_and_continues(self):
        """A protocol we don't handle is a no-op, not an error."""
        manifest = MagicMock()
        manifest.service_id = "test"
        manifest.integration = {"provider_bridge": {"protocol": "unknown_proto"}}
        sd = MagicMock()
        request = MagicMock()

        from augmentum.marketplace.hooks import KNOWN_INTEGRATION_HOOKS
        install_fn = KNOWN_INTEGRATION_HOOKS["provider_bridge"][0]
        # Must not raise
        await install_fn(request, manifest, sd, "user-1")


class TestSubsonicDataclasses:
    def test_search_result_defaults(self):
        sr = SearchResult()
        assert sr.albums == []
        assert sr.songs == []

    def test_album_fields(self):
        a = SubsonicAlbum(id="a1", name="Test", artist="Artist",
                          cover_id="c1", song_count=10, year=2020)
        assert a.name == "Test"
        assert a.song_count == 10

    def test_song_fields(self):
        s = SubsonicSong(id="s1", title="Track", artist="Artist",
                         album="Album", album_id="a1", cover_id="c1",
                         duration=240, track=3, year=2020,
                         content_type="audio/flac")
        assert s.title == "Track"
        assert s.duration == 240
        assert s.content_type == "audio/flac"
