"""Tests for youtube_routes.py — transcript and related videos."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


class TestTranscript:
    def test_invalid_video_id_empty(self, client):
        resp = client.get("/api/youtube/transcript?v=")
        assert resp.status_code == 400
        assert "Invalid video ID" in resp.json()["error"]

    def test_invalid_video_id_short(self, client):
        resp = client.get("/api/youtube/transcript?v=abc")
        assert resp.status_code == 400

    def test_transcript_library_missing(self, client, monkeypatch):
        # Mock the import to raise ImportError
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "youtube_transcript_api":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        # Also mock the http_client to skip oembed
        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=MagicMock(status_code=404))
        client.app.state.http_client = mock_http

        resp = client.get("/api/youtube/transcript?v=dQw4w9WgXcQ")
        assert resp.status_code == 503

    def test_transcript_cache_hit(self, client, monkeypatch):
        from augmentum.proxy import youtube_routes
        cached_data = {
            "video_id": "dQw4w9WgXcQ",
            "title": "Test Video",
            "channel": "Test",
            "transcript": [{"text": "hello", "start": 0.0, "duration": 1.0}],
            "paragraphs": [],
        }
        # Pre-populate cache
        youtube_routes._cache[("dQw4w9WgXcQ", "en")] = (
            cached_data,
            youtube_routes.monotonic(),
        )
        resp = client.get("/api/youtube/transcript?v=dQw4w9WgXcQ")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Test Video"
        # Cleanup
        youtube_routes._cache.clear()


class TestRelatedVideos:
    def test_related_no_query(self, client):
        resp = client.get("/api/youtube/related?q=")
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_related_no_http_client(self, app, client):
        app.state.http_client = None
        resp = client.get("/api/youtube/related?q=lofi")
        assert resp.status_code == 200
        assert resp.json()["results"] == []
