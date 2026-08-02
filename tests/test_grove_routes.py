"""Tests for grove_routes.py — stations, favorites, vitals."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


class TestSomaStations:
    def test_soma_stations(self, client):
        resp = client.get("/api/grove/stations/soma")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert data[0]["source"] == "somafm"
        assert "url" in data[0]


class TestRadioStations:
    def test_radio_search(self, client, monkeypatch):
        # Mock the radio station fetcher to avoid real HTTP calls
        async def _mock_fetch(q, tag, limit):
            return [{"id": "test_station", "name": "Test FM", "url": "http://test.fm/stream"}]
        monkeypatch.setattr(
            "augmentum.proxy.grove_routes._fetch_radio_stations",
            _mock_fetch,
        )
        resp = client.get("/api/grove/stations/radio?q=jazz&limit=5")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestUnifiedSearch:
    def test_search_no_query(self, client, monkeypatch):
        async def _mock_fetch(q, tag, limit):
            return []
        monkeypatch.setattr(
            "augmentum.proxy.grove_routes._fetch_radio_stations",
            _mock_fetch,
        )
        resp = client.get("/api/grove/stations/search")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # With no query, all SomaFM stations returned
        assert len(data) > 0


class TestFavorites:
    def test_get_favorites_no_store(self, client):
        resp = client.get("/api/grove/favorites")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_save_favorites_no_store(self, client):
        resp = client.post("/api/grove/favorites", json=[{"id": "test"}])
        assert resp.status_code == 503

    def test_save_favorites_is_per_user(self, app, client):
        """Authenticated saves go to the per-user store (set_user), never the
        install-wide row — otherwise tenants overwrite each other's favorites.
        The `client` fixture is authenticated as test_user (usr_test)."""
        store = MagicMock()
        store.set_user = AsyncMock()
        store.set = AsyncMock()
        app.state.settings_store = store
        resp = client.post("/api/grove/favorites", json=[{"id": "groovesalad"}])
        assert resp.status_code == 200
        store.set_user.assert_called_once_with(
            "usr_test", "soundscape_favorites", json.dumps([{"id": "groovesalad"}])
        )
        store.set.assert_not_called()

    def test_get_favorites_reads_per_user(self, app, client):
        """Reads resolve per-user (with fallback to the legacy global value)."""
        store = MagicMock()
        store.get_user_or_global = AsyncMock(
            return_value=json.dumps([{"id": "dronezone"}])
        )
        app.state.settings_store = store
        resp = client.get("/api/grove/favorites")
        assert resp.status_code == 200
        assert resp.json() == [{"id": "dronezone"}]
        store.get_user_or_global.assert_called_once_with(
            "usr_test", "soundscape_favorites"
        )


class TestAmbientFavorites:
    def test_get_ambient_favorites_no_store(self, client):
        resp = client.get("/api/grove/ambient-favorites")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_save_ambient_favorites_no_store(self, client):
        resp = client.post("/api/grove/ambient-favorites", json=[])
        assert resp.status_code == 503

    def test_save_ambient_favorites_is_per_user(self, app, client):
        store = MagicMock()
        store.set_user = AsyncMock()
        store.set = AsyncMock()
        app.state.settings_store = store
        resp = client.post("/api/grove/ambient-favorites", json=[{"v": "abc"}])
        assert resp.status_code == 200
        store.set_user.assert_called_once_with(
            "usr_test", "ambient_favorites", json.dumps([{"v": "abc"}])
        )
        store.set.assert_not_called()


class TestYouTubeSearch:
    def test_youtube_search_error(self, client, monkeypatch):
        # Mock httpx to fail
        import httpx
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("fail"))
        monkeypatch.setattr("httpx.AsyncClient", lambda **kw: mock_client)
        resp = client.get("/api/grove/youtube/search?q=lofi")
        assert resp.status_code == 502


class TestSystemVitals:
    def test_vitals_no_ledger(self, client):
        resp = client.get("/api/system/vitals")
        assert resp.status_code == 503

    def test_vitals_success(self, app, client):
        snap = MagicMock(
            gpu_total_mb=8192, gpu_used_mb=4096,
            ram_total_mb=32768, ram_used_mb=16384,
            gpu_name="GPU-B",
            models=[],
        )
        ledger = MagicMock()
        ledger.collect = AsyncMock(return_value=snap)
        app.state.resource_ledger = ledger
        resp = client.get("/api/system/vitals")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gpu_name"] == "GPU-B"
        assert data["gpu_pct"] == 50.0
