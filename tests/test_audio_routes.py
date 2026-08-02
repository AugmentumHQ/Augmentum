"""Tests for audio_routes.py — TTS, STT, provider CRUD, voices."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def _setup_audio_db(app):
    """Set up the audio_providers table in the SQLite backend."""
    import asyncio
    sm = app.state.state_manager
    backend = sm.backend
    conn = backend.conn

    async def _create():
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_providers (
                id TEXT PRIMARY KEY,
                provider_type TEXT NOT NULL,
                name TEXT NOT NULL,
                base_url TEXT NOT NULL,
                api_key TEXT,
                default_model TEXT DEFAULT '',
                default_voice TEXT DEFAULT '',
                is_enabled INTEGER DEFAULT 1,
                is_default INTEGER DEFAULT 0,
                tts_chunking TEXT DEFAULT 'sentence'
            )
        """)
        await conn.commit()

    asyncio.get_event_loop().run_until_complete(_create())
    return conn


class TestProviderCRUD:
    def test_list_providers_empty(self, sqlite_client, app):
        _setup_audio_db(app)
        resp = sqlite_client.get("/api/audio/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_create_provider(self, sqlite_client, app):
        _setup_audio_db(app)
        resp = sqlite_client.post(
            "/api/audio/providers",
            json={
                "id": "kokoro",
                "provider_type": "tts",
                "name": "Kokoro TTS",
                "base_url": "http://localhost:8880",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "kokoro"

    def test_create_provider_invalid_type(self, sqlite_client, app):
        _setup_audio_db(app)
        resp = sqlite_client.post(
            "/api/audio/providers",
            json={
                "id": "bad",
                "provider_type": "invalid",
                "name": "Bad",
                "base_url": "http://localhost:1234",
            },
        )
        assert resp.status_code == 422  # Pydantic validation

    def test_delete_provider_not_found(self, sqlite_client, app):
        _setup_audio_db(app)
        resp = sqlite_client.delete("/api/audio/providers/nonexistent")
        assert resp.status_code == 404


class TestTTSSpeech:
    def test_tts_no_provider(self, sqlite_client, app):
        _setup_audio_db(app)
        resp = sqlite_client.post(
            "/v1/audio/speech",
            json={"input": "Hello world", "voice": "alloy"},
        )
        # No provider configured — expect 503 or 404
        assert resp.status_code in (404, 503)


class TestVoiceList:
    def test_list_voices_no_providers(self, sqlite_client, app):
        _setup_audio_db(app)
        resp = sqlite_client.get("/api/audio/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestBundledProviders:
    def test_list_bundled(self, client):
        resp = client.get("/api/audio/providers/bundled")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


class TestWebUIConfig:
    def test_get_webui(self, sqlite_client, app):
        _setup_audio_db(app)
        resp = sqlite_client.get("/api/audio/providers/webui")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
