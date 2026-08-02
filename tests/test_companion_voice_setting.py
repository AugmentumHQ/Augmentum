"""Companion (Becca) voice setting — per-user, server-side.

Covers the two halves of the wiring:
  - ``_companion_voice_for_user`` resolution order in voice_routes:
    ui.companionVoice → ui.voiceDefaultVoice → "" (provider default)
  - /api/config/ui round-trip for the ``companionVoice`` key

Background: the companion widget opens /ws/voice without a voice param
and its config message never sets one, so before this setting existed
Becca sessions silently fell back to whichever audio_providers row
sorted first — ignoring the user's voice preferences entirely.
"""
from __future__ import annotations

import asyncio

import pytest

from augmentum.proxy.voice_routes import _companion_voice_for_user


class _FakeStore:
    def __init__(self, values: dict[str, str] | None = None, raises: bool = False):
        self._values = values or {}
        self._raises = raises
        self.calls: list[tuple[str, str]] = []

    async def get_user_or_global(self, user_id: str, key: str):
        self.calls.append((user_id, key))
        if self._raises:
            raise RuntimeError("store unavailable")
        return self._values.get(key)


class _FakeAppState:
    def __init__(self, store):
        self.settings_store = store


class TestCompanionVoiceResolution:
    @pytest.mark.asyncio
    async def test_companion_voice_wins_over_default(self):
        state = _FakeAppState(_FakeStore({
            "ui.companionVoice": "pockettts-builtin::alba",
            "ui.voiceDefaultVoice": "kokoro-builtin::af_heart",
        }))
        assert await _companion_voice_for_user(state, "usr_1") == "pockettts-builtin::alba"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_voice(self):
        state = _FakeAppState(_FakeStore({
            "ui.voiceDefaultVoice": "pockettts-builtin::fantine",
        }))
        assert await _companion_voice_for_user(state, "usr_1") == "pockettts-builtin::fantine"

    @pytest.mark.asyncio
    async def test_empty_when_nothing_configured(self):
        state = _FakeAppState(_FakeStore({}))
        assert await _companion_voice_for_user(state, "usr_1") == ""

    @pytest.mark.asyncio
    async def test_whitespace_value_skipped(self):
        state = _FakeAppState(_FakeStore({
            "ui.companionVoice": "   ",
            "ui.voiceDefaultVoice": "alba",
        }))
        assert await _companion_voice_for_user(state, "usr_1") == "alba"

    @pytest.mark.asyncio
    async def test_no_user_id_returns_empty(self):
        state = _FakeAppState(_FakeStore({"ui.companionVoice": "alba"}))
        assert await _companion_voice_for_user(state, "") == ""

    @pytest.mark.asyncio
    async def test_no_store_returns_empty(self):
        class _Bare:
            settings_store = None
        assert await _companion_voice_for_user(_Bare(), "usr_1") == ""

    @pytest.mark.asyncio
    async def test_store_error_degrades_to_empty(self):
        """A broken store must not take down the WS connect path."""
        state = _FakeAppState(_FakeStore(raises=True))
        assert await _companion_voice_for_user(state, "usr_1") == ""

    @pytest.mark.asyncio
    async def test_value_is_stripped(self):
        state = _FakeAppState(_FakeStore({
            "ui.companionVoice": "  pockettts-builtin::alba  ",
        }))
        assert await _companion_voice_for_user(state, "usr_1") == "pockettts-builtin::alba"


class TestCompanionVoiceRoundTrip:
    def test_companion_voice_round_trip(self, sqlite_client):
        """companionVoice persists via PUT /api/config/ui and reads back."""
        from augmentum.state.settings_store import SettingsStore

        app = sqlite_client.app
        backend = app.state.state_manager.backend
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            backend.conn.execute(
                "CREATE TABLE IF NOT EXISTS app_settings "
                "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
            )
        )
        loop.run_until_complete(backend.conn.commit())
        app.state.settings_store = SettingsStore(backend.conn)

        put_resp = sqlite_client.put(
            "/api/config/ui", json={"companionVoice": "pockettts-builtin::alba"},
        )
        assert put_resp.status_code == 200
        assert put_resp.json()["updated"]["companionVoice"] == "pockettts-builtin::alba"

        get_resp = sqlite_client.get("/api/config/ui")
        assert get_resp.status_code == 200
        assert get_resp.json()["companionVoice"] == "pockettts-builtin::alba"
