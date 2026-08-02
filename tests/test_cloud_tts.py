"""Tests for cloud TTS/STT provider presets and Deepgram/ElevenLabs adapters."""

from __future__ import annotations

import pytest

from augmentum.proxy.audio_routes import (
    _KNOWN_VOICES,
    _build_headers,
    _fetch_voices_from_provider,
    _is_deepgram,
    _is_elevenlabs,
    _is_openai_tts,
)

# ---------------------------------------------------------------------------
# Provider detection helpers
# ---------------------------------------------------------------------------


class TestProviderDetection:
    """Tests for _is_deepgram, _is_elevenlabs, and _is_openai_tts helpers."""

    def test_is_deepgram_true(self):
        assert _is_deepgram("https://api.deepgram.com") is True
        assert _is_deepgram("https://api.deepgram.com/v1") is True

    def test_is_deepgram_false(self):
        assert _is_deepgram("https://api.openai.com/v1") is False
        assert _is_deepgram("https://localhost:8880") is False

    def test_is_deepgram_case_insensitive(self):
        assert _is_deepgram("https://API.DEEPGRAM.COM") is True

    def test_is_elevenlabs_true(self):
        assert _is_elevenlabs("https://api.elevenlabs.io/v1") is True
        assert _is_elevenlabs("https://api.elevenlabs.io") is True

    def test_is_elevenlabs_false(self):
        assert _is_elevenlabs("https://api.openai.com/v1") is False
        assert _is_elevenlabs("https://localhost:8880") is False

    def test_is_elevenlabs_case_insensitive(self):
        assert _is_elevenlabs("https://API.ELEVENLABS.IO/v1") is True

    def test_is_openai_true(self):
        assert _is_openai_tts("https://api.openai.com/v1") is True
        assert _is_openai_tts("https://api.openai.com") is True

    def test_is_openai_false(self):
        assert _is_openai_tts("https://api.deepgram.com") is False
        assert _is_openai_tts("https://localhost:8880") is False

    def test_is_openai_case_insensitive(self):
        assert _is_openai_tts("https://API.OPENAI.COM/v1") is True

    def test_providers_mutually_exclusive(self):
        """Each URL should match at most one provider."""
        urls = [
            "https://api.openai.com/v1",
            "https://api.elevenlabs.io/v1",
            "https://api.deepgram.com",
            "https://localhost:8880",
        ]
        for url in urls:
            matches = sum([_is_openai_tts(url), _is_elevenlabs(url), _is_deepgram(url)])
            assert matches <= 1, f"URL {url} matched multiple providers"


# ---------------------------------------------------------------------------
# Auth header format
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    """Tests for _build_headers with provider-specific auth."""

    def test_bearer_for_openai(self):
        h = _build_headers("sk-test", base_url="https://api.openai.com/v1")
        assert h["Authorization"] == "Bearer sk-test"

    def test_xi_api_key_for_elevenlabs(self):
        h = _build_headers("xi-test", base_url="https://api.elevenlabs.io/v1")
        assert h["xi-api-key"] == "xi-test"
        assert "Authorization" not in h

    def test_token_for_deepgram(self):
        h = _build_headers("dg-test", base_url="https://api.deepgram.com")
        assert h["Authorization"] == "Token dg-test"

    def test_no_key(self):
        h = _build_headers(None)
        assert "Authorization" not in h
        assert "xi-api-key" not in h

    def test_bearer_default_no_base_url(self):
        h = _build_headers("test-key")
        assert h["Authorization"] == "Bearer test-key"

    def test_bearer_for_generic(self):
        h = _build_headers("test-key", base_url="https://localhost:8880")
        assert h["Authorization"] == "Bearer test-key"


# ---------------------------------------------------------------------------
# Known voice lists
# ---------------------------------------------------------------------------


class TestKnownVoices:
    """Tests for hardcoded voice lists."""

    def test_openai_voices_13(self):
        voices = _KNOWN_VOICES["openai"]
        assert len(voices) == 13
        ids = {v["id"] for v in voices}
        assert "alloy" in ids
        assert "nova" in ids
        assert "shimmer" in ids
        # New voices
        assert "verse" in ids
        assert "marin" in ids
        assert "cedar" in ids

    def test_deepgram_voices_exist(self):
        voices = _KNOWN_VOICES["deepgram"]
        assert len(voices) >= 12
        ids = {v["id"] for v in voices}
        assert "aura-asteria-en" in ids
        assert "aura-zeus-en" in ids

    def test_voice_dicts_have_id_and_name(self):
        for provider, voices in _KNOWN_VOICES.items():
            for v in voices:
                assert "id" in v, f"Missing id in {provider} voice"
                assert "name" in v, f"Missing name in {provider} voice"

    def test_openai_voices_sorted(self):
        """Voices should be in alphabetical order for UI consistency."""
        voices = _KNOWN_VOICES["openai"]
        ids = [v["id"] for v in voices]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Voice fetching — hardcoded fallbacks
# ---------------------------------------------------------------------------


class TestFetchVoicesFromProvider:
    """Tests for _fetch_voices_from_provider with hardcoded voices."""

    @pytest.mark.asyncio
    async def test_openai_returns_hardcoded(self):
        provider = {
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "default_voice": "alloy",
        }
        voices = await _fetch_voices_from_provider(provider)
        assert len(voices) == len(_KNOWN_VOICES["openai"])
        assert voices[0]["id"] == "alloy"

    @pytest.mark.asyncio
    async def test_deepgram_returns_hardcoded(self):
        provider = {
            "base_url": "https://api.deepgram.com",
            "api_key": "dg-test",
            "default_voice": "aura-asteria-en",
        }
        voices = await _fetch_voices_from_provider(provider)
        assert len(voices) == len(_KNOWN_VOICES["deepgram"])
        assert voices[0]["id"] == "aura-asteria-en"

    @pytest.mark.asyncio
    async def test_hardcoded_returns_copies(self):
        """Ensure we get copies, not references to the shared list."""
        provider = {
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
        }
        voices = await _fetch_voices_from_provider(provider)
        voices[0]["id"] = "modified"
        assert _KNOWN_VOICES["openai"][0]["id"] == "alloy"

    @pytest.mark.asyncio
    async def test_elevenlabs_not_hardcoded(self):
        """ElevenLabs has a real /v1/voices API, so it shouldn't use hardcoded.
        When the API is unreachable, it falls back to default_voice."""
        provider = {
            "base_url": "https://api.elevenlabs.io/v1",
            "api_key": "xi-fake",
            "default_voice": "21m00Tcm4TlvDq8ikWAM",
        }
        # Will fail to connect, should fall back to default_voice
        voices = await _fetch_voices_from_provider(provider)
        assert len(voices) == 1
        assert voices[0]["id"] == "21m00Tcm4TlvDq8ikWAM"

    @pytest.mark.asyncio
    async def test_unknown_provider_fallback_to_default_voice(self):
        """Non-API providers fall back to default_voice when API fails."""
        provider = {
            "base_url": "https://localhost:99999",
            "api_key": None,
            "default_voice": "my-custom-voice",
        }
        voices = await _fetch_voices_from_provider(provider)
        assert len(voices) == 1
        assert voices[0]["id"] == "my-custom-voice"

    @pytest.mark.asyncio
    async def test_unknown_provider_no_default_voice(self):
        provider = {
            "base_url": "https://localhost:99999",
            "api_key": None,
            "default_voice": "",
        }
        voices = await _fetch_voices_from_provider(provider)
        assert voices == []
