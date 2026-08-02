"""Tests for augmentum/voice/pipeline_resolver.py."""

from __future__ import annotations

import pytest

from augmentum.voice.pipeline_resolver import ResolverError, resolve


class TestServerOnlyDefault:
    """With no client caps, every policy except 'local' resolves to server."""

    def test_auto_no_caps_is_server(self):
        assert resolve("vad", "call", client_caps=None, policy="auto") == "server"

    def test_auto_empty_caps_is_server(self):
        assert resolve("stt", "call", client_caps={}, policy="auto") == "server"

    def test_auto_caps_missing_component_is_server(self):
        # Client has TTS but not VAD — VAD still goes to server.
        caps = {"tts": ["pocket-onnx-wasm"]}
        assert resolve("vad", "call", client_caps=caps, policy="auto") == "server"

    def test_server_policy_always_server(self):
        caps = {"vad": ["silero-wasm"]}
        assert resolve("vad", "call", client_caps=caps, policy="server") == "server"

    def test_custom_policy_returns_server(self):
        # Custom defers to legacy routing — resolver returns 'server' so
        # caller's existing dispatch proceeds.
        caps = {"vad": ["silero-wasm"]}
        assert resolve("vad", "call", client_caps=caps, policy="custom") == "server"


class TestClientDispatch:
    def test_auto_with_capable_client(self):
        caps = {"vad": ["silero-wasm"]}
        assert resolve("vad", "call", client_caps=caps, policy="auto") == "client:silero-wasm"

    def test_auto_picks_first_engine(self):
        caps = {"stt": ["moonshine-wasm", "whisper-tiny-wasm"]}
        result = resolve("stt", "call", client_caps=caps, policy="auto")
        assert result == "client:moonshine-wasm"

    def test_local_with_capable_client(self):
        caps = {"vad": ["silero-wasm"]}
        assert resolve("vad", "call", client_caps=caps, policy="local") == "client:silero-wasm"


class TestLocalRequiresCapability:
    """policy='local' raises if client did not advertise the engine."""

    def test_local_no_caps_raises(self):
        with pytest.raises(ResolverError, match="policy=local"):
            resolve("vad", "call", client_caps=None, policy="local")

    def test_local_empty_caps_raises(self):
        with pytest.raises(ResolverError):
            resolve("stt", "companion", client_caps={}, policy="local")

    def test_local_other_component_caps_raises(self):
        # Has TTS, asks for VAD → still raises
        caps = {"tts": ["pocket-onnx-wasm"]}
        with pytest.raises(ResolverError):
            resolve("vad", "call", client_caps=caps, policy="local")

    def test_local_empty_list_raises(self):
        caps = {"vad": []}
        with pytest.raises(ResolverError):
            resolve("vad", "call", client_caps=caps, policy="local")


class TestPinnedProvider:
    """Explicit pin overrides policy + caps."""

    def test_pin_wins_over_server_policy(self):
        result = resolve(
            "tts", "call", client_caps={}, policy="server",
            pinned_provider="fabric:tower:pocket-tts",
        )
        assert result == "fabric:tower:pocket-tts"

    def test_pin_wins_over_local_policy_without_caps(self):
        # Would normally raise — pin bypasses the local-requires-caps check.
        result = resolve(
            "vad", "call", client_caps=None, policy="local",
            pinned_provider="server",
        )
        assert result == "server"

    def test_pin_wins_over_client_dispatch(self):
        caps = {"tts": ["pocket-onnx-wasm"]}
        result = resolve(
            "tts", "call", client_caps=caps, policy="auto",
            pinned_provider="chatterbox-tts",
        )
        assert result == "chatterbox-tts"


class TestSurfaceIndependence:
    """The resolver behaves the same per-surface — the surface is passed
    so the caller can fetch a per-surface policy, but the resolution math
    is the same once policy + caps are determined."""

    def test_companion_surface_same_resolution(self):
        caps = {"vad": ["silero-wasm"]}
        assert (
            resolve("vad", "companion", client_caps=caps, policy="auto")
            == "client:silero-wasm"
        )

    def test_narration_server_policy(self):
        # Real-world: narration defaults to 'server' per config.py
        caps = {"tts": ["pocket-onnx-wasm"]}
        assert resolve("tts", "narration", client_caps=caps, policy="server") == "server"


class TestInputValidation:
    def test_unknown_component_raises(self):
        with pytest.raises(ValueError, match="unknown component"):
            resolve("invalid", "call", policy="auto")

    def test_unknown_surface_raises(self):
        with pytest.raises(ValueError, match="unknown surface"):
            resolve("vad", "invalid", policy="auto")

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="unknown policy"):
            resolve("vad", "call", policy="invalid")


class TestRobustnessToMalformedCaps:
    def test_non_list_caps_value_treated_as_empty(self):
        # Client sent garbage — don't crash, treat as no capability.
        caps = {"vad": "silero-wasm"}  # Should be a list
        assert resolve("vad", "call", client_caps=caps, policy="auto") == "server"

    def test_non_string_engines_filtered(self):
        caps = {"vad": [None, 123, "silero-wasm"]}
        assert (
            resolve("vad", "call", client_caps=caps, policy="auto")
            == "client:silero-wasm"
        )

    def test_whitespace_only_engine_filtered(self):
        caps = {"vad": ["   ", "silero-wasm"]}
        assert (
            resolve("vad", "call", client_caps=caps, policy="auto")
            == "client:silero-wasm"
        )
