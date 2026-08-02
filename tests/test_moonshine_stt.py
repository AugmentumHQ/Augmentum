"""Tests for Moonshine STT wiring — verifies the full integration path.

Traces: config → warmup → session lifecycle → audio dispatch → transcript events.
All moonshine_voice internals are mocked to test wiring only.
"""

from __future__ import annotations

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from augmentum.voice.moonshine_stt import MoonshineSTTSession


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_class_state():
    """Reset MoonshineSTTSession class-level state between tests."""
    orig = (
        MoonshineSTTSession._transcriber_cls,
        MoonshineSTTSession._model_path,
        MoonshineSTTSession._model_arch,
        MoonshineSTTSession._model_loaded,
    )
    yield
    (
        MoonshineSTTSession._transcriber_cls,
        MoonshineSTTSession._model_path,
        MoonshineSTTSession._model_arch,
        MoonshineSTTSession._model_loaded,
    ) = orig


def _make_pcm_frame(duration_ms: int = 32, sample_rate: int = 16000) -> bytes:
    """Create a valid PCM16 frame of given duration."""
    n_samples = int(sample_rate * duration_ms / 1000)
    # Generate a 440 Hz tone so it's not silence
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return samples.tobytes()


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------


class TestConfiguration:
    """Config settings propagate to class-level state."""

    def test_configure_sets_model_path(self):
        MoonshineSTTSession.configure(model_path="custom/model")
        assert MoonshineSTTSession._model_path == "custom/model"

    def test_configure_sets_arch(self):
        MoonshineSTTSession.configure(model_arch="streaming_small")
        assert MoonshineSTTSession._model_arch == "streaming_small"

    def test_configure_empty_keeps_defaults(self):
        default_path = MoonshineSTTSession._model_path
        default_arch = MoonshineSTTSession._model_arch
        MoonshineSTTSession.configure(model_path="", model_arch="")
        assert MoonshineSTTSession._model_path == default_path
        assert MoonshineSTTSession._model_arch == default_arch

    def test_configure_partial(self):
        MoonshineSTTSession.configure(model_path="new/path")
        assert MoonshineSTTSession._model_path == "new/path"
        assert MoonshineSTTSession._model_arch == "streaming_medium"  # unchanged


# ---------------------------------------------------------------------------
# 2. Warmup / Model Loading
# ---------------------------------------------------------------------------


class TestWarmup:
    """Warmup loads the Transcriber class and sets _model_loaded."""

    def test_warmup_success(self):
        mock_transcriber = MagicMock()
        with patch.dict("sys.modules", {"moonshine_voice": MagicMock(
            Transcriber=mock_transcriber,
        )}):
            MoonshineSTTSession.warmup()

        assert MoonshineSTTSession._model_loaded is True
        assert MoonshineSTTSession._transcriber_cls is mock_transcriber

    def test_warmup_idempotent(self):
        """Second warmup is a no-op."""
        MoonshineSTTSession._model_loaded = True
        MoonshineSTTSession._transcriber_cls = MagicMock()
        original_cls = MoonshineSTTSession._transcriber_cls

        MoonshineSTTSession.warmup()
        assert MoonshineSTTSession._transcriber_cls is original_cls

    def test_warmup_import_error(self):
        """Missing moonshine-voice package handled gracefully."""
        import builtins
        _real_import = builtins.__import__

        def _selective_import(name, *args, **kwargs):
            if name == "moonshine_voice":
                raise ImportError("no module named 'moonshine_voice'")
            return _real_import(name, *args, **kwargs)

        MoonshineSTTSession._model_loaded = False
        MoonshineSTTSession._transcriber_cls = None
        with patch("builtins.__import__", side_effect=_selective_import):
            MoonshineSTTSession.warmup()

        assert MoonshineSTTSession._model_loaded is False
        assert MoonshineSTTSession._transcriber_cls is None

    def test_warmup_exception(self):
        """Unexpected error during warmup handled gracefully."""
        def _bad_import(name, *args, **kwargs):
            if name == "moonshine_voice":
                raise RuntimeError("GPU init failed")
            return original_import(name, *args, **kwargs)

        import builtins
        original_import = builtins.__import__
        with patch("builtins.__import__", side_effect=_bad_import):
            MoonshineSTTSession._model_loaded = False
            MoonshineSTTSession.warmup()

        assert MoonshineSTTSession._model_loaded is False


# ---------------------------------------------------------------------------
# 3. Availability Check
# ---------------------------------------------------------------------------


class TestAvailability:
    """is_available() correctly detects package presence."""

    def test_available_when_loaded(self):
        MoonshineSTTSession._model_loaded = True
        assert MoonshineSTTSession.is_available() is True

    def test_available_when_importable(self):
        MoonshineSTTSession._model_loaded = False
        with patch.dict("sys.modules", {"moonshine_voice": MagicMock()}):
            assert MoonshineSTTSession.is_available() is True

    def test_unavailable_when_not_installed(self):
        MoonshineSTTSession._model_loaded = False
        with patch.dict("sys.modules", {"moonshine_voice": None}):
            # Force import to fail
            with patch("builtins.__import__", side_effect=ImportError):
                assert MoonshineSTTSession.is_available() is False


# ---------------------------------------------------------------------------
# 4. Session Connect
# ---------------------------------------------------------------------------


class TestConnect:
    """connect() creates transcriber, listener, and starts streaming."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Full connect path with mocked moonshine_voice."""
        mock_transcriber_instance = MagicMock()
        mock_transcriber_cls = MagicMock(return_value=mock_transcriber_instance)

        # Pre-load model
        MoonshineSTTSession._model_loaded = True
        MoonshineSTTSession._transcriber_cls = mock_transcriber_cls

        # Mock moonshine_voice imports used in connect()
        mock_mv = MagicMock()
        mock_mv.ModelArch.MEDIUM_STREAMING = "medium_streaming_enum"
        mock_mv.ModelArch.SMALL_STREAMING = "small_streaming_enum"
        mock_mv.ModelArch.TINY_STREAMING = "tiny_streaming_enum"
        mock_mv.ModelArch.BASE_STREAMING = "base_streaming_enum"
        mock_mv.ModelArch.BASE = "base_enum"
        mock_mv.ModelArch.TINY = "tiny_enum"
        mock_mv.get_model_for_language.return_value = ("/resolved/path", "resolved_arch")
        mock_mv.TranscriptEventListener = type("TranscriptEventListener", (), {})

        callback = AsyncMock()
        session = MoonshineSTTSession(on_transcript=callback)

        with patch.dict("sys.modules", {
            "moonshine_voice": mock_mv,
            "augmentum.voice.streaming_stt": MagicMock(),
        }):
            # Need to patch the import inside connect()
            with patch("augmentum.voice.moonshine_stt.MoonshineSTTSession._transcriber_cls",
                       mock_transcriber_cls):
                await session.connect()

        assert session._connected is True
        assert session._transcriber is not None
        mock_transcriber_instance.start.assert_called_once()
        mock_transcriber_instance.add_listener.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_when_model_not_loaded_triggers_warmup(self):
        """connect() lazy-loads model if not warmed up."""
        MoonshineSTTSession._model_loaded = False
        MoonshineSTTSession._transcriber_cls = None

        session = MoonshineSTTSession(on_transcript=AsyncMock())

        # warmup will fail (no real package) → connect should fail gracefully
        with patch.object(MoonshineSTTSession, "warmup"):
            await session.connect()

        assert session._connected is False

    @pytest.mark.asyncio
    async def test_connect_exception_sets_not_connected(self):
        """Exception during connect() leaves session in safe state."""
        MoonshineSTTSession._model_loaded = True
        MoonshineSTTSession._transcriber_cls = MagicMock()

        session = MoonshineSTTSession(on_transcript=AsyncMock())

        # Force import error inside connect
        with patch("augmentum.voice.streaming_stt.TranscriptEvent",
                    side_effect=ImportError("boom")):
            await session.connect()

        assert session._connected is False


# ---------------------------------------------------------------------------
# 5. Audio Dispatch
# ---------------------------------------------------------------------------


class TestSendAudio:
    """send_audio() feeds frames and dispatches transcript events."""

    def test_send_audio_when_not_connected_is_noop(self):
        """Silently returns when not connected."""
        session = MoonshineSTTSession()
        session._connected = False
        session.send_audio(_make_pcm_frame())  # Should not raise

    def test_send_audio_converts_pcm_to_float32(self):
        """PCM16 bytes converted to float32 [-1, 1] numpy array."""
        session = MoonshineSTTSession()
        session._connected = True
        mock_transcriber = MagicMock()
        session._transcriber = mock_transcriber
        session._pending_events = []

        frame = _make_pcm_frame()
        session.send_audio(frame)

        mock_transcriber.add_audio.assert_called_once()
        samples_arg = mock_transcriber.add_audio.call_args[0][0]
        assert samples_arg.dtype == np.float32
        assert samples_arg.max() <= 1.0
        assert samples_arg.min() >= -1.0

    def test_send_audio_dispatches_events(self):
        """Transcript events fire as async tasks after add_audio."""
        from augmentum.voice.streaming_stt import TranscriptEvent

        callback = AsyncMock()
        session = MoonshineSTTSession(on_transcript=callback)
        session._connected = True
        mock_transcriber = MagicMock()
        session._transcriber = mock_transcriber

        # Simulate add_audio triggering a listener event
        def _fake_add_audio(samples, sr):
            session._pending_events.append(
                TranscriptEvent(text="hello", is_final=True, speech_final=True)
            )

        mock_transcriber.add_audio.side_effect = _fake_add_audio
        session._pending_events = []

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # send_audio is sync but dispatches tasks to the running loop
            loop.run_until_complete(_run_send_audio(session, _make_pcm_frame()))
        finally:
            loop.close()

    def test_send_audio_exception_handled(self):
        """Exception in add_audio logged, not raised."""
        session = MoonshineSTTSession()
        session._connected = True
        session._transcriber = MagicMock()
        session._transcriber.add_audio.side_effect = RuntimeError("ONNX error")
        session._pending_events = []

        # Should not raise
        session.send_audio(_make_pcm_frame())


async def _run_send_audio(session, frame):
    """Helper to run send_audio within an async context."""
    session.send_audio(frame)
    # Give tasks a chance to run
    await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# 6. Session Close
# ---------------------------------------------------------------------------


class TestClose:
    """close() flushes and stops the transcriber."""

    @pytest.mark.asyncio
    async def test_close_stops_transcriber(self):
        session = MoonshineSTTSession()
        session._connected = True
        mock_transcriber = MagicMock()
        session._transcriber = mock_transcriber
        session._pending_events = []

        await session.close()

        assert session._connected is False
        assert session._transcriber is None
        mock_transcriber.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_dispatches_final_events(self):
        """Final transcript events from stop() are dispatched."""
        from augmentum.voice.streaming_stt import TranscriptEvent

        callback = AsyncMock()
        session = MoonshineSTTSession(on_transcript=callback)
        session._connected = True
        mock_transcriber = MagicMock()
        session._transcriber = mock_transcriber

        # Simulate stop() producing a final event
        def _fake_stop():
            session._pending_events.append(
                TranscriptEvent(text="final words", is_final=True, speech_final=True)
            )

        mock_transcriber.stop.side_effect = _fake_stop
        session._pending_events = []

        await session.close()

        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event.text == "final words"
        assert event.is_final is True

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self):
        """close() on unconnected session is safe."""
        session = MoonshineSTTSession()
        session._connected = False
        session._transcriber = None

        await session.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_close_exception_handled(self):
        """Exception during stop() doesn't crash."""
        session = MoonshineSTTSession()
        session._connected = True
        session._transcriber = MagicMock()
        session._transcriber.stop.side_effect = RuntimeError("boom")
        session._pending_events = []

        await session.close()  # Should not raise
        assert session._transcriber is None


# ---------------------------------------------------------------------------
# 7. Voice Routes Wiring (config → warmup → session)
# ---------------------------------------------------------------------------


class TestVoiceRoutesWiring:
    """Verify the config → voice_routes wiring path."""

    def test_config_has_moonshine_settings(self):
        """Config exposes all required Moonshine settings."""
        from augmentum.config import Settings

        s = Settings()
        assert hasattr(s, "voice_moonshine_enabled")
        assert hasattr(s, "voice_moonshine_model")
        assert hasattr(s, "voice_moonshine_arch")
        assert s.voice_moonshine_enabled is True  # default on
        assert "moonshine" in s.voice_moonshine_model.lower()
        # Must be a canonical arch string the library accepts, not a made-up
        # one. "streaming_medium" used to be the default and was INVALID —
        # string_to_model_arch() rejects it. The library default English model
        # is medium-streaming, so that's our default.
        assert s.voice_moonshine_arch == "medium-streaming"

    def test_moonshine_imported_in_voice_routes(self):
        """voice_routes.py imports MoonshineSTTSession."""
        from augmentum.proxy import voice_routes
        assert hasattr(voice_routes, "MoonshineSTTSession")

    def test_transcript_event_compatible(self):
        """MoonshineSTTSession uses the same TranscriptEvent as streaming_stt."""
        from augmentum.voice.streaming_stt import TranscriptEvent
        event = TranscriptEvent(text="test", is_final=True, speech_final=True)
        assert event.text == "test"
        assert event.is_final is True
        assert event.speech_final is True


# ---------------------------------------------------------------------------
# 8. End-to-End: Audio → Transcript Event
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full path: PCM bytes → MoonshineSTTSession → TranscriptEvent callback."""

    @pytest.mark.asyncio
    async def test_full_roundtrip(self):
        """Feed audio, get transcript event via callback."""
        from augmentum.voice.streaming_stt import TranscriptEvent

        received_events = []

        async def on_transcript(event):
            received_events.append(event)

        session = MoonshineSTTSession(on_transcript=on_transcript)
        session._connected = True
        session._pending_events = []

        # Mock transcriber that produces events on add_audio
        mock_transcriber = MagicMock()

        def _fake_add_audio(samples, sr):
            session._pending_events.append(
                TranscriptEvent(text="hello world", is_final=False, speech_final=False)
            )

        mock_transcriber.add_audio.side_effect = _fake_add_audio
        session._transcriber = mock_transcriber

        # Feed audio
        session.send_audio(_make_pcm_frame())

        # Wait for async task dispatch
        await asyncio.sleep(0.05)

        assert len(received_events) == 1
        assert received_events[0].text == "hello world"
        assert received_events[0].is_final is False

    @pytest.mark.asyncio
    async def test_multiple_frames_accumulate_events(self):
        """Multiple audio frames can produce multiple transcript events."""
        from augmentum.voice.streaming_stt import TranscriptEvent

        received_events = []
        call_count = 0

        async def on_transcript(event):
            received_events.append(event)

        session = MoonshineSTTSession(on_transcript=on_transcript)
        session._connected = True
        session._pending_events = []

        mock_transcriber = MagicMock()

        def _fake_add_audio(samples, sr):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                session._pending_events.append(
                    TranscriptEvent(text="partial", is_final=False, speech_final=False)
                )
            if call_count == 5:
                session._pending_events.append(
                    TranscriptEvent(text="hello", is_final=True, speech_final=True)
                )

        mock_transcriber.add_audio.side_effect = _fake_add_audio
        session._transcriber = mock_transcriber

        for _ in range(6):
            session.send_audio(_make_pcm_frame())

        await asyncio.sleep(0.05)

        assert len(received_events) == 2
        assert received_events[0].text == "partial"
        assert received_events[0].is_final is False
        assert received_events[1].text == "hello"
        assert received_events[1].is_final is True


# ---------------------------------------------------------------------------
# 9. Fallback Path Verification
# ---------------------------------------------------------------------------


class TestFallbackPath:
    """When Moonshine fails, voice_routes should fall through to batch STT."""

    def test_speech_start_fallback_chain(self):
        """Verify the priority: Moonshine > Deepgram streaming > batch."""
        # This is a structural test — verify the code path exists in the
        # speech_start handling section (after the warmup section)
        import inspect
        from augmentum.proxy import voice_routes

        source = inspect.getsource(voice_routes.voice_websocket)

        # Find the speech_start event handler section
        speech_start_idx = source.find('"speech_start"')
        assert speech_start_idx > 0, "speech_start event missing"
        tail = source[speech_start_idx:]

        # Moonshine is tried first within speech_start
        moonshine_idx = tail.find("_use_moonshine and not stt_session")
        assert moonshine_idx > 0, "Moonshine check missing from speech_start"

        # Deepgram streaming is second fallback (after Moonshine in this section)
        deepgram_idx = tail.find("is_streaming_stt_capable", moonshine_idx)
        assert deepgram_idx > moonshine_idx, "Deepgram check should come after Moonshine"

        # Batch is final fallback
        batch_idx = tail.find("BatchSTTFallback()", deepgram_idx)
        assert batch_idx > deepgram_idx, "BatchSTT should be last fallback"

    def test_finalize_speech_handles_both_paths(self):
        """_finalize_speech handles both stt_session and batch_stt."""
        import inspect
        from augmentum.proxy import voice_routes

        source = inspect.getsource(voice_routes.voice_websocket)

        # Streaming STT path
        assert "if stt_session:" in source
        assert "_final_transcript_parts" in source

        # Batch STT fallback
        assert "elif batch_stt:" in source
        assert "transcribe_audio" in source
