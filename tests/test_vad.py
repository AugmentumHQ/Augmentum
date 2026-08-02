"""Tests for voice/vad.py -- voice activity detection."""

from __future__ import annotations

import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from augmentum.voice.vad import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    VadEvent,
    VadProcessor,
    VadState,
)


def _make_silent_frame() -> bytes:
    """Generate one frame of silence (FRAME_BYTES of zero PCM)."""
    return b"\x00" * FRAME_BYTES


def _make_speech_frame() -> bytes:
    """Generate one frame of 440Hz tone to simulate speech."""
    samples = []
    for i in range(FRAME_SAMPLES):
        val = int(16000 * np.sin(2 * np.pi * 440 * i / SAMPLE_RATE))
        samples.append(struct.pack("<h", max(-32768, min(32767, val))))
    return b"".join(samples)


class TestVadConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000

    def test_frame_samples(self):
        assert FRAME_SAMPLES == 512

    def test_frame_bytes(self):
        assert FRAME_BYTES == 1024


class TestVadState:
    def test_states_exist(self):
        assert VadState.IDLE is not None
        assert VadState.SPEECH is not None
        assert VadState.TRAILING is not None


class TestVadProcessor:
    def test_default_construction(self):
        vad = VadProcessor()
        assert vad.speech_threshold == 0.6
        assert vad.silence_duration_ms == 800
        assert vad.min_speech_ms == 250

    def test_reset_clears_state(self):
        vad = VadProcessor()
        vad._state = VadState.SPEECH
        vad._consecutive_speech = 5
        mock_model = MagicMock()
        vad._model = mock_model
        vad.reset()
        assert vad._state == VadState.IDLE
        assert vad._consecutive_speech == 0

    def test_is_speaking_false_when_idle(self):
        vad = VadProcessor()
        assert not vad.is_speaking

    def test_is_speaking_true_when_speech(self):
        vad = VadProcessor()
        vad._state = VadState.SPEECH
        assert vad.is_speaking

    def test_is_speaking_true_when_trailing(self):
        vad = VadProcessor()
        vad._state = VadState.TRAILING
        assert vad.is_speaking

    def test_process_frame_wrong_size_returns_none(self):
        vad = VadProcessor()
        mock_model = MagicMock()
        vad._model = mock_model
        result = vad.process_frame(b"\x00" * 100)  # wrong size
        assert result is None

    def test_process_frame_detects_speech_start(self):
        torch_mock = MagicMock()
        torch_mock.from_numpy.return_value = MagicMock()

        import sys
        orig = sys.modules.get("torch")
        sys.modules["torch"] = torch_mock

        try:
            vad = VadProcessor(min_start_frames=2)
            mock_model = MagicMock(return_value=0.9)  # high probability
            vad._model = mock_model

            # First speech frame
            result1 = vad.process_frame(_make_silent_frame())
            # Second speech frame triggers start (min_start_frames=2)
            result2 = vad.process_frame(_make_silent_frame())

            assert result2 is not None
            assert result2.kind == "speech_start"
        finally:
            if orig is not None:
                sys.modules["torch"] = orig
            else:
                sys.modules.pop("torch", None)

    def test_soft_reset_keeps_prefix(self):
        vad = VadProcessor()
        vad._prefix_buffer = [b"\x00" * FRAME_BYTES]
        vad._state = VadState.SPEECH
        vad._consecutive_speech = 3
        mock_model = MagicMock()
        vad._model = mock_model
        vad.soft_reset()
        assert vad._state == VadState.IDLE
        assert len(vad._prefix_buffer) == 1  # preserved

    def test_get_prefix_audio(self):
        vad = VadProcessor()
        frame = b"\x01" * FRAME_BYTES
        vad._prefix_buffer = [frame, frame]
        prefix = vad.get_prefix_audio()
        assert len(prefix) == FRAME_BYTES * 2
