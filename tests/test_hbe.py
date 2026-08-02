"""Tests for voice/hbe.py -- harmonic bandwidth extension."""

from __future__ import annotations

import numpy as np
import pytest

from augmentum.voice.hbe import _SRC_SR, _TGT_SR, _upsample_simple, extend_bandwidth


class TestExtendBandwidth:
    """Core HBE contract tests."""

    def test_doubles_sample_count(self):
        samples = np.random.randn(48000).astype(np.float32) * 0.5
        result, sr = extend_bandwidth(samples, _SRC_SR)
        # Output should be approximately 2x input length
        assert len(result) == len(samples) * 2

    def test_output_sample_rate_is_48000(self):
        samples = np.random.randn(48000).astype(np.float32) * 0.5
        _, sr = extend_bandwidth(samples, _SRC_SR)
        assert sr == _TGT_SR

    def test_passthrough_non_24khz(self):
        samples = np.random.randn(16000).astype(np.float32)
        result, sr = extend_bandwidth(samples, 16000)
        assert sr == 16000
        np.testing.assert_array_equal(result, samples)

    def test_silence_input_no_crash(self):
        samples = np.zeros(48000, dtype=np.float32)
        result, sr = extend_bandwidth(samples, _SRC_SR)
        assert sr == _TGT_SR
        assert len(result) == 96000

    def test_short_input_upsample_only(self):
        # Shorter than FFT size (2048) -- should just upsample
        samples = np.random.randn(1000).astype(np.float32) * 0.3
        result, sr = extend_bandwidth(samples, _SRC_SR)
        assert sr == _TGT_SR
        assert len(result) == 2000

    def test_output_is_float32(self):
        samples = np.random.randn(48000).astype(np.float32) * 0.5
        result, sr = extend_bandwidth(samples, _SRC_SR)
        assert result.dtype == np.float32

    def test_edge_fading_applied(self):
        samples = np.ones(48000, dtype=np.float32) * 0.5
        result, sr = extend_bandwidth(samples, _SRC_SR)
        # First sample should be faded toward 0
        assert abs(result[0]) < 0.01
        # Last sample should be faded toward 0
        assert abs(result[-1]) < 0.01

    def test_tone_input_produces_output(self):
        t = np.linspace(0, 1, 24000, dtype=np.float32)
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result, sr = extend_bandwidth(tone, _SRC_SR)
        assert sr == _TGT_SR
        assert len(result) == 48000
        # Should have nonzero energy
        assert np.abs(result).max() > 0.01


class TestUpsampleSimple:
    def test_doubles_length(self):
        samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _upsample_simple(samples)
        assert len(result) == 6

    def test_preserves_original_samples(self):
        samples = np.array([1.0, 0.5, -0.5], dtype=np.float32)
        result = _upsample_simple(samples)
        # Even indices are original samples (before fade)
        # For short arrays fade doesn't apply (< 128 samples)
        assert result[0] == pytest.approx(samples[0], abs=0.01)
