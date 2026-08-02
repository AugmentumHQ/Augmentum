"""Tests for voice/smart_turn.py -- learned turn-completion detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from augmentum.voice.smart_turn import (
    _DEFAULT_THRESHOLD,
    _MAX_SAMPLES,
    _SAMPLE_RATE,
    is_available,
    predict_turn_complete,
)


class TestSmartTurnConstants:
    def test_sample_rate(self):
        assert _SAMPLE_RATE == 16000

    def test_max_samples(self):
        assert _MAX_SAMPLES == 8 * 16000

    def test_default_threshold(self):
        assert _DEFAULT_THRESHOLD == 0.5


class TestPredictTurnComplete:
    def test_returns_true_when_not_loaded(self):
        """When model is not loaded, default to turn complete."""
        audio = np.zeros(_MAX_SAMPLES, dtype=np.float32)
        is_complete, prob = predict_turn_complete(audio)
        assert is_complete is True
        assert prob == 1.0

    @patch("augmentum.voice.smart_turn._loaded", True)
    @patch("augmentum.voice.smart_turn._feature_extractor")
    @patch("augmentum.voice.smart_turn._session")
    def test_returns_probability_0_to_1(self, mock_session, mock_extractor):
        mock_features = MagicMock()
        mock_features.input_features = np.random.randn(1, 80, 3000).astype(np.float32)
        mock_extractor.return_value = mock_features

        # Mock ONNX session to return a probability
        mock_session.run.return_value = [np.array([[0.75]])]

        audio = np.random.randn(_MAX_SAMPLES).astype(np.float32)
        is_complete, prob = predict_turn_complete(audio)

        assert 0.0 <= prob <= 1.0
        assert is_complete is True  # 0.75 > 0.5

    @patch("augmentum.voice.smart_turn._loaded", True)
    @patch("augmentum.voice.smart_turn._feature_extractor")
    @patch("augmentum.voice.smart_turn._session")
    def test_short_audio_padded(self, mock_session, mock_extractor):
        mock_features = MagicMock()
        mock_features.input_features = np.random.randn(1, 80, 3000).astype(np.float32)
        mock_extractor.return_value = mock_features
        mock_session.run.return_value = [np.array([[0.3]])]

        short_audio = np.random.randn(8000).astype(np.float32)  # 0.5 seconds
        is_complete, prob = predict_turn_complete(short_audio)

        assert prob == pytest.approx(0.3, abs=0.01)
        assert is_complete is False  # 0.3 < 0.5

    @patch("augmentum.voice.smart_turn._loaded", True)
    @patch("augmentum.voice.smart_turn._feature_extractor")
    @patch("augmentum.voice.smart_turn._session")
    def test_long_audio_truncated(self, mock_session, mock_extractor):
        mock_features = MagicMock()
        mock_features.input_features = np.random.randn(1, 80, 3000).astype(np.float32)
        mock_extractor.return_value = mock_features
        mock_session.run.return_value = [np.array([[0.6]])]

        long_audio = np.random.randn(_MAX_SAMPLES * 3).astype(np.float32)
        is_complete, prob = predict_turn_complete(long_audio)

        assert 0.0 <= prob <= 1.0

    @patch("augmentum.voice.smart_turn._loaded", True)
    @patch("augmentum.voice.smart_turn._feature_extractor")
    @patch("augmentum.voice.smart_turn._session")
    def test_custom_threshold(self, mock_session, mock_extractor):
        mock_features = MagicMock()
        mock_features.input_features = np.random.randn(1, 80, 3000).astype(np.float32)
        mock_extractor.return_value = mock_features
        mock_session.run.return_value = [np.array([[0.7]])]

        audio = np.random.randn(_MAX_SAMPLES).astype(np.float32)
        is_complete, prob = predict_turn_complete(audio, threshold=0.8)
        assert is_complete is False  # 0.7 < 0.8


class TestIsAvailable:
    @patch("augmentum.voice.smart_turn._loaded", False)
    def test_not_available_when_not_loaded(self):
        assert is_available() is False

    @patch("augmentum.voice.smart_turn._loaded", True)
    def test_available_when_loaded(self):
        assert is_available() is True
