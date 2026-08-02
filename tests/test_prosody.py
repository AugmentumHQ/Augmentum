"""Tests for voice/prosody.py -- prosodic cartography and steering."""

from __future__ import annotations

import numpy as np

from augmentum.voice.prosody import (
    _AXIS_DEFINITIONS,
    _EXCLAMATION_MAGNITUDE,
    _QUESTION_MAGNITUDE,
    ProsodyCartographer,
    _quick_sentiment,
    split_prosodic_clauses,
)


def _make_cartographer_with_axes() -> ProsodyCartographer:
    """Create a ProsodyCartographer with pre-computed synthetic axes."""
    cart = ProsodyCartographer()
    # Manually set up axes as unit vectors
    dim = 256
    for _, _, label in _AXIS_DEFINITIONS:
        direction = np.random.randn(dim).astype(np.float32)
        direction /= np.linalg.norm(direction)
        cart._axes[label] = direction
    cart._computed = True
    return cart


class TestAxisDefinitions:
    def test_five_axes_defined(self):
        assert len(_AXIS_DEFINITIONS) == 5

    def test_axis_labels(self):
        labels = {label for _, _, label in _AXIS_DEFINITIONS}
        expected = {"breathiness", "warmth", "depth", "energy", "accent_gb"}
        assert labels == expected

    def test_axis_voices_exist_in_meta(self):
        from augmentum.voice.kokoro_tts import VOICE_META
        for pos, neg, label in _AXIS_DEFINITIONS:
            assert pos in VOICE_META, f"Axis '{label}' positive voice '{pos}' not in VOICE_META"
            assert neg in VOICE_META, f"Axis '{label}' negative voice '{neg}' not in VOICE_META"


class TestSteer:
    def test_question_increases_energy(self):
        cart = _make_cartographer_with_axes()
        base = np.zeros(256, dtype=np.float32)
        result = cart.steer(base, "What do you think?")
        # Should have a non-zero delta in the energy direction
        energy_component = np.dot(result, cart._axes["energy"])
        assert energy_component > 0

    def test_exclamation_increases_warmth(self):
        cart = _make_cartographer_with_axes()
        base = np.zeros(256, dtype=np.float32)
        result = cart.steer(base, "This is amazing!")
        warmth_component = np.dot(result, cart._axes["warmth"])
        assert warmth_component > 0

    def test_neutral_text_minimal_shift(self):
        cart = _make_cartographer_with_axes()
        base = np.random.randn(256).astype(np.float32)
        result = cart.steer(base, "The sky is blue.")
        delta = np.linalg.norm(result - base)
        # Neutral text should produce very small or zero shift
        assert delta < 0.5

    def test_steer_preserves_shape(self):
        cart = _make_cartographer_with_axes()
        base = np.random.randn(256).astype(np.float32)
        result = cart.steer(base, "Hello!")
        assert result.shape == base.shape

    def test_steer_string_input_passthrough(self):
        cart = _make_cartographer_with_axes()
        result = cart.steer("af_heart", "Hello")  # type: ignore[arg-type]
        assert result == "af_heart"

    def test_manual_axes_override(self):
        cart = _make_cartographer_with_axes()
        base = np.zeros(256, dtype=np.float32)
        result = cart.steer(base, "neutral", manual_axes={"warmth": 0.5})
        warmth_component = np.dot(result, cart._axes["warmth"])
        assert warmth_component > 0.3


class TestShiftMagnitudes:
    def test_question_magnitude_range(self):
        assert 0.05 <= _QUESTION_MAGNITUDE <= 0.2

    def test_exclamation_magnitude_range(self):
        assert 0.05 <= _EXCLAMATION_MAGNITUDE <= 0.2


class TestSplitProsodicClauses:
    def test_splits_at_sentence_boundaries(self):
        text = "Hello there. How are you? Fine thanks."
        clauses = split_prosodic_clauses(text)
        assert len(clauses) == 3

    def test_splits_at_semicolons(self):
        text = "First clause; second clause"
        clauses = split_prosodic_clauses(text)
        assert len(clauses) == 2

    def test_single_clause_returns_one(self):
        text = "Just one clause"
        clauses = split_prosodic_clauses(text)
        assert len(clauses) == 1
        assert clauses[0] == "Just one clause"

    def test_empty_string_returns_empty(self):
        assert split_prosodic_clauses("") == []


class TestQuickSentiment:
    def test_positive_text(self):
        score = _quick_sentiment("I love this wonderful day")
        assert score > 0

    def test_negative_text(self):
        score = _quick_sentiment("This is terrible and horrible")
        assert score < 0

    def test_neutral_text(self):
        score = _quick_sentiment("The table is brown")
        assert score == 0.0

    def test_range_is_bounded(self):
        score = _quick_sentiment("love happy joy wonderful amazing")
        assert -1.0 <= score <= 1.0
