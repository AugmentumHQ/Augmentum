"""Smoke tests -- verify every voice module imports and primary classes construct."""

from __future__ import annotations

import importlib

import pytest


class TestVoiceModuleImports:
    """Every module under augmentum/voice/ must import without error."""

    def test_import_tts(self):
        mod = importlib.import_module("augmentum.voice.tts")
        assert mod is not None

    def test_import_kokoro_tts(self):
        mod = importlib.import_module("augmentum.voice.kokoro_tts")
        assert hasattr(mod, "KokoroTTS")
        assert hasattr(mod, "VOICE_META")

    def test_import_stt(self):
        mod = importlib.import_module("augmentum.voice.stt")
        assert mod is not None

    def test_import_moonshine_stt(self):
        mod = importlib.import_module("augmentum.voice.moonshine_stt")
        assert mod is not None

    def test_import_streaming_stt(self):
        mod = importlib.import_module("augmentum.voice.streaming_stt")
        assert mod is not None

    def test_import_audio_processor(self):
        mod = importlib.import_module("augmentum.voice.audio_processor")
        assert mod is not None

    def test_import_denoiser(self):
        mod = importlib.import_module("augmentum.voice.denoiser")
        assert mod is not None

    def test_import_vad(self):
        mod = importlib.import_module("augmentum.voice.vad")
        assert hasattr(mod, "VadProcessor")
        assert hasattr(mod, "VadState")
        assert hasattr(mod, "VadEvent")

    def test_import_emotion(self):
        mod = importlib.import_module("augmentum.voice.emotion")
        assert mod is not None

    def test_import_prosody(self):
        mod = importlib.import_module("augmentum.voice.prosody")
        assert hasattr(mod, "ProsodyCartographer")
        assert hasattr(mod, "split_prosodic_clauses")

    def test_import_hbe(self):
        mod = importlib.import_module("augmentum.voice.hbe")
        assert hasattr(mod, "extend_bandwidth")

    def test_import_smart_turn(self):
        mod = importlib.import_module("augmentum.voice.smart_turn")
        assert hasattr(mod, "predict_turn_complete")
        assert hasattr(mod, "load_model")

    def test_import_speaker(self):
        mod = importlib.import_module("augmentum.voice.speaker")
        assert mod is not None

    def test_import_text_cleaning(self):
        mod = importlib.import_module("augmentum.voice.text_cleaning")
        assert hasattr(mod, "clean_for_tts")

    def test_import_pipeline(self):
        mod = importlib.import_module("augmentum.voice.pipeline")
        assert mod is not None

    def test_import_voice_walk(self):
        mod = importlib.import_module("augmentum.voice.voice_walk")
        assert hasattr(mod, "clone_voice_walk")
        assert hasattr(mod, "WalkResult")
        assert hasattr(mod, "WalkProgress")
