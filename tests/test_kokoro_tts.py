"""Tests for voice/kokoro_tts.py -- voice metadata, blends, lang mapping."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from augmentum.voice.kokoro_tts import (
    RECOMMENDED_BLENDS,
    VOICE_META,
    KokoroTTS,
    _voice_lang,
)


class TestVoiceMeta:
    """VOICE_META catalogue integrity."""

    def test_voice_meta_has_54_entries(self):
        assert len(VOICE_META) == 54

    def test_every_entry_has_required_fields(self):
        required = {"grade", "gender", "lang", "desc"}
        for name, meta in VOICE_META.items():
            missing = required - set(meta.keys())
            assert not missing, f"{name} missing fields: {missing}"

    def test_gender_values_valid(self):
        valid = {"F", "M"}
        for name, meta in VOICE_META.items():
            assert meta["gender"] in valid, f"{name} has invalid gender: {meta['gender']}"

    def test_grade_format(self):
        valid_grades = {"A", "A-", "B-", "C+", "C", "C-", "D+", "D", "D-", "F+"}
        for name, meta in VOICE_META.items():
            assert meta["grade"] in valid_grades, f"{name} has unexpected grade: {meta['grade']}"


class TestRecommendedBlends:
    def test_blends_are_nonempty(self):
        assert len(RECOMMENDED_BLENDS) >= 5

    def test_each_blend_has_required_keys(self):
        required = {"name", "spec", "desc", "gender", "lang"}
        for blend in RECOMMENDED_BLENDS:
            missing = required - set(blend.keys())
            assert not missing, f"Blend '{blend.get('name')}' missing: {missing}"

    def test_blend_specs_reference_valid_voices(self):
        for blend in RECOMMENDED_BLENDS:
            spec = blend["spec"]
            parts = spec.split("+")
            for part in parts:
                voice_name = part.split("*")[0].strip()
                assert voice_name in VOICE_META, f"Blend references unknown voice: {voice_name}"


class TestVoiceLang:
    def test_american_english_prefix(self):
        assert _voice_lang("af_heart") == "en-us"
        assert _voice_lang("am_michael") == "en-us"

    def test_british_english_prefix(self):
        assert _voice_lang("bf_emma") == "en-gb"
        assert _voice_lang("bm_george") == "en-gb"

    def test_japanese_prefix(self):
        assert _voice_lang("jf_alpha") == "ja"

    def test_mandarin_prefix(self):
        assert _voice_lang("zf_xiaobei") == "cmn"

    def test_french_prefix(self):
        assert _voice_lang("ff_siwis") == "fr-fr"

    def test_blend_uses_first_component_lang(self):
        assert _voice_lang("bf_emma*0.6+af_bella*0.4") == "en-gb"

    def test_empty_defaults_to_en_us(self):
        assert _voice_lang("") == "en-us"

    def test_unknown_prefix_defaults_to_en_us(self):
        assert _voice_lang("xx_unknown") == "en-us"


class TestWalkVoiceResolution:
    def test_walk_prefix_loads_npy(self, tmp_path):
        emb = np.random.randn(1, 256).astype(np.float32)
        walk_dir = tmp_path / "voice_walks"
        walk_dir.mkdir()
        np.save(walk_dir / "my_voice.npy", emb)

        tts = KokoroTTS.__new__(KokoroTTS)
        tts._loaded = True
        tts._kokoro = MagicMock()
        tts._voice_cache = {}
        tts._voice_cache_lock = __import__("threading").Lock()

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.data_dir = str(tmp_path)
            result = tts._resolve_voice("walk:my_voice")

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, emb)

    def test_walk_prefix_missing_falls_back(self):
        tts = KokoroTTS.__new__(KokoroTTS)
        tts._loaded = True
        tts._kokoro = MagicMock()
        tts._voice_cache = {}
        tts._voice_cache_lock = __import__("threading").Lock()

        with patch("augmentum.config.settings") as mock_settings:
            mock_settings.data_dir = "/nonexistent"
            result = tts._resolve_voice("walk:missing_voice")

        assert result == "af_heart"
