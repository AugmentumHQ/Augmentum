"""Tests for voice/phoneme_lipsync.py — ARPAbet -> VRM viseme schedules."""

from __future__ import annotations

import pytest

from augmentum.voice.phoneme_lipsync import (
    _SILENCE_TOKENS,
    _VOWELS,
    ARPA_TO_VISEME,
    DEFAULT_DURATIONS_MS,
    _strip_stress,
    is_lang_supported,
    phonemes_to_schedule,
    text_to_phonemes,
    text_to_schedule,
)


class TestArpaToVisemeMap:
    """The ARPAbet -> VRM viseme mapping table."""

    def test_only_targets_vrm_standard_visemes(self):
        """All mapping outputs must be one of {aa, ih, ou, ee, oh, sil}."""
        valid = {"aa", "ih", "ou", "ee", "oh", "sil"}
        unknown = {v for v in ARPA_TO_VISEME.values() if v not in valid}
        assert not unknown, f"non-VRM viseme outputs: {unknown}"

    def test_covers_all_arpabet_vowels(self):
        """Every ARPAbet vowel must have a viseme target (no silent vowels)."""
        missing = _VOWELS - set(ARPA_TO_VISEME.keys())
        assert not missing, f"vowels missing viseme mapping: {missing}"

    def test_vowels_dont_map_to_sil(self):
        """A vowel rendering as closed-mouth would look broken."""
        for v in _VOWELS:
            assert ARPA_TO_VISEME[v] != "sil", f"{v} maps to sil — vowels must show mouth shape"

    def test_bilabials_close_mouth(self):
        """M/B/P must close the mouth (sil) — bilabial closure is visually critical."""
        for c in ("M", "B", "P"):
            assert ARPA_TO_VISEME[c] == "sil", f"{c} should map to sil (bilabial closure)"

    def test_labiodentals_close_mouth(self):
        """F/V should also visually close (labiodental — lower lip to teeth)."""
        for c in ("F", "V"):
            assert ARPA_TO_VISEME[c] == "sil"


class TestDefaultDurations:
    """Per-phoneme baseline duration table."""

    def test_every_mapped_phoneme_has_duration(self):
        """Every entry in ARPA_TO_VISEME must have a baseline duration."""
        missing = set(ARPA_TO_VISEME.keys()) - set(DEFAULT_DURATIONS_MS.keys())
        assert not missing, f"phonemes without duration: {missing}"

    def test_vowels_longer_than_stops(self):
        """Vowels should be longer than stop consonants on average."""
        vowel_avg = sum(DEFAULT_DURATIONS_MS[v] for v in _VOWELS) / len(_VOWELS)
        stops = ("B", "P", "T", "D", "K", "G")
        stop_avg = sum(DEFAULT_DURATIONS_MS[s] for s in stops) / len(stops)
        assert vowel_avg > stop_avg, f"vowels ({vowel_avg}) not longer than stops ({stop_avg})"

    def test_diphthongs_are_longest(self):
        """AY/AW/OY are diphthongs — visually two vowels, so longest baseline."""
        for d in ("AY", "AW", "OY"):
            assert DEFAULT_DURATIONS_MS[d] >= 150, f"{d} should be >= 150ms baseline"

    def test_all_durations_positive(self):
        for ph, dur in DEFAULT_DURATIONS_MS.items():
            assert dur > 0, f"{ph} has non-positive duration: {dur}"


class TestStripStress:
    def test_strips_single_digit(self):
        assert _strip_stress("AH0") == "AH"
        assert _strip_stress("AH1") == "AH"
        assert _strip_stress("AH2") == "AH"

    def test_no_stress_unchanged(self):
        assert _strip_stress("M") == "M"
        assert _strip_stress("HH") == "HH"

    def test_consonant_with_no_digit(self):
        assert _strip_stress("NG") == "NG"


class TestIsLangSupported:
    def test_english_variants_supported(self):
        assert is_lang_supported("en")
        assert is_lang_supported("en-us")
        assert is_lang_supported("en-gb")
        assert is_lang_supported("EN-US")  # case insensitive

    def test_other_languages_not_supported(self):
        assert not is_lang_supported("ja")
        assert not is_lang_supported("zh")
        assert not is_lang_supported("fr")

    def test_empty_lang(self):
        assert not is_lang_supported("")


class TestPhonemesToSchedule:
    """Pure phoneme-to-schedule logic (no g2p_en dependency)."""

    def test_empty_phonemes_yields_silent_schedule(self):
        sched = phonemes_to_schedule([], 1000)
        assert sched["duration_ms"] == 1000
        # Two events: opening sil at t=0 and closing sil at t=duration
        assert len(sched["events"]) == 2
        assert sched["events"][0]["v"] == "sil"
        assert sched["events"][-1]["v"] == "sil"
        assert sched["events"][-1]["t"] == 1000

    def test_zero_duration_yields_minimal_schedule(self):
        sched = phonemes_to_schedule(["HH", "AH0"], 0)
        assert sched["duration_ms"] == 0
        assert len(sched["events"]) == 1
        assert sched["events"][0] == {"t": 0, "v": "sil", "w": 0.0}

    def test_negative_duration_treated_as_zero(self):
        sched = phonemes_to_schedule(["HH"], -100)
        assert sched["duration_ms"] == 0

    def test_first_event_is_silence_at_zero(self):
        sched = phonemes_to_schedule(["HH", "AH0", "L", "OW1"], 1000)
        assert sched["events"][0] == {"t": 0, "v": "sil", "w": 0.0}

    def test_last_event_is_silence_at_duration(self):
        sched = phonemes_to_schedule(["HH", "AH0", "L", "OW1"], 1000)
        last = sched["events"][-1]
        assert last["v"] == "sil"
        assert last["t"] == 1000

    def test_events_monotonically_ordered(self):
        sched = phonemes_to_schedule(
            ["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"], 2000,
        )
        times = [e["t"] for e in sched["events"]]
        assert times == sorted(times), f"events out of order: {times}"

    def test_events_within_audio_duration(self):
        sched = phonemes_to_schedule(
            ["HH", "AH0", "L", "OW1", " ", "W", "ER1", "L", "D"], 2000,
        )
        for ev in sched["events"]:
            assert 0 <= ev["t"] <= 2000, f"event {ev} out of range"

    def test_vowels_get_higher_weight_than_consonants(self):
        sched = phonemes_to_schedule(["AA1", "T"], 1000)
        # Find the AA and T events
        aa_event = next(e for e in sched["events"] if e["v"] == "aa" and e["w"] > 0.0)
        t_event = next(e for e in sched["events"] if e["v"] == "ih" and e["w"] > 0.0)
        assert aa_event["w"] > t_event["w"]

    def test_punctuation_emits_silence_event(self):
        sched = phonemes_to_schedule(["HH", "AH0", ",", "B", "AY1"], 1500)
        # There should be a sil event in the middle (the comma)
        sil_events = [e for e in sched["events"] if e["v"] == "sil"]
        assert len(sil_events) >= 3  # opening, middle (comma), closing

    def test_unknown_phoneme_consumes_time_silently(self):
        # XYZ is not in our table — schedule should still complete
        sched = phonemes_to_schedule(["AA1", "XYZ", "T"], 1000)
        # Last event should still be at duration
        assert sched["events"][-1]["t"] == 1000
        # XYZ should NOT appear as a viseme event
        assert all(e["v"] != "xyz" for e in sched["events"])

    def test_stress_marker_does_not_affect_mapping(self):
        sched_stressed = phonemes_to_schedule(["AA0"], 500)
        sched_no_stress = phonemes_to_schedule(["AA"], 500)
        # Both should produce the same viseme sequence
        v_stressed = [e["v"] for e in sched_stressed["events"]]
        v_no_stress = [e["v"] for e in sched_no_stress["events"]]
        assert v_stressed == v_no_stress

    def test_duration_scaling_fits_audio(self):
        """For a given phoneme sequence, total event timing should fill the audio."""
        phs = ["HH", "EH1", "L", "OW1"] * 5
        sched = phonemes_to_schedule(phs, 3000)
        # Last non-trailing-sil event should be reasonably close to end
        # (within last 30% — the trailing sil ends at exactly 3000)
        assert sched["events"][-1]["t"] == 3000
        # Some event should be past the 50% mark
        midpoint_passed = any(e["t"] > 1500 for e in sched["events"])
        assert midpoint_passed

    def test_short_audio_long_text_compresses(self):
        """Long phoneme sequence in short audio: events compress, no overflow."""
        phs = ["HH", "EH1", "L", "OW1", " ", "W", "ER1", "L", "D"] * 10
        sched = phonemes_to_schedule(phs, 200)
        for ev in sched["events"]:
            assert ev["t"] <= 200, f"event past audio end: {ev}"

    def test_event_schema(self):
        """Every event must have t, v, w fields with correct types."""
        sched = phonemes_to_schedule(["AA1", "M", "P", "IY0"], 800)
        for ev in sched["events"]:
            assert set(ev.keys()) == {"t", "v", "w"}, f"bad event keys: {ev}"
            assert isinstance(ev["t"], int)
            assert isinstance(ev["v"], str)
            assert isinstance(ev["w"], (int, float))
            assert 0.0 <= ev["w"] <= 1.0

    def test_silence_events_have_zero_weight(self):
        sched = phonemes_to_schedule(["HH", "AH0", " ", "B"], 1000)
        for ev in sched["events"]:
            if ev["v"] == "sil":
                assert ev["w"] == 0.0


class TestTextToPhonemes:
    """End-to-end g2p_en integration. Skipped if g2p_en unavailable."""

    @pytest.fixture(autouse=True)
    def _require_g2p(self):
        try:
            from g2p_en import G2p  # noqa: F401
        except ImportError:
            pytest.skip("g2p_en not installed in this environment")

    def test_empty_text_returns_empty_list(self):
        assert text_to_phonemes("") == []
        assert text_to_phonemes("   ") == []

    def test_simple_word_returns_phonemes(self):
        out = text_to_phonemes("hello")
        assert len(out) > 0
        # All entries should be strings
        assert all(isinstance(p, str) for p in out)

    def test_arpabet_phonemes_present(self):
        """Output should contain recognizable ARPAbet phonemes."""
        out = text_to_phonemes("test")
        # "test" -> T EH1 S T (or similar)
        assert "T" in out
        # Some vowel should appear (with stress digit)
        has_vowel = any(_strip_stress(p) in _VOWELS for p in out)
        assert has_vowel

    def test_handles_punctuation(self):
        out = text_to_phonemes("Hello, world!")
        # Should pass through punctuation tokens
        assert any(p in _SILENCE_TOKENS for p in out)


class TestTextToSchedule:
    """End-to-end text -> schedule pipeline."""

    @pytest.fixture(autouse=True)
    def _require_g2p(self):
        try:
            from g2p_en import G2p  # noqa: F401
        except ImportError:
            pytest.skip("g2p_en not installed in this environment")

    def test_unsupported_lang_returns_none(self):
        assert text_to_schedule("konnichiwa", 1000, lang="ja") is None
        assert text_to_schedule("bonjour", 1000, lang="fr") is None

    def test_supported_lang_returns_schedule(self):
        sched = text_to_schedule("hello world", 1500, lang="en-us")
        assert sched is not None
        assert sched["duration_ms"] == 1500
        assert len(sched["events"]) >= 3  # opening sil + content + closing sil

    def test_empty_text_returns_none(self):
        assert text_to_schedule("", 1000, lang="en-us") is None
        assert text_to_schedule("   ", 1000, lang="en-us") is None

    def test_default_lang_is_english(self):
        sched = text_to_schedule("hello", 1000)
        assert sched is not None
