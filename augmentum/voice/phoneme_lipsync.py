"""ARPAbet -> VRM viseme schedule generator for phoneme-driven lip sync.

Uses g2p_en for grapheme-to-phoneme conversion (CMU ARPAbet output),
then maps phonemes to the VRM standard 5-viseme set (aa/ih/ou/ee/oh)
with per-class duration estimates scaled to actual audio length.

Output format: ``{"duration_ms": int, "events": [{"t": ms, "v": name, "w": weight}, ...]}``

Consumed by the frontend lip-sync layer via ``AvatarLipSync.setVisemeSchedule()``.

This module is pure (no side effects, no I/O) and fully unit-testable.
The g2p_en model loads lazily on first call and stays warm for the process
lifetime.
"""

from __future__ import annotations

from typing import Iterable

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Map ARPAbet phonemes (stress stripped) to VRM standard 5-viseme set.
# Vowels carry the dominant visual signal; consonants render as brief
# transients that the lip-sync layer's lerp blends through naturally.
ARPA_TO_VISEME: dict[str, str] = {
    # Open / central vowels -> "aa" (open mouth)
    "AA": "aa", "AE": "aa", "AH": "aa", "ER": "aa",
    # Mid-front / lax vowels -> "ih" (smile / slight teeth)
    "IH": "ih", "EH": "ih", "EY": "ih",
    # High-front tense vowel -> "ee" (wide spread)
    "IY": "ee",
    # Rounded back vowels -> "ou" (rounded lips)
    "UH": "ou", "UW": "ou",
    # Open rounded vowels -> "oh"
    "AO": "oh", "OW": "oh",
    # Diphthongs -> dominant first-half target (Phase 1 simplification)
    "AY": "aa",  # "I" (a -> i)
    "AW": "aa",  # "OW" (a -> u)
    "OY": "oh",  # "OY" (o -> i)

    # Bilabial / labiodental closures -> silence (mouth closes briefly)
    "M": "sil", "B": "sil", "P": "sil", "F": "sil", "V": "sil",
    # Lip-rounded consonants -> "ou"
    "SH": "ou", "ZH": "ou", "CH": "ou", "JH": "ou", "W": "ou",
    # Alveolar / dental -> small "ih"
    "T": "ih", "D": "ih", "S": "ih", "Z": "ih", "N": "ih",
    "L": "ih", "TH": "ih", "DH": "ih", "R": "ih", "Y": "ih",
    # Velar -> small "aa" (back of mouth)
    "K": "aa", "G": "aa", "NG": "aa", "HH": "aa",
}

# Baseline per-phoneme durations in ms. Scaled uniformly to match actual
# audio duration in phonemes_to_schedule. Numbers reflect typical English
# phoneme lengths in conversational speech.
DEFAULT_DURATIONS_MS: dict[str, int] = {
    # Vowels (longer)
    "AA": 130, "AE": 110, "AH": 80,  "ER": 100,
    "IH": 90,  "EH": 100, "EY": 130,
    "IY": 110, "UH": 90,  "UW": 130,
    "AO": 120, "OW": 130,
    # Diphthongs (longer than monophthongs)
    "AY": 160, "AW": 160, "OY": 160,
    # Stops (short)
    "B": 40, "P": 40, "T": 40, "D": 40, "K": 50, "G": 50,
    # Fricatives (medium)
    "F": 80, "V": 70, "S": 90, "Z": 80, "SH": 90, "ZH": 80,
    "TH": 80, "DH": 70, "HH": 50,
    # Affricates
    "CH": 90, "JH": 80,
    # Nasals
    "M": 70, "N": 60, "NG": 80,
    # Liquids / glides
    "L": 60, "R": 70, "W": 60, "Y": 60,
}

# Visible-mouth weights — vowels open more, consonants are briefer transients.
VOWEL_WEIGHT = 0.85
CONSONANT_WEIGHT = 0.45

_VOWELS: frozenset[str] = frozenset({
    "AA", "AE", "AH", "ER", "IH", "EH", "EY",
    "IY", "UH", "UW", "AO", "OW", "AY", "AW", "OY",
})

# g2p_en emits these as standalone tokens for whitespace / punctuation.
_SILENCE_TOKENS: frozenset[str] = frozenset({
    " ", ".", ",", "!", "?", ";", ":", "\n", "\t",
    "-", "(", ")", '"', "'", "...",
})

# Default duration for an inter-word silence token (ms, pre-scaling).
_SILENCE_BASELINE_MS = 60

# Default duration for an unknown phoneme (ms, pre-scaling).
_UNKNOWN_BASELINE_MS = 70

# Leading silence to give the audio playback chain time to ramp up before
# the mouth starts moving. Subtracted from usable duration before scaling.
_LEADING_SILENCE_MS = 30

# Languages we currently support. Phase 1 = English only; non-English
# voices fall back to amplitude lip-sync at the caller.
_SUPPORTED_LANGS: frozenset[str] = frozenset({"en", "en-us", "en-gb"})

_g2p_instance = None  # lazy singleton


def is_lang_supported(lang: str) -> bool:
    """True when text_to_schedule will produce a usable schedule for this lang."""
    return (lang or "").lower() in _SUPPORTED_LANGS


def _strip_stress(phoneme: str) -> str:
    """Remove ARPAbet stress digit suffix: ``AH0`` -> ``AH``."""
    return phoneme.rstrip("0123456789")


def _get_g2p():
    """Return the lazily-initialized G2p instance, or None if unavailable."""
    global _g2p_instance
    if _g2p_instance is None:
        try:
            from g2p_en import G2p
            _g2p_instance = G2p()
        except Exception as exc:
            log.warning("phoneme_lipsync_g2p_unavailable", error=str(exc))
            return None
    return _g2p_instance


def text_to_phonemes(text: str) -> list[str]:
    """Convert text to ARPAbet phoneme stream (preserves stress markers).

    Returns an empty list on empty input or if g2p_en is unavailable.
    Punctuation and whitespace pass through as standalone tokens.
    """
    if not text or not text.strip():
        return []
    g2p = _get_g2p()
    if g2p is None:
        return []
    try:
        return list(g2p(text))
    except Exception as exc:
        log.warning("phoneme_lipsync_g2p_failed", error=str(exc), text_len=len(text))
        return []


def phonemes_to_schedule(
    phonemes: Iterable[str],
    audio_duration_ms: int,
) -> dict:
    """Build a viseme schedule from a phoneme list scaled to audio duration.

    Args:
        phonemes: ARPAbet token stream (with optional stress digits).
        audio_duration_ms: Actual TTS audio length in milliseconds.

    Returns:
        ``{"duration_ms": int, "events": [{"t": ms, "v": viseme_name, "w": weight}]}``
        Events are ordered by ``t``. The first event is always ``sil`` at
        ``t=0`` and the last event is ``sil`` at ``t=duration_ms`` so the
        mouth opens from and returns to closed.
    """
    if audio_duration_ms <= 0:
        return {"duration_ms": 0, "events": [{"t": 0, "v": "sil", "w": 0.0}]}

    phs = [p for p in phonemes if p]

    # Compute baseline durations for each token
    baselines: list[int] = []
    for p in phs:
        if p in _SILENCE_TOKENS:
            baselines.append(_SILENCE_BASELINE_MS)
        else:
            base = _strip_stress(p)
            baselines.append(DEFAULT_DURATIONS_MS.get(base, _UNKNOWN_BASELINE_MS))

    total_baseline = sum(baselines)
    events: list[dict] = [{"t": 0, "v": "sil", "w": 0.0}]

    if total_baseline == 0 or not phs:
        # Empty / silent input — single closed-mouth state across the duration.
        events.append({"t": audio_duration_ms, "v": "sil", "w": 0.0})
        return {"duration_ms": audio_duration_ms, "events": events}

    leading = min(_LEADING_SILENCE_MS, max(0, audio_duration_ms // 4))
    usable_ms = max(1, audio_duration_ms - leading)
    scale = usable_ms / total_baseline
    cursor_ms = leading

    for p, base_dur in zip(phs, baselines):
        # Cumulative-rounding overshoot is bounded by clamping event time to
        # the audio duration. Schedule events past the end would never play
        # anyway and confuse downstream invariants.
        clamped_t = min(cursor_ms, audio_duration_ms)
        scaled = max(1, int(round(base_dur * scale)))

        if p in _SILENCE_TOKENS:
            # Drop a closed-mouth marker at this point — the lerp will
            # collapse mouth shapes during inter-word gaps.
            events.append({"t": clamped_t, "v": "sil", "w": 0.0})
            cursor_ms += scaled
            continue

        base = _strip_stress(p)
        viseme = ARPA_TO_VISEME.get(base)
        if viseme is None:
            # Unknown phoneme — skip the visual event but still consume time.
            cursor_ms += scaled
            continue

        # "sil" means "all mouth shapes at zero" — closure consonants
        # (M/B/P/F/V) emit a sil event with weight 0 so the mouth visibly
        # closes between vowels, matching real lip behavior.
        if viseme == "sil":
            weight = 0.0
        else:
            weight = VOWEL_WEIGHT if base in _VOWELS else CONSONANT_WEIGHT
        events.append({"t": clamped_t, "v": viseme, "w": weight})
        cursor_ms += scaled

    # Always close the mouth at the end of the audio.
    events.append({"t": audio_duration_ms, "v": "sil", "w": 0.0})

    return {"duration_ms": audio_duration_ms, "events": events}


def text_to_schedule(
    text: str,
    audio_duration_ms: int,
    lang: str = "en-us",
) -> dict | None:
    """Convenience: text -> ARPAbet -> viseme schedule.

    Returns None if the language is unsupported or g2p_en is unavailable —
    callers should fall back to amplitude lip-sync in that case.
    """
    if not is_lang_supported(lang):
        return None
    phonemes = text_to_phonemes(text)
    if not phonemes:
        return None
    return phonemes_to_schedule(phonemes, audio_duration_ms)


def phonemes_to_normalized_schedule(phonemes: Iterable[str]) -> dict:
    """Build a viseme schedule with normalized [0.0, 1.0] timing.

    Used when audio duration isn't known at emission time — external TTS
    providers stream chunks, so the server can't tell how long the final
    audio will be until after streaming completes. Instead of waiting,
    we emit the schedule with relative timing and let the client rescale
    it once the audio decoder reports actual duration.

    The leading-silence offset that :func:`phonemes_to_schedule` bakes
    into absolute times is *not* included here — it's a client-side
    concern so the same offset policy (``min(30 ms, duration * 0.25)``)
    applies regardless of which TTS provider produced the audio.

    Returns ``{"events": [{"t_norm": float, "v": str, "w": float}, ...],
    "normalized": True}``. The first event is always ``sil`` at
    ``t_norm=0.0`` and the last is ``sil`` at ``t_norm=1.0``.
    """
    phs = [p for p in phonemes if p]
    events: list[dict] = [{"t_norm": 0.0, "v": "sil", "w": 0.0}]

    if not phs:
        events.append({"t_norm": 1.0, "v": "sil", "w": 0.0})
        return {"events": events, "normalized": True}

    baselines: list[int] = []
    for p in phs:
        if p in _SILENCE_TOKENS:
            baselines.append(_SILENCE_BASELINE_MS)
        else:
            base = _strip_stress(p)
            baselines.append(DEFAULT_DURATIONS_MS.get(base, _UNKNOWN_BASELINE_MS))

    total_baseline = sum(baselines)
    if total_baseline == 0:
        events.append({"t_norm": 1.0, "v": "sil", "w": 0.0})
        return {"events": events, "normalized": True}

    cursor_baseline = 0
    for p, base_dur in zip(phs, baselines):
        t_norm = min(1.0, cursor_baseline / total_baseline)
        if p in _SILENCE_TOKENS:
            events.append({"t_norm": t_norm, "v": "sil", "w": 0.0})
            cursor_baseline += base_dur
            continue

        base = _strip_stress(p)
        viseme = ARPA_TO_VISEME.get(base)
        if viseme is None:
            cursor_baseline += base_dur
            continue

        # "sil" weight 0 for closure consonants (M/B/P/F/V) so the mouth
        # visibly closes between vowels, matching real lip behavior.
        if viseme == "sil":
            weight = 0.0
        else:
            weight = VOWEL_WEIGHT if base in _VOWELS else CONSONANT_WEIGHT
        events.append({"t_norm": t_norm, "v": viseme, "w": weight})
        cursor_baseline += base_dur

    events.append({"t_norm": 1.0, "v": "sil", "w": 0.0})
    return {"events": events, "normalized": True}


def text_to_normalized_schedule(
    text: str,
    lang: str = "en-us",
) -> dict | None:
    """Convenience: text -> ARPAbet -> normalized viseme schedule.

    Returns None if the language is unsupported or g2p_en is unavailable —
    callers should fall back to amplitude lip-sync in that case.
    """
    if not is_lang_supported(lang):
        return None
    phonemes = text_to_phonemes(text)
    if not phonemes:
        return None
    return phonemes_to_normalized_schedule(phonemes)


# ASCII letters plus the punctuation that g2p_en's tokenizer expects.
# Em-dashes, curly quotes, and accented characters intentionally fail
# the gate — they're not bugs we should hide, they're signals that the
# text probably isn't standard English and amplitude is the right
# fallback.
_LOOKS_ENGLISH_KEEP = " .,!?;:'\"-"


def looks_english(text: str) -> bool:
    """Cheap gate: is the text plausibly English (Latin-script ASCII)?

    Used when the calling code doesn't have a language hint from the TTS
    provider — most external providers don't expose one. Conservative by
    design: false negatives (rejecting English text with stylized
    punctuation) just route the sentence to amplitude lip-sync, which is
    a safe fallback. False positives would push g2p_en into producing
    garbage phonemes that the avatar mouth would visibly fail to track.

    Threshold: ``> 90%`` of characters must be ASCII letters or common
    English punctuation. Whitespace is included so multi-word sentences
    pass cleanly.
    """
    if not text:
        return False
    eng_count = sum(
        1 for c in text
        if c.isascii() and (c.isalpha() or c in _LOOKS_ENGLISH_KEEP)
    )
    return eng_count / len(text) > 0.90
