"""Pluggable data sources for the CSM voice-training harness.

A *source* maps a raw dataset on disk into a flat list of ``Record``
(audio_path, text, emotion, speaker_id). This is the seam that makes the
harness reusable: EARS today, a voice actor's labelled studio recordings
tomorrow, with zero changes to prepare_data/train — just a new config and (if
the layout is novel) a small adapter function here.

Add a source = write a function that returns list[Record] and register it in
SOURCES.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Record:
    audio_path: str
    text: str | None       # None -> prepare_data transcribes it (faster-whisper)
    emotion: str | None
    speaker_id: int


def ears_source(cfg: dict) -> list[Record]:
    """EARS (Expressive Anechoic Recordings of Speech), one speaker.

    EARS lays out one directory per speaker (e.g. ``p001``) of 48 kHz wavs
    whose filenames encode the style/emotion — e.g. ``emo_<emotion>_<level>``,
    ``freeform_<...>``, ``sentences_<...>``, ``rainbow_<style>``. Naming can
    vary by release: after extracting, ``ls`` your speaker dir and tweak
    ``_ears_emotion`` below if the leading token differs. Text is left None so
    every clip is transcribed (the read portions also ship official scripts you
    can substitute later for perfect alignment).
    """
    root = Path(cfg["source"]["ears_root"]) / cfg["source"]["ears_speaker"]
    if not root.is_dir():
        raise SystemExit(f"EARS speaker dir not found: {root}")
    spk = int(cfg.get("speaker_id", 0))
    out = [Record(str(w), None, _ears_emotion(w.stem), spk)
           for w in sorted(root.rglob("*.wav"))]
    if not out:
        raise SystemExit(f"No .wav files under {root}")
    return out


# Non-speech file groups — no usable transcript, would poison text->speech
# training. prepare_data drops any clip whose emotion is "__drop__".
_EARS_DROP = ("nonverbal", "vegetative", "melodic")
# Rainbow-passage reading styles -> tag (these are styles, not emotions, but
# the harness treats the tag generically, giving you whisper/loud/etc. knobs).
_EARS_STYLES = {"regular": "neutral", "whisper": "whisper", "loud": "loud",
                "fast": "fast", "slow": "slow",
                "highpitch": "highpitch", "lowpitch": "lowpitch"}


def _ears_emotion(stem: str) -> str:
    """Map an EARS p0NN filename stem to a tag (or '__drop__').

    Verified against speaker p012:
      emo_<emotion>_sentences|freeform  -> <emotion>   (23 emotions)
      rainbow_NN_<style>                -> style tag    (whisper/loud/...)
      sentences_NN | freeform_speech_NN | interjection_* -> neutral
      nonverbal_* | vegetative_* | melodic_*            -> __drop__
    """
    parts = stem.split("_")
    head = parts[0]
    if head in _EARS_DROP:
        return "__drop__"
    if head == "emo" and len(parts) >= 2:
        return parts[1]
    if head == "rainbow" and len(parts) >= 3:
        return _EARS_STYLES.get(parts[2], "neutral")
    # sentences_* (neutral read), freeform_speech_* (improv), interjection_*
    return "neutral"


def csv_source(cfg: dict) -> list[Record]:
    """Generic source — a CSV with columns ``audio`` [, ``text``] [, ``emotion``].

    This is the voice-actor path: hand the harness a folder of wavs plus a
    ``metadata.csv``. ``audio`` paths are resolved against ``source.audio_root``.
    """
    s = cfg["source"]
    base = Path(s.get("audio_root", "") or "")
    spk = int(cfg.get("speaker_id", 0))
    out: list[Record] = []
    with open(s["csv_path"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ap = base / row["audio"] if base.parts else Path(row["audio"])
            out.append(Record(str(ap), row.get("text") or None,
                              row.get("emotion") or None, spk))
    if not out:
        raise SystemExit(f"No rows in {s['csv_path']}")
    return out


SOURCES = {"ears": ears_source, "csv": csv_source}


def load_records(cfg: dict) -> list[Record]:
    t = cfg["source"]["type"]
    if t not in SOURCES:
        raise SystemExit(f"unknown source.type={t!r}; options: {list(SOURCES)}")
    return SOURCES[t](cfg)
