"""Smoke tests for the bundled PocketTTS engine + its wiring.

Tier 1 — module imports, singleton, segmenter, WAV header validity,
clone-path resolver, graceful behaviour without the model files.
Actually loading the model (~236MB download from HF) is a Tier 3
concern; those tests belong under ``tests/live/`` if added.
"""

from __future__ import annotations

import struct

import pytest

# ---------------------------------------------------------------------------
# Module shape
# ---------------------------------------------------------------------------


def test_module_imports_and_constructs():
    from augmentum.voice.pocket_tts import (
        _DEFAULT_VOICE,
        _DEFAULT_VOICE_NAMES,
        _FALLBACK_VOICE_NAMES,
        VOICE_META,
        PocketTTS,
    )
    # Catalog is loaded from the installed package; in the test venv that
    # falls back to the hardcoded list. Either way, the floor is the 8
    # classic voices we shipped from day one.
    assert len(_DEFAULT_VOICE_NAMES) >= len(_FALLBACK_VOICE_NAMES)
    assert _DEFAULT_VOICE in _DEFAULT_VOICE_NAMES
    assert set(VOICE_META) == set(_DEFAULT_VOICE_NAMES)

    eng = PocketTTS.instance()
    assert eng is PocketTTS.instance()           # singleton
    assert eng.is_available is False             # no model in the test env
    # get_voices() returns the built-in catalog regardless of load state —
    # callers use it for picker hydration before forcing a load.
    assert eng.get_voices() == list(_DEFAULT_VOICE_NAMES)
    status = PocketTTS.status()
    assert status["available"] is False
    assert status["voices"] == 0
    assert status["language"] == "english"


def test_load_model_graceful_without_package():
    """Without the ``pocket-tts`` pip dep installed (test venv), the
    engine logs a warning and stays unavailable instead of raising."""
    from augmentum.voice.pocket_tts import PocketTTS
    eng = PocketTTS()                            # fresh instance, not singleton
    eng.load_model()
    assert eng.is_available is False


# ---------------------------------------------------------------------------
# Text segmenter
# ---------------------------------------------------------------------------


def test_text_segmenter_basic():
    from augmentum.voice.pocket_tts import _split_sentences
    assert _split_sentences("") == []
    assert _split_sentences("Hello there. How are you?") == [
        "Hello there.", "How are you?",
    ]


def test_text_segmenter_handles_overlong_input():
    """No input should ever get dropped — overlong runs hard-wrap at the
    char cap so the engine still synthesises every word."""
    from augmentum.voice.pocket_tts import _split_sentences
    long_run = "word " * 400  # ~2000 chars, no sentence punctuation
    parts = _split_sentences(long_run)
    assert parts, "long input must not be dropped"
    assert all(len(p) <= 400 for p in parts)
    rejoined = " ".join(parts).strip()
    assert rejoined.count("word") == 400


# ---------------------------------------------------------------------------
# WAV header (streaming path's contract with the browser decoder)
# ---------------------------------------------------------------------------


def test_wav_header_is_canonical_44_bytes():
    """``_wav_header_sized`` emits a 44-byte canonical header carrying
    the REAL chunk sizes for the buffered-WAV emission path. Sentinel
    sizes were broken on every client that buffered the response into a
    Blob (iOS, HTTP/2 without visible chunked encoding, reverse-proxy
    buffering) so the format now waits for the full synthesis and emits
    a well-formed file."""
    from augmentum.voice.pocket_tts import PocketTTS
    pcm_len = 48_000  # 1 second @ 24 kHz mono int16
    header = PocketTTS._wav_header_sized(24_000, pcm_len)
    assert len(header) == 44

    # RIFF chunk header — size = 36 + data_size (header bytes + payload).
    assert header[0:4] == b"RIFF"
    riff_size = struct.unpack("<I", header[4:8])[0]
    assert riff_size == 36 + pcm_len
    assert header[8:12] == b"WAVE"

    # fmt subchunk — PCM int16 mono
    assert header[12:16] == b"fmt "
    fmt_chunk_size = struct.unpack("<I", header[16:20])[0]
    assert fmt_chunk_size == 16
    audio_format = struct.unpack("<H", header[20:22])[0]
    assert audio_format == 1              # PCM
    channels = struct.unpack("<H", header[22:24])[0]
    assert channels == 1
    sample_rate = struct.unpack("<I", header[24:28])[0]
    assert sample_rate == 24_000
    byte_rate = struct.unpack("<I", header[28:32])[0]
    assert byte_rate == 24_000 * 2        # mono × 2 bytes/sample
    block_align = struct.unpack("<H", header[32:34])[0]
    assert block_align == 2
    bits_per_sample = struct.unpack("<H", header[34:36])[0]
    assert bits_per_sample == 16

    # data subchunk — real size, NOT sentinel.
    assert header[36:40] == b"data"
    data_size = struct.unpack("<I", header[40:44])[0]
    assert data_size == pcm_len


def test_wav_header_varies_with_sample_rate():
    from augmentum.voice.pocket_tts import PocketTTS
    for sr in (16_000, 22_050, 24_000, 48_000):
        header = PocketTTS._wav_header_sized(sr, 0)
        rate_field = struct.unpack("<I", header[24:28])[0]
        byte_rate = struct.unpack("<I", header[28:32])[0]
        assert rate_field == sr
        assert byte_rate == sr * 2


# ---------------------------------------------------------------------------
# Clone-path resolver — used by the Chatterbox drop-in cloning UI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "alba",                       # built-in voice name
    "marius",                     # another built-in
    "/abs/path/sample.wav",       # absolute path
    "./relative/sample.wav",      # relative path
    "~/home/sample.wav",          # tilde-prefixed path
    "sample.wav",                 # filename with audio extension
    "hf://kyutai/tts-voices/x",   # HF URL
    "https://example.com/x.wav",  # HTTPS URL
    "",                           # empty
])
def test_resolve_clone_path_passes_through_non_clone_names(name):
    """Anything that looks like a built-in, a path, a URL, or an audio
    file extension shouldn't trigger the cloned-voice lookup — those
    flow straight to Pocket TTS's own resolver."""
    from augmentum.voice.pocket_tts import PocketTTS
    assert PocketTTS._resolve_clone_path(name) == ""


def test_resolve_clone_path_finds_existing_wav(tmp_path, monkeypatch):
    """A bare name should resolve to a matching file under
    ``{data_dir}/voices/<name>.wav`` so the cloned-voice UI's same-
    library contract works with Pocket TTS as well as Chatterbox."""
    from augmentum.voice.pocket_tts import PocketTTS

    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    clone_file = voices_dir / "myclone.wav"
    clone_file.write_bytes(b"\x00" * 100)        # placeholder

    # Patch the settings.data_dir lookup the resolver reads.
    from augmentum import config as _cfg_mod
    monkeypatch.setattr(_cfg_mod.settings, "data_dir", str(tmp_path))
    resolved = PocketTTS._resolve_clone_path("myclone")
    assert resolved == str(clone_file)


def test_resolve_clone_path_misses_when_file_absent(tmp_path, monkeypatch):
    from augmentum.voice.pocket_tts import PocketTTS
    (tmp_path / "voices").mkdir()
    from augmentum import config as _cfg_mod
    monkeypatch.setattr(_cfg_mod.settings, "data_dir", str(tmp_path))
    assert PocketTTS._resolve_clone_path("does_not_exist") == ""


# ---------------------------------------------------------------------------
# Provider wiring — the picker hydration path
# ---------------------------------------------------------------------------


def test_pocket_is_in_builtin_tts_ids():
    """The voice picker hydrates from ``_BUILTIN_TTS_IDS``; Pocket must
    be there for its voices to surface."""
    from augmentum.proxy.audio_routes import _BUILTIN_TTS_IDS
    assert "pockettts-builtin" in _BUILTIN_TTS_IDS


def test_pocket_is_clone_capable():
    """The Chatterbox-style voice cloning UI shows for clone-capable
    providers. Pocket has WAV-conditioning so it qualifies."""
    from augmentum.proxy.audio_routes import _is_clone_capable_provider
    assert _is_clone_capable_provider({"id": "pockettts-builtin"}) is True
    assert _is_clone_capable_provider({"id": "kokoro-builtin"}) is False
