"""Smoke tests for the streaming WAV writer (audiobook synthesis backbone)."""

from __future__ import annotations

import io
import wave

import pytest


def _mk_wav_blob(nframes: int, val: bytes = b"\x11\x22", rate: int = 24000) -> bytes:
    b = io.BytesIO()
    w = wave.open(b, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(val * nframes)
    w.close()
    return b.getvalue()


def test_extract_wav_data_handles_plain_pcm():
    from augmentum.voice.wav_writer import _extract_wav_data
    assert _extract_wav_data(_mk_wav_blob(10)) == b"\x11\x22" * 10
    # Not a WAV → treated as raw PCM (defensive passthrough).
    assert _extract_wav_data(b"\x00\x01\x02\x03") == b"\x00\x01\x02\x03"


def test_stream_write_and_header_backfill(tmp_path):
    from augmentum.voice.wav_writer import WavWriter
    p = str(tmp_path / "a.wav")
    w = WavWriter(p, sample_rate=24000)
    w.append_wav(_mk_wav_blob(50))
    w.flush_header()
    # Mid-write the file is already a valid WAV reflecting what's been written.
    with wave.open(p, "rb") as r:
        assert r.getnframes() == 50
        assert r.getframerate() == 24000 and r.getnchannels() == 1
    w.append_pcm(b"\x33\x44" * 30)
    w.close()
    with wave.open(p, "rb") as r:
        assert r.getnframes() == 80


def test_resume_appends_to_existing_partial(tmp_path):
    from augmentum.voice.wav_writer import WavWriter
    p = str(tmp_path / "b.wav")
    with WavWriter(p, sample_rate=24000) as w0:
        w0.append_pcm(b"\x00\x01" * 100)
    # Re-open in resume mode and continue from where we left off.
    w = WavWriter(p, sample_rate=24000, resume=True)
    assert w.data_bytes == 200
    w.append_pcm(b"\x00\x02" * 50)
    w.close()
    with wave.open(p, "rb") as r:
        assert r.getnframes() == 150


def test_append_wav_adopts_source_sample_rate(tmp_path):
    # The Kokoro+HBE "deep and slow-mo" class: engine emits 48 kHz WAV,
    # writer constructed at the assumed 24 kHz default. The source WAV's
    # rate is authoritative — the output header must say 48 kHz.
    from augmentum.voice.wav_writer import WavWriter
    p = str(tmp_path / "hbe.wav")
    w = WavWriter(p, sample_rate=24000)
    w.append_wav(_mk_wav_blob(50, rate=48000))
    w.append_wav(_mk_wav_blob(30, rate=48000))
    w.close()
    with wave.open(p, "rb") as r:
        assert r.getframerate() == 48000
        assert r.getnframes() == 80


def test_resume_adopts_partial_file_rate(tmp_path):
    # A crash-resumed file's existing PCM was written at the rate in its
    # header; the resumed writer must not re-label it at the constructor rate.
    from augmentum.voice.wav_writer import WavWriter
    p = str(tmp_path / "res48.wav")
    with WavWriter(p, sample_rate=24000) as w0:
        w0.append_wav(_mk_wav_blob(40, rate=48000))
    w = WavWriter(p, sample_rate=24000, resume=True)
    assert w.sample_rate == 48000
    w.close()


def test_parse_wav_fmt(tmp_path):
    from augmentum.voice.wav_writer import _parse_wav_fmt
    assert _parse_wav_fmt(_mk_wav_blob(5, rate=44100)) == 44100
    assert _parse_wav_fmt(b"not a wav") == 0


def test_close_is_idempotent_and_discard_removes(tmp_path):
    from augmentum.voice.wav_writer import WavWriter
    p = tmp_path / "c.wav"
    w = WavWriter(str(p), sample_rate=24000)
    w.append_pcm(b"\x00\x01" * 10)
    w.close()
    w.close()  # no-op
    assert p.exists()
    w.discard()
    assert not p.exists()


def test_empty_writer_flag(tmp_path):
    from augmentum.voice.wav_writer import WavWriter
    w = WavWriter(str(tmp_path / "d.wav"), sample_rate=24000)
    assert w.empty is True
    w.append_pcm(b"\x00\x01")
    assert w.empty is False
    w.close()


def test_narration_synth_helpers_import():
    """The job handler's reusable helpers (used by the studio long path too) load."""
    from augmentum.jobs.handlers.narration_synth import (
        _chunk_text, _concat_wav, _engine_wav_blobs, _resolve_synth_engine, _partial_path,
    )
    assert _chunk_text("Hi. Bye.") == ["Hi. Bye."]   # short → one chunk
    assert "narration_work" in _partial_path("/data", "u", "file", "x")
    assert callable(_engine_wav_blobs) and callable(_resolve_synth_engine) and callable(_concat_wav)


@pytest.mark.parametrize("fmt", ["wav", "mp3"])
def test_narration_store_progress_columns(fmt):
    """epub_narrations carries the resume checkpoint columns."""
    import sqlite3
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE users(id TEXT PRIMARY KEY)")
    import pathlib
    mig = pathlib.Path("augmentum/state/migrations")
    c.executescript((mig / "148_epub_narration_pairings.sql").read_text())
    c.executescript((mig / "149_epub_narration_progress_checkpoint.sql").read_text())
    cols = {r[1] for r in c.execute("PRAGMA table_info(epub_narrations)")}
    assert {"processed_chunks", "total_chunks"} <= cols
