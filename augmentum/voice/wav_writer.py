"""Streaming WAV writer + WAV→MP3 transcode — for book-length TTS output.

Synthesizing a multi-hour audiobook can't hold the whole PCM in memory, and
a crash mid-job shouldn't restart from chapter one. :class:`WavWriter`
reserves a 44-byte header, appends PCM as it arrives, and backfills the
header on flush — so the file on disk is always a valid (or absent) WAV,
and a half-written file can be resumed (re-open in ``resume`` mode: the
data length is recomputed from the file size, the stale header is
irrelevant until the next flush).

``wav_to_mp3`` does a file→file ffmpeg transcode so the final compressed
deliverable never round-trips through process memory either.

Mono PCM16 only (matches the TTS engines' output). Stdlib + ffmpeg.
"""

from __future__ import annotations

import os
import struct
import subprocess

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_HEADER_BYTES = 44


def _wav_header(data_bytes: int, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_bytes, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits,
        b"data", data_bytes,
    )


class WavWriter:
    """Append-only WAV file whose header is backfilled on flush.

    Usage::

        w = WavWriter(path, sample_rate=24000)            # fresh
        w = WavWriter(path, sample_rate=24000, resume=True)  # continue a partial
        w.append_pcm(pcm16_bytes)        # or w.append_wav(complete_wav_blob)
        w.flush_header()                 # cheap; keeps the file replayable mid-write
        w.close()                        # final header backfill + close
        # w.discard()                    # close + delete the file

    Not thread-safe — drive it from one worker.
    """

    def __init__(self, path: str, *, sample_rate: int = 24_000, channels: int = 1, resume: bool = False) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.channels = channels
        self._closed = False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if resume and os.path.exists(path) and os.path.getsize(path) >= _HEADER_BYTES:
            self._f = open(path, "r+b")
            self.data_bytes = os.path.getsize(path) - _HEADER_BYTES
            # A partial file's PCM was written at the rate in its header —
            # adopt it so the resumed run doesn't re-label existing samples.
            try:
                self._f.seek(0)
                prior = _parse_wav_fmt(self._f.read(_HEADER_BYTES))
                if prior:
                    self.sample_rate = prior
            except Exception:  # noqa: BLE001 — stale header, keep constructor rate
                pass
            self._f.seek(0, os.SEEK_END)
        else:
            self._f = open(path, "wb")
            self._f.write(b"\x00" * _HEADER_BYTES)   # reserve the header
            self.data_bytes = 0

    def append_pcm(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        self._f.write(pcm)
        self.data_bytes += len(pcm)

    def append_wav(self, wav_blob: bytes) -> None:
        """Extract the PCM data chunk from a complete WAV blob and append it.

        The source WAV's sample rate is authoritative: on the first append
        the writer adopts it (the constructor rate is only a fallback for
        raw-PCM appends). TTS engines can emit at rates other than the
        assumed default — e.g. Kokoro with HBE upsamples 24→48 kHz — and
        writing their PCM under the wrong header plays back slow and
        pitched down.
        """
        pcm = _extract_wav_data(wav_blob)
        if not pcm:
            return
        src_rate = _parse_wav_fmt(wav_blob)
        if src_rate and src_rate != self.sample_rate:
            if self.data_bytes == 0:
                self.sample_rate = src_rate
            else:
                # Mid-file rate change: the PCM already written is at the
                # adopted rate — appending a different rate would corrupt
                # playback. Keep writing (audio > silence) but say so.
                log.warning(
                    "wav_writer_rate_mismatch", path=self.path,
                    header_rate=self.sample_rate, blob_rate=src_rate,
                )
        self.append_pcm(pcm)

    def flush_header(self) -> None:
        """Rewrite the header for the bytes written so far (keeps the file valid)."""
        if self._closed:
            return
        pos = self._f.tell()
        self._f.seek(0)
        self._f.write(_wav_header(self.data_bytes, self.sample_rate, self.channels))
        self._f.seek(pos)
        self._f.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._f.seek(0)
            self._f.write(_wav_header(self.data_bytes, self.sample_rate, self.channels))
        except Exception as exc:  # noqa: BLE001
            log.warning("wav_writer_close_header_failed", path=self.path, error=str(exc))
        finally:
            self._f.close()

    def discard(self) -> None:
        """Close and delete the file (e.g. after the artifact has been saved)."""
        self.close()
        try:
            os.remove(self.path)
        except OSError:
            pass

    @property
    def empty(self) -> bool:
        return self.data_bytes == 0

    def __enter__(self) -> WavWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _parse_wav_fmt(wav_bytes: bytes) -> int:
    """Return the sample rate from a WAV blob's fmt chunk, or 0 if absent."""
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return 0
    i = 12
    n = len(wav_bytes)
    while i + 8 <= n:
        cid = wav_bytes[i:i + 4]
        size = struct.unpack_from("<I", wav_bytes, i + 4)[0]
        body = i + 8
        if cid == b"fmt " and body + 8 <= n:
            return struct.unpack_from("<I", wav_bytes, body + 4)[0]
        i = body + size + (size & 1)
    return 0


def _extract_wav_data(wav_bytes: bytes) -> bytes:
    """Return the PCM payload of a WAV blob (walks chunks; tolerates LIST/INFO)."""
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        # Not a WAV — assume it's already raw PCM (defensive).
        return wav_bytes
    i = 12
    n = len(wav_bytes)
    while i + 8 <= n:
        cid = wav_bytes[i:i + 4]
        size = struct.unpack_from("<I", wav_bytes, i + 4)[0]
        body = i + 8
        if cid == b"data":
            return wav_bytes[body:body + size] if body + size <= n else wav_bytes[body:]
        i = body + size + (size & 1)   # chunks are word-aligned
    return b""


def wav_to_mp3(in_path: str, out_path: str, *, bitrate_kbps: int = 128) -> bool:
    """Transcode a WAV file to MP3 in place via ffmpeg (file→file, no memory).

    Returns True on success. Falls back to leaving the WAV alone (caller
    keeps using it) on any failure.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", in_path,
        "-codec:a", "libmp3lame", "-b:a", f"{int(bitrate_kbps)}k",
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            log.warning(
                "wav_to_mp3_failed", returncode=proc.returncode,
                stderr=(proc.stderr or b"")[:300].decode(errors="replace"),
            )
            try:
                os.remove(out_path)
            except OSError:
                pass
            return False
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except FileNotFoundError:
        log.warning("wav_to_mp3_no_ffmpeg")
        return False
    except Exception as exc:  # noqa: BLE001
        log.warning("wav_to_mp3_error", error=str(exc))
        return False
