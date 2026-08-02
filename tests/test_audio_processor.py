"""Tests for the voice audio preprocessor (AGC + normalization)."""

from __future__ import annotations

import numpy as np

from augmentum.voice.audio_processor import AudioProcessor, normalize_pcm

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32 ms at 16 kHz
FRAME_BYTES = FRAME_SAMPLES * 2


def _make_pcm(amplitude: float = 0.1, samples: int = FRAME_SAMPLES) -> bytes:
    """Generate a sine wave PCM16 buffer at the given amplitude (0-1)."""
    t = np.linspace(0, samples / SAMPLE_RATE, samples, dtype=np.float32)
    wave = (amplitude * 32767 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    return wave.tobytes()


def _rms(pcm: bytes) -> float:
    """Compute RMS of PCM16 bytes."""
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


# ---------------------------------------------------------------------------
# normalize_pcm
# ---------------------------------------------------------------------------


class TestNormalizePcm:
    def test_quiet_audio_is_boosted(self):
        quiet = _make_pcm(amplitude=0.05)
        normalized = normalize_pcm(quiet, target_peak=0.9)
        assert _rms(normalized) > _rms(quiet) * 5

    def test_loud_audio_unchanged(self):
        loud = _make_pcm(amplitude=0.95)
        result = normalize_pcm(loud, target_peak=0.9)
        # Should be returned unchanged (already above 0.9 * 0.95 threshold)
        assert result == loud

    def test_silent_audio_unchanged(self):
        silent = b"\x00" * FRAME_BYTES
        result = normalize_pcm(silent)
        assert result == silent

    def test_empty_audio_unchanged(self):
        result = normalize_pcm(b"")
        assert result == b""

    def test_preserves_length(self):
        pcm = _make_pcm(amplitude=0.1)
        result = normalize_pcm(pcm)
        assert len(result) == len(pcm)

    def test_no_clipping(self):
        """Even with very quiet input, normalization should not clip."""
        quiet = _make_pcm(amplitude=0.01)
        normalized = normalize_pcm(quiet, target_peak=0.9)
        samples = np.frombuffer(normalized, dtype=np.int16)
        assert np.max(np.abs(samples)) <= 32767

    def test_custom_target(self):
        quiet = _make_pcm(amplitude=0.05)
        norm_low = normalize_pcm(quiet, target_peak=0.5)
        norm_high = normalize_pcm(quiet, target_peak=0.9)
        assert _rms(norm_high) > _rms(norm_low)


# ---------------------------------------------------------------------------
# AudioProcessor (numpy AGC fallback)
# ---------------------------------------------------------------------------


class TestAudioProcessorNumpy:
    """Test the numpy AGC fallback (no webrtc-noise-gain)."""

    def test_init_without_wng(self):
        proc = AudioProcessor(agc_enabled=True, ns_enabled=False)
        # Force init (lazy)
        result = proc.process_frame(_make_pcm(0.05))
        assert len(result) == FRAME_BYTES

    def test_quiet_speech_is_boosted(self):
        proc = AudioProcessor(agc_enabled=True, ns_enabled=False)
        quiet = _make_pcm(amplitude=0.02)

        # Feed several frames to let the AGC adapt
        for _ in range(10):
            result = proc.process_frame(quiet)

        # After adaptation, output should be louder
        assert _rms(result) > _rms(quiet)

    def test_agc_does_not_amplify_silence(self):
        proc = AudioProcessor(agc_enabled=True, ns_enabled=False)
        silent = b"\x00" * FRAME_BYTES
        result = proc.process_frame(silent)
        assert result == silent

    def test_disabled_is_passthrough(self):
        proc = AudioProcessor(agc_enabled=False, ns_enabled=False)
        pcm = _make_pcm(amplitude=0.05)
        result = proc.process_frame(pcm)
        assert result == pcm

    def test_reset_clears_gain(self):
        proc = AudioProcessor(agc_enabled=True, ns_enabled=False)
        # Warm up
        for _ in range(10):
            proc.process_frame(_make_pcm(0.02))
        assert proc._numpy_gain != 1.0
        proc.reset()
        assert proc._numpy_gain == 1.0

    def test_no_clipping(self):
        proc = AudioProcessor(
            agc_enabled=True, ns_enabled=False, agc_target_dbfs=-6.0,
        )
        # Feed moderately loud audio — AGC should boost but not clip
        pcm = _make_pcm(amplitude=0.3)
        for _ in range(20):
            result = proc.process_frame(pcm)
        samples = np.frombuffer(result, dtype=np.int16)
        assert np.max(np.abs(samples)) <= 32767

    def test_gain_adapts_over_time(self):
        """Gain should increase over multiple quiet frames."""
        proc = AudioProcessor(agc_enabled=True, ns_enabled=False)
        quiet = _make_pcm(amplitude=0.01)

        gains = []
        for _ in range(20):
            proc.process_frame(quiet)
            gains.append(proc._numpy_gain)

        # Gain should be increasing
        assert gains[-1] > gains[0]


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigDefaults:
    def test_default_threshold(self):
        from augmentum.config import settings
        assert settings.voice_vad_speech_threshold == 0.5

    def test_agc_enabled_by_default(self):
        from augmentum.config import settings
        assert settings.voice_audio_agc is True

    def test_normalize_enabled_by_default(self):
        from augmentum.config import settings
        assert settings.voice_stt_normalize is True


# ---------------------------------------------------------------------------
# webrtc-noise-gain 10 ms framing
#
# The pipeline feeds 32 ms (1024-byte) frames but Process10ms only accepts
# 320-byte frames, and 1024 is not a multiple of 320.  The old code emitted
# the 64-byte remainder unprocessed, splicing raw audio into every frame at
# 31.25 Hz.  These tests pin the carry behaviour.
# ---------------------------------------------------------------------------


class _FakeChunk:
    def __init__(self, audio: bytes) -> None:
        self.audio = audio
        self.is_speech = True


class _FakeWng:
    """Stand-in for webrtc_noise_gain.AudioProcessor.

    Marks every processed sample with a per-chunk sequence number, so the
    output stream reveals both unprocessed passthrough (sample value 0, which
    never appears in the input) and any lost or duplicated chunk.
    """

    def __init__(self) -> None:
        self.calls = 0

    def Process10ms(self, pcm: bytes) -> _FakeChunk:  # noqa: N802 — upstream name
        assert len(pcm) == 320, f"expected a 10 ms frame, got {len(pcm)} bytes"
        self.calls += 1
        marked = np.full(160, self.calls, dtype=np.int16)
        return _FakeChunk(marked.tobytes())


def _wng_processor() -> tuple[AudioProcessor, _FakeWng]:
    proc = AudioProcessor(agc_enabled=True, ns_enabled=True)
    fake = _FakeWng()
    proc._wng = fake
    proc._use_wng = True
    proc._initialized = True
    return proc, fake


class TestWngFraming:
    def test_output_length_always_matches_input(self):
        """The VAD downstream requires exactly 1024 bytes back."""
        proc, _ = _wng_processor()
        for _ in range(20):
            out = proc.process_frame(_make_pcm(amplitude=0.5))
            assert len(out) == FRAME_BYTES

    def test_no_unprocessed_audio_is_spliced_in(self):
        """Regression: the 64-byte remainder used to pass through raw.

        Every sample the fake emits is non-zero and equal to its chunk index,
        so a sample carrying an input value means raw audio leaked through.
        """
        proc, _ = _wng_processor()
        raw_marker = np.full(FRAME_SAMPLES, 12345, dtype=np.int16).tobytes()

        collected = []
        for _ in range(20):
            collected.append(proc.process_frame(raw_marker))

        samples = np.frombuffer(b"".join(collected), dtype=np.int16)
        assert not np.any(samples == 12345), "raw input leaked into the output"

    def test_no_samples_lost_or_duplicated(self):
        """Chunk markers must appear in order, 160 samples each."""
        proc, fake = _wng_processor()

        collected = []
        for _ in range(25):
            collected.append(proc.process_frame(_make_pcm(amplitude=0.5)))

        samples = np.frombuffer(b"".join(collected), dtype=np.int16)
        # Leading priming silence, then the processed stream.
        assert np.all(samples[:160] == 0)
        stream = samples[160:]
        # Markers are strictly non-decreasing and advance by exactly 1.
        boundaries = np.flatnonzero(np.diff(stream)) + 1
        steps = np.diff(stream)[np.flatnonzero(np.diff(stream))]
        assert np.all(steps == 1), "chunk sequence skipped or repeated"
        # Every completed run is exactly one 10 ms chunk long.
        run_lengths = np.diff(np.concatenate(([0], boundaries)))
        assert np.all(run_lengths == 160)
        assert fake.calls >= 25 * FRAME_BYTES // 320

    def test_exact_10ms_frames_take_the_zero_latency_path(self):
        proc, fake = _wng_processor()
        out = proc.process_frame(_make_pcm(samples=160))
        assert len(out) == 320
        assert fake.calls == 1
        assert not proc._wng_primed

    def test_reset_clears_carry(self):
        proc, _ = _wng_processor()
        proc.process_frame(_make_pcm(amplitude=0.5))
        proc.reset()
        assert not proc._wng_in
        assert not proc._wng_out
        assert not proc._wng_primed
