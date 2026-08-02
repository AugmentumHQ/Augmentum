"""Real-time audio preprocessing for voice chat — AGC, noise suppression,
and normalization to improve STT accuracy on quiet speakers.

Processing chain (per 10 ms frame):
  1. Noise suppression — reduce background noise
  2. Automatic Gain Control — boost quiet speech to a consistent level

Per-utterance (before batch STT):
  3. Peak normalization — scale the full utterance so the loudest sample
     hits a target amplitude (default 0.9)

Uses ``webrtc-noise-gain`` when available (WebRTC NS + AGC, <1% CPU).
Falls back to a pure-numpy AGC implementation that's good enough for
the VAD-sensitivity problem even without noise suppression.
"""

from __future__ import annotations

import numpy as np

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# 16 kHz mono, 16-bit PCM
SAMPLE_RATE = 16000
# webrtc-noise-gain works on 10 ms frames
# 10ms at 16kHz = 160 samples × 2 bytes = 320 bytes
_WNG_FRAME_BYTES = 320


# ---------------------------------------------------------------------------
# Utterance-level normalization (used before batch STT)
# ---------------------------------------------------------------------------


def normalize_pcm(
    pcm_bytes: bytes,
    target_peak: float = 0.9,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Peak-normalize a PCM16 buffer so the loudest sample hits *target_peak*.

    This is the single highest-impact preprocessing step for quiet audio
    going into Whisper — community benchmarks show 15-30% WER improvement.

    Returns the same length PCM16 bytes.  If the audio is silent (all zeros)
    or already loud enough, returns the original bytes unchanged.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return pcm_bytes

    peak = np.max(np.abs(samples))
    if peak < 1.0:
        # Completely silent — nothing to normalize
        return pcm_bytes

    current_peak_ratio = peak / 32767.0
    if current_peak_ratio >= target_peak * 0.95:
        # Already loud enough — skip to avoid unnecessary quantization noise
        return pcm_bytes

    gain = (target_peak * 32767.0) / peak
    normalized = np.clip(samples * gain, -32767, 32767).astype(np.int16)
    return normalized.tobytes()


# ---------------------------------------------------------------------------
# Real-time frame processor (wraps webrtc-noise-gain or numpy fallback)
# ---------------------------------------------------------------------------


class AudioProcessor:
    """Per-frame audio preprocessor for the voice pipeline.

    Applies noise suppression and AGC to every PCM frame before it reaches
    the VAD and STT stages.  Designed for 16 kHz mono 16-bit PCM.

    Parameters:
        agc_enabled:     Enable automatic gain control
        ns_enabled:      Enable noise suppression
        agc_target_dbfs: Target speech level in dBFS (e.g. -16 means speech
                         should average around -16 dBFS)
        ns_level:        Noise suppression aggressiveness (0-3 for webrtc,
                         ignored for numpy fallback)
    """

    def __init__(
        self,
        *,
        agc_enabled: bool = True,
        ns_enabled: bool = True,
        agc_target_dbfs: float = -16.0,
        ns_level: int = 2,
    ) -> None:
        self._agc_enabled = agc_enabled
        self._ns_enabled = ns_enabled
        self._agc_target_dbfs = agc_target_dbfs
        self._ns_level = ns_level

        self._wng: object | None = None  # webrtc-noise-gain AudioProcessor
        self._use_wng = False
        self._numpy_gain: float = 1.0  # Adaptive gain for numpy fallback
        self._initialized = False

        # Carry buffers for the webrtc 10 ms framing.  The pipeline feeds us
        # 32 ms (1024-byte) frames, which is NOT a multiple of 320, so every
        # call leaves a remainder.  These keep that remainder in the stream
        # instead of splicing unprocessed audio into the output.
        self._wng_in = bytearray()   # Unprocessed input awaiting a full 10 ms
        self._wng_out = bytearray()  # Processed output awaiting collection
        self._wng_primed = False

    def _init(self) -> None:
        """Lazy-init: try webrtc-noise-gain, fall back to numpy.

        The webrtc-noise-gain API:
          AudioProcessor(auto_gain_dbfs: int, noise_suppression_level: int)
          - auto_gain_dbfs: 0 = disabled, 1-31 = target level in dBFS
          - noise_suppression_level: 0 = disabled, 1-4 = aggressiveness
          .Process10ms(pcm_bytes) → ProcessedAudioChunk(.audio, .is_speech)
        """
        if self._initialized:
            return
        self._initialized = True

        if not self._agc_enabled and not self._ns_enabled:
            return

        try:
            from webrtc_noise_gain import AudioProcessor as WngAP

            # Map our config to the constructor parameters
            # auto_gain_dbfs: 0=disabled, otherwise target in dBFS (positive int, typically 3-20)
            agc_val = int(abs(self._agc_target_dbfs)) if self._agc_enabled else 0
            # noise_suppression_level: 0=disabled, 1=low, 2=moderate, 3=high, 4=very high
            ns_val = self._ns_level if self._ns_enabled else 0

            self._wng = WngAP(agc_val, ns_val)
            self._use_wng = True
            log.info("audio_processor_init", backend="webrtc-noise-gain",
                     agc_dbfs=agc_val, ns_level=ns_val)
        except ImportError:
            log.info("audio_processor_init", backend="numpy_agc",
                     agc=self._agc_enabled, ns=False,
                     note="pip install webrtc-noise-gain for noise suppression")
        except Exception as exc:
            log.warning("audio_processor_wng_failed", error=str(exc),
                        fallback="numpy_agc")
            self._use_wng = False

    def process_frame(self, pcm_bytes: bytes) -> bytes:
        """Process one PCM16 frame.  Returns processed PCM16 bytes.

        Accepts any frame size — internally splits into 10 ms chunks for
        webrtc-noise-gain (which requires exactly 10 ms frames), or
        processes the whole frame with the numpy fallback.

        Never raises — returns the original audio on any error so the
        voice pipeline stays alive even if audio processing fails.
        """
        try:
            if not self._initialized:
                self._init()
        except Exception as exc:
            log.warning("audio_processor_init_failed", error=str(exc))
            self._initialized = True  # Don't retry
            return pcm_bytes

        if not self._agc_enabled and not self._ns_enabled:
            return pcm_bytes

        try:
            if self._use_wng:
                return self._process_wng(pcm_bytes)
            if self._agc_enabled:
                return self._process_numpy_agc(pcm_bytes)
            return pcm_bytes
        except Exception as exc:
            log.warning("audio_processor_frame_error", error=str(exc))
            return pcm_bytes

    # --- webrtc-noise-gain path ---

    def _process_wng(self, pcm_bytes: bytes) -> bytes:
        """Process via webrtc-noise-gain (10 ms frame chunks).

        Process10ms expects exactly 320 bytes (10ms at 16kHz, 160 samples × 2 bytes).
        Returns ProcessedAudioChunk with .audio (bytes) and .is_speech (bool).

        The voice pipeline sends 32 ms (1024-byte) frames and the VAD
        downstream requires the returned frame to be exactly as long as the
        input, so we cannot simply emit whole 10 ms chunks.  Instead we carry
        the sub-frame remainder across calls and prime the output by one
        10 ms chunk of silence, which guarantees the output buffer can always
        satisfy a same-length read (worst-case shortfall for 1024/320 is 256
        bytes).  Cost: a one-time 10 ms latency at stream start.
        """
        if not pcm_bytes:
            return pcm_bytes

        # Exact-size callers with no carry in flight keep the zero-latency path.
        if (
            len(pcm_bytes) == _WNG_FRAME_BYTES
            and not self._wng_in
            and not self._wng_out
        ):
            return self._wng.Process10ms(pcm_bytes).audio

        if not self._wng_primed:
            self._wng_primed = True
            self._wng_out.extend(b"\x00" * _WNG_FRAME_BYTES)

        self._wng_in.extend(pcm_bytes)
        while len(self._wng_in) >= _WNG_FRAME_BYTES:
            chunk = bytes(self._wng_in[:_WNG_FRAME_BYTES])
            del self._wng_in[:_WNG_FRAME_BYTES]
            self._wng_out.extend(self._wng.Process10ms(chunk).audio)

        want = len(pcm_bytes)
        if len(self._wng_out) < want:
            # Should be unreachable given the priming above; stay in sync by
            # returning the input rather than emitting a short frame.
            log.warning("audio_processor_wng_underrun",
                        want=want, have=len(self._wng_out))
            return pcm_bytes

        out = bytes(self._wng_out[:want])
        del self._wng_out[:want]
        return out

    # --- Numpy AGC fallback ---

    def _process_numpy_agc(self, pcm_bytes: bytes) -> bytes:
        """Simple adaptive AGC using numpy — no noise suppression.

        Tracks the recent speech level and applies gain to bring it toward
        the target.  Uses a slow attack / fast release envelope to avoid
        pumping on transients.
        """
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if len(samples) == 0:
            return pcm_bytes

        # Measure frame RMS level
        rms = np.sqrt(np.mean(samples ** 2))
        if rms < 10.0:
            # Near-silence — don't adapt gain (would amplify noise)
            return pcm_bytes

        # Target RMS from dBFS
        target_rms = 32767.0 * (10.0 ** (self._agc_target_dbfs / 20.0))

        # Desired gain for this frame
        desired_gain = target_rms / rms

        # Clamp gain to reasonable range (don't amplify >30 dB or attenuate >6 dB)
        desired_gain = max(0.5, min(desired_gain, 32.0))

        # Smooth gain changes: slow attack (ramp up slowly to avoid
        # amplifying noise bursts), fast release (pull back quickly
        # to avoid clipping)
        if desired_gain > self._numpy_gain:
            # Attack: slow ramp up (~300 ms time constant at 32 ms frames)
            alpha = 0.1
        else:
            # Release: fast pull back (~64 ms time constant)
            alpha = 0.5
        self._numpy_gain += alpha * (desired_gain - self._numpy_gain)

        # Apply gain with clipping protection
        amplified = np.clip(
            samples * self._numpy_gain, -32767, 32767,
        ).astype(np.int16)
        return amplified.tobytes()

    def reset(self) -> None:
        """Reset internal state (call between voice sessions)."""
        self._numpy_gain = 1.0
        self._wng_in.clear()
        self._wng_out.clear()
        self._wng_primed = False
