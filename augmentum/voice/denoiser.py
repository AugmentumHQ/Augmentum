"""DTLN real-time speech enhancement with band-pass preprocessing.

Integrates into the voice processing pipeline between raw audio capture
and VAD/STT.  The DTLN (Dual-signal Transformation LSTM Network) uses
two ONNX models in sequence to suppress non-stationary noise (crowd
chatter, TV, traffic) that WebRTC NS cannot handle.

Processing chain per 32 ms frame:
  1. Biquad highpass filter (80 Hz) — removes rumble, handling noise
  2. DTLN model 1 — STFT-domain magnitude masking via LSTM
  3. DTLN model 2 — time-domain enhancement via LSTM
  4. Output: clean 512-sample PCM16 frame

Latency: ~1 ms per frame on CPU via ONNX Runtime.
Models: https://github.com/breizhn/DTLN (~3.7 MB total)
Paper: https://arxiv.org/abs/2005.07551
"""

from __future__ import annotations

import math
import os

import numpy as np

from augmentum.utils.logging import get_logger
from augmentum.utils.paths import resolve_model_dir

log = get_logger(__name__)

# DTLN model constants (must match training configuration)
_BLOCK_LEN = 512     # FFT window size (samples)
_BLOCK_SHIFT = 128   # Hop size (samples)
_SAMPLE_RATE = 16000

# Default model location — /home/augmentum/.dtln in the Docker container
# (Dockerfile bakes weights there), platform-appropriate user cache dir
# on native installs. See augmentum/utils/paths.py.
_DEFAULT_MODEL_DIR = resolve_model_dir("dtln")


# ---------------------------------------------------------------------------
# Biquad Highpass Filter
# ---------------------------------------------------------------------------


class BiquadHighpass:
    """Stateful 2nd-order Butterworth highpass filter.

    Removes low-frequency rumble (HVAC, wind, handling noise, foot traffic)
    before neural processing.  Uses Direct Form II Transposed — no scipy
    dependency, runs in pure numpy.

    At 80 Hz cutoff / 16 kHz sample rate, this attenuates:
      - 40 Hz (HVAC fundamental):  ~-12 dB
      - 60 Hz (mains hum):         ~-7 dB
      - 80 Hz (cutoff):            ~-3 dB
      - 100 Hz+ (speech):          ~0 dB (passthrough)
    """

    def __init__(self, cutoff_hz: float = 80.0, sample_rate: int = _SAMPLE_RATE):
        # Butterworth biquad coefficients (Audio EQ Cookbook)
        w0 = 2.0 * math.pi * cutoff_hz / sample_rate
        cos_w0 = math.cos(w0)
        alpha = math.sin(w0) / (2.0 * 0.7071)  # Q = 1/√2 for Butterworth

        a0 = 1.0 + alpha
        self._b0 = (1.0 + cos_w0) / 2.0 / a0
        self._b1 = -(1.0 + cos_w0) / a0
        self._b2 = (1.0 + cos_w0) / 2.0 / a0
        self._a1 = -2.0 * cos_w0 / a0
        self._a2 = (1.0 - alpha) / a0

        # Filter state (Direct Form II Transposed)
        self._z1 = 0.0
        self._z2 = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply highpass filter to float32 sample array.  Maintains state
        across calls for continuous streaming."""
        out = np.empty_like(samples)
        z1, z2 = self._z1, self._z2
        b0, b1, b2 = self._b0, self._b1, self._b2
        a1, a2 = self._a1, self._a2
        for i in range(len(samples)):
            x = float(samples[i])
            y = b0 * x + z1
            z1 = b1 * x - a1 * y + z2
            z2 = b2 * x - a2 * y
            out[i] = y
        self._z1, self._z2 = z1, z2
        return out

    def reset(self) -> None:
        self._z1 = 0.0
        self._z2 = 0.0


# ---------------------------------------------------------------------------
# DTLN Neural Denoiser
# ---------------------------------------------------------------------------


class DTLNDenoiser:
    """Real-time neural speech denoiser using DTLN ONNX models.

    Processes 512-sample (32 ms at 16 kHz) PCM16 frames, maintaining
    LSTM hidden states across frames for temporal noise tracking.

    The denoiser uses two models in sequence per hop:
      - Model 1: STFT-domain magnitude masking (estimate noise mask)
      - Model 2: Time-domain signal enhancement (refine output)

    Each 512-sample frame is processed as 4 hops of 128 samples with
    overlap-add reconstruction.

    LSTM states persist across frames — the model learns the background
    noise profile over time, improving quality the longer it runs.
    """

    def __init__(self, model_dir: str = ""):
        self._model_dir = model_dir or _DEFAULT_MODEL_DIR
        self._sess1 = None
        self._sess2 = None
        self._loaded = False

        # Input/output names (discovered from model metadata)
        self._input_names_1: list[str] = []
        self._input_names_2: list[str] = []

        # LSTM states (shape discovered from model, persist across frames)
        self._states1: np.ndarray | None = None
        self._states2: np.ndarray | None = None

        # Overlap-add buffers
        self._in_buffer = np.zeros(_BLOCK_LEN, dtype=np.float32)
        self._out_buffer = np.zeros(_BLOCK_LEN, dtype=np.float32)

    def load_model(self) -> None:
        """Load ONNX models.  Call once before processing.

        Safe to call multiple times — subsequent calls are no-ops.
        If models aren't found or ONNX Runtime isn't available,
        logs a warning and the denoiser becomes a passthrough.
        """
        if self._loaded:
            return

        m1_path = os.path.join(self._model_dir, "model_1.onnx")
        m2_path = os.path.join(self._model_dir, "model_2.onnx")

        if not os.path.exists(m1_path) or not os.path.exists(m2_path):
            log.warning("dtln_models_not_found", path=self._model_dir,
                        note="Run with DTLN models in /home/augmentum/.dtln/ "
                             "or set voice_denoise_model_dir")
            return

        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._sess1 = ort.InferenceSession(m1_path, sess_options=opts)
            self._sess2 = ort.InferenceSession(m2_path, sess_options=opts)

            # Discover input names dynamically (robust across DTLN export versions)
            self._input_names_1 = [inp.name for inp in self._sess1.get_inputs()]
            self._input_names_2 = [inp.name for inp in self._sess2.get_inputs()]

            # Initialize LSTM states from model metadata
            # Model inputs: [audio_feature, lstm_state]
            # State shape varies by export but is typically (2, 1, 128, 2)
            state_shape_1 = self._sess1.get_inputs()[1].shape
            state_shape_2 = self._sess2.get_inputs()[1].shape

            # Replace any symbolic dimensions with concrete values
            def _resolve_shape(shape):
                return tuple(d if isinstance(d, int) else 1 for d in shape)

            self._states1 = np.zeros(_resolve_shape(state_shape_1), dtype=np.float32)
            self._states2 = np.zeros(_resolve_shape(state_shape_2), dtype=np.float32)

            self._loaded = True
            log.info("dtln_loaded", path=self._model_dir,
                     model1_inputs=self._input_names_1,
                     model2_inputs=self._input_names_2,
                     state1_shape=list(self._states1.shape),
                     state2_shape=list(self._states2.shape))

        except ImportError:
            log.warning("dtln_onnxruntime_not_available",
                        note="pip install onnxruntime for neural denoising")
        except Exception as exc:
            log.warning("dtln_load_error", error=str(exc))

    @property
    def is_available(self) -> bool:
        return self._loaded

    def reset(self) -> None:
        """Reset LSTM states and buffers.

        Call on voice session start.  Do NOT call on every speech_start —
        the LSTM benefits from continuous noise floor tracking.
        """
        if self._states1 is not None:
            self._states1.fill(0)
        if self._states2 is not None:
            self._states2.fill(0)
        self._in_buffer.fill(0)
        self._out_buffer.fill(0)

    def process_frame(self, pcm_bytes: bytes) -> bytes:
        """Denoise a 512-sample PCM16 frame.

        Returns enhanced PCM16 bytes (same length as input).
        Falls back to passthrough on any error.
        """
        if not self._loaded:
            return pcm_bytes

        try:
            return self._process(pcm_bytes)
        except Exception as exc:
            log.debug("dtln_frame_error", error=str(exc))
            return pcm_bytes

    def _process(self, pcm_bytes: bytes) -> bytes:
        # Convert PCM16 → float32 [-1, 1]
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(samples) != _BLOCK_LEN:
            log.debug("dtln_frame_size_mismatch", expected=_BLOCK_LEN, got=len(samples))
            return pcm_bytes  # Non-standard frame size — passthrough

        # Process 4 hops of 128 samples through the DTLN pipeline
        enhanced_out = np.zeros(_BLOCK_LEN, dtype=np.float32)

        for hop in range(_BLOCK_LEN // _BLOCK_SHIFT):
            start = hop * _BLOCK_SHIFT
            new_samples = samples[start:start + _BLOCK_SHIFT]

            # Shift input buffer, append new samples
            self._in_buffer[:-_BLOCK_SHIFT] = self._in_buffer[_BLOCK_SHIFT:]
            self._in_buffer[-_BLOCK_SHIFT:] = new_samples

            # --- Model 1: STFT-domain magnitude masking ---
            spec = np.fft.rfft(self._in_buffer)
            mag = np.abs(spec).astype(np.float32).reshape(1, 1, -1)
            phase = np.angle(spec)

            out_1 = self._sess1.run(
                None,
                {
                    self._input_names_1[0]: mag,
                    self._input_names_1[1]: self._states1,
                },
            )
            mask = out_1[0]
            self._states1 = out_1[1]

            # Apply mask and reconstruct time-domain signal
            estimated_complex = (mag * mask).squeeze() * np.exp(1j * phase)
            estimated_block = np.fft.irfft(estimated_complex).astype(np.float32)

            # --- Model 2: Time-domain enhancement ---
            block_input = estimated_block.reshape(1, 1, -1)

            out_2 = self._sess2.run(
                None,
                {
                    self._input_names_2[0]: block_input,
                    self._input_names_2[1]: self._states2,
                },
            )
            enhanced_block = out_2[0].squeeze()
            self._states2 = out_2[1]

            # Overlap-add reconstruction
            self._out_buffer[:-_BLOCK_SHIFT] = self._out_buffer[_BLOCK_SHIFT:]
            self._out_buffer[-_BLOCK_SHIFT:] = 0.0
            self._out_buffer += enhanced_block

            # Extract this hop's output
            enhanced_out[start:start + _BLOCK_SHIFT] = self._out_buffer[:_BLOCK_SHIFT]

        # Convert back to PCM16
        enhanced_pcm = np.clip(enhanced_out * 32768.0, -32767, 32767).astype(np.int16)
        return enhanced_pcm.tobytes()


# ---------------------------------------------------------------------------
# Combined Speech Enhancement Pipeline
# ---------------------------------------------------------------------------


class SpeechEnhancer:
    """Combined pipeline: highpass filter → DTLN neural denoiser.

    Wraps both stages with a single ``process_frame()`` API that slots
    into the voice pipeline before the existing AudioProcessor (AGC).

    Processing order:
      1. Biquad highpass (80 Hz) — removes sub-speech rumble
      2. DTLN denoiser — neural noise suppression
      3. → continues to AudioProcessor (AGC) → VAD → STT

    If DTLN models aren't available, falls back to highpass-only.
    """

    def __init__(
        self,
        *,
        highpass_hz: int = 80,
        model_dir: str = "",
        denoise_enabled: bool = True,
    ):
        self._highpass: BiquadHighpass | None = None
        self._denoiser: DTLNDenoiser | None = None

        if highpass_hz > 0:
            self._highpass = BiquadHighpass(cutoff_hz=highpass_hz)

        # Each stage is independently switchable — a caller that wants
        # highpass-only must be able to get it without the neural stage.
        if denoise_enabled:
            self._denoiser = DTLNDenoiser(model_dir=model_dir)

    def load_model(self) -> None:
        """Load DTLN models.  Safe to call multiple times."""
        if self._denoiser:
            self._denoiser.load_model()

    @property
    def is_available(self) -> bool:
        """True if at least the highpass filter is active."""
        return self._highpass is not None or (
            self._denoiser is not None and self._denoiser.is_available
        )

    @property
    def has_neural(self) -> bool:
        """True if DTLN neural denoiser loaded successfully."""
        return self._denoiser is not None and self._denoiser.is_available

    def reset(self) -> None:
        """Reset all internal state (call on voice session start)."""
        if self._highpass:
            self._highpass.reset()
        if self._denoiser:
            self._denoiser.reset()

    def process_frame(self, pcm_bytes: bytes) -> bytes:
        """Enhance a 512-sample PCM16 frame.

        Applies highpass → DTLN in sequence.  Never raises — returns
        original audio on any error to keep the voice pipeline alive.
        """
        result = pcm_bytes

        # Stage 1: Highpass filter (removes rumble)
        if self._highpass:
            try:
                samples = np.frombuffer(result, dtype=np.int16).astype(np.float32)
                filtered = self._highpass.process(samples)
                result = np.clip(filtered, -32767, 32767).astype(np.int16).tobytes()
            except Exception as exc:
                log.debug("highpass_error", error=str(exc))

        # Stage 2: DTLN neural denoiser
        if self._denoiser and self._denoiser.is_available:
            result = self._denoiser.process_frame(result)

        return result
