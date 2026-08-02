"""Speaker verification using WeSpeaker (ONNX runtime).

Provides voice enrollment (create a voiceprint from speech samples) and
real-time speaker verification (check if a speech segment matches the
enrolled user).  Runs entirely on CPU via ONNX Runtime, ~10-30 ms per
verification — fast enough to gate audio between VAD and STT.

Model: VoxCeleb-trained ResNet34 ONNX (~28 MB) downloaded from HuggingFace
on first use and cached locally in the data directory.  Custom ONNX models
can be supplied via the ``onnx_path`` parameter.

Embedding: 256-dim speaker vector, cosine similarity scoring.

Preprocessing best practices (implemented here):
  - **VAD trimming**: Strip silence from start/end of audio before embedding
    extraction.  Silence dilutes speaker features and degrades scores.
  - **Peak normalization**: Normalize audio amplitude before extraction so
    volume differences between enrollment and verification don't hurt scores.
  - **Multi-sample averaging**: Enrollment averages L2-normalized embeddings
    from 3+ diverse speech samples for a robust voiceprint.
  - **Quality-adaptive threshold**: High-quality enrollments (consistent
    samples) tolerate stricter thresholds; lower quality gets more lenient.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# 16 kHz mono PCM16 — same format as our Silero VAD pipeline
SAMPLE_RATE = 16000
# Minimum audio duration (seconds) for reliable embedding
MIN_ENROLLMENT_SECONDS = 1.5
MIN_VERIFY_SECONDS = 1.0
# Minimum successful enrollment samples for a robust voiceprint
MIN_ENROLLMENT_SAMPLES = 3


@dataclass
class VoicePrint:
    """A stored speaker voiceprint (embedding + metadata)."""

    embedding: list[float]       # 256-dim speaker embedding
    enrolled_at: float = 0.0     # Unix timestamp
    sample_count: int = 0        # Number of enrollment samples used
    quality_score: float = 0.0   # Self-consistency score (0-1)

    def to_json(self) -> str:
        return json.dumps({
            "embedding": self.embedding,
            "enrolled_at": self.enrolled_at,
            "sample_count": self.sample_count,
            "quality_score": self.quality_score,
        })

    @classmethod
    def from_json(cls, data: str) -> VoicePrint:
        d = json.loads(data)
        return cls(
            embedding=d["embedding"],
            enrolled_at=d.get("enrolled_at", 0.0),
            sample_count=d.get("sample_count", 0),
            quality_score=d.get("quality_score", 0.0),
        )


_WESPEAKER_HF_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM"
    "/resolve/main/voxceleb_resnet34_LM.onnx"
)
_WESPEAKER_MODEL_NAME = "voxceleb_resnet34_LM.onnx"


def _ensure_model_cached(cache_dir: str | Path) -> Path:
    """Download the WeSpeaker ONNX model from HuggingFace if not already cached.

    Returns the path to the cached model file.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    model_path = cache_path / _WESPEAKER_MODEL_NAME

    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        log.debug("wespeaker_model_cached", path=str(model_path))
        return model_path

    log.info("wespeaker_model_downloading", url=_WESPEAKER_HF_URL)
    from urllib.request import urlretrieve

    tmp_path = model_path.with_suffix(".tmp")
    try:
        urlretrieve(_WESPEAKER_HF_URL, str(tmp_path))
        tmp_path.rename(model_path)
        log.info("wespeaker_model_cached", path=str(model_path),
                 size_mb=f"{model_path.stat().st_size / 1_048_576:.1f}")
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return model_path


class SpeakerVerifier:
    """WeSpeaker speaker verification engine (ONNX runtime).

    Usage::

        verifier = SpeakerVerifier()
        verifier.load_model()

        # Enrollment: compute voiceprint from 3-5 speech samples
        voiceprint = verifier.enroll([sample1_pcm, sample2_pcm, sample3_pcm])

        # Verification: check if speech matches the enrolled user
        score = verifier.verify(speech_pcm, voiceprint)
        if score >= verifier.effective_threshold(voiceprint):
            # It's the enrolled user
    """

    def __init__(
        self,
        *,
        model_dir: str = "",
        onnx_path: str = "",
        verify_threshold: float = 0.55,
    ) -> None:
        self._model_dir = model_dir
        self._onnx_path = onnx_path
        self.verify_threshold = verify_threshold
        self._model: Any = None
        self._session: Any = None  # onnxruntime InferenceSession
        self._loaded = False

    def load_model(self) -> None:
        """Lazy-load the WeSpeaker ONNX model.

        Attempts to load in order:
          1. Custom onnx_path if provided
          2. wespeakerruntime package (if installed, uses its own model)
          3. Download from HuggingFace, cache locally, load via onnxruntime
        """
        if self._loaded:
            return

        # 1. Custom ONNX path — load directly via onnxruntime
        if self._onnx_path:
            self._load_onnx_direct(self._onnx_path)
            return

        # 2. Try wespeakerruntime package (may fail to download from Shanghai)
        try:
            import wespeakerruntime as wespeaker
            self._model = wespeaker.Speaker(lang="en")
            self._loaded = True
            log.info("speaker_verifier_loaded", model="wespeaker_resnet34",
                     source="wespeakerruntime")
            return
        except ImportError:
            log.debug("wespeakerruntime_not_installed")
        except Exception:
            log.debug("wespeakerruntime_failed", exc_info=True)

        # 3. Download from HuggingFace and load via onnxruntime
        from augmentum.config import settings

        cache_dir = Path(self._model_dir) if self._model_dir else Path(settings.data_dir) / "models" / "wespeaker"
        try:
            model_path = _ensure_model_cached(cache_dir)
            self._load_onnx_direct(str(model_path))
        except Exception as exc:
            log.warning("speaker_verifier_load_error", error=str(exc))
            raise

    def _load_onnx_direct(self, onnx_path: str) -> None:
        """Load the ONNX model directly via onnxruntime."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        self._session = ort.InferenceSession(onnx_path, sess_options=so)
        self._loaded = True
        log.info("speaker_verifier_loaded", model="wespeaker_resnet34",
                 source="onnxruntime_direct", path=onnx_path)

    def _pcm_to_float(self, pcm_bytes: bytes) -> np.ndarray:
        """Convert raw PCM16 bytes to float32 numpy array [-1, 1]."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    def _extract_embedding(self, pcm_bytes: bytes) -> np.ndarray | None:
        """Extract a 256-dim speaker embedding from PCM16 audio.

        Applies two preprocessing steps before feature extraction:
          1. **VAD trimming** — strip leading/trailing silence using energy-based
             detection.  WeSpeaker's own docs recommend Silero VAD; we use a
             lightweight energy approach here to avoid loading a second model
             instance (Silero is already running in the voice pipeline).
          2. **Peak normalization** — scale so the loudest sample hits 0.9.
             Volume mismatch between enrollment and verification degrades
             cosine similarity.

        Robustness: if VAD-trim leaves the audio below MIN_VERIFY_SECONDS
        but the original input was long enough, we retry extraction on
        the original (untrimmed) audio. This catches the "quiet edges
        confused energy VAD" case where the user really did speak for
        long enough but the trim was over-eager. Without this fallback,
        the verify path returns 0.0 and the user sees "Voice not
        recognized" when the real fault is audio handling, not speaker
        mismatch.

        Returns None if the audio is too short or the model isn't loaded.
        """
        if not self._loaded:
            self.load_model()

        original_duration = len(pcm_bytes) / (SAMPLE_RATE * 2)

        # First attempt: trimmed audio (preferred — silence dilutes scores)
        trimmed = _vad_trim_pcm(pcm_bytes)
        trimmed_duration = len(trimmed) / (SAMPLE_RATE * 2)

        if trimmed_duration >= MIN_VERIFY_SECONDS:
            emb = self._run_extract(_peak_normalize_pcm(trimmed))
            if emb is not None:
                return emb
            # ONNX/wespeaker failed — fall through to retry-on-original

        # Second attempt: fall back to the untrimmed audio if the original
        # was long enough. Skips the energy-VAD step that just rejected us.
        if original_duration >= MIN_VERIFY_SECONDS:
            log.debug("speaker_extract_retry_untrimmed",
                      original=f"{original_duration:.2f}s",
                      trimmed=f"{trimmed_duration:.2f}s")
            return self._run_extract(_peak_normalize_pcm(pcm_bytes))

        log.debug("speaker_audio_too_short",
                  original=f"{original_duration:.2f}s",
                  trimmed=f"{trimmed_duration:.2f}s")
        return None

    def _run_extract(self, pcm_bytes: bytes) -> np.ndarray | None:
        """Dispatch to the loaded backend (ONNX or wespeakerruntime).

        Returns None on any backend error so the caller can fall back
        to a different preprocessing strategy.
        """
        try:
            if self._session is not None:
                return self._extract_via_onnx(pcm_bytes)
            return self._extract_via_wespeaker(pcm_bytes)
        except Exception as exc:
            log.warning("speaker_embedding_error", error=str(exc))
            return None

    def _extract_via_wespeaker(self, pcm_bytes: bytes) -> np.ndarray | None:
        """Extract embedding via wespeakerruntime (file-based API)."""
        import os
        import tempfile

        wav_bytes = _pcm_to_wav(pcm_bytes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        try:
            embedding = self._model.extract_embedding(tmp_path)
            if isinstance(embedding, np.ndarray):
                return embedding.flatten()
            return np.array(embedding).flatten()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _extract_via_onnx(self, pcm_bytes: bytes) -> np.ndarray | None:
        """Extract embedding directly via onnxruntime with numpy fbank."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        # WeSpeaker expects samples scaled by 2^15 (int16 range)
        feats = _compute_fbank_np(samples, sample_rate=SAMPLE_RATE)
        if feats is None or feats.shape[0] < 5:
            return None
        # CMN (cepstral mean normalization, no variance normalization)
        feats = feats - np.mean(feats, axis=0)
        # Add batch dimension: [1, T, 80]
        feats = np.expand_dims(feats, 0).astype(np.float32)
        outputs = self._session.run(["embs"], {"feats": feats})
        return outputs[0].flatten()

    def enroll(self, audio_samples: list[bytes]) -> VoicePrint | None:
        """Create a voiceprint from multiple PCM16 audio samples.

        Each sample should be 2-5 seconds of clear speech from the user.
        The final embedding is the L2-normalized mean of all sample embeddings.

        Requires at least ``MIN_ENROLLMENT_SAMPLES`` (3) successful embeddings
        for a robust voiceprint.  More samples = better averaging of natural
        vocal variation.

        Returns None if enrollment fails (bad audio, model error).
        """
        embeddings: list[np.ndarray] = []

        for i, sample in enumerate(audio_samples):
            emb = self._extract_embedding(sample)
            if emb is not None:
                embeddings.append(emb)
                log.debug("enrollment_sample_ok", sample=i,
                          duration=f"{len(sample) / (SAMPLE_RATE * 2):.1f}s")
            else:
                log.warning("enrollment_sample_failed", sample=i)

        if len(embeddings) < MIN_ENROLLMENT_SAMPLES:
            log.warning("enrollment_insufficient_samples",
                        good=len(embeddings), required=MIN_ENROLLMENT_SAMPLES,
                        total=len(audio_samples))
            return None

        # L2-normalize each embedding before averaging
        normed = []
        for emb in embeddings:
            n = np.linalg.norm(emb)
            normed.append(emb / n if n > 0 else emb)

        # Average and re-normalize
        mean_emb = np.mean(normed, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm

        # Quality score: mean pairwise cosine similarity between samples
        # Higher = more consistent enrollment audio
        quality = _pairwise_similarity(embeddings)

        voiceprint = VoicePrint(
            embedding=mean_emb.tolist(),
            enrolled_at=time.time(),
            sample_count=len(embeddings),
            quality_score=round(quality, 4),
        )

        log.info("enrollment_complete",
                 samples=len(embeddings),
                 quality=f"{quality:.4f}",
                 effective_threshold=f"{self.effective_threshold(voiceprint):.4f}")

        return voiceprint

    def effective_threshold(self, voiceprint: VoicePrint) -> float:
        """Compute the verification threshold adjusted for enrollment quality.

        High-quality enrollments (consistent samples, quality > 0.85) produce
        tighter embedding clusters, so we can raise the threshold to reduce
        false accepts.  Lower-quality enrollments need a more forgiving
        threshold to avoid false rejects.

        Adjustment range: ±0.05 from the base threshold.
        """
        quality = voiceprint.quality_score
        if quality >= 0.85:
            # High quality — stricter threshold (fewer false accepts)
            return min(self.verify_threshold + 0.05, 0.80)
        if quality <= 0.65:
            # Low quality — more lenient (fewer false rejects)
            return max(self.verify_threshold - 0.05, 0.35)
        return self.verify_threshold

    def verify(
        self, pcm_bytes: bytes, voiceprint: VoicePrint,
    ) -> float:
        """Check if a speech segment matches the enrolled voiceprint.

        Returns cosine similarity score (0.0 - 1.0).
        Use ``effective_threshold(voiceprint)`` for the quality-adjusted
        threshold, or compare against ``verify_threshold`` for the base.
        """
        emb = self._extract_embedding(pcm_bytes)
        if emb is None:
            return 0.0

        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        enrolled = np.array(voiceprint.embedding)
        norm_e = np.linalg.norm(enrolled)
        if norm_e > 0:
            enrolled = enrolled / norm_e

        # Cosine similarity on unit vectors — range [-1, 1], clamped to [0, 1].
        # Note: official WeSpeaker CLI uses (score+1)/2 normalization.
        # Our thresholds are calibrated against this raw cosine scale.
        score = float(np.dot(emb, enrolled))
        return max(0.0, score)

    def is_enrolled_speaker(
        self, pcm_bytes: bytes, voiceprint: VoicePrint,
    ) -> bool:
        """Convenience: returns True if the speaker matches.

        Uses quality-adaptive threshold from ``effective_threshold()``.
        """
        return self.verify(pcm_bytes, voiceprint) >= self.effective_threshold(voiceprint)


def _pairwise_similarity(embeddings: list[np.ndarray]) -> float:
    """Compute mean pairwise cosine similarity for quality scoring."""
    if len(embeddings) < 2:
        return 1.0

    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            a = embeddings[i] / (np.linalg.norm(embeddings[i]) + 1e-8)
            b = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
            sims.append(float(np.dot(a, b)))

    return sum(sims) / len(sims) if sims else 0.0


# ---------------------------------------------------------------------------
# Audio preprocessing for embedding extraction
# ---------------------------------------------------------------------------

# VAD trim: energy threshold in RMS (float32 range [-1, 1])
# Frames below this are considered silence.
_VAD_ENERGY_THRESHOLD = 0.01
# Frame size for energy-based VAD (20 ms at 16 kHz = 320 samples)
_VAD_FRAME_SAMPLES = 320
# Minimum non-silent frames to keep (don't trim everything)
_VAD_MIN_SPEECH_FRAMES = 16  # ~320 ms


def _vad_trim_pcm(pcm_bytes: bytes) -> bytes:
    """Strip leading and trailing silence from PCM16 audio.

    Uses a simple energy-based approach (RMS per 20 ms frame) rather than
    loading a second Silero VAD instance.  This is the single most impactful
    preprocessing step for speaker embedding quality — WeSpeaker's own
    documentation recommends VAD trimming before feature extraction.

    Returns the trimmed PCM16 bytes, or the original if trimming would
    remove too much audio.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if len(samples) < _VAD_FRAME_SAMPLES:
        return pcm_bytes

    # Compute RMS energy per frame
    n_frames = len(samples) // _VAD_FRAME_SAMPLES
    if n_frames == 0:
        return pcm_bytes

    frames = samples[:n_frames * _VAD_FRAME_SAMPLES].reshape(n_frames, _VAD_FRAME_SAMPLES)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Find first and last frames above energy threshold
    active = np.where(rms >= _VAD_ENERGY_THRESHOLD)[0]
    if len(active) < _VAD_MIN_SPEECH_FRAMES:
        # Too little speech — return original to avoid bad embedding
        return pcm_bytes

    start_frame = max(0, active[0] - 2)    # Keep 2 frames (~40 ms) margin
    end_frame = min(n_frames, active[-1] + 3)  # Keep 3 frames (~60 ms) margin

    start_sample = start_frame * _VAD_FRAME_SAMPLES
    end_sample = end_frame * _VAD_FRAME_SAMPLES

    # Convert back to PCM16 bytes
    trimmed_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    trimmed = trimmed_samples[start_sample:end_sample]

    trimmed_dur = len(trimmed) / SAMPLE_RATE
    orig_dur = len(trimmed_samples) / SAMPLE_RATE
    if trimmed_dur < orig_dur * 0.3:
        # Trimmed more than 70% — likely bad VAD, return original
        log.debug("vad_trim_too_aggressive",
                  original=f"{orig_dur:.1f}s", trimmed=f"{trimmed_dur:.1f}s")
        return pcm_bytes

    log.debug("vad_trim_applied",
              original=f"{orig_dur:.1f}s", trimmed=f"{trimmed_dur:.1f}s",
              removed_pct=f"{(1 - trimmed_dur / orig_dur) * 100:.0f}%")

    return trimmed.tobytes()


def _peak_normalize_pcm(
    pcm_bytes: bytes, target_peak: float = 0.9,
) -> bytes:
    """Peak-normalize PCM16 so the loudest sample hits *target_peak*.

    Volume mismatch between enrollment and verification sessions degrades
    cosine similarity scores.  Normalizing both to the same peak level
    removes this variable.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(samples) == 0:
        return pcm_bytes

    peak = np.max(np.abs(samples))
    if peak < 1.0:
        return pcm_bytes  # Silent

    current_peak_ratio = peak / 32767.0
    if current_peak_ratio >= target_peak * 0.95:
        return pcm_bytes  # Already loud enough

    gain = (target_peak * 32767.0) / peak
    normalized = np.clip(samples * gain, -32767, 32767).astype(np.int16)
    return normalized.tobytes()


# ---------------------------------------------------------------------------
# Lightweight fbank feature extraction (numpy only, no torchaudio/kaldi)
# ---------------------------------------------------------------------------
# Matches wespeakerruntime's _compute_fbank parameters:
#   80 mel bins, 25 ms frame, 10 ms shift, hamming window, no dither.


def _compute_fbank_np(
    samples: np.ndarray,
    sample_rate: int = 16000,
    num_mel_bins: int = 80,
    frame_length_ms: int = 25,
    frame_shift_ms: int = 10,
) -> np.ndarray | None:
    """Compute log-Mel filterbank features matching Kaldi's fbank defaults.

    Args:
        samples: float32 audio scaled to int16 range (±32768).
        sample_rate: Audio sample rate (16000 Hz expected).
        num_mel_bins: Number of Mel filterbank channels.
        frame_length_ms: Frame length in milliseconds.
        frame_shift_ms: Frame shift (hop) in milliseconds.

    Returns:
        Log-Mel fbank features [T, num_mel_bins] or None if too short.
    """
    frame_length = int(sample_rate * frame_length_ms / 1000)
    frame_shift = int(sample_rate * frame_shift_ms / 1000)
    n_fft = 1
    while n_fft < frame_length:
        n_fft *= 2

    if len(samples) < frame_length:
        return None

    # Frame the signal
    n_frames = 1 + (len(samples) - frame_length) // frame_shift
    indices = (
        np.arange(frame_length)[None, :]
        + np.arange(n_frames)[:, None] * frame_shift
    )
    frames = samples[indices]

    # Apply Hamming window
    window = np.hamming(frame_length).astype(np.float32)
    frames = frames * window

    # FFT → power spectrum
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = np.abs(spectrum) ** 2

    # Mel filterbank
    mel_filters = _mel_filterbank(num_mel_bins, n_fft, sample_rate)
    mel_spec = np.dot(power, mel_filters.T)

    # Log with floor to avoid log(0)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)

    return log_mel.astype(np.float32)


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    num_bins: int, n_fft: int, sample_rate: int,
) -> np.ndarray:
    """Build a Mel filterbank matrix [num_bins, n_fft//2 + 1]."""
    n_freqs = n_fft // 2 + 1
    low_mel = _hz_to_mel(0)
    high_mel = _hz_to_mel(sample_rate / 2)

    mel_points = np.linspace(low_mel, high_mel, num_bins + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_bins, n_freqs), dtype=np.float32)
    for i in range(num_bins):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center > left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filters[i, j] = (right - j) / (right - center)

    return filters


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw PCM16 mono bytes in a minimal WAV header."""
    data_size = len(pcm_bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,                  # Subchunk1Size (PCM)
        1,                   # AudioFormat (PCM = 1)
        1,                   # NumChannels (mono)
        sample_rate,
        sample_rate * 2,     # ByteRate
        2,                   # BlockAlign
        16,                  # BitsPerSample
        b"data",
        data_size,
    )
    return header + pcm_bytes
