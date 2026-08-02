"""SmartTurn v3 — learned turn-completion detection.

Runs after Silero VAD detects silence to predict whether the user has
finished their turn or is just pausing mid-thought (e.g., "um", thinking).

Uses Pipecat's open-weights SmartTurn v3.2 model (BSD 2-Clause license):
  - Whisper Tiny encoder backbone + linear classifier head (~8M params)
  - ONNX inference, ~12ms on CPU, no PyTorch/CUDA dependency
  - Input: up to 8 seconds of 16kHz mono float32 audio
  - Output: probability that the turn is complete (threshold 0.5)

Model downloaded from HuggingFace on first use (~8MB).
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import numpy as np

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16000
_MAX_SECONDS = 8
_MAX_SAMPLES = _MAX_SECONDS * _SAMPLE_RATE
_DEFAULT_THRESHOLD = 0.5

# Model singleton
_lock = threading.Lock()
_session = None
_feature_extractor = None
_loaded = False
_model_path: str = ""


def _get_model_dir() -> Path:
    """Return the directory for SmartTurn model storage."""
    from augmentum.config import settings
    return Path(settings.data_dir) / "models" / "smart-turn"


def _ensure_model() -> str:
    """Return the SmartTurn ONNX model path, downloading if not cached.

    Lookup order:
      1. Image-baked location ``/home/augmentum/.smart-turn/`` — populated by
         the Dockerfile so first voice call has zero network latency.
      2. Persistent data-dir cache (legacy path / user-downloaded fallback).
      3. HuggingFace download to the data-dir cache.
    """
    prebaked = Path("/home/augmentum/.smart-turn/smart-turn-v3.2-cpu.onnx")
    if prebaked.exists() and prebaked.stat().st_size > 1_000_000:
        return str(prebaked)

    model_dir = _get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "smart-turn-v3.2-cpu.onnx"

    if model_file.exists() and model_file.stat().st_size > 1_000_000:
        return str(model_file)

    log.info("smart_turn_downloading")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="pipecat-ai/smart-turn-v3",
            filename="smart-turn-v3.2-cpu.onnx",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
        log.info("smart_turn_downloaded", path=path,
                 size_mb=f"{os.path.getsize(path) / 1_048_576:.1f}")
        return path
    except ImportError:
        # No huggingface_hub — try direct download
        import urllib.request
        url = ("https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/"
               "smart-turn-v3.2-cpu.onnx")
        tmp = str(model_file) + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, str(model_file))
        log.info("smart_turn_downloaded", path=str(model_file))
        return str(model_file)


def load_model() -> bool:
    """Load the SmartTurn ONNX model. Thread-safe, idempotent."""
    global _session, _feature_extractor, _loaded, _model_path

    if _loaded:
        return True

    with _lock:
        if _loaded:
            return True

        try:
            import onnxruntime as ort

            _model_path = _ensure_model()

            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.log_severity_level = 3

            _session = ort.InferenceSession(_model_path, sess_options=so)

            # Load WhisperFeatureExtractor for log-mel spectrogram computation
            from transformers import WhisperFeatureExtractor
            _feature_extractor = WhisperFeatureExtractor(chunk_length=_MAX_SECONDS)

            _loaded = True
            log.info("smart_turn_loaded", model=_model_path)
            return True

        except ImportError as exc:
            log.warning("smart_turn_import_error", error=str(exc),
                        note="pip install onnxruntime transformers for SmartTurn")
            return False
        except Exception as exc:
            log.warning("smart_turn_load_error", error=str(exc))
            return False


def is_available() -> bool:
    """True if the SmartTurn model is loaded and ready."""
    return _loaded


def predict_turn_complete(
    audio_float32: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
) -> tuple[bool, float]:
    """Predict whether the user's turn is complete.

    Args:
        audio_float32: Mono 16kHz float32 array [-1, 1], up to 8 seconds
                       of the current turn's speech (including trailing silence).
        threshold: Probability threshold for "complete" (default 0.5).

    Returns:
        (is_complete, probability) — probability is the raw sigmoid output.
    """
    if not _loaded or _session is None or _feature_extractor is None:
        # Not available — default to "complete" (don't block the turn)
        return True, 1.0

    # Truncate to last 8 seconds (keep the ending, which has the
    # prosodic cues for turn completion)
    if len(audio_float32) > _MAX_SAMPLES:
        audio_float32 = audio_float32[-_MAX_SAMPLES:]

    # Pad short audio at the beginning (audio at the end)
    if len(audio_float32) < _MAX_SAMPLES:
        padding = _MAX_SAMPLES - len(audio_float32)
        audio_float32 = np.pad(audio_float32, (padding, 0), mode="constant")

    # Whisper feature extraction → log-mel spectrogram
    inputs = _feature_extractor(
        audio_float32,
        sampling_rate=_SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=_MAX_SAMPLES,
        truncation=True,
        do_normalize=True,
    )

    input_features = inputs.input_features.astype(np.float32)

    # ONNX inference
    outputs = _session.run(None, {"input_features": input_features})
    probability = float(outputs[0][0].item())

    return probability > threshold, probability


async def predict_turn_complete_async(
    audio_float32: np.ndarray,
    threshold: float = _DEFAULT_THRESHOLD,
) -> tuple[bool, float]:
    """Async wrapper — runs inference in a thread to avoid blocking the event loop."""
    return await asyncio.to_thread(predict_turn_complete, audio_float32, threshold)
