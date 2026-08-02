"""Wake-word detection for per-avatar voice entry.

Two surfaces live here:

  ``training`` — Kokoro-synthesized positive samples + synthesized
  negatives, a small CRNN trained on mel-spectrogram windows, exported
  to ONNX. Runs server-side as a JobRunner handler so a 5-10 minute
  training job doesn't block the request loop.

  ``service`` — Runtime inference. ``WakeWordService.feed(pcm, source_id)``
  takes PCM16 frames at 16 kHz and emits ``wake.detected`` on the
  PresenceBus when the trained phrase is recognized. Source-agnostic;
  the same service handles browser-routed mic via /ws/voice today and
  fabric-routed audio from another node later.

The contract Phase 2 of the fabric layer will adopt: this module
exposes a clean Python API (no fabric/* imports) and publishes via
the canonical PresenceBus. Fabric capability extractors wrap, they
don't refactor.
"""

from __future__ import annotations

from augmentum.voice.wake_word.service import (
    WakeDetection,
    WakeWordDetector,
    WakeWordService,
    get_or_create_service,
    load_models_from_db,
)

__all__ = [
    "WakeDetection",
    "WakeWordDetector",
    "WakeWordService",
    "get_or_create_service",
    "load_models_from_db",
]
