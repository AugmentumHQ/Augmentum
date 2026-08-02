"""Vision provider abstraction.

A `vision_provider` is the unified entry point for any code that
wants to caption, describe, OCR, or VQA an image. Two implementations
ship:

- :class:`PrimaryVisionProvider` — routes through the currently
  loaded primary model when that model is vision-capable (paired
  with an mmproj). Best quality; reuses VRAM already in use.

- :class:`SmolVLMProvider` — sibling :class:`LlamaServerManager`
  serving SmolVLM 256M (175 MB Q8_0 base + 104 MB mmproj). Always
  available regardless of the primary slot. Default-CPU; opt-in GPU.

The :class:`VisionRouter` picks one based on workload hints:

- ``interactive`` (chat image upload): primary if available, else
  SmolVLM
- ``background`` (screen index, file-index captioning, security
  cam first-pass): always SmolVLM so the primary's KV cache stays
  clean for the user's active conversation

The unprecedented part of this design is the *workload hint*. Every
other local-LLM tool routes by "what model is loaded" — Augmentum
routes by "what is this work for". Background pipelines never steal
KV slots from the user.
"""

from __future__ import annotations

from augmentum.vision.provider import (
    ClassifierVisionProvider,
    PrimaryVisionProvider,
    SmolVLMConfig,
    SmolVLMProvider,
    SmolVLMSibling,
    VisionProvider,
)
from augmentum.vision.router import VisionRouter, Workload

__all__ = [
    "ClassifierVisionProvider",
    "PrimaryVisionProvider",
    "SmolVLMConfig",
    "SmolVLMProvider",
    "SmolVLMSibling",
    "VisionProvider",
    "VisionRouter",
    "Workload",
]
