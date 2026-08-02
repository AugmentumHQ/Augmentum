"""Primitive adapters — stateless capabilities exposed to the runtime.

Primitives wrap existing services (TTS, STT, image gen, browse, files,
code exec, voice mixer, memory recall/consolidate, drift audit writer,
game agent orchestrator, XR scene controller). Each primitive is a
single-call, stateless capability. Subagents may compose them.

Same registry gating as subagents, with its own flag
``companion_primitive_registry_active`` (added in Unit F if not yet
present — see TODO at the bottom of this module).
"""

# Auto-register all primitive adapters at package-import time.
from augmentum.companion_runtime.primitives import (  # noqa: F401
    browse,
    code_dispatch,
    code_exec,
    drift_audit_writer,
    files,
    game_agent,
    image_gen,
    memory_consolidate,
    memory_recall,
    stt,
    tts,
    voice_mixer,
    xr_scene,
)
from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry

__all__ = [
    "PrimitiveBase",
    "PrimitiveContext",
    "PrimitiveResult",
    "PrimitiveRegistry",
]
