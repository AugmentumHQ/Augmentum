"""Voice-mixer primitive — placeholder.

Sprint 0 found no standalone voice mixer in the codebase
(``augmentum.xr.session.VoiceBridge`` exists but is a single-stream
bridge, not a mixer). This primitive registers a stub so Sprint 3
dispatch can list it, but invocation returns a clean
``unimplemented`` error until a real mixer ships.
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry


class VoiceMixerPrimitive(PrimitiveBase):
    name = "voice_mixer"
    description = (
        "Multi-speaker voice mixing (placeholder — feature not yet "
        "implemented in the audio pipeline)."
    )

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        return PrimitiveResult(
            ok=False,
            error="voice_mixer_unimplemented",
            metadata={"note": "No mixer in the audio pipeline yet; "
                      "ship a real implementation before invoking."},
        )


PrimitiveRegistry.register(VoiceMixerPrimitive)
