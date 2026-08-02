"""STT primitive — speech-to-text.

Wraps the existing STT path. ``call(audio_bytes=..., format=...)``
returns transcribed text. The underlying service handles provider
selection (whisper.cpp / external) — this adapter is a thin pass-through.
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class STTPrimitive(PrimitiveBase):
    name = "stt"
    description = "Transcribe audio bytes to text."

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        audio = kwargs.get("audio_bytes")
        if not audio:
            return PrimitiveResult(ok=False, error="stt: empty audio_bytes")

        # Lazy import — voice.stt may pull large optional deps.
        try:
            from augmentum.voice import stt as _stt_module
        except Exception as exc:
            return PrimitiveResult(ok=False, error=f"stt_import_failed: {exc!s}")

        transcribe = (
            getattr(_stt_module, "transcribe_audio", None)
            or getattr(_stt_module, "transcribe", None)
        )
        if transcribe is None:
            return PrimitiveResult(
                ok=False,
                error="stt: no transcribe entry point in augmentum.voice.stt",
                metadata={"note": "Wire the actual STT entry name in Sprint 3"},
            )

        try:
            result = await transcribe(audio, **{
                k: v for k, v in kwargs.items() if k != "audio_bytes"
            })
        except Exception as exc:
            log.exception("stt_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"stt_failed: {exc!s}")

        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        return PrimitiveResult(ok=True, payload=text)


PrimitiveRegistry.register(STTPrimitive)
