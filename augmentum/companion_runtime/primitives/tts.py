"""TTS primitive — wraps ``augmentum.voice.tts.prefetch_tts_audio``.

Stateless: one ``call(text=..., voice=..., speed=...)`` returns a list
of audio chunks (bytes). Higher-layer streaming (LiveKit, WebRTC, XR
voice bridge) consumes the chunks separately.
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


class TTSPrimitive(PrimitiveBase):
    name = "tts"
    description = "Synthesize speech audio bytes from text."

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        text = kwargs.get("text", "").strip()
        if not text:
            return PrimitiveResult(ok=False, error="tts: empty text")

        try:
            from augmentum.voice.tts import prefetch_tts_audio
        except Exception as exc:
            return PrimitiveResult(ok=False, error=f"tts_import_failed: {exc!s}")

        app_state = getattr(ctx.runtime, "_app_state", None)
        sqlite_backend = getattr(app_state, "sqlite_backend", None) if app_state else None
        if sqlite_backend is None:
            return PrimitiveResult(ok=False, error="tts: no sqlite_backend on runtime")

        voice = kwargs.get("voice", "")
        speed = float(kwargs.get("speed", 1.0))

        try:
            # backend.connect() returns None (initializer, not a context
            # manager) — use the live connection (audit 2026-06-17).
            conn = sqlite_backend.conn
            chunks = await prefetch_tts_audio(
                text, conn, voice=voice, speed=speed,
            )
        except Exception as exc:
            log.exception("tts_failed", error=str(exc))
            return PrimitiveResult(ok=False, error=f"tts_failed: {exc!s}")

        return PrimitiveResult(
            ok=True,
            payload=chunks,
            metadata={"chunk_count": len(chunks), "voice": voice, "speed": speed},
        )


PrimitiveRegistry.register(TTSPrimitive)
