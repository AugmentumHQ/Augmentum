"""Companion voice bridge — turns ``plan.say`` strings into audible speech.

The slow-path planner emits ``say`` as text. This module turns each
non-empty utterance into PCM/MP3 audio via whichever TTS provider
Augmentum has configured (Kokoro, Pocket TTS, Chatterbox, etc.), then
hands the bytes to the surface adapter so they ride down the bridge
WebSocket alongside ``action`` and ``frame`` traffic.

Why a thin bridge module: every TTS provider question is already
solved inside ``augmentum/proxy/audio_routes.py:tts_synthesize_bytes``.
We delegate; we don't re-implement provider routing, voice resolution,
or text cleaning. The bridge is just the glue that lets the
game-agent orchestrator stay provider-agnostic.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# Default MIME mapping for the response_format passed to TTS. Both
# Augmentum's built-in engines and the OpenAI-shape providers we
# proxy honour ``mp3`` as a stable lowest-common-denominator. The
# browser <audio> element decodes it without WebCodecs.
_FORMAT_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "opus": "audio/opus",
}


class VoiceBridge:
    """Resolve text -> audio bytes via Augmentum's TTS layer.

    Use when:
    - The orchestrator wants to speak a string on behalf of the
      companion. Construct one bridge per session (cheap) or share
      one across all sessions (also fine — instances are stateless).

    Expects:
    - ``get_state_manager_conn`` is a zero-arg callable returning the
      aiosqlite connection used by Augmentum's state manager, or
      ``None`` when the state manager is not yet wired. The lazy
      form mirrors how the LLM bridge resolves the provider registry
      — keeps the bridge constructible before startup events finish.

    Returns:
    - From :meth:`synthesize`, a tuple ``(audio_bytes, mime)`` or
      ``(None, "")`` when the operator has disabled TTS or no
      provider is configured. Callers should treat ``None`` as
      "skip the audio frame" rather than an error.
    """

    def __init__(
        self,
        get_state_manager_conn: Callable[[], Any],
        *,
        default_voice: str = "",
        speed: float = 1.0,
        response_format: str = "mp3",
    ) -> None:
        self._get_conn = get_state_manager_conn
        self._voice = default_voice
        self._speed = speed
        self._format = response_format

    @property
    def mime(self) -> str:
        return _FORMAT_MIME.get(self._format, "application/octet-stream")

    async def synthesize(
        self, text: str, *, voice: str | None = None,
    ) -> tuple[bytes | None, str]:
        """Turn ``text`` into audio. Tolerant of every failure mode.

        Reasons this may return ``(None, "")`` and NOT raise:
        - TTS feature flag is off (``audio_tts_enabled = False``).
        - No TTS provider is configured.
        - The provider returned no audio (no model, empty voice).
        - The synthesis call itself raised (logged, swallowed).

        The bridge never lets a TTS failure take down a game-agent
        session — the user is still playing, the agent is still
        planning, only the voice cuts out.

        @param voice:
            Per-call voice override. Use when a session has its own
            companion identity (character card) and the bridge is
            shared across sessions. ``None`` falls back to the
            bridge's constructor-supplied ``default_voice``.
        """

        cleaned = (text or "").strip()
        if not cleaned:
            return (None, "")

        conn = self._get_conn()
        if conn is None:
            log.debug("game_agent.voice.no_conn")
            return (None, "")

        # Imported lazily to avoid an import cycle: audio_routes pulls
        # in a lot of FastAPI surface that we don't need at module
        # import time.
        from augmentum.proxy.audio_routes import tts_synthesize_bytes

        chosen_voice = voice if voice is not None else self._voice
        try:
            audio, _is_builtin = await tts_synthesize_bytes(
                conn,
                cleaned,
                voice=chosen_voice,
                speed=self._speed,
                response_format=self._format,
            )
        except Exception as exc:  # noqa: BLE001
            # tts_synthesize_bytes raises HTTPException on
            # provider/config errors. Log and degrade — the
            # companion just falls silent.
            log.warning(
                "game_agent.voice.synthesize_failed",
                error=str(exc),
                text_len=len(cleaned),
                voice=chosen_voice,
            )
            return (None, "")

        if not audio:
            return (None, "")
        return (audio, self.mime)

    async def synthesize_b64(
        self, text: str, *, voice: str | None = None,
    ) -> tuple[str, str]:
        """Same as :meth:`synthesize` but base64-encodes the bytes.

        Convenience wrapper because every consumer ships the audio
        over WS as ``bytes_b64``.
        """

        audio, mime = await self.synthesize(text, voice=voice)
        if audio is None:
            return ("", "")
        return (base64.b64encode(audio).decode("ascii"), mime)


__all__ = ["VoiceBridge"]
