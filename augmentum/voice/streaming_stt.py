"""Streaming Speech-to-Text client — sends audio frames in real-time,
receives partial and final transcripts over WebSocket.

Supports Deepgram (native WebSocket streaming) with automatic fallback
to batch STT for providers that don't support streaming.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.http_client import normalize_base_url
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TranscriptEvent:
    """Transcript event from the streaming STT provider."""

    text: str
    is_final: bool          # Transcription is finalized (won't change)
    speech_final: bool      # Utterance boundary detected (speaker stopped)
    duration: float = 0.0   # Audio segment duration in seconds (0 if unknown)


class StreamingSTTSession:
    """Manages a streaming STT WebSocket connection to Deepgram.

    Usage::

        session = StreamingSTTSession(api_key="...", model="nova-3")
        await session.connect()
        await session.send_audio(pcm_bytes) # call per frame
        # Events arrive via on_transcript callback
        await session.close()
    """

    def __init__(
        self,
        *,
        base_url: str = "wss://api.deepgram.com",
        api_key: str,
        model: str = "nova-3",
        language: str = "en",
        endpointing_ms: int = 200,
        sample_rate: int = 16000,
        encoding: str = "linear16",
        on_transcript: Any = None,
    ) -> None:
        self._base_url = normalize_base_url(base_url)
        self._api_key = api_key
        self._model = model
        self._language = language
        self._endpointing_ms = endpointing_ms
        self._sample_rate = sample_rate
        self._encoding = encoding
        self.on_transcript = on_transcript  # async callable(TranscriptEvent)

        self._ws: Any = None
        self._receive_task: asyncio.Task | None = None
        self._connected = False

    async def connect(self) -> None:
        """Open WebSocket connection to Deepgram streaming endpoint."""
        try:
            import websockets
        except ImportError:
            log.warning("streaming_stt_no_websockets",
                        msg="pip install websockets for streaming STT")
            raise

        params = (
            f"model={self._model}"
            f"&language={self._language}"
            f"&encoding={self._encoding}"
            f"&sample_rate={self._sample_rate}"
            f"&channels=1"
            f"&interim_results=true"
            f"&endpointing={self._endpointing_ms}"
            f"&utterance_end_ms=1000"
            f"&vad_events=true"
        )

        # Normalize URL scheme
        ws_url = self._base_url
        if ws_url.startswith("https://"):
            ws_url = "wss://" + ws_url[8:]
        elif ws_url.startswith("http://"):
            ws_url = "ws://" + ws_url[7:]
        elif not ws_url.startswith("ws"):
            ws_url = "wss://" + ws_url

        url = f"{ws_url}/v1/listen?{params}"
        headers = {"Authorization": f"Token {self._api_key}"}

        self._ws = await websockets.connect(url, additional_headers=headers)
        self._connected = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._receive_task.add_done_callback(self._on_receive_done)
        log.info("streaming_stt_connected", model=self._model)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Send a PCM audio frame to the STT provider.

        Non-blocking — schedules the WebSocket send as a background
        task and returns immediately. Coroutine-typed so callers can
        treat the Deepgram path and the local Moonshine path
        identically (``await stt_session.send_audio(frame)``).
        """
        if self._ws and self._connected:
            task = asyncio.create_task(self._ws.send(pcm_bytes))
            task.add_done_callback(self._on_send_done)

    @staticmethod
    def _on_send_done(task: asyncio.Task) -> None:
        """Log exceptions from fire-and-forget audio sends."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.warning("streaming_stt_send_error", error=str(exc))

    @staticmethod
    def _on_receive_done(task: asyncio.Task) -> None:
        """Log unexpected exits from the receive loop."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.warning("streaming_stt_receive_error", error=str(exc))

    async def close(self) -> None:
        """Gracefully close the STT connection."""
        self._connected = False
        if self._ws:
            try:
                # Send Deepgram close message
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                # Brief wait for final transcript
                await asyncio.sleep(0.1)
            except Exception as exc:
                log.debug("streaming_stt_close_message_failed", error=str(exc))
            try:
                await self._ws.close()
            except Exception as exc:
                log.debug("streaming_stt_ws_close_failed", error=str(exc))
            self._ws = None
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except (asyncio.CancelledError, Exception):
                pass
        log.debug("streaming_stt_closed")

    async def _receive_loop(self) -> None:
        """Background task that reads Deepgram WebSocket messages."""
        try:
            while self._connected:
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(), timeout=15.0,
                    )
                except TimeoutError:
                    log.warning("streaming_stt_receive_timeout")
                    break

                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    log.warning("streaming_stt_malformed_message",
                                preview=str(message)[:200])
                    continue

                msg_type = data.get("type", "")

                if msg_type == "Results":
                    channel = data.get("channel", {})
                    alt = (channel.get("alternatives") or [{}])[0]
                    transcript = alt.get("transcript", "").strip()

                    if not transcript:
                        continue

                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)

                    event = TranscriptEvent(
                        text=transcript,
                        is_final=is_final,
                        speech_final=speech_final,
                    )

                    if self.on_transcript:
                        await self.on_transcript(event)

                elif msg_type == "UtteranceEnd":
                    # Deepgram utterance_end event — fire speech_final
                    if self.on_transcript:
                        await self.on_transcript(TranscriptEvent(
                            text="", is_final=True, speech_final=True,
                        ))

                elif msg_type == "Metadata":
                    log.debug("streaming_stt_metadata",
                              request_id=data.get("request_id", ""))

                elif msg_type == "Error":
                    log.warning("streaming_stt_error",
                                message=data.get("message", ""))

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if self._connected:
                log.warning("streaming_stt_receive_error", error=str(exc))
        finally:
            self._connected = False


def is_streaming_stt_capable(base_url: str) -> bool:
    """Check if the STT provider supports WebSocket streaming."""
    url = (base_url or "").lower()
    return "deepgram.com" in url


_MAX_FRAMES = 30000  # ~16 minutes at 32ms/frame


@dataclass
class BatchSTTFallback:
    """Accumulates audio frames and transcribes in one batch call.

    Used when the STT provider doesn't support streaming (OpenAI Whisper,
    Groq, etc.).  Audio is buffered and sent as a single POST request
    when ``transcribe()`` is called.
    """

    frames: list[bytes] = field(default_factory=list)

    def add_frame(self, pcm_bytes: bytes) -> None:
        if len(self.frames) >= _MAX_FRAMES:
            return
        self.frames.append(pcm_bytes)

    def get_audio(self) -> bytes:
        """Return all accumulated audio as a single PCM buffer."""
        return b"".join(self.frames)

    def clear(self) -> None:
        self.frames.clear()

    @property
    def duration_ms(self) -> float:
        """Approximate duration based on frame count (32 ms per frame)."""
        return len(self.frames) * 32.0
