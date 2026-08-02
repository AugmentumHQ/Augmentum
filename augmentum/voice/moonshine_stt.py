"""In-process streaming STT using Moonshine.

Provides the same interface as StreamingSTTSession (Deepgram) so the voice
routes can use either interchangeably.  Runs the Moonshine model locally —
no network hop, native partial transcripts, ~150-270ms latency on CPU.

The model is loaded once (class-level) and shared across all voice sessions.
Each session calls start()/stop() on the shared transcriber for clean state.

Requires: pip install moonshine-voice
Docs: https://github.com/moonshine-ai/moonshine
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_SAMPLE_RATE = 16000


class MoonshineSTTSession:
    """In-process streaming STT session using Moonshine.

    Same interface as ``StreamingSTTSession`` (Deepgram):
      - ``connect()`` — initialize transcriber
      - ``send_audio(pcm_bytes)`` — feed 16-bit PCM frames
      - ``close()`` — flush and stop

    Transcripts arrive via the ``on_transcript`` async callback using the
    same ``TranscriptEvent`` dataclass from ``streaming_stt.py``.

    The Moonshine model is loaded once at class level and shared across
    sessions.  Call ``MoonshineSTTSession.warmup()`` during voice connection
    startup to pre-load the model in the background.
    """

    # Class-level shared model state
    _shared_transcriber = None
    _model_loaded = False
    _model_lock = threading.Lock()

    # Dedicated transcriber for non-streaming batch transcription. Kept
    # SEPARATE from _shared_transcriber on purpose: the streaming path drives
    # _shared_transcriber through start()/add_audio()/stop() every turn, and an
    # add_audio worker abandoned on its 5s timeout keeps mutating that object
    # after the turn ends — wedging it so transcribe_without_streaming returns
    # empty instantly. The batch fallback is precisely the path that must
    # RESCUE an empty stream, so it cannot ride the same (possibly wedged)
    # instance. This one is never touched by the streaming lifecycle.
    _batch_transcriber = None
    _batch_lock = threading.Lock()

    # Resolved model path/arch from warmup(), reused to build the batch
    # transcriber without re-resolving (and re-downloading) the model.
    _resolved_path: Any = None
    _resolved_arch: Any = None

    def __init__(
        self,
        *,
        on_transcript: Any = None,
    ) -> None:
        self.on_transcript = on_transcript  # async callable(TranscriptEvent)
        self._listener: Any = None
        self._connected = False
        self._pending_events: list = []

    # ------------------------------------------------------------------
    # Class-level model management
    # ------------------------------------------------------------------

    _configured_model = ""
    _configured_arch = ""

    @classmethod
    def configure(cls, model_path: str = "", model_arch: str = "") -> None:
        """Set model label/arch before loading.  Call during app startup.

        ``model_arch`` is authoritative for size selection — it must be a
        canonical moonshine_voice arch string (e.g. ``medium-streaming``,
        ``small-streaming``, ``tiny-streaming``). ``warmup()`` parses it and
        asks the library for that specific arch, falling back to the library
        default for English if it's empty or unparseable. ``model_path`` is a
        human-readable label used only in logs.
        """
        cls._configured_model = model_path
        cls._configured_arch = model_arch

    @classmethod
    def warmup(cls) -> None:
        """Pre-load the Moonshine model.  Thread-safe, idempotent.

        Creates the shared Transcriber instance. This absorbs the ONNX
        GPU discovery delay (~1-20s) during startup, not during the first
        speech segment when latency matters.
        """
        if cls._model_loaded:
            return

        with cls._model_lock:
            if cls._model_loaded:
                return

            try:
                from moonshine_voice import (
                    Transcriber,
                    get_model_for_language,
                    string_to_model_arch,
                )

                # Honor the configured arch (size) when it parses; otherwise
                # fall back to the library's English default. get_model_for_
                # language resolves the cache path and downloads on demand.
                wanted_arch = None
                arch_str = (cls._configured_arch or "").strip()
                if arch_str:
                    try:
                        wanted_arch = string_to_model_arch(arch_str)
                    except Exception as exc:
                        log.warning("moonshine_bad_arch_string",
                                    configured=arch_str,
                                    note="falling back to default English model",
                                    error=str(exc))

                # get_model_for_language returns (path, ModelArch). Passing
                # wanted_model_arch=None yields the default for the language.
                resolved_path, resolved_arch = get_model_for_language(
                    "en", wanted_model_arch=wanted_arch
                )

                cls._shared_transcriber = Transcriber(
                    model_path=resolved_path,
                    model_arch=resolved_arch,
                )
                cls._resolved_path = resolved_path
                cls._resolved_arch = resolved_arch
                cls._model_loaded = True
                log.info("moonshine_model_loaded",
                         label=cls._configured_model or "(default)",
                         requested_arch=arch_str or "(default)",
                         model=str(resolved_path),
                         arch=str(resolved_arch))
            except ImportError:
                log.warning("moonshine_not_available",
                            note="pip install moonshine-voice for local streaming STT")
            except Exception as exc:
                log.warning("moonshine_load_error", error=str(exc))

    @classmethod
    def is_available(cls) -> bool:
        """True if moonshine-voice is installed and model can be loaded."""
        if cls._model_loaded:
            return True
        try:
            import moonshine_voice  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def get_batch_transcriber(cls):
        """Return the dedicated batch transcriber, building it on first use.

        Isolated from the streaming ``_shared_transcriber`` so a wedged
        streaming session can never poison non-streaming batch calls (the
        fallback that rescues an empty stream). Reuses the model path/arch
        resolved by ``warmup()``; returns ``None`` if the model can't load.

        Callers MUST hold ``cls._batch_lock`` around the actual
        ``transcribe_without_streaming`` call — the native ONNX session is
        not reentrant, so concurrent batch transcribes (multi-user) must
        serialize on this one instance.
        """
        if cls._batch_transcriber is not None:
            return cls._batch_transcriber

        if not cls._model_loaded:
            cls.warmup()
        if cls._resolved_path is None or cls._resolved_arch is None:
            return None

        with cls._batch_lock:
            if cls._batch_transcriber is not None:
                return cls._batch_transcriber
            try:
                from moonshine_voice import Transcriber

                cls._batch_transcriber = Transcriber(
                    model_path=cls._resolved_path,
                    model_arch=cls._resolved_arch,
                )
                log.info("moonshine_batch_transcriber_loaded",
                         model=str(cls._resolved_path),
                         arch=str(cls._resolved_arch))
            except Exception as exc:
                log.warning("moonshine_batch_transcriber_load_error", error=str(exc))
                return None
        return cls._batch_transcriber

    # ------------------------------------------------------------------
    # Per-session lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize a streaming STT session.

        Reuses the shared Transcriber (pre-created in warmup).
        Clears old listeners before adding new ones to prevent accumulation.
        """
        if not self._model_loaded:
            await asyncio.to_thread(self.warmup)

        if not self._model_loaded or self._shared_transcriber is None:
            log.warning("moonshine_connect_failed", reason="model not loaded")
            return

        try:
            from moonshine_voice import TranscriptEventListener

            from augmentum.voice.streaming_stt import TranscriptEvent

            transcriber = self._shared_transcriber
            session_ref = self

            # CRITICAL: Remove all old listeners before adding new ones.
            # Without this, listeners accumulate across sessions and events
            # fire into dead session references.
            transcriber.remove_all_listeners()

            self._pending_events = []
            self._last_partial_text = ""  # Dedup: track last emitted text

            class _Listener(TranscriptEventListener):
                def on_line_started(self, event):
                    # New speech segment — reset partial tracking
                    session_ref._last_partial_text = ""

                def on_line_text_changed(self, event):
                    # Moonshine sends the FULL cumulative line text on every
                    # callback, not deltas.  Only emit if text actually changed.
                    text = (event.line.text or "").strip()
                    dur = getattr(event.line, "duration", 0.0) or 0.0
                    if text and text != session_ref._last_partial_text:
                        session_ref._last_partial_text = text
                        session_ref._pending_events.append(
                            TranscriptEvent(
                                text=text,
                                is_final=False,
                                speech_final=False,
                                duration=dur,
                            )
                        )

                def on_line_completed(self, event):
                    text = (event.line.text or "").strip()
                    if not text:
                        return
                    dur = getattr(event.line, "duration", 0.0) or 0.0
                    # If text_changed already fired with identical text on
                    # this same frame, replace the pending partial with a
                    # final instead of appending a duplicate.
                    replaced = False
                    for i in range(len(session_ref._pending_events) - 1, -1, -1):
                        evt = session_ref._pending_events[i]
                        if not evt.is_final and evt.text == text:
                            session_ref._pending_events[i] = TranscriptEvent(
                                text=text,
                                is_final=True,
                                speech_final=True,
                                duration=dur,
                            )
                            replaced = True
                            break
                    if not replaced:
                        session_ref._pending_events.append(
                            TranscriptEvent(
                                text=text,
                                is_final=True,
                                speech_final=True,
                                duration=dur,
                            )
                        )
                    session_ref._last_partial_text = ""

            self._listener = _Listener()
            transcriber.add_listener(self._listener)
            transcriber.start()
            self._connected = True

            log.info("moonshine_session_started")

        except Exception as exc:
            log.warning("moonshine_connect_error", error=str(exc))
            self._connected = False

    async def send_audio(self, pcm_bytes: bytes) -> None:
        """Feed a PCM16 audio frame to Moonshine.

        Runs the native add_audio call in a worker thread via
        asyncio.to_thread so it never blocks the event loop. Bounded
        by a 5s timeout — if Moonshine hangs, we abandon the thread
        (it's a daemon worker so it dies with the process) and return.
        Awaiting serially also preserves frame ordering, which the
        native Transcriber's internal state depends on.

        Callbacks from the C++ listener fire during add_audio and push
        into self._pending_events; we dispatch them as async tasks
        once the worker returns.
        """
        if not self._connected or not self._shared_transcriber:
            return

        try:
            # Convert PCM16 bytes → float32 [-1, 1]
            samples = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            # Official API: add_audio(List[float], sample_rate)
            self._pending_events.clear()

            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self._shared_transcriber.add_audio,
                        samples.tolist(), _SAMPLE_RATE,
                    ),
                    timeout=5.0,
                )
            except TimeoutError:
                # to_thread workers can't be cancelled — the add_audio
                # call keeps running but we stop waiting. Matches the
                # previous t.join(timeout=5.0) + abandon semantics.
                log.warning("moonshine_add_audio_timeout", timeout_s=5.0)
                return

            # Dispatch any transcript events as async tasks
            if self._pending_events and self.on_transcript:
                for event in self._pending_events:
                    log.debug("moonshine_transcript_event",
                              text=event.text[:100] if event.text else "",
                              is_final=event.is_final)
                    asyncio.create_task(self.on_transcript(event))
                self._pending_events.clear()

        except Exception as exc:
            log.warning("moonshine_audio_error", error=str(exc))

    async def close(self) -> None:
        """Stop the transcriber and flush remaining transcript."""
        self._connected = False
        transcriber = self._shared_transcriber
        if transcriber:
            try:
                # stop() flushes any remaining partial transcript
                self._pending_events.clear()
                transcriber.stop()

                # Dispatch final flush events
                if self._pending_events and self.on_transcript:
                    for event in self._pending_events:
                        log.info("moonshine_flush_event",
                                 text=event.text[:100] if event.text else "",
                                 is_final=event.is_final)
                        await self.on_transcript(event)
                    self._pending_events.clear()

                # Clean up listener to prevent accumulation
                if self._listener:
                    try:
                        transcriber.remove_all_listeners()
                    except Exception as exc:
                        log.debug("moonshine_listener_cleanup_failed", error=str(exc))
                    self._listener = None

            except Exception as exc:
                log.warning("moonshine_close_error", error=str(exc))

        log.info("moonshine_session_closed")
