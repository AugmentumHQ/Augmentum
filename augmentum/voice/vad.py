"""Server-side Voice Activity Detection using Silero VAD v6.

Processes 16 kHz mono PCM16 audio frames (512 samples = 32 ms each)
and emits speech start/end events.  Runs entirely on CPU, <1 ms per frame.

Install: ``pip install silero-vad``  (MIT license, ~2 MB model)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Silero VAD processes 512-sample frames at 16 kHz (32 ms each).
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32 ms
FRAME_BYTES = FRAME_SAMPLES * 2  # 16-bit PCM = 2 bytes/sample


class VadState(Enum):
    """Current VAD state machine position."""

    IDLE = auto()       # No speech detected
    SPEECH = auto()     # Actively speaking
    TRAILING = auto()   # Silence after speech — waiting for timeout


@dataclass
class VadEvent:
    """Event emitted by the VAD processor."""

    kind: str           # "speech_start" | "speech_end" | "speech_discard"
    timestamp: float    # monotonic timestamp


@dataclass
class VadProcessor:
    """Stateful Silero VAD wrapper with speech boundary detection.

    Parameters:
        speech_threshold: probability above which a frame is "speech" (0.0–1.0)
        silence_duration_ms: ms of silence after speech to trigger end-of-speech
        min_speech_ms: minimum speech duration to count as real (not noise)
        prefix_padding_ms: audio to keep before speech_start for STT context
    """

    speech_threshold: float = 0.6
    silence_duration_ms: int = 800
    min_speech_ms: int = 250
    prefix_padding_ms: int = 300
    min_start_frames: int = 3  # consecutive speech frames to trigger start (~96ms)

    # --- Internal state ---
    _model: object = field(default=None, repr=False)
    _state: VadState = field(default=VadState.IDLE, init=False)
    _speech_start_time: float = field(default=0.0, init=False)
    _last_speech_time: float = field(default=0.0, init=False)
    _prefix_buffer: list[bytes] = field(default_factory=list, init=False)
    _prefix_max_frames: int = field(default=0, init=False)
    _consecutive_speech: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._prefix_max_frames = max(
            1, int(self.prefix_padding_ms / 32),  # 32 ms per frame
        )

    def load_model(self) -> None:
        """Lazy-load the Silero VAD model."""
        if self._model is not None:
            return
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad(onnx=True)
            log.info("silero_vad_loaded", backend="onnx")
        except ImportError:
            try:
                import torch
                self._model, _ = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", onnx=True,
                )
                log.info("silero_vad_loaded", backend="torch_hub_onnx")
            except Exception:
                log.warning("silero_vad_unavailable")
                raise

    def reset(self) -> None:
        """Reset VAD state between connections."""
        self._state = VadState.IDLE
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
        self._prefix_buffer.clear()
        self._consecutive_speech = 0
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception as exc:
                log.debug("vad_reset_states_failed", path="reset", error=str(exc))

    def soft_reset(self) -> None:
        """Reset speech state without clearing prefix buffer.

        Use for echo suppression — keeps pre-speech context for the next
        real speech event while discarding the false-start state.
        """
        self._state = VadState.IDLE
        self._consecutive_speech = 0
        if self._model is not None:
            try:
                self._model.reset_states()
            except Exception as exc:
                log.debug("vad_reset_states_failed", path="soft_reset", error=str(exc))

    def process_frame(self, pcm_bytes: bytes) -> VadEvent | None:
        """Process one 32 ms PCM16 frame.  Returns a VadEvent or None.

        Args:
            pcm_bytes: exactly FRAME_BYTES (1024) bytes of 16-bit LE PCM at 16 kHz.
        """
        if self._model is None:
            self.load_model()

        if len(pcm_bytes) != FRAME_BYTES:
            log.warning("vad_frame_size_mismatch",
                        expected=FRAME_BYTES, got=len(pcm_bytes))
            return None

        now = time.monotonic()

        # Convert bytes → float32 tensor
        import torch

        samples = np.frombuffer(pcm_bytes[:FRAME_BYTES], dtype=np.int16)
        audio = torch.from_numpy(samples.astype(np.float32) / 32768.0)

        prob = float(self._model(audio, SAMPLE_RATE))
        is_speech = prob >= self.speech_threshold

        if self._state == VadState.IDLE:
            # Maintain rolling prefix buffer
            self._prefix_buffer.append(pcm_bytes)
            if len(self._prefix_buffer) > self._prefix_max_frames:
                self._prefix_buffer.pop(0)

            if is_speech:
                self._consecutive_speech += 1
                if self._consecutive_speech >= self.min_start_frames:
                    self._state = VadState.SPEECH
                    self._speech_start_time = now
                    self._last_speech_time = now
                    self._consecutive_speech = 0
                    return VadEvent(kind="speech_start", timestamp=now)
            else:
                self._consecutive_speech = 0

        elif self._state == VadState.SPEECH:
            if is_speech:
                self._last_speech_time = now
            else:
                self._state = VadState.TRAILING

        elif self._state == VadState.TRAILING:
            if is_speech:
                # Speech resumed — back to active
                self._state = VadState.SPEECH
                self._last_speech_time = now
            else:
                silence_ms = (now - self._last_speech_time) * 1000
                if silence_ms >= self.silence_duration_ms:
                    speech_ms = (self._last_speech_time - self._speech_start_time) * 1000
                    self._state = VadState.IDLE
                    self._prefix_buffer.clear()

                    if speech_ms < self.min_speech_ms:
                        return VadEvent(kind="speech_discard", timestamp=now)
                    return VadEvent(kind="speech_end", timestamp=now)

        return None

    def get_prefix_audio(self) -> bytes:
        """Return buffered prefix audio (pre-speech context for STT)."""
        return b"".join(self._prefix_buffer)

    @property
    def is_speaking(self) -> bool:
        return self._state in (VadState.SPEECH, VadState.TRAILING)

    @property
    def voiced_ms(self) -> float:
        """Milliseconds of actual voiced span in the current segment.

        ``is_speaking`` stays True through TRAILING (post-speech silence
        up to ``silence_duration_ms``), so wall-clock time since
        speech_start overstates how much real speech occurred. This is
        the same quantity the end-of-segment discard check uses
        (``speech_ms < min_speech_ms``) — gates that decide "was that
        real speech?" (e.g. barge-in confirmation) must measure this,
        not elapsed time, or a sub-min_speech_ms blip plus trailing
        silence can confirm an interrupt that the segment-end check
        then discards.
        """
        if self._state in (VadState.SPEECH, VadState.TRAILING):
            return max(0.0, (self._last_speech_time - self._speech_start_time) * 1000.0)
        return 0.0
