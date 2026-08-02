"""Presence pipeline — stateful conversation orchestrator for the companion.

The companion presence path is the audio surface where Becca's expression
lives in real time. Unlike the voice-call overlay (which manages a single
focused exchange), the presence path stays alive whenever she's "in the
room" — wake-word listening, push-to-talk, ambient interjection.

What this subsystem owns:
  - The state machine that coordinates STT, turn detection, LLM generation,
    TTS synthesis, interruption, and backchannel dispatch as a single
    conversation (not five independent subsystems firing in sequence).
  - The Mimi-codec audio history substrate that survives server restarts
    and remains compatible with any future Kyutai-family model swap
    (PocketTTS / Moshi / CSM).
  - Companion-shaped LLM framing — short responses, conversational
    fillers, interruption-aware recovery.

What this subsystem does NOT own:
  - Audio acquisition (lives in ui/scripts/voice/mic-device.js + the
    existing AudioWorklet path).
  - The companion runtime, dispatch tree, growth loop, or verb catalog
    (those stay in augmentum/companion/companion.py and growth/).
  - Voice-call mode (that's voice.js + voice_routes.py, untouched).

Phase plan (per the 2026-06 design pass):
  Phase 1: state machine + pipeline orchestrator + WS endpoint stub  ← this
  Phase 2: Mimi audio history substrate + PocketTTS token capture
  Phase 3: streaming chunker + LLM/TTS streaming integration
  Phase 4: Smart Turn v3 + speculative LLM kickoff
  Phase 5: tiered VAD + interruption coordination
  Phase 6: backchannel policy + pre-synthesized clip dispatch
  Phase 7: companion system prompt frame + settings UI

Each phase ships independently and the pipeline orchestrator carries the
state model that lets the later phases plug in without architectural churn.
"""

from __future__ import annotations

from augmentum.companion.presence.audio_history import (
    DEFAULT_RETENTION_DAYS,
    DEFAULT_WINDOW_SECONDS,
    MimiAudioHistory,
    SPEAKER_BECCA,
    SPEAKER_USER,
    Turn,
    deserialize_mimi_tokens,
    serialize_mimi_tokens,
)
from augmentum.companion.presence.pipeline import (
    PresencePipeline,
    StateTransition,
)
from augmentum.companion.presence.state import (
    PresenceContext,
    PresenceState,
    VALID_TRANSITIONS,
    is_valid_transition,
)

__all__ = [
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_WINDOW_SECONDS",
    "MimiAudioHistory",
    "PresenceContext",
    "PresencePipeline",
    "PresenceState",
    "SPEAKER_BECCA",
    "SPEAKER_USER",
    "StateTransition",
    "Turn",
    "VALID_TRANSITIONS",
    "deserialize_mimi_tokens",
    "is_valid_transition",
    "serialize_mimi_tokens",
]
