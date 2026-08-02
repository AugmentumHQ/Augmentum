"""Standalone audio capabilities (TTS, tones) — distinct from media playback."""

from __future__ import annotations

from augmentum.devices.capability import ActionSchema, Capability


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="audio.tts_say@1",
        label="Speak Text",
        description="Synthesize and speak text on a device's speaker.",
        actions=(
            ActionSchema(
                name="say",
                description="Speak the given text. Returns once playback starts (or on failure).",
                args_schema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                        "voice": {"type": "string"},
                        "language": {"type": "string"},
                        "rate": {"type": "number"},
                        "volume": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                },
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={"type": "object"},
        events=("started", "ended", "error"),
        lm_tools=("say",),
    ),
    Capability(
        id="audio.tone_play@1",
        label="Play Tone",
        description="Play a short alert tone or named sound.",
        actions=(
            ActionSchema(
                name="play_tone",
                description="Play a basic tone.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "frequency_hz": {"type": "number", "minimum": 20, "maximum": 20000},
                        "duration_ms": {"type": "integer", "minimum": 50},
                        "volume": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                },
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="play_named",
                description="Play a named device-supported sound (chime, alarm, etc).",
                args_schema={
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "volume": {"type": "integer", "minimum": 0, "maximum": 100},
                    },
                },
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={"type": "object"},
        events=("ended",),
        lm_tools=("play_named",),
    ),
)
