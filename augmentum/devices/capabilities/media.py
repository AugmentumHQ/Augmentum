"""Media-playback capabilities."""

from __future__ import annotations

from augmentum.devices.capability import ActionSchema, Capability

_AUDIO_PLAY_ACTIONS: tuple[ActionSchema, ...] = (
    ActionSchema(
        name="play",
        description="Begin playback of a content URL on the device.",
        args_schema={
            "type": "object",
            "required": ["content_url"],
            "properties": {
                "content_url": {"type": "string"},
                "content_type": {"type": "string"},
                "title": {"type": "string"},
                "poster_url": {"type": "string"},
                "start_time_s": {"type": "number", "minimum": 0},
                "metadata": {"type": "object"},
            },
        },
        returns_schema={"type": "object"},
        is_stateful=True,
    ),
    ActionSchema(
        name="pause",
        description="Pause active playback.",
        args_schema={"type": "object"},
        returns_schema={"type": "object"},
        is_stateful=True,
    ),
    ActionSchema(
        name="resume",
        description="Resume paused playback.",
        args_schema={"type": "object"},
        returns_schema={"type": "object"},
        is_stateful=True,
    ),
    ActionSchema(
        name="stop",
        description="Stop playback and tear down the session.",
        args_schema={"type": "object"},
        returns_schema={"type": "object"},
        is_stateful=True,
    ),
    ActionSchema(
        name="seek",
        description="Seek to a specific position (seconds from start).",
        args_schema={
            "type": "object",
            "required": ["position_s"],
            "properties": {
                "position_s": {"type": "number", "minimum": 0},
            },
        },
        returns_schema={"type": "object"},
        is_stateful=True,
    ),
    ActionSchema(
        name="set_volume",
        description="Set output volume level (0-100).",
        args_schema={
            "type": "object",
            "required": ["level"],
            "properties": {
                "level": {"type": "integer", "minimum": 0, "maximum": 100},
            },
        },
        returns_schema={"type": "object"},
    ),
    ActionSchema(
        name="set_mute",
        description="Mute or unmute output.",
        args_schema={
            "type": "object",
            "required": ["muted"],
            "properties": {
                "muted": {"type": "boolean"},
            },
        },
        returns_schema={"type": "object"},
    ),
)


_AUDIO_PLAY_STATE: dict = {
    "type": "object",
    "properties": {
        "current_time_s": {"type": "number"},
        "duration_s": {"type": "number"},
        "is_paused": {"type": "boolean"},
        "is_muted": {"type": "boolean"},
        "volume_level": {"type": ["integer", "null"]},
        "title": {"type": "string"},
        "receiver_state": {"type": "string"},
    },
}


_VIDEO_EXTRA_ACTIONS: tuple[ActionSchema, ...] = (
    ActionSchema(
        name="set_audio_track",
        description="Switch the active audio track.",
        args_schema={
            "type": "object",
            "required": ["track_index"],
            "properties": {"track_index": {"type": "integer", "minimum": 0}},
        },
        returns_schema={"type": "object"},
    ),
    ActionSchema(
        name="set_subtitle_track",
        description="Switch active subtitle track. Pass -1 to disable.",
        args_schema={
            "type": "object",
            "required": ["track_index"],
            "properties": {"track_index": {"type": "integer", "minimum": -1}},
        },
        returns_schema={"type": "object"},
    ),
)


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="media.audio_play@1",
        label="Audio Playback",
        description="Stream and control audio on a device.",
        actions=_AUDIO_PLAY_ACTIONS,
        state_schema=_AUDIO_PLAY_STATE,
        events=("state_changed", "ended", "error"),
        lm_tools=("play", "pause", "resume", "stop", "seek"),
    ),
    Capability(
        id="media.video_play@1",
        label="Video Playback",
        description="Stream and control video on a device.",
        actions=_AUDIO_PLAY_ACTIONS + _VIDEO_EXTRA_ACTIONS,
        state_schema=_AUDIO_PLAY_STATE,
        events=("state_changed", "ended", "error"),
        extends="media.audio_play@1",
        lm_tools=("play", "pause", "resume", "stop", "seek"),
    ),
    Capability(
        id="media.queue@1",
        label="Playback Queue",
        description="Manage a playback queue of media items.",
        actions=(
            ActionSchema(
                name="add",
                description="Append item to the queue.",
                args_schema={
                    "type": "object",
                    "required": ["content_url"],
                    "properties": {
                        "content_url": {"type": "string"},
                        "title": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                },
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="next",
                description="Skip to the next queued item.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="previous",
                description="Skip to the previous queued item.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="clear",
                description="Empty the queue.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "current_index": {"type": "integer"},
            },
        },
        events=("queue_changed",),
        lm_tools=("add", "next", "previous"),
    ),
)
