"""Augmentum-internal surface capabilities.

Treat augmentum's own UI panels as devices in the same registry. The
chat surface, the avatar canvas, the browse panel are first-class device
targets — `display.image_show@1` routes to a TV OR the chat panel via
the same call.

This file declares augmentum-specific capabilities that don't fit the
generic display/audio/lighting buckets.
"""

from __future__ import annotations

from augmentum.devices.capability import ActionSchema, Capability

CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="surface.show_artifact@1",
        label="Show Artifact",
        description="Push a structured artifact (image, document, code, gallery) to an augmentum surface.",
        actions=(
            ActionSchema(
                name="show",
                description="Render an artifact on the target surface.",
                args_schema={
                    "type": "object",
                    "required": ["artifact_type"],
                    "properties": {
                        "artifact_type": {"type": "string"},
                        "payload": {"type": "object"},
                        "title": {"type": "string"},
                    },
                },
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
            ActionSchema(
                name="dismiss",
                description="Remove the artifact from the surface.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {
                "artifact_type": {"type": "string"},
                "title": {"type": "string"},
            },
        },
        events=("shown", "dismissed", "interacted"),
        lm_tools=("show",),
    ),
    Capability(
        id="avatar.set_pose@1",
        label="Avatar Pose",
        description="Drive the VRM avatar's pose / expression.",
        actions=(
            ActionSchema(
                name="apply",
                description="Apply a pose by name or by parameters.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "pose_name": {"type": "string"},
                        "expression": {"type": "string"},
                        "transition_ms": {"type": "integer", "minimum": 0},
                    },
                },
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={"type": "object"},
        events=("pose_changed",),
        lm_tools=("apply",),
    ),
    Capability(
        id="surface.follow_state@1",
        label="Follow Surface State",
        description="Subscribe a screen, browser, or app surface to a durable Augmentum surface session.",
        actions=(
            ActionSchema(
                name="join",
                description="Join a surface session as a controller, display, or observer.",
                args_schema={
                    "type": "object",
                    "required": ["session_id", "role"],
                    "properties": {
                        "session_id": {"type": "string"},
                        "role": {"type": "string"},
                        "participant_id": {"type": "string"},
                        "label": {"type": "string"},
                    },
                },
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
            ActionSchema(
                name="leave",
                description="Leave or dismiss the current surface session.",
                args_schema={"type": "object", "properties": {"session_id": {"type": "string"}}},
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "revision": {"type": "integer"},
                "role": {"type": "string"},
            },
        },
        events=("joined", "left", "state_changed"),
        lm_tools=("join",),
    ),
    Capability(
        id="display.comic_read@1",
        label="Comic Reader Display",
        description="Render token-gated comic pages and follow reader scroll/page state.",
        actions=(
            ActionSchema(
                name="open",
                description="Open a comic/webtoon surface on this display.",
                args_schema={
                    "type": "object",
                    "required": ["session_id"],
                    "properties": {
                        "session_id": {"type": "string"},
                        "receiver_url": {"type": "string"},
                        "file_id": {"type": "string"},
                        "title": {"type": "string"},
                    },
                },
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
            ActionSchema(
                name="set_view_state",
                description="Apply page, scroll, and zoom state to the reader display.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer", "minimum": 1},
                        "scroll_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                        "zoom": {"type": "number", "minimum": 0.25, "maximum": 4},
                    },
                },
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
            ActionSchema(
                name="clear",
                description="Close the reader display.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
        ),
        state_schema={"type": "object"},
        events=("opened", "view_state_applied", "closed"),
        extends="surface.follow_state@1",
        lm_tools=("open", "set_view_state"),
    ),
    Capability(
        id="input.touch_scroll@1",
        label="Touch Scroll Input",
        description="Expose touch, wheel, or gesture scroll deltas as a normalized control stream.",
        actions=(
            ActionSchema(
                name="scroll",
                description="Patch reader or viewport scroll state from a touch controller.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "page": {"type": "integer", "minimum": 1},
                        "scroll_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                        "delta_y": {"type": "number"},
                    },
                },
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
        ),
        state_schema={"type": "object"},
        events=("scrolled",),
        lm_tools=("scroll",),
    ),
    Capability(
        id="input.gamepad_bridge@1",
        label="Gamepad Input Bridge",
        description="Expose phone, browser, or physical gamepad state to a paired game stream.",
        actions=(
            ActionSchema(
                name="apply",
                description="Patch normalized buttons, axes, and player routing for a game stream.",
                args_schema={
                    "type": "object",
                    "required": ["session_id"],
                    "properties": {
                        "session_id": {"type": "string"},
                        "player": {"type": "string"},
                        "buttons": {"type": "object"},
                        "axes": {"type": "object"},
                        "sequence": {"type": "integer", "minimum": 0},
                    },
                },
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
        ),
        state_schema={"type": "object"},
        events=("input_applied", "controller_connected", "controller_disconnected"),
        lm_tools=("apply",),
    ),
)
