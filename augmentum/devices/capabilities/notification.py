"""Notification delivery capabilities."""

from __future__ import annotations

from augmentum.devices.capability import ActionSchema, Capability

CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="notification.send@1",
        label="Send Notification",
        description="Push a notification to a device (phone, smart display, etc).",
        actions=(
            ActionSchema(
                name="send",
                description="Send a notification with a title and body.",
                args_schema={
                    "type": "object",
                    "required": ["body"],
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                        "image_url": {"type": "string"},
                        "deep_link": {"type": "string"},
                    },
                },
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={"type": "object"},
        events=("delivered", "error"),
        lm_tools=("send",),
    ),
)
