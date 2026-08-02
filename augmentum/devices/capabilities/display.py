"""Display capabilities — image, text, and web content on screens."""

from __future__ import annotations

from augmentum.devices.capability import ActionSchema, Capability


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="display.image_show@1",
        label="Show Image",
        description="Display a still image on a screen.",
        actions=(
            ActionSchema(
                name="show",
                description="Display an image. Stateful for displays that hold the image until cleared.",
                args_schema={
                    "type": "object",
                    "required": ["image_url"],
                    "properties": {
                        "image_url": {"type": "string"},
                        "caption": {"type": "string"},
                        "duration_s": {"type": "number", "minimum": 0},
                    },
                },
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
            ActionSchema(
                name="clear",
                description="Stop showing the current image.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {
                "image_url": {"type": "string"},
                "caption": {"type": "string"},
                "displayed_since": {"type": "number"},
            },
        },
        events=("shown", "cleared"),
        lm_tools=("show",),
    ),
    Capability(
        id="display.text_show@1",
        label="Show Text",
        description="Display text or a notification on a screen.",
        actions=(
            ActionSchema(
                name="show",
                description="Display a text message.",
                args_schema={
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string"},
                        "title": {"type": "string"},
                        "duration_s": {"type": "number", "minimum": 0},
                        "priority": {"type": "string", "enum": ["low", "normal", "high"]},
                    },
                },
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="clear",
                description="Stop showing the current text.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "title": {"type": "string"},
            },
        },
        events=("shown", "cleared"),
        lm_tools=("show",),
    ),
    Capability(
        id="display.web_show@1",
        label="Show Web Page",
        description="Load and display a URL on a screen.",
        actions=(
            ActionSchema(
                name="load_url",
                description="Load a web URL on the device.",
                args_schema={
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {"type": "string"},
                    },
                },
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
            ActionSchema(
                name="clear",
                description="Close the loaded page.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
                is_stateful=True,
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
        },
        events=("loaded", "cleared", "error"),
        lm_tools=("load_url",),
    ),
)
