"""Lighting capabilities — on/off, color, brightness."""

from __future__ import annotations

from augmentum.devices.capability import ActionSchema, Capability

CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="lighting.set_state@1",
        label="Light On/Off",
        description="Turn a light on or off.",
        actions=(
            ActionSchema(
                name="on",
                description="Turn the light on.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="off",
                description="Turn the light off.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="toggle",
                description="Flip the light's on/off state.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {"is_on": {"type": "boolean"}},
        },
        events=("state_changed",),
        lm_tools=("on", "off", "toggle"),
    ),
    Capability(
        id="lighting.set_color@1",
        label="Light Color",
        description="Set the color of a light.",
        actions=(
            ActionSchema(
                name="apply",
                description="Set the light to a specific color (hex, rgb, hs, or color-temperature kelvin).",
                args_schema={
                    "type": "object",
                    "properties": {
                        "hex": {"type": "string"},
                        "rgb": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 255},
                            "minItems": 3,
                            "maxItems": 3,
                        },
                        "hue": {"type": "number", "minimum": 0, "maximum": 360},
                        "saturation": {"type": "number", "minimum": 0, "maximum": 100},
                        "kelvin": {"type": "integer", "minimum": 1000, "maximum": 10000},
                        "transition_ms": {"type": "integer", "minimum": 0},
                    },
                },
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {
                "hex": {"type": "string"},
                "rgb": {"type": "array"},
                "kelvin": {"type": ["integer", "null"]},
            },
        },
        events=("state_changed",),
        lm_tools=("apply",),
    ),
    Capability(
        id="lighting.set_brightness@1",
        label="Light Brightness",
        description="Set the brightness of a light (0-100).",
        actions=(
            ActionSchema(
                name="apply",
                description="Set brightness as a percentage.",
                args_schema={
                    "type": "object",
                    "required": ["level"],
                    "properties": {
                        "level": {"type": "integer", "minimum": 0, "maximum": 100},
                        "transition_ms": {"type": "integer", "minimum": 0},
                    },
                },
                returns_schema={"type": "object"},
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {"level": {"type": "integer"}},
        },
        events=("state_changed",),
        lm_tools=("apply",),
    ),
)
