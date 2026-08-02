"""Sensor capabilities — read-only state."""

from __future__ import annotations

from augmentum.devices.capability import ActionSchema, Capability

CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="sensor.read_state@1",
        label="Sensor",
        description="Read current sensor state (temperature, motion, etc).",
        actions=(
            ActionSchema(
                name="read",
                description="Return current sensor reading.",
                args_schema={"type": "object"},
                returns_schema={
                    "type": "object",
                    "properties": {
                        "value": {},
                        "unit": {"type": "string"},
                        "ts": {"type": "number"},
                    },
                },
            ),
        ),
        state_schema={
            "type": "object",
            "properties": {
                "value": {},
                "unit": {"type": "string"},
                "ts": {"type": "number"},
            },
        },
        events=("state_changed",),
        lm_tools=("read",),
    ),
)
