"""Switch / on-off-style capabilities."""

from __future__ import annotations

from augmentum.devices.capability import ActionSchema, Capability

CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        id="switch.toggle@1",
        label="Switch",
        description="Generic on/off switch (smart plug, fan, etc).",
        actions=(
            ActionSchema(
                name="on",
                description="Turn the switch on.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="off",
                description="Turn the switch off.",
                args_schema={"type": "object"},
                returns_schema={"type": "object"},
            ),
            ActionSchema(
                name="toggle",
                description="Flip the switch state.",
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
)
