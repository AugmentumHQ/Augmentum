"""Typed capability contracts.

A capability is a wire-protocol-agnostic description of what a device can
do. Drivers declare which capabilities they implement; the registry uses
capabilities to validate invocations and project LLM tools.

Capability IDs include a version suffix (`media.video_play@1`) so a future
revision can ship alongside the previous one. Capabilities can extend a
parent capability (single inheritance only) so `media.video_play` can
inherit the action surface of `media.audio_play` without duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ActionSchema:
    """One callable action on a capability."""

    name: str
    description: str
    args_schema: dict[str, Any] = field(default_factory=dict)
    returns_schema: dict[str, Any] = field(default_factory=dict)
    is_stateful: bool = False
    expected_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": dict(self.args_schema or {}),
            "returns_schema": dict(self.returns_schema or {}),
            "is_stateful": bool(self.is_stateful),
            "expected_latency_ms": int(self.expected_latency_ms or 0),
        }


@dataclass(frozen=True, slots=True)
class Capability:
    """Typed contract a driver implements to expose a device behavior."""

    id: str
    label: str
    description: str
    actions: tuple[ActionSchema, ...] = ()
    state_schema: dict[str, Any] = field(default_factory=dict)
    events: tuple[str, ...] = ()
    extends: str = ""
    lm_tools: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def get_action(self, name: str) -> ActionSchema | None:
        for action in self.actions:
            if action.name == name:
                return action
        return None

    def is_stateful(self) -> bool:
        return any(action.is_stateful for action in self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "actions": [action.to_dict() for action in self.actions],
            "state_schema": dict(self.state_schema or {}),
            "events": list(self.events),
            "extends": self.extends,
            "lm_tools": list(self.lm_tools),
            "extra": dict(self.extra or {}),
        }
