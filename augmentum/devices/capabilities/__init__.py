"""Capability catalog.

Each capability is a typed contract — wire-protocol-agnostic — that one
or more drivers can implement. Capabilities are registered at import time;
lookups are O(1) via the `_ALL` dict.

Add a new capability by writing a module under this package, declaring
`Capability` instances at module level, and listing them in
`_REGISTERED_CAPABILITIES` below.
"""

from __future__ import annotations

from augmentum.devices.capabilities import (
    audio,
    display,
    lighting,
    media,
    notification,
    sensor,
    surface,
    switch,
)
from augmentum.devices.capability import Capability

_REGISTERED_CAPABILITIES: tuple[Capability, ...] = (
    *media.CAPABILITIES,
    *display.CAPABILITIES,
    *audio.CAPABILITIES,
    *lighting.CAPABILITIES,
    *switch.CAPABILITIES,
    *sensor.CAPABILITIES,
    *notification.CAPABILITIES,
    *surface.CAPABILITIES,
)


_ALL: dict[str, Capability] = {cap.id: cap for cap in _REGISTERED_CAPABILITIES}


def get_capability(capability_id: str) -> Capability | None:
    return _ALL.get(str(capability_id or "").strip())


def list_capabilities() -> list[Capability]:
    return list(_REGISTERED_CAPABILITIES)


def has_capability(capability_id: str) -> bool:
    return str(capability_id or "").strip() in _ALL


def resolve_action(capability_id: str, action_name: str) -> tuple[Capability, ...] | None:
    """Walk the `extends` chain to find the action.

    Returns the (capability, action) pair if found; None otherwise.
    Used by the registry when a driver implements `media.video_play@1`
    but the action being invoked is inherited from `media.audio_play@1`.
    """
    seen: set[str] = set()
    cap = get_capability(capability_id)
    while cap is not None and cap.id not in seen:
        seen.add(cap.id)
        action = cap.get_action(action_name)
        if action is not None:
            return (cap, action)
        if not cap.extends:
            return None
        cap = get_capability(cap.extends)
    return None


__all__ = [
    "get_capability",
    "has_capability",
    "list_capabilities",
    "resolve_action",
]
