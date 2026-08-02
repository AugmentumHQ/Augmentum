"""Device substrate — capabilities, drivers, and a single registry.

This package is the foundation Cast-to-TV, smart lights, sensors, and
augmentum's own UI surfaces all consume. Each device is bound to a driver
that implements one wire protocol; capabilities are typed, versioned
contracts the rest of the system speaks.

See `docs/superpowers/specs/2026-05-07-device-substrate-design.md` for the
architectural rationale.
"""

from __future__ import annotations

from augmentum.devices.capability import (
    ActionSchema,
    Capability,
)
from augmentum.devices.device import (
    Device,
    DiscoveredDevice,
)
from augmentum.devices.driver import (
    DeviceDriver,
    DriverContext,
)
from augmentum.devices.events import (
    Event,
    EventBus,
)
from augmentum.devices.invocation import (
    InvocationContext,
    InvocationError,
    InvocationResult,
    PairResult,
)
from augmentum.devices.sessions import (
    CapabilitySession,
    SessionRuntime,
)

__all__ = [
    "ActionSchema",
    "Capability",
    "CapabilitySession",
    "Device",
    "DeviceDriver",
    "DiscoveredDevice",
    "DriverContext",
    "Event",
    "EventBus",
    "InvocationContext",
    "InvocationError",
    "InvocationResult",
    "PairResult",
    "SessionRuntime",
]
