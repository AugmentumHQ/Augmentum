"""Driver protocol — one wire protocol per implementation.

Drivers declare which capabilities they support and implement a small,
async-first interface for discovery, invocation, snapshot, and event
streaming. Drivers that need long-lived listeners (mDNS responder, MQTT
subscriber) launch them in `start()` and tear them down in `stop()`.

The registry never reaches inside a driver — it dispatches by `device.driver`
and lets the driver own its connection state and protocol details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from augmentum.devices.device import Device, DiscoveredDevice
    from augmentum.devices.invocation import (
        Event,
        InvocationContext,
        InvocationResult,
        PairResult,
    )


@dataclass(slots=True)
class DriverContext:
    """Shared resources passed to drivers at startup.

    Drivers should hold a reference and reuse the http_client + bus rather
    than opening fresh ones — pooled connections + bounded subscriber
    queues are configured here.
    """

    http_client: Any = None  # httpx.AsyncClient at runtime
    event_bus: Any = None    # EventBus
    settings_get: Any = None  # async (key) -> value
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DeviceDriver(Protocol):
    """Wire-protocol implementation that powers a class of devices.

    Methods are async because every interesting transport (TLS sockets,
    SOAP HTTP, mDNS multicast, MQTT) wants to be non-blocking. Drivers
    that have nothing to do for `start`/`stop` simply return.
    """

    id: str
    label: str
    description: str
    capabilities: tuple[str, ...]
    discovery_modes: tuple[str, ...]
    requires_pairing: bool
    supports_passive_discovery: bool

    async def start(self, ctx: DriverContext) -> None: ...

    async def stop(self) -> None: ...

    async def discover(
        self,
        *,
        timeout_s: float = 3.0,
        user_id: str = "",
    ) -> list[DiscoveredDevice]: ...

    async def probe(
        self,
        *,
        host: str,
        port: int | None = None,
        hint: dict[str, Any] | None = None,
    ) -> DiscoveredDevice | None: ...

    async def invoke(
        self,
        device: Device,
        capability: str,
        action: str,
        args: dict[str, Any],
        ctx: InvocationContext,
    ) -> InvocationResult: ...

    async def snapshot(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> dict[str, Any] | None: ...

    async def subscribe(
        self,
        device: Device,
        capability: str,
        ctx: InvocationContext,
    ) -> AsyncIterator[Event]: ...

    async def pair_start(
        self,
        device: Device,
        ctx: InvocationContext,
    ) -> PairResult: ...

    async def pair_complete(
        self,
        device: Device,
        code: str,
        ctx: InvocationContext,
    ) -> PairResult: ...
