"""Bounded-queue event bus.

Drivers publish `Event`s; UI panels and automations subscribe. Each
subscriber gets its own bounded asyncio.Queue — slow consumers drop the
oldest event rather than blocking the publisher or growing the queue
unbounded.

Subscriptions are user-scoped: a subscriber sees events only for devices
owned by their user_id. Scoping is enforced on subscribe, not on each
publish, so the publisher path stays cheap.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from augmentum.devices.invocation import Event
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


_DEFAULT_QUEUE_SIZE = 100


class _Subscription:
    __slots__ = ("queue", "user_id", "device_id", "capability_id", "_drop_count")

    def __init__(
        self,
        *,
        user_id: str,
        device_id: str | None,
        capability_id: str | None,
        queue_size: int,
    ) -> None:
        self.queue: asyncio.Queue[Event | None] = asyncio.Queue(maxsize=max(1, queue_size))
        self.user_id = user_id
        self.device_id = device_id or None
        self.capability_id = capability_id or None
        self._drop_count = 0

    def matches(self, event: Event) -> bool:
        if event.user_id and event.user_id != self.user_id:
            return False
        if self.device_id and event.device_id != self.device_id:
            return False
        if self.capability_id and event.capability_id != self.capability_id:
            return False
        return True

    def offer(self, event: Event) -> None:
        if self.queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self._drop_count += 1
        with contextlib.suppress(asyncio.QueueFull):
            self.queue.put_nowait(event)


class EventBus:
    """Process-local pub/sub for device events."""

    def __init__(self, *, default_queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._subscriptions: set[_Subscription] = set()
        self._default_queue_size = max(1, int(default_queue_size or _DEFAULT_QUEUE_SIZE))

    async def publish(self, event: Event) -> None:
        if not isinstance(event, Event):
            return
        for sub in list(self._subscriptions):
            if sub.matches(event):
                sub.offer(event)

    def subscribe(
        self,
        *,
        user_id: str,
        device_id: str | None = None,
        capability_id: str | None = None,
        queue_size: int | None = None,
    ) -> AsyncIterator[Event]:
        sub = _Subscription(
            user_id=user_id,
            device_id=device_id,
            capability_id=capability_id,
            queue_size=queue_size or self._default_queue_size,
        )
        self._subscriptions.add(sub)

        async def _iter() -> AsyncIterator[Event]:
            try:
                while True:
                    event = await sub.queue.get()
                    if event is None:
                        return
                    yield event
            finally:
                self._subscriptions.discard(sub)

        return _iter()

    def close_all(self) -> None:
        """Send sentinel to every subscriber on shutdown."""
        for sub in list(self._subscriptions):
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(None)
        self._subscriptions.clear()
