"""Server→UI pub/sub for cross-feature live updates.

Distinct from ``status_bus`` (per-request engine stage events) and the
per-feature streams (model download progress, coder run, voice). This is
the channel admin/CRUD operations broadcast on so the UI can keep
provider lists, model catalogs, settings, and similar surfaces fresh
without polling or page reloads.

Single in-process bus, one SSE endpoint, per-subscriber bounded queue.
Server-scoped events (admin CRUD on shared tables like ``providers``)
publish with ``user_id=""`` and reach every subscriber. User-scoped
events publish with a specific ``user_id`` and only reach that user's
subscribers. Slow clients are never allowed to backpressure the
publishers — a full queue drops the event with a warning log.

Wiring sites in this commit:
    * provider_routes.py — providers.added / updated / deleted
    * models routes / jobs — models.install_started / install_complete

To add a new topic: import ``publish`` in the handler, fire it on the
success path. The UI module that cares listens via the DOM CustomEvent
``system-event:<topic>`` dispatched by ``ui/scripts/system-events.js``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class SystemEvent:
    topic: str
    data: dict[str, Any]
    user_id: str = ""  # "" = broadcast
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)


_QUEUE_MAX = 256


class Subscription:
    """One client's view of the bus. Use as an async context manager."""

    __slots__ = ("user_id", "queue", "id", "_bus")

    def __init__(self, user_id: str, bus: SystemEventBus) -> None:
        self.user_id = user_id
        self.queue: asyncio.Queue[SystemEvent] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self.id = uuid.uuid4().hex[:8]
        self._bus = bus

    async def __aenter__(self) -> Subscription:
        self._bus._register(self)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._bus._unregister(self)


class SystemEventBus:
    def __init__(self) -> None:
        self._subs: set[Subscription] = set()

    def _register(self, sub: Subscription) -> None:
        self._subs.add(sub)

    def _unregister(self, sub: Subscription) -> None:
        self._subs.discard(sub)

    def publish(self, topic: str, data: dict[str, Any], *, user_id: str = "") -> None:
        event = SystemEvent(topic=topic, data=data, user_id=user_id)
        for sub in list(self._subs):
            if event.user_id and sub.user_id != event.user_id:
                continue
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning(
                    "system_events_subscriber_dropped",
                    sub_id=sub.id, topic=topic, user_id=sub.user_id,
                )

    def subscribe(self, user_id: str) -> Subscription:
        return Subscription(user_id, self)


_bus = SystemEventBus()


def publish(topic: str, data: dict[str, Any] | None = None, *, user_id: str = "") -> None:
    """Fire-and-forget event broadcast. Safe to call from sync code."""
    _bus.publish(topic, dict(data or {}), user_id=user_id)


def subscribe(user_id: str) -> Subscription:
    return _bus.subscribe(user_id)


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/system", tags=["system"])

_KEEPALIVE_S = 15.0


@router.get("/events")
async def system_events_stream(request: Request) -> StreamingResponse:
    """Long-lived SSE feed of every event the current user can see.

    Browsers auto-reconnect on EventSource drop, so we don't replay
    history; clients re-fetch the affected collection (provider list,
    model catalog, etc.) on receipt as the source of truth.
    """
    user = request.scope.get("user")
    user_id = getattr(user, "id", "") if user else ""

    async def event_stream():
        yield ": connected\n\n"
        async with subscribe(user_id) as sub:
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(sub.queue.get(), timeout=_KEEPALIVE_S)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                payload = json.dumps({
                    "id": event.id,
                    "topic": event.topic,
                    "data": event.data,
                    "ts": event.ts,
                })
                yield f"event: {event.topic}\ndata: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
