"""In-memory pub/sub hub for live bug-finder progress events.

The orchestrator emits structured events at each stage boundary +
subagent completion. The SSE route at ``/api/bug-finder/runs/{run_id}/
events`` subscribes a queue per client and drains it as the run
progresses. The whole substrate is process-local — runs that span
restarts surface their final state via the persisted row, not the
stream.

Hub lifecycle:
* First publish for a run_id creates the hub.
* Each subscriber gets its own ``asyncio.Queue`` so a slow client
  can't backpressure the publisher.
* ``terminal=True`` events also signal hub close: subscribers drain
  their queues then exit; ``cleanup`` removes the hub from the parent
  dict so the next run with the same id starts fresh.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    """One structured event in the bug-finder progress stream."""

    kind: str
    """One of: ``stage`` (stage transition), ``subagent_start`` /
    ``subagent_complete`` (one subagent boundary), ``findings`` (running
    totals), ``done`` (terminal — stream closes after this event)."""

    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    terminal: bool = False

    def to_sse(self) -> str:
        """Format as a single SSE ``data:`` block. Newlines in the
        payload are JSON-escaped, so a single ``data:`` line is safe."""
        import json
        return f"event: {self.kind}\ndata: {json.dumps(self.payload)}\n\n"


class BugFinderStreamHub:
    """Per-run pub/sub. Cheap to construct; bounded by subscriber count."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._subscribers: list[asyncio.Queue[StreamEvent]] = []
        self._closed = False
        # Replay buffer so a subscriber connecting mid-run gets the
        # events they missed (e.g. user closes + reopens the panel).
        # Cap at 200 entries — at one event per stage + per subagent,
        # that's ~120 events for a 40-chunk run, comfortably under cap.
        self._history: list[StreamEvent] = []

    def subscribe(self) -> asyncio.Queue[StreamEvent]:
        """Register a new subscriber. Replays buffered history."""
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue(maxsize=512)
        for evt in self._history:
            try:
                queue.put_nowait(evt)
            except asyncio.QueueFull:
                break
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[StreamEvent]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    def publish(self, event: StreamEvent) -> None:
        """Push an event to every subscriber + the replay buffer.

        Non-blocking — when a subscriber's queue is full the event is
        dropped for that subscriber only. The hub does not close on
        publish failures: terminal events should always reach the
        history buffer so late subscribers still see the run finished.
        """
        if self._closed and not event.terminal:
            return
        self._history.append(event)
        if len(self._history) > 200:
            del self._history[: len(self._history) - 200]
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        if event.terminal:
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# ---------------------------------------------------------------------------
# Registry: app.state.bug_finder_streams keyed by run_id
# ---------------------------------------------------------------------------


def get_or_create_hub(
    registry: dict[str, BugFinderStreamHub], run_id: str,
) -> BugFinderStreamHub:
    """Lazy-init helper. Pass ``app.state.bug_finder_streams`` as the
    registry. Safe to call concurrently from one event loop because
    dict reads/writes are atomic under the GIL."""
    hub = registry.get(run_id)
    if hub is None:
        hub = BugFinderStreamHub(run_id)
        registry[run_id] = hub
    return hub


def drop_hub(
    registry: dict[str, BugFinderStreamHub], run_id: str,
) -> None:
    """Called when a stream closes and the hub's history is no longer
    needed. We retire closed hubs eagerly so the registry doesn't grow
    unbounded over long-lived processes."""
    registry.pop(run_id, None)
