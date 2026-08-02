"""Per-run event bus for the foundry — the spine of the live theater.

The theater UI is only worth building if the user can *watch* the agent work.
That needs a realtime feed of what each stage is doing: code being generated,
the Blender render resolving, the game booting, the agent playing and deciding,
defects found, the score climbing, a regeneration kicking off.

``FoundryEventBus`` is a tiny in-memory pub/sub, one per run: the loop pushes
structured events; the SSE endpoint drains them to any number of subscribers
(the theater, plus late joiners who get the backlog first so a reload doesn't
lose the story so far). No external broker — same single-process assumption
the game-agent session registry already makes.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

# Event type vocabulary (documented so the theater and the loop stay in sync):
#   run_start     {passes, dimension, title}
#   pass_start    {index}
#   generating    {index}                     — coder loop dispatched
#   generated     {index, slug, violations}   — files emitted + validated
#   asset_render  {index, image}              — Blender render, base64 PNG data URL
#   play_start    {index, session_id, log_url}— autonomous play began
#   observation   {index, text}               — a decision/note (echoed from play)
#   pass_scored   {index, score, score_per_min, defects: [{kind,severity,detail}]}
#   regenerating  {index}                     — feeding defects into next pass
#   done          {improved, passes: [...] }
#   error         {message}


class FoundryEventBus:
    """Fan-out event queue for one foundry run.

    Keeps a bounded backlog so a subscriber that connects mid-run (or after a
    page reload) can replay what already happened before tailing live. Marked
    closed when the run ends so subscribers terminate cleanly.
    """

    def __init__(self, *, backlog_cap: int = 500) -> None:
        self._backlog: list[dict[str, Any]] = []
        self._backlog_cap = backlog_cap
        self._subscribers: set[asyncio.Queue[dict[str, Any] | None]] = set()
        self._closed = False

    def emit(self, type: str, **data: Any) -> None:
        """Publish an event. Non-blocking; safe to call from the loop."""
        if self._closed:
            return
        event = {"type": type, **data}
        self._backlog.append(event)
        if len(self._backlog) > self._backlog_cap:
            # Keep the head (run_start / early passes) AND recent tail — drop
            # the middle so a long run's story stays legible without unbounded
            # memory. Simpler: trim oldest beyond cap (head matters less than
            # live tail for a reconnecting viewer).
            self._backlog = self._backlog[-self._backlog_cap:]
        for q in list(self._subscribers):
            # a stalled subscriber never blocks the loop
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(event)

    def close(self) -> None:
        """Signal end-of-stream to all current + future subscribers."""
        self._closed = True
        for q in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(None)

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        """Yield the backlog, then live events, until the run closes."""
        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=1000)
        # Snapshot the backlog before registering so we neither miss nor
        # duplicate the boundary event.
        backlog = list(self._backlog)
        self._subscribers.add(q)
        try:
            for event in backlog:
                yield event
            if self._closed:
                return
            while True:
                event = await q.get()
                if event is None:  # close sentinel
                    return
                yield event
        finally:
            self._subscribers.discard(q)
