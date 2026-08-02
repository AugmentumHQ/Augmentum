"""Live self-edit runs — the guided, streamed view of a self-edit in flight.

A self-edit is not one agent call; it's a *pipeline*: pick a target → ground it in
the scanner's findings → spin an isolated candidate worktree → drive a (possibly
escalating) agent → boot-smoke + the confirm oracle + an audit diff → a verdict.
The blocking ``/propose`` returns only the final verdict, minutes later. This
module makes that pipeline observable as it happens, so the user is *guided* —
the way coder mode shows its work, but richer: the ladder rungs, every verifier,
the score moving.

Two pieces:

* **A live run** (``LiveRun``) — a server-owned object (survives a viewer
  refresh): a monotonic event buffer (for instant replay-on-attach) + a
  subscriber bus (for live tailing) + a terminal result. ``LiveRunManager``
  holds them by id and evicts each a few minutes after it finishes.

* **A context-scoped progress sink** — ``emit_progress(event)`` reads a
  :class:`contextvars.ContextVar` set for the duration of one run's task. This
  is how the deep pipeline code (orchestrator / escalate / the edit driver)
  feeds rich events to the bus WITHOUT threading a ``progress=`` parameter
  through every signature. Outside a run (e.g. the synchronous ``/propose``
  preview, or tests) the sink is unset and ``emit_progress`` is a no-op, so the
  instrumentation is free when nobody's watching.

No import of the orchestrator / coder layers here (they import *this*), so it
stays dependency-free and unit-testable with a fake coro.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from collections.abc import Awaitable, Callable
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Set to a run's ``emit`` for the lifetime of its task; None elsewhere.
_SINK: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "selfedit_progress_sink", default=None)

# Keep a finished run live this long so a late/re-attaching viewer still gets the
# full buffer + the terminal frame from memory (the durable copy is the archive).
_EVICT_GRACE_S = 300
# Cap the in-memory buffer so a pathological run can't grow without bound; the
# oldest events drop (the verdict + recent activity are what matter on attach).
_MAX_BUFFER = 2000


class LiveRun:
    """One self-edit run in flight: event buffer + subscriber bus + result."""

    def __init__(self, run_id: str, *, user_id: str, title: str = "",
                 target: str = "", ladder: list[str] | None = None) -> None:
        self.run_id = run_id
        self.user_id = user_id
        self.title = title
        self.target = target
        self.ladder = ladder or []
        self.status = "running"
        self.events: list[dict] = []
        self.seq = 0
        self.result: dict | None = None
        self.finished = asyncio.Event()
        self.task_handle: asyncio.Task | None = None
        self._subs: set[asyncio.Queue] = set()

    # -- bus -----------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def emit(self, event: dict) -> None:
        """Stamp a seq, buffer it (for replay), fan out to live subscribers.
        Never raises — a broken subscriber can't stall the run."""
        self.seq += 1
        ev = {"seq": self.seq, **event}
        self.events.append(ev)
        if len(self.events) > _MAX_BUFFER:
            del self.events[: len(self.events) - _MAX_BUFFER]
        for q in list(self._subs):
            with contextlib.suppress(Exception):  # slow/broken subscriber, drop
                q.put_nowait(ev)

    def finish(self, result: dict) -> None:
        """Mark terminal, broadcast the ``done`` frame, release waiters. Idempotent."""
        if self.finished.is_set():
            return
        self.status = result.get("status", "done")
        self.result = result
        self.seq += 1
        done = {"seq": self.seq, "kind": "done", **result}
        self.events.append(done)
        for q in list(self._subs):
            with contextlib.suppress(Exception):
                q.put_nowait(done)
        self.finished.set()

    def snapshot(self, *, since: int = 0) -> dict:
        return {
            "run_id": self.run_id, "title": self.title, "target": self.target,
            "ladder": self.ladder, "status": self.status,
            "events": [e for e in self.events if e.get("seq", 0) > since],
            "result": self.result, "seq": self.seq,
        }


class LiveRunManager:
    """Holds live self-edit runs by id; evicts each a few minutes post-finish."""

    def __init__(self) -> None:
        self._runs: dict[str, LiveRun] = {}

    def get(self, run_id: str) -> LiveRun | None:
        return self._runs.get(run_id)

    def list_active(self, *, user_id: str) -> list[dict]:
        return [
            {"run_id": r.run_id, "title": r.title, "target": r.target,
             "status": r.status}
            for r in self._runs.values()
            if r.user_id == user_id and not r.finished.is_set()
        ]

    def create(self, run_id: str, **kw) -> LiveRun:
        run = LiveRun(run_id, **kw)
        self._runs[run_id] = run
        return run

    async def stop(self, run_id: str, *, user_id: str) -> bool:
        run = self._runs.get(run_id)
        if (run is None or run.user_id != user_id or run.task_handle is None
                or run.finished.is_set()):
            return False
        run.task_handle.cancel()
        return True

    def _schedule_evict(self, run_id: str) -> None:
        async def _evict() -> None:
            await asyncio.sleep(_EVICT_GRACE_S)
            self._runs.pop(run_id, None)
        with contextlib.suppress(RuntimeError):  # no running loop (e.g. in a test)
            asyncio.create_task(_evict())


def get_live_run_manager(app_state: Any) -> LiveRunManager:
    """Lazily attach a single self-edit LiveRunManager to app.state (its OWN, not
    the coder run manager — distinct ids, distinct lifecycle)."""
    mgr = getattr(app_state, "selfedit_live_manager", None)
    if mgr is None:
        mgr = LiveRunManager()
        app_state.selfedit_live_manager = mgr
    return mgr


def emit_progress(event: dict) -> None:
    """Emit a pipeline/agent event to the current run's bus, if one is scoped.

    No-op outside a live run — so the synchronous ``/propose`` preview and tests
    pay nothing. Called from orchestrator / escalate / the edit driver."""
    sink = _SINK.get(None)
    if sink is None:
        return
    with contextlib.suppress(Exception):  # instrumentation must never break the run
        sink(event)


def launch_live_run(
    app_state: Any, *, user_id: str, run_id: str, title: str, target: str,
    ladder: list[str], coro_factory: Callable[[], Awaitable[dict]],
) -> LiveRun:
    """Create a server-owned run and start ``coro_factory()`` on a task that
    outlives any viewer. The progress sink is scoped to that task, so every
    ``emit_progress`` inside the pipeline lands on this run's bus. The coro's
    returned dict becomes the terminal ``done`` payload. Returns immediately."""
    mgr = get_live_run_manager(app_state)
    run = mgr.create(run_id, user_id=user_id, title=title, target=target, ladder=ladder)
    run.emit({"kind": "run", "run_id": run_id, "title": title, "target": target,
              "ladder": ladder})

    async def _work() -> None:
        token = _SINK.set(run.emit)
        try:
            result = await coro_factory()
            run.finish(result if isinstance(result, dict) else {"status": "done", "ok": True})
        except asyncio.CancelledError:
            run.finish({"status": "cancelled", "ok": False, "error": "stopped by user"})
            raise
        except Exception as exc:  # noqa: BLE001 — never leave a run hanging
            log.warning("selfedit_live_run_crashed", run_id=run_id, error=repr(exc))
            run.finish({"status": "failed", "ok": False, "error": repr(exc)})
        finally:
            _SINK.reset(token)
            mgr._schedule_evict(run_id)

    def _on_done(t: asyncio.Task) -> None:
        # Safety net: if the task ended WITHOUT ``_work`` running its body — e.g.
        # cancelled before it ever started (CancelledError thrown at coroutine
        # entry, so the in-body except never fires) — finish the run anyway, so a
        # viewer's ``finished.wait()`` always releases. No-op once finished.
        if run.finished.is_set():
            return
        if t.cancelled():
            run.finish({"status": "cancelled", "ok": False, "error": "stopped by user"})
        else:
            exc = t.exception()
            run.finish({"status": "failed", "ok": False,
                        "error": repr(exc) if exc else "run ended unexpectedly"})

    run.task_handle = asyncio.create_task(_work())
    run.task_handle.add_done_callback(_on_done)
    return run
