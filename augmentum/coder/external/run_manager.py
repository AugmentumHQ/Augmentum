"""In-process registry for live Claude Code runs.

The keystone that makes a run a *server-owned object* instead of a side effect of
one browser request. Before this, a run executed inside the SSE generator's
lifetime, so closing the tab / refreshing cancelled it. Now:

* the work runs in a manager-owned ``asyncio.Task`` that outlives any viewer;
* viewers *subscribe* to a per-run event bus (attach/detach freely);
* an explicit ``stop()`` cancels the task.

Durability still lives in SQLite (``run_store``) — this registry only holds the
*live* tail. A viewer attaching to a finished/evicted run replays from the DB.
Everything an integration needs (Becca-driven, voice, mobile, scheduled) is
built on a run that no longer depends on a connection being open.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class Run:
    """One live run: its task handle, terminal state, and subscriber bus."""

    def __init__(
        self, run_id: str, *, workspace_id: str, user_id: str,
        task: str, resumed_from: str = "",
    ) -> None:
        self.run_id = run_id
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.task = task
        self.resumed_from = resumed_from
        self.status = "running"
        self.finished = asyncio.Event()
        self.final: dict | None = None          # the terminal 'done' payload
        self.task_handle: asyncio.Task | None = None
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, item: dict) -> None:
        """Fan an event out to every current subscriber (non-blocking)."""
        for q in list(self._subs):
            try:
                q.put_nowait(item)
            except Exception:  # noqa: BLE001 — a slow/broken subscriber can't stall the run
                pass

    def finish(self, final: dict) -> None:
        """Mark terminal, broadcast the 'done' frame, release waiters. Idempotent."""
        if self.finished.is_set():
            return
        self.status = final.get("status", "done")
        self.final = final
        self.publish({"kind": "done", **final})
        self.finished.set()


class RunManager:
    """Holds live runs by id. Evicts each shortly after it finishes (the DB keeps
    the durable copy, so late attachers fall back to ``run_store``)."""

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def create(self, run_id: str, **kw) -> Run:
        run = Run(run_id, **kw)
        self._runs[run_id] = run
        return run

    def start(self, run: Run, work: Callable[[Run], Awaitable[None]]) -> None:
        """Schedule ``work(run)`` on a manager-owned task. The task survives
        viewer disconnects; only ``stop()`` (or completion) ends it."""
        async def _wrap() -> None:
            try:
                await work(run)
            except asyncio.CancelledError:
                run.finish({
                    "status": "cancelled", "ok": False,
                    "error": "stopped by user", "files_changed": [],
                })
                raise
            except Exception as exc:  # noqa: BLE001 — never leave a run hanging
                log.warning("claude_run_task_crashed", run_id=run.run_id, error=repr(exc))
                run.finish({
                    "status": "failed", "ok": False,
                    "error": repr(exc), "files_changed": [],
                })
            finally:
                # Evict now — durable state is in SQLite; in-flight subscribers
                # already hold their queues (they received the 'done' frame).
                self._runs.pop(run.run_id, None)

        run.task_handle = asyncio.create_task(_wrap())

    async def stop(self, run_id: str) -> bool:
        """Cancel a live run. Returns False if it isn't running here."""
        run = self._runs.get(run_id)
        if run is None or run.task_handle is None or run.finished.is_set():
            return False
        run.task_handle.cancel()
        return True


def get_run_manager(app_state) -> RunManager:
    """Lazily attach a single RunManager to app.state."""
    mgr = getattr(app_state, "claude_run_manager", None)
    if mgr is None:
        mgr = RunManager()
        app_state.claude_run_manager = mgr
    return mgr
