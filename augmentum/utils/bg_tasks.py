"""Tracked background tasks — keep references so Python's GC doesn't
silently drop fire-and-forget asyncio work mid-execution.

Without this, the pattern::

    asyncio.create_task(self._persist_and_clear(snap))

returns a task that no one holds a reference to. Python's GC is free
to collect that task before it finishes — silently dropping the work
(and swallowing any exceptions it would have raised). This has been
observed across the codebase as missing telemetry, dropped wake
events, vanished artifact builds, and stale cast session markers.

Usage::

    from augmentum.utils.bg_tasks import track
    track(self._persist_and_clear(snap))

The helper adds the task to a module-level ``set``, then auto-removes
it via ``add_done_callback`` when it finishes — so the set stays small
(only currently-running tasks live in it), but Python's GC sees a
strong reference for the lifetime of the work.

Returns the task so callers can ``await`` it or cancel it if they need
to. Most callers won't.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Module-level set holding refs to in-flight tasks. Cleared by the
# done-callback on each task. NOT a leak: ``add_done_callback`` runs
# exactly once per task, regardless of how it finishes (success,
# exception, cancellation).
_BG_TASKS: set[asyncio.Task] = set()


def _reap(task: asyncio.Task) -> None:
    """Done-callback: drop the ref AND surface a swallowed exception.

    A fire-and-forget task that raised would otherwise vanish (asyncio
    only prints "Task exception was never retrieved" at GC, if ever) —
    log it at warning so a failing background task is visible (audit
    2026-06-17). Cancellation is normal teardown, not an error.
    """
    _BG_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning(
            "bg_task_failed",
            name=task.get_name(), error=str(exc)[:200], exc_info=exc,
        )


def track(
    coro_or_task: Coroutine[Any, Any, Any] | asyncio.Task,
    *,
    name: str = "",
) -> asyncio.Task:
    """Schedule a coroutine OR register an existing task; return the task.

    Accepts either:
      * a coroutine (the common case) — wraps it in ``asyncio.create_task``
      * an existing ``asyncio.Task`` — just registers it for ref-keeping

    The accepted-task form is for callers who already created a task
    (e.g. needing a custom name) and just want the GC-safety guarantee.

    ``name`` labels the task so the failure log identifies the source.
    """
    task = (
        coro_or_task
        if isinstance(coro_or_task, asyncio.Task)
        else asyncio.create_task(coro_or_task, name=name or None)
    )
    _BG_TASKS.add(task)
    task.add_done_callback(_reap)
    return task


def in_flight_count() -> int:
    """Diagnostic: how many tracked background tasks are currently
    running. Useful for telemetry / debugging task accumulation."""
    return len(_BG_TASKS)
