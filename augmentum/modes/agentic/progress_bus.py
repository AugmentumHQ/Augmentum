"""Per-task progress bus.

Tools running deep inside an agentic flow (e.g. ``_auto_illustrate``
emitting per-chapter image events) need a way to push live updates back
to the inspector panel without threading a queue reference through every
intermediate function.

A ``ContextVar`` carries the active task's ``progress_callback`` for the
duration of the chain step. The agentic handler binds it before invoking
the chain; tools call :func:`emit_progress` to publish events; the
binding is automatically released when the step finishes.

Events published here ride the same meta envelope every other agentic
chunk uses (``mode: "agentic"``, ``task_id``), so the frontend's
existing meta router picks them up without special plumbing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

# Type alias for clarity. The callback receives a payload dict that gets
# merged into the meta envelope under whatever key the caller chooses
# (e.g. ``ebook_plan``, ``chapter_illustration``).
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]

_active_callback: ContextVar[ProgressCallback | None] = ContextVar(
    "_active_callback", default=None,
)


def bind_callback(callback: ProgressCallback | None):
    """Bind a progress callback for the current async context.

    Returns the ContextVar token so the caller can ``reset()`` it after
    the protected region finishes. Idempotent: passing ``None`` clears.
    """
    return _active_callback.set(callback)


def reset_binding(token) -> None:
    """Release a binding established by :func:`bind_callback`."""
    _active_callback.reset(token)


async def emit_progress(payload: dict[str, Any]) -> None:
    """Publish a progress event from a tool.

    No-op when no callback is bound (e.g. tool invoked outside the
    agentic handler — direct API call, unit test, passthrough mode).
    Tools should always call this rather than checking themselves so
    instrumentation stays uniform.
    """
    cb = _active_callback.get()
    if cb is None:
        return
    try:
        await cb(payload)
    except Exception:
        # Progress events are best-effort. A broken inspector should
        # never break the underlying tool execution.
        pass
