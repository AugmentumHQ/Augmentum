"""Handler-facing API for background jobs.

A handler receives one JobContext per run. Everything the handler needs
to cooperate with the runner (report progress, check for cancel, read
its payload, offload blocking work) goes through this object.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")


class JobCancelled(Exception):
    """Raised by ``JobContext.check_cancel`` when the user asked to cancel.

    Handlers should let this propagate. The runner catches it, marks
    the row ``cancelled``, and moves on. Handlers MAY catch it briefly
    to do a fast cleanup, but must re-raise.
    """


class JobRetryable(Exception):
    """Signal a transient failure that should be retried next tick.

    Handlers raise this for errors that are likely to resolve on retry
    (network blip, GPU OOM from transient contention). The runner will
    flip the row back to ``pending`` if ``attempts < max_attempts``,
    otherwise terminate it as ``failed`` like any other exception.
    """


class JobContext:
    """Interface the runner passes to each handler invocation.

    ``payload`` is the handler's input (pre-decoded from JSON by the
    store). ``update_progress`` writes to the DB immediately — callers
    should throttle themselves (every chunk / every N seconds), not
    every loop iteration, to avoid write amplification.
    """

    def __init__(
        self,
        *,
        job_id: str,
        user_id: str,
        job_type: str,
        payload: dict[str, Any],
        store,  # JobsStore — untyped to avoid a circular import
    ) -> None:
        self.job_id = job_id
        self.user_id = user_id
        self.job_type = job_type
        self.payload = payload
        self._store = store

    async def update_progress(self, progress: float, stage: str = "") -> None:
        """Record progress (0.0-1.0) and optionally update the stage label.

        A handler that can't estimate progress cleanly should still call
        this periodically with the same ``progress`` value and a new
        ``stage`` string — the updated_at bump is how the UI knows the
        job is alive vs. stalled.
        """
        await self._store.update_progress(
            self.job_id, progress=progress, stage=stage,
        )

    async def check_cancel(self) -> None:
        """Raise JobCancelled if the user asked to cancel.

        Call between meaningful units of work (per chunk, per chapter,
        per N seconds). Not every tight-loop iteration — each call hits
        the DB.
        """
        if await self._store.is_cancel_requested(self.job_id):
            raise JobCancelled(self.job_id)

    async def run_in_thread(
        self, fn: Callable[..., _T], *args: Any, **kwargs: Any,
    ) -> _T:
        """Run a blocking function on a thread pool so the event loop stays live.

        The job runner shares the FastAPI event loop — a handler that
        blocks (subprocess.run, model.transcribe, time.sleep, a tight
        pure-Python loop, etc.) will freeze chat streaming, WebSocket
        pings, and every other request until it returns. This helper
        wraps ``asyncio.to_thread`` so the common case is a one-liner:

            result = await ctx.run_in_thread(subprocess.run, [...], check=True)
            pcm = await ctx.run_in_thread(moonshine_model.transcribe, audio)

        For long external processes, prefer ``asyncio.create_subprocess_exec``
        directly — it's natively non-blocking and supports streaming stdout.
        Reserve run_in_thread for library calls that don't offer an async
        variant.
        """
        return await asyncio.to_thread(fn, *args, **kwargs)
