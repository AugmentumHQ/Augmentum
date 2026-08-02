"""Background-job worker loop + handler registry.

JobRunner is a single-worker polling loop. It pulls the next pending job
from the store, looks up the handler for its ``job_type``, and runs it
with a JobContext. Handlers report progress through the context and
cooperate with cancellation via ``check_cancel``.

Concurrency: one job at a time by default. This matches the dominant
consumer profile (CPU/GPU-bound, e.g. transcription, media transcode).
Future consumers whose work is I/O-bound could justify an asyncio
semaphore slotting scheme, but we don't need it yet and shouldn't build
for it speculatively.

Idempotency: handlers MUST be idempotent. On server restart any job
caught mid-run is re-queued (see JobsStore.requeue_crashed). A handler
that writes partial state without being able to resume it will produce
duplicate work or corrupted output. If a handler can't meet this bar,
it should persist its own checkpoint in the payload and read it on
re-entry.

Non-blocking contract: the runner shares the FastAPI event loop. A
handler that blocks the loop (synchronous subprocess, pure-Python CPU
burn, blocking I/O) will freeze every other request — chat streams,
WebSocket pings, even health checks. Handlers MUST either be natively
async or delegate blocking work via ``ctx.run_in_thread`` (wraps
asyncio.to_thread) or ``asyncio.create_subprocess_exec``. The runner
itself is verified non-blocking by ``tests/test_jobs_responsiveness.py``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from augmentum.jobs.context import JobCancelled, JobContext, JobRetryable
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.jobs.monitor import JobMonitor

log = get_logger(__name__)

# A handler takes a context and returns an optional dict to store as
# ``result``. Handlers that don't want to return anything return None.
JobHandler = Callable[[JobContext], Awaitable[dict | None]]

# Module-level registry. Handlers register on import; the runner reads
# this map at dispatch time so hot-reload during dev doesn't require a
# runner restart.
_HANDLERS: dict[str, JobHandler] = {}


def register_handler(job_type: str, handler: JobHandler) -> None:
    """Register a handler for a job_type. Replaces any prior registration.

    Replacement (rather than error-on-duplicate) is intentional: dev
    hot-reload re-imports the module and re-registers the handler. In
    production nothing re-imports, so the semantics are the same.
    """
    if not job_type:
        raise ValueError("register_handler requires a non-empty job_type")
    _HANDLERS[job_type] = handler
    log.debug("job_handler_registered", job_type=job_type)


def get_handler(job_type: str) -> JobHandler | None:
    return _HANDLERS.get(job_type)


class JobRunner:
    """Owns the worker task. One instance per app.

    Poll interval is short because the store is local SQLite — no network
    round-trip. Jobs that need real-time dispatch can bypass the poll by
    calling ``wake()`` after enqueueing (cheap: sets an event).
    """

    # Idle poll cadence. The runner is event-driven via ``_wakeup`` —
    # ``enqueue`` wakes it immediately, so this poll is purely a safety net
    # for the rare case the wake event is missed. Was 2s; bumping to 30s
    # cuts the ~30 idle DB hits/minute (each a SELECT + UPDATE on the
    # shared aiosqlite worker thread that competed with auth/state/chat
    # reads) without affecting how fast a real enqueue is picked up.
    _POLL_INTERVAL_S: float = 30.0
    _IDLE_BACKOFF_S: float = 5.0

    def __init__(
        self,
        store,  # store: JobsStore (untyped for import cycle)
        *,
        monitor: JobMonitor | None = None,
    ) -> None:
        self._store = store
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._wakeup = asyncio.Event()
        self._current_job_id: str | None = None
        # Reliability monitor — emits per-job terminal events so
        # downstream surfaces (notifications, follow-up handlers) can
        # react. Optional so older tests / call sites that construct a
        # bare runner still work. See augmentum/jobs/monitor.py.
        self._monitor = monitor

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the worker loop. Idempotent — safe to call twice."""
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._loop(), name="job-runner")
        self._task.add_done_callback(self._on_task_done)
        log.info("job_runner_started")

    async def stop(self) -> None:
        """Request graceful stop. The in-flight job (if any) finishes."""
        self._stopping = True
        self._wakeup.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30.0)
            except TimeoutError:
                log.warning("job_runner_stop_timeout")
                self._task.cancel()

    def wake(self) -> None:
        """Nudge the loop out of its poll sleep. Call after enqueueing."""
        self._wakeup.set()

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Restart the loop if it died unexpectedly (not during stop)."""
        if self._stopping:
            return
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("job_runner_crashed_restarting", error=str(exc))
            # Schedule a restart on the next event loop iteration.
            asyncio.get_event_loop().call_soon(self.start)

    # ── Main loop ─────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                job = await self._store.claim_next_pending()
            except Exception as exc:
                log.warning("job_claim_failed", error=str(exc))
                await self._sleep(self._IDLE_BACKOFF_S)
                continue

            if job is None:
                # Queue empty — sleep until woken or timeout.
                await self._sleep(self._POLL_INTERVAL_S)
                continue

            await self._run_one(job)

    async def _sleep(self, timeout: float) -> None:
        """Sleep up to ``timeout`` seconds, interruptible by ``wake()``."""
        try:
            await asyncio.wait_for(self._wakeup.wait(), timeout=timeout)
        except TimeoutError:
            pass
        finally:
            self._wakeup.clear()

    async def _run_one(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        job_type = job["job_type"]
        self._current_job_id = job_id

        handler = get_handler(job_type)
        if handler is None:
            # Unknown type — fail terminally (retrying won't help).
            log.warning(
                "job_unknown_type", job_id=job_id, job_type=job_type,
            )
            await self._store.mark_failed(
                job_id,
                error=f"No handler registered for job_type '{job_type}'",
            )
            self._current_job_id = None
            return

        ctx = JobContext(
            job_id=job_id,
            user_id=job["user_id"],
            job_type=job_type,
            payload=job.get("payload") or {},
            store=self._store,
        )

        log.info(
            "job_started",
            job_id=job_id,
            job_type=job_type,
            user_id=job["user_id"],
            attempt=job.get("attempts"),
        )

        # Snapshot before handler runs so the terminal event carries the
        # original payload regardless of any handler-side mutations.
        original_payload = dict(ctx.payload) if ctx.payload else None
        try:
            result = await handler(ctx)
            await self._store.mark_completed(job_id, result=result)
            log.info("job_completed", job_id=job_id, job_type=job_type)
            await self._emit_terminal(
                job_id, job["user_id"], job_type,
                outcome="completed",
                result=result,
                payload=original_payload,
            )
        except JobCancelled:
            await self._store.mark_cancelled(job_id)
            log.info("job_cancelled", job_id=job_id, job_type=job_type)
            await self._emit_terminal(
                job_id, job["user_id"], job_type,
                outcome="cancelled",
                payload=original_payload,
            )
        except JobRetryable as exc:
            # Retryable failure with attempts remaining does NOT emit a
            # terminal event — the job goes back to pending. Only the
            # final-attempt failure (which lands in mark_failed without
            # retryable=True) is terminal. To keep the contract clean,
            # we let the next attempt run; if it terminates we emit then.
            await self._store.mark_failed(
                job_id, error=str(exc), retryable=True,
            )
            log.info(
                "job_retryable_failure",
                job_id=job_id,
                job_type=job_type,
                error=str(exc),
            )
        except Exception as exc:
            await self._store.mark_failed(job_id, error=str(exc))
            log.warning(
                "job_failed",
                job_id=job_id,
                job_type=job_type,
                error=str(exc),
                exc_info=True,
            )
            await self._emit_terminal(
                job_id, job["user_id"], job_type,
                outcome="failed",
                error=str(exc),
                payload=original_payload,
            )
        finally:
            self._current_job_id = None

    async def _emit_terminal(
        self,
        job_id: str,
        user_id: str,
        job_type: str,
        *,
        outcome: str,
        result: dict | None = None,
        error: str = "",
        payload: dict | None = None,
    ) -> None:
        """Emit a terminal event via the monitor if one is wired.

        Build Plan Phase 1.5 reliability contract: the runner is the
        single emitter of post-success / post-failure / post-cancel
        events. Monitor listeners route to notification surfaces and
        follow-up handlers. Failure to emit (no monitor wired) is not
        an error — older deployments may run without a monitor and
        still work fine.
        """
        if self._monitor is None:
            return
        from augmentum.jobs.monitor import JobTerminalEvent
        event = JobTerminalEvent(
            job_id=job_id,
            user_id=user_id,
            job_type=job_type,
            outcome=outcome,
            completed_at=int(time.time()),
            result=result,
            error=error,
            payload=payload,
        )
        try:
            await self._monitor.emit_terminal(event)
        except Exception:
            # Defensive — monitor listeners are individually isolated,
            # but the monitor itself raising is non-fatal for the
            # runner. Log loudly so the gap surfaces.
            log.warning(
                "job_terminal_emit_failed",
                job_id=job_id,
                job_type=job_type,
                outcome=outcome,
                exc_info=True,
            )
