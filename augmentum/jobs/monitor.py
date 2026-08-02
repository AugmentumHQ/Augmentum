"""Job reliability monitor — terminal-state guarantees + stale sweeper.

Augmentum's job runner (``augmentum/jobs/runner.py``) handles the happy
path well: success → ``mark_completed``, retryable failure →
``mark_failed(retryable=True)`` with attempt count, cancel →
``mark_cancelled``, hard failure → ``mark_failed``. What it doesn't do:

  * Guarantee the user **always hears back** on terminal state. Today
    the runner logs and the store row updates — but if the user is
    away from the chat surface when their transcription / media gen /
    dream cycle finishes, there's no notification mechanism.

  * Catch **stalled** jobs. If the worker dies mid-handler (OOM,
    container kill -9, segfault in a native dep), the row stays
    ``running`` until the next ``requeue_crashed`` at start. A
    container that runs for weeks without restart can accumulate
    rows that look running but aren't.

  * Allow **handlers to chain**. A dream cycle that finishes wants to
    enqueue a memory-tier promotion sweep; there's no in-process
    mechanism to do this beyond the handler explicitly enqueueing the
    next job (which means every dependency relationship has to be
    open-coded).

This module adds those three pieces with a single small surface:

  * :class:`JobMonitor` — subscribable terminal-state event emitter +
    stale sweeper loop + follow-up registry.
  * :class:`JobTerminalEvent` — the canonical "this job finished"
    payload that listeners receive.

Build Plan Phase 1.5: reliability contract. The contract is
*the user always hears back* — every job that reaches a terminal state
(success, failure, cancellation, timeout) emits exactly one event.
Downstream notification surfaces (Web Push, browser notification,
status bus, Connect message) subscribe to that event. The monitor
itself does NOT route to any specific notification channel — that
keeps it independent of the channel layer's availability.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Terminal-state event ────────────────────────────────────────────────────


@dataclass(frozen=True)
class JobTerminalEvent:
    """The canonical "this job is done" payload.

    Exactly one of these is emitted per job per terminal state. The
    ``outcome`` field is the discriminator:

    * ``"completed"`` — handler returned normally; ``result`` holds the
      payload the handler returned (None if the handler returned None).
    * ``"failed"`` — handler raised; ``error`` is the message. Note
      this is the TERMINAL failure — retryable failures with attempts
      remaining do NOT produce a terminal event (the job goes back to
      pending and tries again).
    * ``"cancelled"`` — operator or runtime requested cancellation.
    * ``"timed_out"`` — the stale sweeper observed the job exceeded
      its terminal-state-deadline; the row is now marked failed but
      from the monitor's perspective the outcome is "timed_out" to
      distinguish operator failure from runtime failure.
    """

    job_id: str
    user_id: str
    job_type: str
    outcome: str  # "completed" | "failed" | "cancelled" | "timed_out"
    completed_at: int
    result: dict | None = None
    error: str = ""
    payload: dict | None = None


# Listener callable shape. Async so listeners can do real work
# (notification dispatch, follow-up handler invocation, etc.) without
# blocking the monitor loop.
TerminalListener = Callable[[JobTerminalEvent], Awaitable[None]]


# ── Stale-sweep configuration ───────────────────────────────────────────────

# Default deadline for a single job's runtime before the sweeper
# considers it stalled. Handlers that legitimately take longer (large
# media transcode, multi-hour dream cycle) MUST publish progress at
# least this often via the JobContext — the sweeper uses
# ``updated_at`` to detect liveness, not raw start time.
_DEFAULT_RUNTIME_DEADLINE_S = 30 * 60  # 30 minutes
_DEFAULT_SWEEP_INTERVAL_S = 60  # 1 minute


# ── Monitor ─────────────────────────────────────────────────────────────────


class JobMonitor:
    """Owns the reliability contract for the jobs subsystem.

    One instance per app, wired on startup. The runner calls
    :meth:`emit_terminal` whenever a job reaches a terminal state; the
    sweep loop watches for stalled jobs and force-terminates them.

    Listener registration is open-ended — multiple subscribers can
    listen for the same event (a Web Push dispatcher, a status bus
    publisher, a downstream handler chain). Listeners that raise are
    isolated (the failure logs but doesn't stop other listeners or
    the sweep loop).
    """

    def __init__(
        self,
        store,  # JobsStore (untyped to avoid import cycle)
        *,
        runtime_deadline_s: int = _DEFAULT_RUNTIME_DEADLINE_S,
        sweep_interval_s: int = _DEFAULT_SWEEP_INTERVAL_S,
    ) -> None:
        self._store = store
        self._runtime_deadline_s = runtime_deadline_s
        self._sweep_interval_s = sweep_interval_s
        self._listeners: list[TerminalListener] = []
        # Per-job follow-ups: job_id → list of callables fired exactly
        # once when that job emits a terminal event. Pruned after fire.
        self._follow_ups: dict[str, list[TerminalListener]] = {}
        self._task: asyncio.Task | None = None
        self._stopping = False

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin the stale-sweep loop. Idempotent."""
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._sweep_loop(), name="job-monitor")
        log.info(
            "job_monitor_started",
            runtime_deadline_s=self._runtime_deadline_s,
            sweep_interval_s=self._sweep_interval_s,
        )

    async def stop(self) -> None:
        """Shut down the sweep loop cleanly."""
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("job_monitor_stopped")

    # ── Listener registration ────────────────────────────────────────

    def subscribe(self, listener: TerminalListener) -> None:
        """Register a global listener for terminal events.

        Called for every job that terminates regardless of type / user.
        Use for cross-cutting concerns: notification routing, audit
        logging, metrics.
        """
        self._listeners.append(listener)

    def add_follow_up(self, job_id: str, callback: TerminalListener) -> None:
        """Register a one-shot follow-up tied to a specific job.

        Fires exactly once when the named job emits a terminal event,
        then is dropped from the registry. Use for handler chains
        (dream cycle complete → memory promotion).
        """
        self._follow_ups.setdefault(job_id, []).append(callback)

    # ── Terminal event emission ──────────────────────────────────────

    async def emit_terminal(self, event: JobTerminalEvent) -> None:
        """Emit a terminal event. Called by the runner after the store
        row is updated.

        Listener order:
          1. Per-job follow-ups (one-shot, drained from registry).
          2. Global subscribers.

        Each listener is awaited in turn; a raised exception logs and
        is swallowed so it doesn't stop the next listener. This is
        the "user always hears back" guarantee — no listener can
        silently break the chain by raising.
        """
        # Follow-ups fire first because they're the canonical handler-
        # chain signal (dream complete → memory promotion). Drained
        # atomically so a re-entrant follow-up registration doesn't
        # fire twice for the same job.
        followups = self._follow_ups.pop(event.job_id, [])
        for cb in followups:
            await self._invoke_safely(cb, event, kind="follow_up")
        for listener in list(self._listeners):
            await self._invoke_safely(listener, event, kind="subscriber")

    async def _invoke_safely(
        self,
        callback: TerminalListener,
        event: JobTerminalEvent,
        *,
        kind: str,
    ) -> None:
        """Run a single listener; swallow + log any exception."""
        try:
            await callback(event)
        except Exception:
            log.warning(
                "job_terminal_listener_failed",
                kind=kind,
                job_id=event.job_id,
                job_type=event.job_type,
                outcome=event.outcome,
                exc_info=True,
            )

    # ── Stale sweeper ────────────────────────────────────────────────

    async def _sweep_loop(self) -> None:
        """Force-terminate jobs whose handler has been running past the
        runtime deadline.

        The runner shares the FastAPI event loop, so a single hung
        handler blocks every other job. The sweeper can't rescue the
        current handler in-process (asyncio doesn't support
        cooperative timeout of unrelated tasks without help), but it
        CAN ensure the DB row reflects reality — so the next runner
        start doesn't re-process a row that's actually dead.
        """
        while not self._stopping:
            try:
                await asyncio.sleep(self._sweep_interval_s)
                if self._stopping:
                    return
                await self._sweep_once()
            except asyncio.CancelledError:
                return
            except Exception:
                log.warning("job_monitor_sweep_error", exc_info=True)

    async def _sweep_once(self) -> None:
        """One pass of the stale-sweep. Public for tests."""
        try:
            stalled = await self._find_stalled_jobs()
        except Exception:
            log.warning("job_monitor_stalled_query_failed", exc_info=True)
            return
        for row in stalled:
            await self._force_terminate(row)

    async def _find_stalled_jobs(self) -> list[dict[str, Any]]:
        """Return rows whose ``status='running'`` and ``updated_at`` is
        older than the runtime deadline.

        Uses a raw SELECT through the store's underlying connection so
        we don't have to add a new store API for this single use case.
        """
        cutoff = int(time.time()) - self._runtime_deadline_s
        conn = getattr(self._store, "_conn", None)
        if conn is None:
            return []
        cur = await conn.execute(
            """SELECT id, user_id, job_type, payload, updated_at
                 FROM background_jobs
                WHERE status = 'running'
                  AND updated_at < ?""",
            (cutoff,),
        )
        rows = await cur.fetchall()
        await cur.close()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append({
                "id": row[0],
                "user_id": row[1],
                "job_type": row[2],
                "payload": row[3],
                "updated_at": row[4],
            })
        return out

    async def _force_terminate(self, row: dict[str, Any]) -> None:
        """Mark a stalled row as failed and emit the terminal event.

        The 'timed_out' outcome on the event distinguishes runtime
        timeout from handler-raised failure so notification surfaces
        can word the user-facing message appropriately ("the dream
        cycle timed out, want me to retry?" vs. "the dream cycle
        failed: <error>").
        """
        import json as _json
        log.warning(
            "job_force_terminated_stale",
            job_id=row["id"],
            job_type=row["job_type"],
            user_id=row["user_id"],
            updated_at=row["updated_at"],
        )
        await self._store.mark_failed(
            row["id"],
            error=(
                f"Job runtime exceeded {self._runtime_deadline_s}s "
                "without progress update — force-terminated by the "
                "reliability monitor. Re-enqueue with a fresh job if "
                "you still want this work done."
            ),
        )
        payload: dict | None = None
        try:
            if row.get("payload"):
                payload = _json.loads(row["payload"])
        except Exception:
            payload = None
        event = JobTerminalEvent(
            job_id=row["id"],
            user_id=row["user_id"],
            job_type=row["job_type"],
            outcome="timed_out",
            completed_at=int(time.time()),
            error="runtime deadline exceeded",
            payload=payload,
        )
        await self.emit_terminal(event)
