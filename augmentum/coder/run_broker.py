"""In-process broker that decouples a Coder agent loop from the HTTP
request that started it.

The agent loop runs as a detached ``asyncio.Task`` that pushes
``InternalStreamChunk`` objects into a per-run ring buffer. The HTTP
stream becomes a *subscription* to that buffer — client disconnect
(mobile screen sleep, tab switch, fetch abort) just ends the
subscription; the agent keeps running and writing to the persistent
ledger.

Reattach contract:

- Live broker entry exists → subscribe to its stream from ``since_seq``.
  Recent chunks come from the ring buffer; new ones from a wakeup
  ``asyncio.Event``.
- Broker entry gone (run completed or evicted) → route falls back to
  the structured ledger (``coder_turn_events``) plus the final
  assistant message from ``coder_sessions.conversation``.
- Run finished with an error (agent exception / cancellation) → the
  subscription ends with a synthetic terminal chunk (``done=True``,
  augmentum ``status`` of ``error``/``cancelled`` plus the error
  text) so the client is never left with a silently-truncated stream.

Server restart kills all in-flight runs by definition (the tasks are
in-memory). A startup sweep marks any ``coder_turn_runs`` row stuck in
``status='running'`` as ``status='cancelled'`` so the UI shows
``interrupted`` instead of spinning forever.

Cancellation: ``cancel(run_id)`` sets a flag *and* cancels the
detached task. The handler's existing ``CancelledError`` cleanup
(``cancel_workspace_execs``) still fires inside the task so in-flight
shell execs get SIGTERM. From the request-lifecycle point of view, the
client closing the fetch is no longer a cancel signal — the UI must
call ``POST /api/coder/runs/{id}/cancel`` to actually stop the run.
"""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import InternalStreamChunk

log = get_logger(__name__)


class WorkspaceBusyError(RuntimeError):
    """Raised by :meth:`CoderRunBroker.start_run` when another live run
    already owns the same ``(user_id, workspace_id)``.

    Exists to close the check-then-start TOCTOU window: callers used to
    poll ``get_active_for_workspace`` and then ``start_run`` several
    awaits later, so two agents could end up mutating one workspace.
    The atomic check now lives inside ``start_run`` under the broker
    lock; this exception is the reject signal. Callers with a retry
    loop (the background-job handler) catch it and requeue; the
    interactive path lets it surface as a turn error ("workspace busy").
    """

    def __init__(self, *, workspace_id: str, holder_run_id: str) -> None:
        super().__init__(
            f"workspace {workspace_id} is owned by live run "
            f"{holder_run_id}",
        )
        self.workspace_id = workspace_id
        self.holder_run_id = holder_run_id


def _ownership_ok(entry: RunBrokerEntry, expected_user_id: str | None) -> bool:
    """Tenant guard for broker mutators (defense-in-depth).

    ``expected_user_id=None`` skips the check (trusted in-process
    callers: the handler, rewind, shutdown). When provided and it
    doesn't match the entry's owner, the mutator refuses via its normal
    failure return value — the routes already 404 unowned runs via the
    ledger lookup, so a mismatch here means a bug or a forgotten gate,
    and we log it loudly rather than mutate another tenant's run.
    Isolation is a ground rule, not a tunable.
    """
    if expected_user_id is None or entry.user_id == expected_user_id:
        return True
    log.warning(
        "coder_run_broker_ownership_mismatch",
        run_id=entry.run_id,
        owner_user_id=entry.user_id,
        expected_user_id=expected_user_id,
    )
    return False


# Ring buffer cap. A typical coder turn produces a few hundred chunks
# (one per content delta + per tool event + per phase change). 2000
# covers most multi-minute turns without unbounded memory growth on
# pathological loops. When the cap is exceeded, oldest entries are
# evicted and any subscriber reading from an evicted seq gets a single
# ``buffer_overflow`` marker so the UI can fall back to "see the
# ledger" rather than silently losing prose.
_DEFAULT_BUFFER_CAP = 2000

# How long a finished entry hangs around in the registry before
# ``_sweep_stale`` removes it. Long enough for a reconnecting mobile
# client (screen-sleep, walk-into-elevator) to find the entry and
# replay the tail; short enough that long-completed runs don't pin
# memory. The ledger is the durable record either way.
_RETAIN_FINISHED_SECONDS = 600.0

# Soft inactivity ceiling on an active run. If the agent stops
# emitting chunks for this long *and* has no subscribers, the broker
# logs a warning. We don't auto-kill — coder turns can legitimately
# block on long-running shell execs (apt-get install, cargo build) —
# but the warning surfaces the case for operators.
_ACTIVE_QUIET_WARN_SECONDS = 900.0

# Cap on per-run cooperative inbox depth. A wedged or paused agent
# could otherwise accumulate hundreds of typed messages from an
# impatient user; we reject beyond this with a clear error so the
# UI can prompt the user to either resume or cancel. Picked to
# comfortably exceed any realistic "burst of clarifications" while
# bounding memory growth on the pathologic case.
_INBOX_CAP = 25


def _make_set_event() -> asyncio.Event:
    """Construct an asyncio.Event that starts in the SET state.

    Used as the pause gate's default: a handler that never observes a
    pause never blocks on this event. Pause flips the event to clear;
    resume sets it again to release the awaiter.
    """
    evt = asyncio.Event()
    evt.set()
    return evt


@dataclass(slots=True)
class _BufferedChunk:
    seq: int
    timestamp: float
    chunk: InternalStreamChunk


@dataclass(slots=True)
class RunBrokerEntry:
    """Per-run state held by the broker."""

    run_id: str
    user_id: str
    workspace_id: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    task: asyncio.Task | None = None
    # Default cap is for entries constructed directly (tests, tooling);
    # the broker always passes its own ``deque(maxlen=self._buffer_cap)``
    # through the constructor in start_run.
    buffer: deque[_BufferedChunk] = field(default_factory=lambda: deque(maxlen=_DEFAULT_BUFFER_CAP))
    seq: int = 0
    new_chunk_event: asyncio.Event = field(default_factory=asyncio.Event)
    done: bool = False
    cancel_requested: bool = False
    # Free-text hint for *why* the run was cancelled. THE canonical
    # vocabulary (cancel()'s docstring defers here — keep in sync
    # with the handler's _resolve_cancel_reason renderer):
    #   user_cancel (Esc / Cancel button) · slash_clear ·
    #   slash_compact · new_turn_started · page_unload ·
    #   server_shutdown (broker.shutdown) · paused_timeout
    #   (sweep_paused_timeouts) · user_rewind (/rewind route) ·
    #   background_timeout / background_job_error (headless
    #   background-mission job handler).
    # The handler reads it in its CancelledError path so the next
    # turn's <prior_turns> block can tell the model the turn ended
    # because (e.g.) the user pressed Stop — not because the model
    # itself decided to stop. Defaults to an empty string until
    # cancel() is called; the handler maps empty → "user_cancel" so
    # legacy callsites still produce a readable summary.
    cancel_reason: str = ""
    error: str = ""
    subscriber_count: int = 0
    last_activity_at: float = field(default_factory=time.time)
    # Live :class:`~augmentum.coder.turn_snapshot.TurnSnapshot` for this
    # run. Attached by the handler at turn start so the /rewind route
    # can restore files mid-flight or just-after without reaching into
    # the (detached, possibly already gone) handler instance. Typed as
    # ``object`` to avoid a runtime import cycle — the broker never
    # touches snapshot internals, just hands the reference back to the
    # rewind module. The reference outlives the run (held until the
    # entry is evicted, ~600s post-finish), so rewind works for both
    # in-flight and just-completed turns. After eviction, rewind falls
    # back to the snapshot stored on the matching ReviewBundle in
    # app.state.review_registry.
    turn_snapshot: object | None = None

    # ── Cooperative interjection inbox ───────────────────────────────
    # FIFO of user messages submitted via POST /api/coder/runs/{id}/
    # interject while the run is in flight. The handler drains entries
    # at two distinct points depending on their ``mode``:
    #
    #   * ``mode="steer"`` — drained at the NEXT iteration boundary in
    #     the agent loop. Used when the user wants to redirect what the
    #     agent is doing right now. Trade-off: feels responsive when
    #     iterations are quick, but a long-running shell exec can delay
    #     the inject by 30s+ since we don't interrupt tool calls
    #     mid-flight. If no boundary ever comes (the user steered while
    #     the model was writing its final response), the entry is NOT
    #     stranded: the end-of-turn drain promotes it into the
    #     queue-followup chain so it becomes the next turn's prompt.
    #   * ``mode="queue"`` — drained at end-of-turn (after the natural
    #     loop exit). Becomes the user content of a brand-new turn
    #     (fresh run_id). Used for "I have a follow-up I want you to do
    #     after this finishes". This is the SAFE default — landed as
    #     the #1 documented complaint about CC's queue is that the
    #     end-of-pause drain derails mid-task work.
    #
    # The third mode in the UI vocabulary — ``interrupt`` — does NOT
    # land here. It cancels via the existing ``/cancel`` route and
    # starts a new turn explicitly, the same path a fresh user message
    # has always taken.
    #
    # Entry schema:
    #   {"id": str, "content": str, "attachments": list[dict] | None,
    #    "mode": "queue" | "steer", "queued_at": float,
    #    "delivered_at": float | None}
    #
    # Bounded at _INBOX_CAP entries; overflow rejects with an error so
    # a paused/wedged agent doesn't accumulate unbounded backlog. The
    # handler is responsible for marking ``delivered_at`` on each
    # entry it consumes — leaves a trail for "your message was queued
    # at T, delivered at T+12s" UX surfacing.
    pending_user_messages: list[dict] = field(default_factory=list)

    # ── Cooperative pause/resume ──────────────────────────────────────
    # Pause: handler observes ``paused`` at iteration boundary and
    # awaits ``pause_event`` until cleared. Resume sets the event +
    # clears the flag. Queued/steered messages remain in the inbox
    # while paused; they drain on the next iteration after resume.
    # Implementation detail: the event STARTS set so a handler that
    # never sees a pause never blocks.
    paused: bool = False
    pause_event: asyncio.Event = field(default_factory=lambda: _make_set_event())
    # Wall-clock timestamp when ``paused`` last flipped True. Cleared
    # to 0.0 on resume. The paused-timeout sweep reads this to
    # auto-cancel runs that have been paused longer than
    # ``coder_max_paused_seconds`` — prevents a forgotten pause from
    # holding container resources + broker memory indefinitely.
    paused_at: float = 0.0
    # Latch for the sweeper's quiet-active warning: set when the
    # warning fires so it logs ONCE per quiet spell instead of every
    # 60s sweep; reset by _push when the run produces output again
    # (a new quiet spell warns anew).
    quiet_warned: bool = False

    def _push(self, chunk: InternalStreamChunk) -> _BufferedChunk:
        self.seq += 1
        entry = _BufferedChunk(seq=self.seq, timestamp=time.time(), chunk=chunk)
        self.buffer.append(entry)
        self.last_activity_at = entry.timestamp
        self.quiet_warned = False
        # Pulse the event so all subscribers wake up. Wakeup pattern:
        # set → clear at the start of each subscriber's wait cycle so
        # an event set during processing doesn't get lost.
        self.new_chunk_event.set()
        return entry

    def _finish(self, error: str = "") -> None:
        self.done = True
        self.finished_at = time.time()
        self.error = error
        self.new_chunk_event.set()

    def buffer_min_seq(self) -> int:
        if not self.buffer:
            return 0
        return self.buffer[0].seq


class CoderRunBroker:
    """Process-global registry of detached coder-mode agent runs."""

    def __init__(self, buffer_cap: int = _DEFAULT_BUFFER_CAP) -> None:
        self._runs: dict[str, RunBrokerEntry] = {}
        self._lock = asyncio.Lock()
        self._buffer_cap = buffer_cap
        self._sweeper_task: asyncio.Task | None = None
        self._shutdown = False

    # ------------------------------------------------------------------
    # Lifecycle hooks (start_sweeper / shutdown)
    # ------------------------------------------------------------------

    def start_sweeper(self) -> None:
        """Spawn the periodic stale-entry sweeper.

        Idempotent — safe to call multiple times. Skipped at module
        import time because asyncio.create_task requires a running
        loop; the proxy startup hook is the right call site.
        """
        if self._sweeper_task is not None and not self._sweeper_task.done():
            return
        try:
            self._sweeper_task = asyncio.create_task(
                self._sweep_loop(), name="coder-run-broker-sweeper",
            )
        except RuntimeError:
            # No running loop yet (import time / sync test setup).
            # start_run() re-invokes this method — it always runs
            # inside a loop, so the sweeper self-heals on the first
            # real run even if the boot wiring never called us.
            self._sweeper_task = None

    # How long shutdown() waits for cancelled run tasks to finish
    # unwinding. Covers the handler's CancelledError/finally cleanup
    # (interruption summary, ledger finish, SIGTERM to workspace
    # execs) — generous, because cancel_workspace_execs does real
    # Docker round-trips; bounded, so a wedged cleanup can't hold the
    # whole process-exit hostage.
    _SHUTDOWN_DRAIN_TIMEOUT_S = 15.0

    async def shutdown(self) -> None:
        """Cancel all in-flight runs and WAIT (bounded) for their tasks
        to finish unwinding. Called on app shutdown so we don't leak
        tasks past the process exit.

        Contract: cancelling without awaiting would let the event loop
        close while the handlers' CancelledError cleanup (SIGTERM to
        in-flight workspace execs, ledger finish) is still pending —
        that cleanup would silently never run. We gather the tasks with
        ``return_exceptions=True`` (they end in CancelledError by
        design) under a ``wait_for`` ceiling so a wedged cleanup can't
        block exit forever.

        The startup sweep marks any row left in ``status='running'`` as
        cancelled on next boot — this method is the in-process leg of
        the same defense.
        """
        self._shutdown = True
        async with self._lock:
            entries = list(self._runs.values())
        pending: list[asyncio.Task] = []
        for entry in entries:
            entry.cancel_requested = True
            # Tag the reason so the handler's interruption-summary
            # path can render "server was shutting down" in the next
            # turn's prior_turns block — distinct from a user-driven
            # cancel.
            if not entry.cancel_reason:
                entry.cancel_reason = "server_shutdown"
            # Release a paused run's gate so the cancellation can
            # actually reach the handler's awaiter.
            entry.pause_event.set()
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()
                pending.append(entry.task)
        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
            pending.append(self._sweeper_task)
        if not pending:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=self._SHUTDOWN_DRAIN_TIMEOUT_S,
            )
        except TimeoutError:
            still = sum(1 for t in pending if not t.done())
            log.warning(
                "coder_run_broker_shutdown_drain_timeout",
                pending_tasks=still,
                timeout_s=self._SHUTDOWN_DRAIN_TIMEOUT_S,
            )

    # ------------------------------------------------------------------
    # Run dispatch
    # ------------------------------------------------------------------

    async def start_run(
        self,
        *,
        run_id: str,
        user_id: str,
        workspace_id: str,
        agent: Callable[
            [RunBrokerEntry], AsyncIterator[InternalStreamChunk],
        ] | Callable[
            [RunBrokerEntry], Awaitable[AsyncIterator[InternalStreamChunk]],
        ],
        exclusive_workspace: bool = True,
    ) -> RunBrokerEntry:
        """Register a new run and spawn the detached task.

        ``agent`` is a callable taking the broker entry and returning
        an async iterator of chunks. The broker pumps the iterator and
        pushes each chunk into the buffer. Passing a callable (instead
        of an already-iterating generator) lets us hold off on the
        first ``await`` until the task runs — important because some
        coder setup (state restore, ledger row) belongs inside the
        detached scope, not the request scope that called start_run.

        ``exclusive_workspace`` (default True) enforces one live run
        per ``(user_id, workspace_id)`` ATOMICALLY, under the broker
        lock — the authoritative fix for the poll-then-start TOCTOU
        where two agents could end up mutating one workspace. Raises
        :class:`WorkspaceBusyError` when a live holder exists. A run
        with ``cancel_requested`` set is exempt: the interactive
        cancel→new-turn flow (reason ``new_turn_started``) starts the
        next turn while the old task is still unwinding, and that
        overlap-with-teardown has always been the accepted contract.
        """
        if self._shutdown:
            raise RuntimeError("Broker is shutting down")
        entry = RunBrokerEntry(
            run_id=run_id,
            user_id=user_id,
            workspace_id=workspace_id,
            buffer=deque(maxlen=self._buffer_cap),
        )
        # Self-healing sweeper: idempotent, and start_run always runs
        # inside a loop — finished entries can't leak just because the
        # boot wiring forgot (or failed) to call start_sweeper().
        self.start_sweeper()
        async with self._lock:
            if run_id in self._runs:
                raise ValueError(f"Run {run_id} already registered")
            if exclusive_workspace:
                for other in self._runs.values():
                    if (
                        other.user_id == user_id
                        and other.workspace_id == workspace_id
                        and not other.done
                        and not other.cancel_requested
                    ):
                        raise WorkspaceBusyError(
                            workspace_id=workspace_id,
                            holder_run_id=other.run_id,
                        )
            self._runs[run_id] = entry
        entry.task = asyncio.create_task(
            self._pump(entry, agent), name=f"coder-run-{run_id}",
        )
        return entry

    async def _pump(
        self,
        entry: RunBrokerEntry,
        agent: Callable[
            [RunBrokerEntry], AsyncIterator[InternalStreamChunk],
        ] | Callable[
            [RunBrokerEntry], Awaitable[AsyncIterator[InternalStreamChunk]],
        ],
    ) -> None:
        """Drive the agent iterator and feed chunks into the buffer.

        Exceptions are caught and recorded on the entry rather than
        propagating — a detached task that raises just logs and dies,
        with no observer to see the traceback. Subscribers see ``done``
        + ``error`` and surface it in the UI.
        """
        try:
            result = agent(entry)
            # isawaitable, not iscoroutine: the declared contract is
            # Awaitable[AsyncIterator], which includes Futures and
            # custom __await__ objects — iscoroutine silently treated
            # those as the iterator itself.
            if inspect.isawaitable(result):
                # Caller returned an awaitable that resolves to an iterator
                # (e.g. an async generator wrapped in an async def setup).
                iterator = await result
            else:
                iterator = result
            async for chunk in iterator:
                if entry.cancel_requested:
                    # Honor an explicit cancel between chunks — gives
                    # the agent a chance to finish its current
                    # operation cleanly before we tear down. The
                    # exception is delivered INTO the generator's
                    # frame (see _throw_cancel_into); always raises.
                    await self._throw_cancel_into(entry, iterator)
                entry._push(chunk)
        except asyncio.CancelledError:
            log.info(
                "coder_run_broker_cancelled",
                run_id=entry.run_id,
                workspace_id=entry.workspace_id,
            )
            entry._finish(error="cancelled" if entry.cancel_requested else "")
            raise
        except Exception as exc:
            log.warning(
                "coder_run_broker_agent_error",
                run_id=entry.run_id,
                workspace_id=entry.workspace_id,
                error=str(exc),
                exc_info=True,
            )
            entry._finish(error=str(exc))
        else:
            entry._finish()

    @staticmethod
    async def _throw_cancel_into(
        entry: RunBrokerEntry,
        iterator: AsyncIterator[InternalStreamChunk],
    ) -> None:
        """Deliver a cooperative cancel INTO the agent generator's frame.

        Contract: this method never returns normally — it always ends
        by raising ``CancelledError`` (or whatever the generator turned
        it into) so ``_pump``'s cancel path runs.

        Raising ``CancelledError`` in the pump's own frame (the old
        code) left the agent generator SUSPENDED at its yield; it was
        only closed later, by GC, with ``GeneratorExit`` — which
        bypasses the handler's ``except asyncio.CancelledError`` block
        in ``_run_agent_with_ledger`` (interruption summary, ledger
        status 'cancelled' vs 'error', queue_dropped marker) and makes
        its unwind-time yields illegal. ``athrow`` runs that cleanup
        now, deterministically, in this task. The generator is allowed
        to yield trailing marker chunks while unwinding (the handler
        yields ``queue_dropped`` before re-raising); we push those into
        the buffer so subscribers still see them.
        """
        athrow = getattr(iterator, "athrow", None)
        if athrow is None:
            # Plain async iterator — no generator frame to unwind; the
            # pump-frame raise is all there is.
            raise asyncio.CancelledError()
        try:
            chunk = await athrow(asyncio.CancelledError())
            while True:
                entry._push(chunk)
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            # Generator swallowed the cancel and finished — still a
            # cancelled run from the broker's point of view.
            pass
        raise asyncio.CancelledError()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> RunBrokerEntry | None:
        return self._runs.get(run_id)

    def attach_snapshot(self, run_id: str, snapshot: object) -> bool:
        """Bind a TurnSnapshot to the entry for ``run_id``.

        Called by the handler immediately after :meth:`start_run` so the
        rewind route can reach the snapshot without the handler instance.
        Idempotent: re-attaching the same snapshot is a no-op; attaching
        a different one replaces it (handler restart after compaction is
        the realistic case). Returns ``True`` when bound, ``False`` when
        the run is unknown or already evicted.
        """
        entry = self._runs.get(run_id)
        if entry is None:
            return False
        entry.turn_snapshot = snapshot
        return True

    def enqueue_user_message(
        self,
        run_id: str,
        *,
        content: str,
        attachments: list[dict] | None = None,
        mode: str = "queue",
        expected_user_id: str | None = None,
    ) -> dict | None:
        """Append a cooperative interjection to the inbox.

        Returns the inserted entry on success (so the route can echo
        the id back to the client), or ``None`` when the run is
        unknown, evicted, finished, the inbox is at capacity, or
        ``expected_user_id`` is provided and isn't the run's owner
        (see ``_ownership_ok``).

        Mode must be one of ``queue`` (drains at end-of-turn, becomes
        a new turn) or ``steer`` (drains at the next iteration
        boundary within the current turn). Unknown modes fall back to
        ``queue`` — the safe default. The ``interrupt`` mode in the
        UI vocabulary uses the existing ``/cancel`` route and does
        not pass through here.
        """
        entry = self._runs.get(run_id)
        if entry is None or entry.done:
            return None
        if not _ownership_ok(entry, expected_user_id):
            return None
        mode = (mode or "queue").strip().lower()
        if mode not in ("queue", "steer"):
            mode = "queue"
        if len(entry.pending_user_messages) >= _INBOX_CAP:
            log.warning(
                "coder.inbox_capacity_reached",
                run_id=run_id,
                cap=_INBOX_CAP,
                mode=mode,
            )
            return None
        msg = {
            "id": uuid.uuid4().hex[:12],
            "content": (content or ""),
            "attachments": list(attachments or []),
            "mode": mode,
            "queued_at": time.time(),
            "delivered_at": None,
        }
        entry.pending_user_messages.append(msg)
        log.info(
            "coder.inbox_enqueued",
            run_id=run_id,
            mode=mode,
            msg_id=msg["id"],
            queue_depth=len(entry.pending_user_messages),
            attachment_count=len(msg["attachments"]),
        )
        return dict(msg)

    def drain_user_messages(
        self,
        run_id: str,
        *,
        mode: str | None = None,
        expected_user_id: str | None = None,
    ) -> list[dict]:
        """Pop and return pending inbox entries for ``run_id``.

        ``expected_user_id`` (optional) refuses with ``[]`` — without
        draining — when it doesn't match the run's owner.

        When ``mode`` is set, only entries matching that mode are
        drained (the handler calls this with ``"steer"`` at iteration
        boundaries and with ``"queue"`` at end-of-turn). When
        ``mode`` is None, drains everything in FIFO order — used by
        the rewind/cancel paths to flush the inbox.

        Each drained entry is stamped with ``delivered_at`` so the UI
        can surface delivery latency on the original message bubble.
        """
        entry = self._runs.get(run_id)
        if entry is None or not entry.pending_user_messages:
            return []
        if not _ownership_ok(entry, expected_user_id):
            return []
        now = time.time()
        kept: list[dict] = []
        drained: list[dict] = []
        for msg in entry.pending_user_messages:
            if mode is not None and msg.get("mode") != mode:
                kept.append(msg)
                continue
            msg["delivered_at"] = now
            drained.append(msg)
        entry.pending_user_messages = kept
        if drained:
            log.info(
                "coder.inbox_drained",
                run_id=run_id,
                drained=len(drained),
                remaining=len(kept),
                filter_mode=mode or "*",
            )
        return drained

    def inbox_depth(self, run_id: str, *, mode: str | None = None) -> int:
        """How many entries are pending — optionally filtered by mode."""
        entry = self._runs.get(run_id)
        if entry is None:
            return 0
        if mode is None:
            return len(entry.pending_user_messages)
        return sum(1 for m in entry.pending_user_messages if m.get("mode") == mode)

    def pause(
        self, run_id: str, *, expected_user_id: str | None = None,
    ) -> bool:
        """Flag the run as paused — the handler awaits at the next
        iteration boundary until :meth:`resume` is called.

        Returns True iff the run existed, was active, and transitioned
        to paused (idempotent re-pause returns False so the caller can
        detect no-op; so does an ``expected_user_id`` owner mismatch).
        Pause does NOT cancel in-flight tool execution; the agent will
        finish whatever it's currently running, then block at the next
        boundary check.
        """
        entry = self._runs.get(run_id)
        if entry is None or entry.done:
            return False
        if not _ownership_ok(entry, expected_user_id):
            return False
        if entry.paused:
            return False
        entry.paused = True
        entry.pause_event.clear()
        entry.paused_at = time.time()
        log.info("coder.run_paused", run_id=run_id)
        return True

    def resume(
        self, run_id: str, *, expected_user_id: str | None = None,
    ) -> bool:
        """Clear the pause flag and release the awaiter.

        Idempotent — calling resume on a non-paused run returns False
        but doesn't error (so does an ``expected_user_id`` owner
        mismatch). Returns True only when an actual pause was cleared.
        """
        entry = self._runs.get(run_id)
        if entry is None or entry.done:
            return False
        if not _ownership_ok(entry, expected_user_id):
            return False
        if not entry.paused:
            return False
        entry.paused = False
        entry.pause_event.set()
        entry.paused_at = 0.0
        log.info("coder.run_resumed", run_id=run_id)
        return True

    async def await_pause_gate(self, run_id: str) -> None:
        """Block until the run is not paused.

        Called by the handler at each iteration boundary. No-op when
        the run is unpaused (the event starts set). When paused, the
        coroutine awaits the event so the loop yields cleanly to other
        tasks — no busy-wait, no polling.
        """
        entry = self._runs.get(run_id)
        if entry is None:
            return
        # Hot path: not paused → event is set → wait() returns immediately.
        await entry.pause_event.wait()

    def latest_for_workspace(
        self, *, user_id: str, workspace_id: str,
    ) -> RunBrokerEntry | None:
        """Return the most-recent entry (running OR done-and-retained)
        for ``(user_id, workspace_id)``.

        Differs from :meth:`get_active_for_workspace` by including
        finished entries that haven't been evicted yet — exactly the
        target shape for the /rewind route, which works against either
        a still-running turn or one that just completed. Returns
        ``None`` only when the broker has nothing remembered for this
        workspace at all.
        """
        match: RunBrokerEntry | None = None
        for entry in self._runs.values():
            if entry.user_id != user_id or entry.workspace_id != workspace_id:
                continue
            if match is None or entry.started_at > match.started_at:
                match = entry
        return match

    def get_active_for_workspace(
        self, *, user_id: str, workspace_id: str,
    ) -> RunBrokerEntry | None:
        """Return the in-flight entry for this workspace, if any.

        Used by the UI on mount: if a run is still going, attach to
        the broker stream instead of starting a new turn. Returns
        ``None`` once the run has finished — at that point the UI
        falls back to the ledger replay endpoint.
        """
        for entry in self._runs.values():
            if (
                entry.user_id == user_id
                and entry.workspace_id == workspace_id
                and not entry.done
            ):
                return entry
        return None

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    async def subscribe(
        self,
        run_id: str,
        *,
        since_seq: int = 0,
    ) -> AsyncIterator[_BufferedChunk]:
        """Yield buffered chunks past ``since_seq`` then tail live ones.

        Three emission cases:

        - ``since_seq`` falls within the current buffer window →
          replay the slice ``(since_seq, max_seq]`` then wait for
          new chunks.
        - ``since_seq`` is older than ``buffer_min_seq`` (ring
          overflowed during disconnect) → yield a synthetic
          ``buffer_overflow`` marker so the UI can render
          "reconnected, some streamed content was lost" without
          silently losing chunks. Then replay the whole buffer.
        - ``since_seq >= max_seq`` → no replay, tail live only.

        Iterator exits when ``entry.done`` is set AND the buffer has
        been fully drained past the subscriber's cursor. If the run
        finished with an error (agent exception or cancellation), a
        synthetic terminal chunk (``done=True`` + augmentum status +
        error text) is emitted after the drain so the subscriber never
        ends the stream unaware that the run failed. Its seq is
        ``entry.seq + 1`` — past every real chunk, so seq-deduping
        clients keep it, and a reconnect from that cursor skips it.
        """
        entry = self._runs.get(run_id)
        if entry is None:
            return
        entry.subscriber_count += 1
        try:
            cursor = since_seq
            # Detect overflow: chunks between the caller's cursor and
            # the oldest buffered chunk were evicted from the ring.
            # A cursor of 0 means "from the beginning" — it overflowed
            # iff the buffer no longer starts at seq 1, hence the
            # ``min_seq > cursor + 1`` form. Synthesize a sentinel
            # chunk so the frontend knows to refresh from the ledger
            # if it wants the missed prose. Importing here keeps the
            # broker importable without the models layer (helpful for
            # standalone testing).
            from augmentum.models.base import InternalStreamChunk

            min_seq = entry.buffer_min_seq()
            if min_seq and min_seq > cursor + 1:
                marker = InternalStreamChunk(
                    content_delta="",
                    done=False,
                    augmentum={
                        "status": "buffer_overflow",
                        "run_id": entry.run_id,
                        "lost_from_seq": cursor + 1,
                        "lost_to_seq": min_seq - 1,
                    },
                )
                # The marker occupies a seq the subscriber has never
                # seen (the newest evicted one) so clients that dedupe
                # by seq still deliver it, and it stays strictly below
                # every chunk about to be replayed from the buffer.
                yield _BufferedChunk(
                    seq=min_seq - 1,
                    timestamp=time.time(),
                    chunk=marker,
                )
                cursor = min_seq - 1

            while True:
                # Replay any buffered chunks past the cursor.
                replayed = False
                # Snapshot to a list to avoid mutation-during-iter.
                pending = [b for b in entry.buffer if b.seq > cursor]
                for buf in pending:
                    yield buf
                    cursor = buf.seq
                    replayed = True

                if entry.done and not replayed:
                    # Drained. If the run ended in an error (agent
                    # exception or cancellation), the subscriber must
                    # hear about it — the ordinary chunk flow carries
                    # no terminal signal for failure paths. Skip if
                    # the caller's cursor is already past the terminal
                    # seq (it saw this chunk on a previous connection).
                    terminal_seq = entry.seq + 1
                    if entry.error and cursor < terminal_seq:
                        status = (
                            "cancelled" if entry.error == "cancelled" else "error"
                        )
                        yield _BufferedChunk(
                            seq=terminal_seq,
                            timestamp=time.time(),
                            chunk=InternalStreamChunk(
                                content_delta="",
                                done=True,
                                augmentum={
                                    "status": status,
                                    "run_id": entry.run_id,
                                    "error": entry.error,
                                    "final_state": True,
                                },
                            ),
                        )
                    return

                # Wait for the next push (or for the run to finish).
                # Clear-before-wait so a chunk pushed BETWEEN our
                # snapshot above and the wait below still wakes us.
                entry.new_chunk_event.clear()
                if any(b.seq > cursor for b in entry.buffer) or entry.done:
                    # Race: a chunk landed (or run finished) between
                    # the snapshot and the clear. Loop without waiting.
                    continue
                try:
                    await entry.new_chunk_event.wait()
                except asyncio.CancelledError:
                    return
        finally:
            entry.subscriber_count = max(0, entry.subscriber_count - 1)

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(
        self,
        run_id: str,
        *,
        reason: str = "user_cancel",
        expected_user_id: str | None = None,
    ) -> bool:
        """Request cancellation of ``run_id`` with a ``reason`` hint.

        Returns True if the run existed and was active; False
        otherwise (including an ``expected_user_id`` owner mismatch —
        see ``_ownership_ok``). The actual stop is async — the
        detached task sees the flag at the next chunk boundary or hits
        the ``Task.cancel()`` and raises ``CancelledError`` mid-await.

        ``reason`` is recorded on the entry so the handler's
        CancelledError handler can read it and surface it in the
        next turn's ``<prior_turns>`` block — the model otherwise
        sees a silent gap and may try to resume the abandoned
        plan. Use one of the canonical reasons — the single source
        of truth is the vocabulary listed on
        ``RunBrokerEntry.cancel_reason``; arbitrary strings are
        accepted but the renderer falls back to a generic label.
        """
        entry = self._runs.get(run_id)
        if entry is None or entry.done:
            return False
        if not _ownership_ok(entry, expected_user_id):
            return False
        entry.cancel_requested = True
        # Don't overwrite a reason that was already set (e.g. a
        # later double-cancel shouldn't clobber the first one).
        if not entry.cancel_reason:
            entry.cancel_reason = (reason or "user_cancel").strip()[:50] or "user_cancel"
        if entry.task is not None and not entry.task.done():
            entry.task.cancel()
        return True

    def sweep_paused_timeouts(self, *, max_paused_seconds: float) -> int:
        """Cancel runs that have been paused longer than the threshold.

        Walks the broker's run registry once; for any active entry
        with ``paused=True`` AND ``time.time() - paused_at >
        max_paused_seconds``, calls ``cancel(run_id, reason='paused_timeout')``.
        The handler's CancelledError path will then surface a
        structured turn-summary entry so the next turn's prior_turns
        block tells the model "the prior turn was abandoned mid-pause."

        ``max_paused_seconds <= 0`` is a no-op (timeout disabled by
        operator). Returns the count of runs cancelled.
        """
        if max_paused_seconds <= 0:
            return 0
        now = time.time()
        cancelled = 0
        for entry in list(self._runs.values()):
            if entry.done or not entry.paused:
                continue
            if entry.cancel_requested:
                continue  # already cancelled; CancelledError pending — don't double-count
            if entry.paused_at <= 0:
                continue  # paused flag set without a timestamp — defensive
            paused_for = now - entry.paused_at
            if paused_for < max_paused_seconds:
                continue
            log.warning(
                "coder.run_paused_timeout",
                run_id=entry.run_id,
                workspace_id=entry.workspace_id,
                paused_for_s=round(paused_for, 1),
                threshold_s=max_paused_seconds,
            )
            if self.cancel(entry.run_id, reason="paused_timeout"):
                # Release the pause gate so the cancelled task can
                # actually progress to its CancelledError — without
                # this the await_pause_gate stays blocked and the
                # cancel never lands.
                entry.pause_event.set()
                cancelled += 1
        return cancelled

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    async def _sweep_loop(self) -> None:
        try:
            while not self._shutdown:
                await asyncio.sleep(60.0)
                await self._sweep_once()
        except asyncio.CancelledError:
            return

    async def _sweep_once(self) -> None:
        now = time.time()
        drop: list[str] = []
        async with self._lock:
            for run_id, entry in self._runs.items():
                if (
                    entry.done
                    and entry.finished_at is not None
                    and now - entry.finished_at > _RETAIN_FINISHED_SECONDS
                ):
                    drop.append(run_id)
                elif (
                    not entry.done
                    # Paused runs are quiet by USER CHOICE — the
                    # paused-timeout sweep owns that case; warning on
                    # them here is noise.
                    and not entry.paused
                    and entry.subscriber_count == 0
                    and now - entry.last_activity_at > _ACTIVE_QUIET_WARN_SECONDS
                    # Latch: warn once per quiet spell, not once per
                    # 60s sweep. _push resets it on new activity.
                    and not entry.quiet_warned
                ):
                    entry.quiet_warned = True
                    log.warning(
                        "coder_run_broker_quiet_active",
                        run_id=run_id,
                        idle_seconds=int(now - entry.last_activity_at),
                    )
            for run_id in drop:
                self._runs.pop(run_id, None)

    # ------------------------------------------------------------------
    # Stats (introspection / debugging)
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        active = sum(1 for e in self._runs.values() if not e.done)
        finished = sum(1 for e in self._runs.values() if e.done)
        return {
            "total": len(self._runs),
            "active": active,
            "finished_retained": finished,
        }


async def sweep_orphan_running_runs(conn) -> int:
    """One-shot startup pass: mark every coder-run row stuck in a
    non-terminal state as interrupted.

    Called from the server startup path right after the broker is
    constructed. The rows can be stuck because the previous process
    held the in-memory task (broker entry / RunManager task / subagent
    loop) and crashed or was killed without reaching the finish path.
    Without this sweep the UI would see a "running" turn forever and
    refuse to start a new one.

    Covers the whole zombie class, not just the turn ledger:

    * ``coder_turn_runs``     — ``status='running'`` → ``'cancelled'``
      with ``finish_reason='server_restart'`` (the vocabulary the
      handler's cancel path already uses).
    * ``coder_subagent_runs`` — ``stop_reason='running'`` →
      ``'server_restart'`` + ``completed_at`` stamped. Previously
      unswept: a subagent spawned via ``task_dispatch`` that died with
      the process kept its breadcrumb open forever.
    * ``claude_runs``         — external-coder rows in
      ``status='running'`` → ``'failed'`` with the same
      "interrupted (server restarted)" error the lazy list-time
      reconcile in ``external_coder_routes.py`` writes, so both paths
      produce identical rows. The lazy reconcile only fires when a
      user lists that workspace's runs; the boot sweep closes the gap.

    Safety: all coder run execution is in-process asyncio (broker
    tasks, RunManager tasks, subagent loops) — there is no heartbeat
    or lease column because runs cannot survive the process, and there
    is no multi-instance deployment sharing the SQLite file. At boot,
    therefore, *every* non-terminal row is by definition orphaned.
    Deliberately not user-scoped: this is server-level maintenance
    across all tenants, mirroring each table's own vocabulary.

    Returns the total number of rows updated across all tables.
    """
    if conn is None:
        return 0
    now = time.time()
    total = 0
    counts: dict[str, int] = {}

    async def _sweep(table: str, sql: str, params: tuple) -> None:
        nonlocal total
        try:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            swept = int(cursor.rowcount or 0)
        except Exception as exc:
            log.warning("coder_orphan_sweep_failed", table=table, error=str(exc))
            # The connection is SHARED (the app's single aiosqlite
            # conn). A failed execute/commit can leave an open
            # transaction that would then swallow or corrupt the next
            # caller's work — roll it back before returning.
            try:
                await conn.rollback()
            except Exception:
                log.warning(
                    "coder_orphan_sweep_rollback_failed", table=table,
                    exc_info=True,
                )
            return
        if swept:
            counts[table] = swept
            total += swept

    await _sweep(
        "coder_turn_runs",
        """
        UPDATE coder_turn_runs
        SET status = 'cancelled',
            completed_at = COALESCE(completed_at, ?),
            updated_at = ?,
            finish_reason = COALESCE(NULLIF(finish_reason, ''), 'server_restart')
        WHERE status = 'running'
        """,
        (now, now),
    )
    await _sweep(
        "coder_subagent_runs",
        """
        UPDATE coder_subagent_runs
        SET stop_reason = 'server_restart',
            stop_detail = 'interrupted (server restarted)',
            completed_at = COALESCE(completed_at, ?)
        WHERE stop_reason = 'running'
        """,
        (int(now),),
    )
    await _sweep(
        "claude_runs",
        """
        UPDATE claude_runs
        SET status = 'failed',
            error = 'interrupted (server restarted)',
            updated_at = datetime('now')
        WHERE status = 'running'
        """,
        (),
    )
    if total:
        log.info("coder_orphan_runs_swept_detail", **counts)
    return total
