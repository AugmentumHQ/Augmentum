"""Single egress point for all chat-surface output in Coder mode.

Every piece of text that reaches the user goes through ``emit()`` or
``emit_relay()``. Two reasons this matters:

1. **One place for sanitizers.** The 2026-04-21 DTLN-preamble leak (where
   planner monologue like "The user is asking me..." streamed to chat)
   would have been a one-line fix here instead of per-yield-site surgery.
   Future sanitizers (redaction, length caps, rate limiting) land in one
   spot.

2. **Schema enforcement for ``augmentum`` metadata.** Previously the
   ``{phase, status}`` dict was invented ad-hoc at each of ~60 yield
   sites. ``Phase`` and ``Status`` below are the exhaustive vocabulary;
   new values must be added here before they can be emitted.

The ``_meta_chunk`` staticmethod on ``CoderHandler`` remains for callers
that want a content-less chunk — it now delegates to ``emit()``.
"""
from __future__ import annotations

from typing import Any, Literal, get_args

from augmentum.models.base import InternalStreamChunk

# First line of the per-turn runtime-context carrier message (see
# ``CoderHandler._build_runtime_carrier_message`` and the native-loop
# sibling in phase_act). Shared so consumers that need to RECOGNIZE a
# carrier — e.g. compaction, which must not mistake it for the task
# definition or condense stale per-turn state into history — match the
# exact bytes the builders emit.
RUNTIME_CARRIER_HEADER = "[Augmentum runtime context — not user dialogue]"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

Phase = Literal[
    "planning",
    "executing",
    "passthrough",
    "conversational",
    "waiting",
    # ``completing`` — fires after the agent loop has finished its work
    # but BEFORE the broker marks the run done. Used today by the
    # cooperative-handler's end-of-turn queue drain so the frontend
    # can react to ``queue_followup`` chunks while the turn is still
    # technically alive (auto-chain the next turn from queued messages).
    "completing",
]

# Exhaustive list of statuses emitted across the handler + phase mixins
# + legacy. Grep-collected 2026-04-21. Any new status emitted without
# adding it here raises ValueError in :func:`emit` / :func:`emit_relay`
# (overridable via AUGMENTUM_STRICT_METADATA=0 for emergency bypass).
#
# Adding a status: update this Literal AND the ``_VALID_STATUSES`` set
# below. Removing: drop from both. Renaming: same as removing + adding
# — downstream UI code may depend on the exact string.
Status = Literal[
    # Lifecycle
    "started",
    "complete",
    "strategy",
    "streaming",
    "done",
    "error",
    "recoverable_error",   # transient backend failure (429/5xx); UI shows Try Again pill
    "retrying",            # mid-backoff during a transient retry (UI shows "retrying in Ns")
    "escalated_to_buddy",  # stagnation tripped; remaining iters run on workspace.bug_finder_verifier_model
    "rate_limited",
    "short_circuit",
    # Streaming sub-states — emitted on transitions only (not per-chunk)
    # so the UI can label dead-air windows ("Waiting for model…",
    # "Reasoning…") instead of showing an ambiguous spinner. Wired in
    # through :class:`StreamProgressTracker`. See
    # ``feedback_backend_processing_reassurance`` memory for the scope
    # rule that mandates these across every streaming surface.
    "awaiting_first_token",
    "thinking",
    "responding",
    # Engine stage lifecycle relay (model_load / model_swap /
    # slot_restore / prefill). The backend yields stage_start /
    # stage_complete dicts (status_bus.Stage) before the first token;
    # _stream_and_parse forwards them under this status so
    # coder-stream.js can drive the real prefill progress bar
    # (coder-progress.js polls /api/engine/v2/prefill_progress while a
    # ``prefill`` stage is active). Without the relay the user stared
    # at a frozen label for the whole 30-330s prefill window.
    "stage",
    # Live reasoning relay — carries the model's chain-of-thought text in
    # ``thinking_delta`` while an act/plan LLM call is in flight, coalesced
    # by :class:`ReasoningRelay` (~2-4 chunks/s, never per-token) so the
    # wire, the broker ring buffer, and the DOM stay cheap. High-frequency
    # by design: the turn ledger's ``observe_chunk`` skips this status so
    # it never becomes per-delta SQLite writes, and the frontend routes it
    # to the collapsible reasoning block instead of the status pill.
    "reasoning_delta",
    # Plan phase
    "question",
    "planning",
    "continuation",
    "tier_classified",     # Phase 1 — eval-visible tier metadata
    "skipped_reflex",      # Phase 1.3 — REFLEX short-circuit marker
    # Execution — canonical + hybrid loops
    "observation_refresh",
    "compaction",
    "tool_call",
    "tool_result",
    "tool_error",
    "empty_model_stop_retry",
    "continuation_nudge",
    "progress_without_action_nudge",
    "operate_evidence_nudge",
    "stagnation_nudge",
    "unclaimed_code_block_nudge",
    "content_loop_nudge",
    # Raw XML tool-call markup found in the stop candidate's reasoning
    # or prose — the calls never executed (wrong channel), so the stop
    # was built on actions that never ran. See
    # handler._has_leaked_tool_markup (2026-07-02, Qwythos-9B).
    "leaked_tool_markup_nudge",
    "inspection_loop_nudge",
    "silent_success_nudge",
    "identical_result_nudge",
    "same_file_edit_nudge",
    "probe_no_signal_nudge",
    # Duplicate-read ladder (duplicate_calls.py, 2026-07-06): windowed
    # identical-call nudge, then in-place context repair (duplicate
    # results stubbed + reorientation note) — recovery, not a break.
    "duplicate_call_nudge",
    "loop_reorient",
    # Command-carousel ladder + turn-progress ceiling (2026-07-07) and
    # code-intel adoption nudges — registered late: the emitters landed
    # without vocabulary entries, the exact drift class the
    # loop-health coordinator (coder/loop_health.py) now guards. The
    # coordinator's suppression telemetry status lives here too.
    "command_carousel_nudge",
    "flaky_test_nudge",
    "progress_stall_nudge",
    "symbol_grep_nudge",
    "single_read_nudge",
    "loop_health_suppressed",
    "failing_shell_nudge",
    "task_stale_nudge",
    "populated_repo_nudge",
    "read_only_nudge",
    "power_activated",
    "reflection",
    "budget",
    # Independent completion check before honoring a stop (see
    # coder/goal_judge.py + phase_act's judge re-entry loop). Emitted
    # with {ok, impossible, reason, attempt} extras. Added 2026-07-02:
    # the emitter landed in 1747c0b without this vocabulary entry, so
    # the drift guard below errored the TAIL of every coder run.
    "goal_judge",
    # Held-out verification gate (coder/verify_command.py) + Qwen-Code-style
    # next-speaker classifier (coder/next_speaker.py) — both second-guess an
    # accepted write-stop before honoring it. Emitted from phase_act's stop
    # path; registered here 2026-07-24 (they'd been failing every gated run).
    "verify_command",
    "next_speaker_check",
    # Cooperative turn handling (2026-05-30)
    "steer_delivered",   # iteration-boundary drain of mode="steer" inbox entries
    "queue_followup",    # end-of-turn drain of the whole inbox (queue + undelivered steers); payload.messages = drained entries
    "queue_dropped",     # cancel/error path: inbox flushed; UI flips queued badges to "dropped"
    # Subagent live activity feed (2026-05-31)
    "subagent_progress",  # per-iteration / per-tool snapshot from a running subagent
    "subagent_cancelled", # explicit cancel hit a running subagent
    # Execution — termination reasons
    "max_iterations_reached",
    "tasks_completed",
    "finish_task_called",
    "repeat_stopped",
    "no_progress",
    "validation_error_break",
    "test_failure_streak_break",
    "same_file_edit_break",
    "action_stagnation_break",
    "inspection_loop_break",
    "no_write_progress_break",
    "progress_stall_break",
    "escalation_exhausted",
    "fallback_summary",
    # Legacy strategies (decompose / architect / mission / direct / react)
    "in_progress",
    "decomposing",
    "step_start",
    "step_complete",
    "step_warning",
    "fixing",
    "shell_output",
    "test_result",
    "verification_failed",
    "architect_reasoning",
    "no_tools",
    "mission_log",
    "mission_started",
    "mission_completed",
    "mission_failed",
    "mission_replanned",
    "promise_started",
    "promise_fulfilled",
    "promise_rejected",
    "promise_retry",
    "promise_verifying",
    "promise_decomposed",
    "planner_fallback",
    "fallback_unresolved",
]


# Runtime-validatable frozensets DERIVED from the Literals above — the
# single source of truth is the Literal, so the two can never drift apart.
# Previously these were a hand-maintained parallel copy; every few weeks an
# emitter landed a status in phase_act/handler, someone added it to the
# Literal but forgot this set (or vice-versa), and the strict guard below
# errored the tail of every run (goal_judge 2026-07-02, command_carousel,
# code-intel, verify_command/next_speaker_check 2026-07-24). ``get_args``
# on the Literal removes that whole failure mode: add to the Literal, done.
_VALID_PHASES: frozenset[str] = frozenset(get_args(Phase))
_VALID_STATUSES: frozenset[str] = frozenset(get_args(Status))


# Statuses/phases we've already warned about — one log line per novel
# value for the process lifetime, so a drifted status emitted every
# iteration doesn't flood the logs.
_WARNED_DRIFT: set[str] = set()


def _validate_metadata(phase: str, status: str) -> None:
    """Check ``phase``/``status`` against the exhaustive Literal vocabulary.

    Degrade, don't abort. A ``{phase, status}`` label is TELEMETRY — it
    drives a UI pill, not the coding work. An unregistered value must NEVER
    take down the user's turn. So the default is: log one warning per novel
    drift and let the chunk through with its label intact (the UI falls back
    to a generic spinner for an unknown status — cosmetic, not fatal).

    History: this guard used to ``raise`` by default, so any emitter that
    landed a new status without updating :data:`Status` errored the tail of
    every run (see the derived-set comment above). That's exactly the
    "fix the class" trap — a telemetry mismatch shouldn't be able to kill a
    turn. Tier 2 (Literal-derived sets) closes the add-to-one-not-the-other
    drift; this closes the brand-new-string drift.

    Strict mode — ``AUGMENTUM_STRICT_METADATA`` set to a truthy value
    (``1``/``true``/``yes``/``on``) — restores the hard raise for tests/CI so
    drift fails loud there, where a red test is the right feedback. Default
    (unset) is lenient.
    """
    import os as _os
    strict = _os.environ.get("AUGMENTUM_STRICT_METADATA", "").lower() in (
        "1", "true", "yes", "on",
    )
    problems: list[str] = []
    if phase not in _VALID_PHASES:
        problems.append(
            f"phase={phase!r} not in chat_egress.Phase Literal "
            f"(valid: {sorted(_VALID_PHASES)})"
        )
    if status not in _VALID_STATUSES:
        problems.append(
            f"status={status!r} not in chat_egress.Status Literal "
            f"(add it to the Status Literal in chat_egress.py)"
        )
    if not problems:
        return
    msg = "; ".join(problems)
    if strict:
        raise ValueError(f"chat_egress metadata drift: {msg}")
    key = f"{phase}|{status}"
    if key not in _WARNED_DRIFT:
        _WARNED_DRIFT.add(key)
        import structlog
        structlog.get_logger(__name__).warning(
            "chat_egress_metadata_drift", problems=problems,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def emit(
    content: str = "",
    *,
    phase: Phase,
    status: Status,
    model: str,
    thinking: str = "",
    role: str | None = None,
    finish_reason: str | None = None,
    usage: Any = None,
    done: bool = False,
    extra: dict | None = None,
) -> InternalStreamChunk:
    """Build an originated chat chunk.

    Use for content the handler itself produces — notifications,
    errors, synthesized prose, meta events. For forwarding a backend
    chunk, use :func:`emit_relay` so the source chunk's metadata is
    preserved.
    """
    _validate_metadata(phase, status)
    augmentum: dict[str, Any] = {"mode": "coder", "phase": phase, "status": status}
    if extra:
        augmentum.update(extra)

    return InternalStreamChunk(
        content_delta=content,
        thinking_delta=thinking,
        role=role,
        finish_reason=finish_reason,
        usage=usage,
        model=model,
        done=done,
        augmentum=augmentum,
    )


def emit_relay(
    source: InternalStreamChunk,
    *,
    phase: Phase,
    status: Status,
    model_fallback: str,
    content_override: str | None = None,
    thinking_override: str | None = None,
    extra: dict | None = None,
) -> InternalStreamChunk:
    """Relay a backend chunk with our ``augmentum`` metadata applied.

    ``content_override`` / ``thinking_override``: pass ``""`` to suppress
    the source field, a string to replace it, or ``None`` to keep the
    original value. Used when a sanitizer has already processed the
    content (e.g. plan-phase preamble stripping).
    """
    _validate_metadata(phase, status)
    augmentum: dict[str, Any] = {"mode": "coder", "phase": phase, "status": status}
    if extra:
        augmentum.update(extra)

    return InternalStreamChunk(
        content_delta=(
            source.content_delta if content_override is None else content_override
        ),
        thinking_delta=(
            source.thinking_delta if thinking_override is None else thinking_override
        ),
        role=source.role,
        finish_reason=source.finish_reason,
        usage=source.usage,
        model=source.model or model_fallback,
        done=source.done,
        augmentum=augmentum,
    )


# ---------------------------------------------------------------------------
# Streaming progress tracker
# ---------------------------------------------------------------------------


class StreamProgressTracker:
    """Emits sub-state transition chunks (``awaiting_first_token`` →
    ``thinking`` → ``responding``) so the UI can label dead-air windows
    instead of showing an ambiguous spinner.

    Transitions only: one meta-chunk per state change, not per delta.
    A non-reasoning model skips straight from ``awaiting_first_token``
    to ``responding``; a reasoning model goes through ``thinking`` first
    and may flip back and forth on some architectures (rare).

    Usage::

        tracker = StreamProgressTracker()
        yield tracker.begin(phase="planning", model=req.model)
        async for chunk in backend.chat_stream(req):
            prog = tracker.update(chunk, phase="planning", model=req.model)
            if prog is not None:
                yield prog
            # ...existing per-chunk processing...
    """

    __slots__ = ("_substate",)

    def __init__(self) -> None:
        self._substate: str | None = None

    def begin(self, *, phase: Phase, model: str) -> InternalStreamChunk:
        """Emit the initial ``awaiting_first_token`` marker.

        Call immediately before the ``async for chunk in backend.chat_stream``
        loop — after the request has been dispatched but before any data
        is back. Marks the start of the TTFT / prefix-eval window.
        """
        self._substate = "awaiting_first_token"
        return emit(phase=phase, status="awaiting_first_token", model=model)

    def update(
        self,
        chunk: InternalStreamChunk,
        *,
        phase: Phase,
        model: str,
    ) -> InternalStreamChunk | None:
        """Return a transition chunk if the sub-state changed, else None.

        Detection is purely field-based — backends already separate
        reasoning tokens into ``thinking_delta`` (handled by
        ``augmentum/utils/thinking.py`` family parsers on ingress) so
        this tracker doesn't re-parse. Chunks with neither field (pure
        tool-call deltas, finish-reason-only chunks) don't move state.
        """
        new: str | None
        if chunk.thinking_delta:
            new = "thinking"
        elif chunk.content_delta:
            new = "responding"
        else:
            new = None
        if new is None or new == self._substate:
            return None
        self._substate = new
        return emit(phase=phase, status=new, model=model)

    @property
    def substate(self) -> str | None:
        return self._substate


# ---------------------------------------------------------------------------
# Live reasoning relay (coalescer)
# ---------------------------------------------------------------------------


class ReasoningRelay:
    """Coalesce per-token reasoning deltas into cheap ``reasoning_delta`` chunks.

    Backends emit ``thinking_delta`` roughly per token (~20-60/s). Relaying
    each one to the client would mean one NDJSON line, one broker
    ring-buffer slot, and one DOM append per token — the exact
    per-token-work class behind the 2026-06 coder freeze fixes. This
    coalescer batches text and flushes when either

    * the pending buffer reaches ``min_chars``, or
    * ``max_latency_s`` has elapsed since the last flush (checked on
      chunk arrival — no timers; if no chunks arrive there is nothing
      new to show anyway),

    which lands at ~2-4 chunks/second on a fast model while a slow model
    still feels live. **Text is only ever batched, never dropped** — the
    caller MUST call :meth:`flush` at stream end / before transition
    events / on error so no reasoning is lost.
    """

    __slots__ = ("_phase", "_model", "_pending", "_last_flush",
                 "_min_chars", "_max_latency_s")

    def __init__(
        self,
        *,
        phase: Phase,
        model: str,
        min_chars: int = 64,
        max_latency_s: float = 0.25,
    ) -> None:
        import time as _time
        self._phase: Phase = phase
        self._model = model
        self._pending: list[str] = []
        self._last_flush = _time.monotonic()
        self._min_chars = min_chars
        self._max_latency_s = max_latency_s

    def add(self, text: str) -> InternalStreamChunk | None:
        """Buffer ``text``; return a chunk when a flush threshold is hit."""
        if not text:
            return None
        import time as _time
        self._pending.append(text)
        pending_len = sum(len(p) for p in self._pending)
        if (
            pending_len >= self._min_chars
            or (_time.monotonic() - self._last_flush) >= self._max_latency_s
        ):
            return self.flush()
        return None

    def flush(self) -> InternalStreamChunk | None:
        """Emit everything buffered (or None if nothing is pending)."""
        import time as _time
        self._last_flush = _time.monotonic()
        if not self._pending:
            return None
        text = "".join(self._pending)
        self._pending = []
        return emit(
            phase=self._phase,
            status="reasoning_delta",
            model=self._model,
            thinking=text,
        )
