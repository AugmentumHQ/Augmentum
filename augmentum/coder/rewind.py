"""Coder turn rewind — undo the most recent turn's workspace changes
and pop the matching conversation/state entries.

The user-visible model:

  1. AI did a bad turn (hallucination, broken edit, wrong direction).
  2. User clicks Rewind.
  3. Workspace files revert to pre-turn state.
  4. Last user + assistant messages disappear from the conversation
     (handled frontend-side after this module returns success).
  5. The corresponding ``turn_summaries`` entry is dropped so the
     next turn's ``<prior_turns>`` block doesn't carry the trace of
     work that no longer exists.

What rewind does NOT undo (deliberate scope):

  * Observations written via the ``observe`` tool — the JSONL ledger
    at /workspace/.augmentum/observations.jsonl is append-only and
    represents durable learnings, not turn-local state.
  * ``working_set`` / ``files_read`` additions — session-level
    bookkeeping; tracking which fields were added by which turn is a
    bigger lift than v1 warrants.
  * Side effects outside the workspace: HTTP requests, started
    services, db_inspect writes to external databases, git pushes.
    Surfaced as warnings in the response so the user knows about
    them.

Scope decision: most-recent turn only. Multi-step undo would need a
snapshot stack with cascading restore order; not in v1.

Dispatch entry: :func:`rewind_last_turn`. The /rewind route is a thin
wrapper; the bulk of the work is here so the logic can be tested in
isolation.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# How long to wait for a cancelled run's broker task to actually
# settle (set ``done=True``) before we proceed with state mutation.
# The handler's CancelledError path persists state during teardown,
# so racing it risks load → mutate → overwrite-from-handler. 5s is
# generous; typical settle is <100ms after Task.cancel() fires.
_CANCEL_SETTLE_TIMEOUT_S: float = 5.0


@dataclass
class RewindOutcome:
    """Result of a rewind attempt.

    ``ok`` is False only when the most-recent turn cannot be found at
    all (no broker entry, no review bundle). Partial-success cases —
    some files un-restored, no summary to pop — return ``ok=True``
    with the partial results surfaced in ``warnings`` and
    ``irreversible_paths`` for the user to act on.

    ``mode`` echoes the requested rewind scope back so the caller (UI
    toast, audit log) can show "rewound conversation only" vs the
    fuller "both" message without re-deriving it.
    """

    ok: bool
    mode: str = "both"
    run_id: str = ""
    cancelled_in_flight: bool = False
    restored_paths: list[str] = field(default_factory=list)
    irreversible_paths: list[str] = field(default_factory=list)
    turn_summary_popped: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "run_id": self.run_id,
            "cancelled_in_flight": self.cancelled_in_flight,
            "restored_paths": list(self.restored_paths),
            "irreversible_paths": list(self.irreversible_paths),
            "turn_summary_popped": self.turn_summary_popped,
            "warnings": list(self.warnings),
            "error": self.error,
        }


# Valid rewind scopes. Documented at the function level too; kept here
# so other consumers (route layer, frontend handshake) can import the
# canonical list rather than copy-pasting it.
REWIND_MODES: frozenset[str] = frozenset({"both", "files", "conv"})


# ---------------------------------------------------------------------------
# Snapshot resolution
# ---------------------------------------------------------------------------

def _snapshot_from_broker(app_state: Any, *, user_id: str, workspace_id: str):
    """Find the most-recent (running or finished-retained) broker entry
    for ``(user_id, workspace_id)`` and return its ``(entry, snapshot)``.

    Returns ``(None, None)`` when no entry exists or the entry's snapshot
    was never attached (legacy runs from before this feature shipped).
    """
    broker = getattr(app_state, "coder_run_broker", None)
    if broker is None:
        return None, None
    entry = broker.latest_for_workspace(
        user_id=user_id, workspace_id=workspace_id,
    )
    if entry is None:
        return None, None
    return entry, entry.turn_snapshot


def _snapshot_from_review_registry(
    app_state: Any, *, user_id: str, workspace_id: str,
):
    """Fallback path: locate the most-recent pending ReviewBundle for
    this workspace and return its snapshot.

    Used when the broker entry has been evicted (>10min after finish)
    but the user hasn't accepted/rejected the review bundle yet.
    Returns ``(bundle, snapshot)`` or ``(None, None)``.
    """
    registry = getattr(app_state, "review_registry", None)
    if registry is None:
        return None, None
    bundles = registry.pending_for(user_id)
    candidates = [
        b for b in bundles if b.workspace_id == workspace_id
    ]
    if not candidates:
        return None, None
    # Most recent first by created_at.
    candidates.sort(key=lambda b: b.created_at, reverse=True)
    bundle = candidates[0]
    return bundle, bundle.snapshot


# ---------------------------------------------------------------------------
# Cancel + settle
# ---------------------------------------------------------------------------

async def _cancel_and_settle(
    app_state: Any, *, run_id: str,
) -> tuple[bool, bool]:
    """Request cancellation of the broker entry and wait for it to
    finish.

    Returns ``(cancelled, settled)``:

    * ``cancelled`` — True iff ``broker.cancel()`` reported the run
      was active and the flag was set.
    * ``settled`` — True iff the entry transitioned to ``done`` within
      :data:`_CANCEL_SETTLE_TIMEOUT_S`. False means we gave up
      waiting; the caller should still proceed but log a warning, and
      can decide whether to skip state mutation to avoid the race.
    """
    broker = getattr(app_state, "coder_run_broker", None)
    if broker is None:
        return False, True
    entry = broker.get(run_id)
    if entry is None or entry.done:
        return False, True

    cancelled = broker.cancel(run_id, reason="user_rewind")
    if not cancelled:
        return False, True

    deadline = time.monotonic() + _CANCEL_SETTLE_TIMEOUT_S
    while time.monotonic() < deadline:
        if entry.done:
            return True, True
        await asyncio.sleep(0.05)
    log.warning(
        "coder.rewind_cancel_settle_timeout",
        run_id=run_id, waited_s=_CANCEL_SETTLE_TIMEOUT_S,
    )
    return True, False


# ---------------------------------------------------------------------------
# State mutation
# ---------------------------------------------------------------------------

def _pop_matching_turn_summary(state: Any, run_id: str) -> bool:
    """Remove the trailing turn_summary entry that belongs to ``run_id``.

    Two policies, in order:

    1. Strict: if the last entry has ``turn_id == run_id``, pop it.
       This is the post-fix path — every new summary stamps its
       originating turn id.
    2. Fallback: if the last entry has no ``turn_id`` (older
       persisted summaries from before the field landed) and we have
       no signal that the summary belongs to a *different* run, pop
       it. The realistic failure mode here is benign — at worst a
       previous turn's summary disappears from the next prompt; the
       workspace state is still correct, and the cap of 10 means the
       data was about to roll off anyway.

    Returns True iff an entry was popped.
    """
    summaries = getattr(state, "turn_summaries", None) or []
    if not summaries:
        return False
    last = summaries[-1]
    if not isinstance(last, dict):
        return False
    stamped = last.get("turn_id")
    if stamped:
        if stamped == run_id:
            summaries.pop()
            return True
        # Stamped but doesn't match — probably the rewound turn never
        # wrote a summary (cancelled before turn-end), and the last
        # summary is from a *previous* turn. Leave it alone.
        return False
    # Unstamped: fall through to "pop the last one" heuristic.
    summaries.pop()
    return True


def _reset_per_request_state(state: Any) -> None:
    """Clear the per-request scratchpads the rewound turn populated.

    Mirrors the per-request branch of
    ``CoderHandler._reset_for_new_request`` (without the snapshot
    setup, which doesn't apply here). Session-level bookkeeping
    (files_read mtimes, working_set, tool_calls_made) is left alone —
    same scope decision as the docstring at the top of this module.
    """
    # Plan + tasks + mission — the rewound turn's plan is now stale.
    state.plan = ""
    state.plan_steps = []
    state.current_step = 0
    state.step_outputs = {}
    state.mission = []
    state.tasks = []
    # Loop-termination + completion contract — if the rewound turn set
    # finish_requested = True, leaving it on would short-circuit the
    # next turn's first iteration.
    state.finish_requested = False
    state.finish_summary = ""
    if hasattr(state, "clear_pending_objective_contract"):
        state.clear_pending_objective_contract()
    # Recent error ledgers — clear the ring buffers that may have
    # collected entries during the rewound turn. Leave
    # recent_tool_failures (cross-turn memory) alone.
    state.recent_validation_errors.clear()
    state.recent_tool_calls.clear()
    state.consecutive_failures = 0
    state.error = None
    state.current_intent = None


# ---------------------------------------------------------------------------
# Ledger update
# ---------------------------------------------------------------------------

async def _mark_run_rewound(conn: Any, *, run_id: str, user_id: str) -> bool:
    """Tag the ``coder_turn_runs`` row as rewound.

    Uses ``status='cancelled'`` (a known status the UI already
    understands) with ``finish_reason='user_rewind'`` so the next
    turn's prior_turns block can tell the model "the previous turn
    was rewound by the user". Status 'rewound' would be cleaner but
    requires every consumer to learn a new value; cancelled +
    finish_reason is back-compatible.

    Returns True on a successful update; False on any error or when
    the row doesn't exist.
    """
    if conn is None:
        return False
    try:
        now = time.time()
        cursor = await conn.execute(
            """
            UPDATE coder_turn_runs
            SET status = 'cancelled',
                completed_at = COALESCE(completed_at, ?),
                updated_at = ?,
                finish_reason = 'user_rewind'
            WHERE id = ? AND user_id = ?
            """,
            (now, now, run_id, user_id),
        )
        await conn.commit()
        return int(cursor.rowcount or 0) > 0
    except Exception as exc:
        log.warning("coder.rewind_run_mark_failed", run_id=run_id, error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def rewind_last_turn(
    *,
    user_id: str,
    workspace_id: str,
    app_state: Any,
    mode: str = "both",
) -> RewindOutcome:
    """Rewind the most recent coder turn for ``(user_id, workspace_id)``.

    Modes (matching Claude Code's converged vocabulary):

    * ``both`` (default) — restore files AND drop the conversation/
      state for the turn. The "I want this turn to never have
      happened" rewind.
    * ``files`` — restore files only. Keep the conversation history
      AND ``turn_summaries`` entries intact. Useful when the model's
      analysis was correct but its edits were wrong: keep the chat
      so the model remembers the discussion, but throw out the bad
      file changes. The matching frontend behaviour is to LEAVE the
      assistant message in place (no message removal).
    * ``conv`` — drop conversation/state, KEEP the files on disk.
      The "poisoned context" rewind: model made the right edits but
      its assumption that landed in conversation is wrong and would
      pollute every subsequent turn. Frontend drops the user+assistant
      pair from the chat tree; backend pops the turn_summary +
      clears per-request scratchpads but does NOT call
      ``snapshot.restore()``.

    Steps:
      1. Find the most-recent broker entry (running or finished-retained).
         Fall back to the most-recent pending ReviewBundle if none.
      2. If the run is in-flight, cancel it and wait for settle.
      3. Restore every snapshotted path via :meth:`TurnSnapshot.restore`
         (skipped when ``mode == "conv"``).
      4. Load CoderState, pop the matching turn_summary, clear per-
         request scratchpads, save (skipped when ``mode == "files"``).
      5. Mark the ``coder_turn_runs`` row as rewound (always).
      6. Resolve the review bundle (if any) as ``status='rewound'``
         (always).

    The caller (HTTP route) takes care of dropping the matching
    conversation messages — the chat tree's source of truth is
    frontend-side, and there's no clean way for this module to know
    which nodes correspond to the rewound turn without parsing the
    tree. The frontend gates the message-removal step on ``mode``
    (only ``both`` / ``conv`` drop messages; ``files`` keeps them).
    """
    mode = (mode or "both").strip().lower()
    if mode not in REWIND_MODES:
        mode = "both"
    outcome = RewindOutcome(ok=False, mode=mode)

    # ---- 1. Find the snapshot ------------------------------------------
    entry, snapshot = _snapshot_from_broker(
        app_state, user_id=user_id, workspace_id=workspace_id,
    )
    bundle = None
    run_id = ""
    session_id = ""
    if entry is not None:
        run_id = entry.run_id
        # session_id isn't on the broker entry directly — fetch from the
        # ledger so we can persist the popped state back to the right row.
        session_id = await _session_id_for_run(app_state, run_id, user_id)
        if snapshot is None:
            # Broker entry exists but no snapshot was attached — legacy
            # run from before this feature shipped. Try the review
            # registry path; if that also fails, surface the limitation.
            bundle, snapshot = _snapshot_from_review_registry(
                app_state, user_id=user_id, workspace_id=workspace_id,
            )
    else:
        bundle, snapshot = _snapshot_from_review_registry(
            app_state, user_id=user_id, workspace_id=workspace_id,
        )
        if bundle is not None:
            run_id = bundle.turn_id
            session_id = bundle.session_id

    if snapshot is None:
        outcome.error = (
            "Nothing to rewind — no recent turn snapshot is held in "
            "memory. Server restart since the last turn drops snapshots."
        )
        return outcome

    outcome.run_id = run_id

    # ---- 2. Cancel + settle if in-flight -------------------------------
    if entry is not None and not entry.done:
        cancelled, settled = await _cancel_and_settle(app_state, run_id=run_id)
        outcome.cancelled_in_flight = cancelled
        if not settled:
            outcome.warnings.append(
                "Run cancellation didn't settle within timeout — "
                "state changes may race with the handler's teardown."
            )

    # ---- 3. Restore files (skipped for mode=conv) ----------------------
    if mode in ("both", "files"):
        try:
            touched = list(snapshot.touched_paths)
            failed = await snapshot.restore(touched)
            restored = [p for p in touched if p not in failed]
            outcome.restored_paths = restored
            outcome.irreversible_paths = list(failed)
        except Exception as exc:
            log.warning(
                "coder.rewind_restore_failed",
                run_id=run_id, mode=mode, error=str(exc),
            )
            outcome.error = f"File restore failed: {exc}"
            return outcome
    else:
        # conv-only: workspace stays as-is. The user explicitly chose
        # to keep the edits — typically because the edits were correct
        # but the conversation context was poisoned by a bad
        # assumption the model wrote down.
        outcome.warnings.append(
            "Workspace files left as-is — conv-only rewind drops "
            "conversation/state but keeps the edits on disk."
        )

    # ---- 4. Pop matching turn_summary + reset scratchpads -------------
    # Skipped for mode=files: the user wants to keep the conversation
    # AND the turn_summary AND the in-progress plan; only the file
    # edits get rolled back. Frontend matches: it does NOT remove the
    # assistant message bubble when mode=files.
    if mode in ("both", "conv"):
        state_manager = getattr(app_state, "state_manager", None)
        if state_manager is not None and session_id:
            try:
                state = await state_manager.load_coder_state(
                    session_id, user_id=user_id,
                )
                if state is not None:
                    outcome.turn_summary_popped = _pop_matching_turn_summary(
                        state, run_id,
                    )
                    _reset_per_request_state(state)
                    await state_manager.save_coder_state(
                        session_id, state, user_id=user_id,
                    )
            except Exception as exc:
                log.warning(
                    "coder.rewind_state_update_failed",
                    run_id=run_id, session_id=session_id, error=str(exc),
                )
                outcome.warnings.append(
                    "Files restored but conversation state could not be "
                    "rolled back fully — next turn may show stale prior_turns."
                )

    # ---- 5. Mark the run row -------------------------------------------
    conn = _conn_from_app_state(app_state)
    if conn is not None and run_id:
        await _mark_run_rewound(conn, run_id=run_id, user_id=user_id)

    # ---- 6. Resolve the review bundle if any ---------------------------
    registry = getattr(app_state, "review_registry", None)
    if registry is not None and run_id:
        try:
            registry.resolve(run_id, status="rewound")
        except Exception:
            # resolve returns None on missing — never raises in normal
            # flow. Catch defensively in case future registry changes
            # introduce one.
            log.debug("coder.rewind_bundle_resolve_failed", exc_info=True)

    # Side-effect warning — surfaced when files were actually restored
    # (mode=both or mode=files). Skipped for mode=conv because the
    # workspace stayed put, so the "side effects aren't undone"
    # framing doesn't apply.
    if mode in ("both", "files"):
        outcome.warnings.append(
            "Workspace files restored, but side effects outside the "
            "workspace (HTTP requests, started services, external DB "
            "writes, git pushes) are not undone."
        )

    outcome.ok = True
    log.info(
        "coder.rewind_ok",
        run_id=run_id,
        cancelled=outcome.cancelled_in_flight,
        restored=len(outcome.restored_paths),
        irreversible=len(outcome.irreversible_paths),
        summary_popped=outcome.turn_summary_popped,
    )
    return outcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _session_id_for_run(
    app_state: Any, run_id: str, user_id: str,
) -> str:
    """Look up the session_id stored on a coder_turn_runs row."""
    conn = _conn_from_app_state(app_state)
    if conn is None or not run_id:
        return ""
    try:
        cursor = await conn.execute(
            "SELECT session_id FROM coder_turn_runs WHERE id = ? AND user_id = ?",
            (run_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return ""
        return str(row[0] or "")
    except Exception:
        return ""


def _conn_from_app_state(app_state: Any) -> Any:
    """Pull the aiosqlite connection off the state manager."""
    sm = getattr(app_state, "state_manager", None)
    if sm is None:
        return None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None)
