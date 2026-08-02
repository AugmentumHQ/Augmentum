"""Proactive bridges — external completions feed her initiative queue.

Wiring program Phase 6 (2026-06-12). The initiative queue, scorer,
and surfacing pipeline shipped long ago, but only INTERNAL features
(unresolved journal, unfinished creations) ever fed it — a finished
transcription or a confirmed bug landed in a table and waited for the
user to go looking. These bridges hand external events to the same
queue the scorer already drains, so "your audiobook finished — want
it?" becomes possible without new delivery machinery.

Gating: ``presence_mode.autonomy_allowed()`` — the SILENT floor means
no autonomous writes at all, honored here at enqueue time (same
posture as the wondering/synthesis writers).
"""

from __future__ import annotations

import json
import time
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Job types whose completion is internal plumbing, not something she
# should consider mentioning. Sweeps and scheduled maintenance are
# noise; user-initiated work (transcription, downloads, conversions)
# is signal.
_QUIET_JOB_TYPES = frozenset({
    "signal_aggregation", "memory_compaction", "dream_cycle",
    "media_sync", "community_feed_refresh",
    # coder_background_run emits its own richer 'coder_run_completed'
    # initiative (jobs/handlers/coder_background_run.py::_emit_run_perception)
    # — suppress the generic 'job_finished' one so she doesn't hear it twice.
    "coder_background_run",
})


async def enqueue_external_initiative(
    conn: Any,
    *,
    companion_id: str,
    target_user_id: str,
    kind: str,
    payload: dict[str, Any],
    score: float = 0.7,
) -> int | None:
    """Write one external-event proposal into companion_initiative_queue.

    Returns the row id, or None when gated/failed. Same column shape
    initiative.enqueue uses, plus target_user_id (the queue is
    companion-scoped; the target says whose event it was).
    """
    from augmentum.companion_runtime.presence_mode import autonomy_allowed
    if not autonomy_allowed():
        log.debug("external_initiative_gated_silent", kind=kind)
        return None
    if not target_user_id:
        return None
    try:
        cur = await conn.execute(
            "INSERT INTO companion_initiative_queue "
            "(companion_id, proposed_at, kind, payload, score, status, "
            " target_user_id) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            (
                companion_id,
                time.time(),
                kind,
                json.dumps(payload, separators=(",", ":")),
                float(score),
                target_user_id,
            ),
        )
        row_id = cur.lastrowid
        await conn.commit()
        await cur.close()
        log.info(
            "external_initiative_enqueued",
            kind=kind, user_id=target_user_id, row_id=row_id,
        )
        return row_id
    except Exception:  # noqa: BLE001
        log.warning("external_initiative_enqueue_failed", kind=kind, exc_info=True)
        return None


def _conn_from_app_state(app_state: Any):
    sm = getattr(app_state, "state_manager", None)
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None)


def _companion_id(app_state: Any) -> str:
    runtime = getattr(app_state, "companion_runtime", None)
    return getattr(runtime, "companion_id", "") or "becca"


def make_job_terminal_listener(app_state: Any):
    """Build the JobMonitor listener that bridges terminal job events
    into the initiative queue. Resolves runtime/conn lazily at event
    time so registration order against companion startup is moot."""

    async def _on_terminal(event: Any) -> None:
        outcome = getattr(event, "outcome", "")
        if outcome not in ("completed", "failed", "timed_out"):
            return
        job_type = getattr(event, "job_type", "") or ""
        if job_type in _QUIET_JOB_TYPES:
            return
        conn = _conn_from_app_state(app_state)
        if conn is None:
            return
        # Failures are slightly more urgent than completions — the
        # user is probably waiting on the thing that just broke.
        score = 0.72 if outcome != "completed" else 0.66
        await enqueue_external_initiative(
            conn,
            companion_id=_companion_id(app_state),
            target_user_id=getattr(event, "user_id", "") or "",
            kind="job_finished",
            payload={
                "job_id": getattr(event, "job_id", ""),
                "job_type": job_type,
                "outcome": outcome,
                "error": (getattr(event, "error", "") or "")[:200],
            },
            score=score,
        )

    return _on_terminal


async def bridge_signal_results(
    app_state: Any, results: dict[str, dict[str, int]],
) -> int:
    """After a signal-aggregation pass, enqueue one initiative per
    user who gained NEW open signals. Returns proposals written."""
    conn = _conn_from_app_state(app_state)
    if conn is None:
        return 0
    written = 0
    for user_id, counts in (results or {}).items():
        new_total = sum(int(v or 0) for v in (counts or {}).values())
        if new_total <= 0:
            continue
        row_id = await enqueue_external_initiative(
            conn,
            companion_id=_companion_id(app_state),
            target_user_id=user_id,
            kind="signals_found",
            payload={"new_signals": new_total, "by_source": dict(counts)},
            score=0.6,
        )
        if row_id is not None:
            written += 1
    return written
