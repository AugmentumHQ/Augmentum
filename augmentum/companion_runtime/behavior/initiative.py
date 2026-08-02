"""Initiative scorer + queue.

Periodically scores potential "Becca-volunteered" thoughts and writes
proposals to ``companion_initiative_queue`` (migration 156). Each
proposal has a kind, payload, score, and status. The runtime
surfaces high-scoring proposals at the right moment — for example,
when the owner re-engages after a quiet period the queue is consulted.

Score features (sprint plan §6):
- time since last interaction (longer → higher; gap matters)
- unresolved journal threads
- unfinished creations awaiting input
- mutual-influence observations awaiting acknowledgement
- household state changes (Sprint 7+; degrades to 0 today)

Threshold: ``companion_initiative_threshold`` (default 0.62). Below
that the proposal is recorded but stays in ``status='pending'``;
above it the runtime may surface immediately as a bus event.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.companion_runtime.scoping import owner_clause
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


@dataclass(slots=True)
class Proposal:
    """A scored thought ready to be queued."""
    kind: str
    payload: dict
    score: float


# ── Feature extractors ───────────────────────────────────────────────

async def _time_since_last_interaction(
    runtime: CompanionRuntime, *, owner_user_id: str = "",
) -> float:
    """Hours since the most recent journal/interaction event. Capped
    at 24h so a long absence doesn't dominate utility forever.

    Pre-Piece-7' bug: this used to query ``companion_journal.ts`` which
    doesn't exist — migration 154 names the column ``created_at`` (ISO
    TEXT). The try/except swallowed the OperationalError and the
    feature silently returned 0.0 always. Fixed to use the real column
    and compute the delta via SQLite's strftime so we don't drag a
    parser into the hot path. Scoped to the owner so one user's activity
    doesn't drive another's initiative (audit 2026-06-17).
    """
    backend = runtime.backend
    frag, p = owner_clause(owner_user_id)
    try:
        cur = await backend.conn.execute(
            "SELECT (strftime('%s', 'now') - strftime('%s', MAX(created_at))) "
            f"FROM companion_journal WHERE companion_id = ? {frag}",
            (runtime.companion_id, *p),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return 0.0
    if not row or row[0] is None:
        # No journal entries yet — treat as "long absence" so first-run
        # initiative scoring isn't suppressed.
        return 1.0
    try:
        delta_s = float(row[0])
    except (TypeError, ValueError):
        return 0.0
    hours = max(0.0, delta_s / 3600.0)
    return min(hours / 24.0, 1.0)


async def _unresolved_journal(
    runtime: CompanionRuntime, *, owner_user_id: str = "",
) -> float:
    """Journal entries flagged as ``unfinished`` or ``wondering``
    in the last 7 days.

    Same fix as :func:`_time_since_last_interaction` — switched from
    the non-existent ``ts`` column to the real ``created_at``. Uses
    SQLite's ``datetime('now', '-7 days')`` so we don't have to format
    a threshold timestamp Python-side. We also count both
    ``wondering`` and ``unfinished`` entry types (per migration 154's
    canonical kinds — both are explicitly "thread not yet closed").
    Owner-scoped (audit 2026-06-17).
    """
    backend = runtime.backend
    frag, p = owner_clause(owner_user_id)
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal "
            "WHERE companion_id = ? "
            "  AND entry_type IN ('wondering', 'unfinished') "
            "  AND created_at > datetime('now', '-7 days') "
            f"  AND COALESCE(suppressed, 0) = 0 {frag}",
            (runtime.companion_id, *p),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return 0.0
    count = (row[0] if row else 0) or 0
    return min(count / 5.0, 1.0)


async def _unfinished_creations(
    runtime: CompanionRuntime, *, owner_user_id: str = "",
) -> float:
    backend = runtime.backend
    frag, p = owner_clause(owner_user_id)
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_creations "
            f"WHERE companion_id = ? AND shared_at IS NULL {frag}",
            (runtime.companion_id, *p),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return 0.0
    count = (row[0] if row else 0) or 0
    return min(count / 3.0, 1.0)


async def _unacked_observations(
    runtime: CompanionRuntime, *, owner_user_id: str = "",
) -> float:
    backend = runtime.backend
    frag, p = owner_clause(owner_user_id)
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_observations "
            f"WHERE companion_id = ? AND surfaced = 0 {frag}",
            (runtime.companion_id, *p),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return 0.0
    count = (row[0] if row else 0) or 0
    return min(count / 4.0, 1.0)


_FEATURE_WEIGHTS = {
    "time_since": 0.30,
    "unresolved_journal": 0.25,
    "unfinished_creations": 0.20,
    "unacked_observations": 0.25,
}


# ── Public API ───────────────────────────────────────────────────────

async def score(runtime: CompanionRuntime) -> Proposal:
    """Compute a single initiative proposal for the current moment.

    The proposal's ``kind`` is derived from which feature dominates:
    if unfinished_creations leads, propose ``share_creation``; if
    unresolved_journal, propose ``revisit_thread``; etc. This keeps
    proposals concrete enough that Sprint 4a's surfacing has something
    to render (Sprint 5 actually renders).
    """
    owner = runtime.owner_user_id or ""
    f_time = await _time_since_last_interaction(runtime, owner_user_id=owner)
    f_journal = await _unresolved_journal(runtime, owner_user_id=owner)
    f_creations = await _unfinished_creations(runtime, owner_user_id=owner)
    f_obs = await _unacked_observations(runtime, owner_user_id=owner)

    total = (
        f_time * _FEATURE_WEIGHTS["time_since"]
        + f_journal * _FEATURE_WEIGHTS["unresolved_journal"]
        + f_creations * _FEATURE_WEIGHTS["unfinished_creations"]
        + f_obs * _FEATURE_WEIGHTS["unacked_observations"]
    )

    # Choose proposal kind by dominant feature
    dom = max(
        ("revisit_thread", f_journal),
        ("share_creation", f_creations),
        ("surface_observation", f_obs),
        ("reach_out_after_quiet", f_time),
        key=lambda kv: kv[1],
    )
    return Proposal(
        kind=dom[0],
        payload={
            "features": {
                "time_since_24h_norm": round(f_time, 3),
                "unresolved_journal": round(f_journal, 3),
                "unfinished_creations": round(f_creations, 3),
                "unacked_observations": round(f_obs, 3),
            },
        },
        score=round(total, 3),
    )


async def enqueue(runtime: CompanionRuntime, proposal: Proposal) -> int | None:
    """Write the proposal to ``companion_initiative_queue``.

    Returns the row id, or None if the write failed.
    """
    backend = runtime.backend
    payload_json = json.dumps(proposal.payload, separators=(",", ":"))
    # Write the owner (mig 179 added the column) so the queue row is
    # attributable + isolatable per user (audit 2026-06-17).
    owner = runtime.owner_user_id or ""
    try:
        cur = await backend.conn.execute(
            "INSERT INTO companion_initiative_queue "
            "(companion_id, user_id, proposed_at, kind, payload, score, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.companion_id,
                owner,
                time.time(),
                proposal.kind,
                payload_json,
                proposal.score,
                "pending",
            ),
        )
        row_id = cur.lastrowid
        await backend.conn.commit()
        await cur.close()
    except Exception:
        log.exception("initiative_enqueue_failed")
        return None
    return row_id


async def step(runtime: CompanionRuntime) -> Proposal | None:
    """One iteration: score, write if non-trivial, surface if above
    threshold.

    Resource cap (Piece 7'): self-gates on
    ``companion_initiative_min_interval_s`` (default 60s) so the tick
    loop's 5-30s cadence doesn't fire 4 SQL SELECTs per tick. The
    cap timestamp lives on ``runtime.last_initiative_score_at``;
    skipping returns None without touching the DB.

    Master kill-switch: ``companion_initiative_enabled`` (default
    False during rollout). When disabled, returns None without
    consuming the interval window.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_initiative_enabled", False):
        return None

    # Interval gate — read once into a local to keep the comparison
    # cheap. ``time.time()`` is microsecond-fast; the read of the
    # runtime attribute is a pure attr lookup.
    min_interval_s = float(
        getattr(settings, "companion_initiative_min_interval_s", 60.0)
    )
    now = time.time()
    last_at = float(getattr(runtime, "last_initiative_score_at", 0.0))
    if last_at and (now - last_at) < min_interval_s:
        return None

    base_threshold = float(getattr(settings, "companion_initiative_threshold", 0.62))
    proposal = await score(runtime)
    # Stamp the score-at AFTER the SQL fan-out so a failure mid-score
    # doesn't lock the next pass out. We update on success so the cap
    # is honest about "time since last completed scoring."
    runtime.last_initiative_score_at = now

    # Sprint 7 — feedback bias adjusts the effective threshold.
    # bias > 1.0 (user engaged) → lower threshold (easier to surface).
    # bias < 1.0 (user dismissed) → raise threshold (harder).
    # Reach for the bias only when enabled to avoid an unnecessary DB
    # hit on the hot path when the feature is off.
    threshold = base_threshold
    if getattr(settings, "companion_feedback_bias_enabled", False):
        owner = getattr(runtime, "owner_user_id", "") or ""
        if owner:
            try:
                from augmentum.companion_runtime import feedback as _fb
                bias = await _fb.aggregate_bias(runtime, user_id=owner)
                # bias range [0.5, 2.0]; dividing threshold by bias means
                # higher bias = lower effective threshold (easier surface).
                threshold = base_threshold / max(bias, 0.1)
            except Exception:
                log.debug("initiative_bias_lookup_failed", exc_info=True)

    # Skip writing if the score is effectively zero — keeps the queue
    # from filling with all-zero rows during idle periods.
    if proposal.score < 0.05:
        return proposal
    row_id = await enqueue(runtime, proposal)
    if proposal.score >= threshold and row_id is not None:
        await runtime.bus.publish_topic(
            "initiative.surfaced",
            {"id": row_id, "kind": proposal.kind, "score": proposal.score},
            source_companion_id=runtime.companion_id,
        )
    return proposal


__all__ = ["Proposal", "enqueue", "score", "step"]
