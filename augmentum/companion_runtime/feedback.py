"""Feedback bias — learn what kinds of notes the user values.

Sprint 7, Aletheia × Augmentum arc Piece 15.

When the user acts on a note (Pull it together / Good to know / Mute
this topic), the action endpoint writes a row to
``companion_note_feedback``. This module reads that history and computes
a per-topic-signature bias multiplier the initiative scoring uses to
boost topics the user engages with and damp topics they dismiss.

Bias function — simple, robust:

  recent_window = 14 days
  surfaced_weight = +1.0  (user pulled it together / opened a ref)
  acknowledged_weight = +0.2  (user accepted it without acting)
  muted_weight = -1.5  (user explicitly suppressed the thread)

  raw_score = sum(weight × decay(age))  per kind
  multiplier = clamp(1.0 + raw_score / N, 0.5, 2.0)

where ``decay(age) = exp(-age_days / 7)`` so recent feedback dominates,
and N is a normalizer (default 5 — five strong signals to fully
saturate).

Floor 0.5 / ceiling 2.0: never zero out a topic from feedback alone
(mute is the hard switch), never blow it up to infinity. **Mute is
the explicit kill — feedback is the soft signal.**

Multiplier resets toward 1.0 over time (the decay factor is what does
this — no separate reset code needed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Window for feedback aggregation. Older entries don't count.
FEEDBACK_WINDOW_DAYS: int = 14

# Per-kind weights.
KIND_WEIGHTS: dict[str, float] = {
    "surfaced": 1.0,         # pull it together / opened ref / resonate
    "acknowledged": 0.2,     # good to know
    "dismissed": -0.3,       # not now, not relevant — softer than mute
    "muted": -1.5,           # mute this topic
}

# Normalizer — five strong signals worth of evidence saturates.
NORMALIZER: float = 5.0

# Bias bounds. Floor never zero (mute is the hard switch), ceiling
# never unboundedly large.
BIAS_FLOOR: float = 0.5
BIAS_CEILING: float = 2.0


@dataclass(frozen=True, slots=True)
class TopicFeedback:
    """Aggregate feedback for a single topic signature."""
    multiplier: float
    surfaced_count: int
    acknowledged_count: int
    dismissed_count: int
    muted_count: int


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


async def record(
    runtime: CompanionRuntime,
    *,
    note_id: int,
    user_id: str,
    kind: str,
) -> bool:
    """Write a feedback row. Idempotent at the user-intent level —
    the same user clicking 'acknowledged' twice writes two rows,
    which the bias function down-weights via the decay.

    Best-effort: never raises, never blocks the action it's recording.
    """
    if not note_id or not user_id or not kind:
        return False
    try:
        await runtime.backend.conn.execute(
            "INSERT INTO companion_note_feedback "
            "(note_id, user_id, companion_id, kind) "
            "VALUES (?, ?, ?, ?)",
            (note_id, user_id, runtime.companion_id, kind),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning(
            "feedback_record_failed",
            note_id=note_id, user_id=user_id, kind=kind, exc_info=True,
        )
        return False

    # Best-effort Today regen — note feedback is one of the meaningful
    # event surfaces. Debounced inside maybe_regenerate so a flurry of
    # clicks doesn't thrash the LLM. Never blocks the feedback write.
    try:
        from augmentum.companion_runtime import today as _today
        await _today.maybe_regenerate(runtime, user_id=user_id)
    except Exception:
        log.debug("today_regen_from_feedback_failed", exc_info=True)
    return True


async def aggregate_bias(
    runtime: CompanionRuntime,
    *,
    user_id: str,
) -> float:
    """Return a global bias multiplier in [BIAS_FLOOR, BIAS_CEILING].

    Aggregates all feedback in the window — this is the GLOBAL bias
    that says "the user generally engages / generally dismisses
    autonomous output." Per-topic bias is a future polish; the global
    version is enough to modulate initiative thresholds.

    Score formula: weighted sum with exponential decay over 7d
    half-life. ``raw_score = Σ kind_weight × exp(-age_days / 7)``.
    Normalized by NORMALIZER (5 signals to saturate). Clamped.
    """
    if not user_id:
        return 1.0

    try:
        cur = await runtime.backend.conn.execute(
            f"""
            SELECT kind,
                   (julianday('now') - julianday(recorded_at)) AS age_days
            FROM companion_note_feedback
            WHERE user_id = ? AND companion_id = ?
              AND recorded_at > datetime('now', '-{FEEDBACK_WINDOW_DAYS} days')
            """,
            (user_id, runtime.companion_id),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.warning("feedback_aggregate_failed", exc_info=True)
        return 1.0

    if not rows:
        return 1.0

    raw_score = 0.0
    for kind, age_days in rows:
        weight = KIND_WEIGHTS.get(str(kind or ""), 0.0)
        age = max(0.0, float(age_days or 0.0))
        decay = math.exp(-age / 7.0)  # half-life ~5d
        raw_score += weight * decay

    multiplier = 1.0 + (raw_score / NORMALIZER)
    return _clamp(multiplier, BIAS_FLOOR, BIAS_CEILING)


async def feedback_summary(
    runtime: CompanionRuntime,
    *,
    user_id: str,
) -> TopicFeedback:
    """Read-only view of the user's recent feedback distribution.
    Used by Observatory to render 'how she's been received' charts.
    """
    if not user_id:
        return TopicFeedback(
            multiplier=1.0,
            surfaced_count=0,
            acknowledged_count=0,
            dismissed_count=0,
            muted_count=0,
        )

    counts = {"surfaced": 0, "acknowledged": 0, "dismissed": 0, "muted": 0}
    try:
        cur = await runtime.backend.conn.execute(
            f"""
            SELECT kind, COUNT(*) FROM companion_note_feedback
            WHERE user_id = ? AND companion_id = ?
              AND recorded_at > datetime('now', '-{FEEDBACK_WINDOW_DAYS} days')
            GROUP BY kind
            """,
            (user_id, runtime.companion_id),
        )
        for kind, count in await cur.fetchall():
            if kind in counts:
                counts[kind] = int(count)
        await cur.close()
    except Exception:
        log.warning("feedback_counts_query_failed", exc_info=True)

    multiplier = await aggregate_bias(runtime, user_id=user_id)
    return TopicFeedback(
        multiplier=multiplier,
        surfaced_count=counts["surfaced"],
        acknowledged_count=counts["acknowledged"],
        dismissed_count=counts["dismissed"],
        muted_count=counts["muted"],
    )


__all__ = [
    "TopicFeedback",
    "record",
    "aggregate_bias",
    "feedback_summary",
    "FEEDBACK_WINDOW_DAYS",
    "BIAS_FLOOR",
    "BIAS_CEILING",
]
