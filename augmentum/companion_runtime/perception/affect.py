"""Affect perception loop (Lane 2 §1).

Builds per-user baseline distributions over personality facet activations,
compares short-window patterns against longer-window references, and
detects texture shifts (displacement, narrowing, asymmetry) that earn
becoming candidate noticings.

This module does NOT voice noticings on its own. Candidates flow into the
journal at confidence='early'/'normal' and may eventually graduate to
the relationship doc + companion_observations table. Initiative scorer
(``behavior/initiative.py``) reads those graduated noticings and decides
whether to surface — with hard rate-limits (≤ 1 unprompted care-surface
per day, 4h cooldown after the user has just opened up, no time-of-day
weighting, etc.).

Sprint F ships:
  - build_baseline (single window)
  - rebuild_all_baselines (nightly job)
  - perceive() — single perception step producing 0-1 CandidateNotice
  - the texture math (displacement, narrowing, asymmetry, novel_week)

History: previously lived at ``augmentum/companion_runtime/perception.py``
as a module sibling to the ``perception/`` package. Python's package
resolution shadowed the file, making every export here unreachable by
any import path — silently breaking the whole affect pipeline. Moved
2026-06-05 as Phase 0 of the verbs architecture rollout (see
``docs/superpowers/specs/2026-06-05-companion-verbs-architecture-design.md``
Phase 0). New canonical path:
``augmentum.companion_runtime.perception.affect``.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Window definitions (Lane 2 §1.2)
WINDOWS_DAYS = (7, 30, 180)

# Half-lives per window (~half the window for exponential decay weighting)
WINDOW_HALF_LIFE_HOURS = {7: 12, 30 * 24 // 24 * 7: 24 * 3.5, 30: 24 * 14, 180: 24 * 90}
# Practical map — keyed by window_days
HALF_LIFE_HOURS = {7: 24 * 3.5, 30: 24 * 14, 180: 24 * 90}

# Trust gate — below this turn_count for the 30d baseline, no
# perception-loop surfacing (Lane 2 §1.3).
TRUST_GATE_MIN_TURNS = 60

# Texture detection minimums
MIN_TURNS_FOR_SHORT_WINDOW = 12
DISPLACEMENT_SIGMA_THRESHOLD = -1.5
NARROWING_RATIO_THRESHOLD = 0.7
ASYMMETRY_RATIO_THRESHOLD = 1.6
REQUIRED_TEXTURES = 2  # 2-of-3 textures must concurrently fire


# Facet clusters (Lane 2 §1.5(a))
WARM_CLUSTER: frozenset[str] = frozenset({
    "warm", "playful", "delighted", "curious", "openhanded", "alert",
})
HARD_CLUSTER: frozenset[str] = frozenset({
    "unsure", "not_okay", "tired", "withholding", "frustrated", "still",
})
COGNITIVE_CLUSTER: frozenset[str] = frozenset({
    "rigorous", "decisive", "exploratory", "patient", "curious",
    "skeptical", "contemplative",
})
AFFECT_CLUSTER: frozenset[str] = WARM_CLUSTER | HARD_CLUSTER


@dataclass(frozen=True, slots=True)
class BaselineDistribution:
    user_id: str
    companion_id: str
    window_days: int
    facet_mean: dict[str, float] = field(default_factory=dict)
    facet_stddev: dict[str, float] = field(default_factory=dict)
    activation_density: float = 0.0
    turn_count: int = 0


@dataclass(frozen=True, slots=True)
class CandidateNotice:
    """Output of a perception step. Zero or one per user per night.

    Written to the journal as ``entry_type='noticing'`` with
    ``confidence='normal'``; consolidation may bump to 'firm' on
    repetition (Lane 2 §2.1).
    """
    user_id: str
    companion_id: str
    textures: list[tuple[str, float]]
    active_hard_facets: list[str]
    contextual_memory_ids: list[str]
    contextual_memory_summary: str
    confidence: float
    detected_at: float


# ── Baseline build ───────────────────────────────────────────────────

async def build_baseline(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    window_days: int,
) -> BaselineDistribution:
    """Query personality_facet_activations and aggregate into a baseline.

    SQL aggregation: one GROUP BY query per window. Results stored in
    companion_affect_baselines (migration 164).
    """
    backend = runtime.backend
    cutoff_clause = f"datetime('now', '-{int(window_days)} days')"

    cur = await backend.conn.execute(
        f"""
        SELECT facet, AVG(intensity) AS mean, COUNT(*) AS n
        FROM personality_facet_activations
        WHERE user_id = ? AND companion_id = ?
          AND activated_at >= {cutoff_clause}
        GROUP BY facet
        """,
        (user_id, runtime.companion_id),
    )
    rows = await cur.fetchall()
    facet_mean: dict[str, float] = {}
    facet_n: dict[str, int] = {}
    for r in rows:
        facet_mean[r[0]] = float(r[1] or 0.0)
        facet_n[r[0]] = int(r[2] or 0)

    # Stddev — separate pass; cheap given the small facet vocabulary.
    facet_stddev: dict[str, float] = {}
    for facet, mean in facet_mean.items():
        cur = await backend.conn.execute(
            f"""
            SELECT intensity FROM personality_facet_activations
            WHERE user_id = ? AND companion_id = ? AND facet = ?
              AND activated_at >= {cutoff_clause}
            """,
            (user_id, runtime.companion_id, facet),
        )
        intensities = [float(r[0]) for r in await cur.fetchall()]
        if len(intensities) >= 2:
            m = sum(intensities) / len(intensities)
            var = sum((x - m) ** 2 for x in intensities) / (len(intensities) - 1)
            facet_stddev[facet] = math.sqrt(var)
        else:
            facet_stddev[facet] = 0.0

    cur = await backend.conn.execute(
        f"""
        SELECT COUNT(DISTINCT turn_id) FROM personality_facet_activations
        WHERE user_id = ? AND companion_id = ?
          AND activated_at >= {cutoff_clause}
          AND turn_id IS NOT NULL
        """,
        (user_id, runtime.companion_id),
    )
    turn_row = await cur.fetchone()
    turn_count = int(turn_row[0]) if turn_row and turn_row[0] is not None else 0

    activation_density = (
        sum(facet_n.values()) / max(turn_count, 1)
        if turn_count > 0 else 0.0
    )

    # Persist into companion_affect_baselines (UPSERT pattern).
    await backend.conn.execute(
        """
        INSERT INTO companion_affect_baselines
          (user_id, companion_id, window_days, facet_mean_json,
           facet_stddev_json, activation_density, turn_count,
           last_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT (user_id, companion_id, window_days) DO UPDATE SET
          facet_mean_json = excluded.facet_mean_json,
          facet_stddev_json = excluded.facet_stddev_json,
          activation_density = excluded.activation_density,
          turn_count = excluded.turn_count,
          last_updated_at = excluded.last_updated_at
        """,
        (
            user_id, runtime.companion_id, window_days,
            json.dumps(facet_mean), json.dumps(facet_stddev),
            activation_density, turn_count,
        ),
    )
    await backend.conn.commit()

    return BaselineDistribution(
        user_id=user_id,
        companion_id=runtime.companion_id,
        window_days=window_days,
        facet_mean=facet_mean,
        facet_stddev=facet_stddev,
        activation_density=activation_density,
        turn_count=turn_count,
    )


async def load_baseline(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    window_days: int,
) -> BaselineDistribution | None:
    """Load a stored baseline. Returns None if it hasn't been built yet."""
    backend = runtime.backend
    cur = await backend.conn.execute(
        """
        SELECT facet_mean_json, facet_stddev_json, activation_density, turn_count
        FROM companion_affect_baselines
        WHERE user_id = ? AND companion_id = ? AND window_days = ?
        """,
        (user_id, runtime.companion_id, window_days),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    try:
        mean = json.loads(row[0] or "{}")
        stddev = json.loads(row[1] or "{}")
    except json.JSONDecodeError:
        mean, stddev = {}, {}
    return BaselineDistribution(
        user_id=user_id,
        companion_id=runtime.companion_id,
        window_days=window_days,
        facet_mean={k: float(v) for k, v in mean.items()},
        facet_stddev={k: float(v) for k, v in stddev.items()},
        activation_density=float(row[2] or 0.0),
        turn_count=int(row[3] or 0),
    )


async def rebuild_all_baselines(runtime: CompanionRuntime) -> dict[str, int]:
    """Nightly job — rebuild all three windows for every user that has
    any activations recorded with this companion. Returns
    ``{user_id: turn_count_30d}`` for monitoring.
    """
    backend = runtime.backend
    cur = await backend.conn.execute(
        """
        SELECT DISTINCT user_id FROM personality_facet_activations
        WHERE companion_id = ? AND user_id != ''
        """,
        (runtime.companion_id,),
    )
    user_ids = [r[0] for r in await cur.fetchall()]
    result: dict[str, int] = {}
    for user_id in user_ids:
        try:
            for w in WINDOWS_DAYS:
                bl = await build_baseline(runtime, user_id=user_id, window_days=w)
                if w == 30:
                    result[user_id] = bl.turn_count
        except Exception:
            log.exception("rebuild_baseline_failed", user_id=user_id)
    log.info("baselines_rebuilt", users=len(result), companion_id=runtime.companion_id)
    return result


# ── Texture detection (Lane 2 §1.5) ──────────────────────────────────

def _cluster_sum(facet_mean: dict[str, float], cluster: frozenset[str]) -> float:
    return sum(facet_mean.get(f, 0.0) for f in cluster)


def displacement(short: BaselineDistribution, long_: BaselineDistribution) -> float:
    """Warm-vs-hard axis displacement. Positive = warmer than baseline,
    negative = harder. Z-score-ish; pooled stddev across both clusters.
    """
    short_warm = _cluster_sum(short.facet_mean, WARM_CLUSTER)
    short_hard = _cluster_sum(short.facet_mean, HARD_CLUSTER)
    long_warm = _cluster_sum(long_.facet_mean, WARM_CLUSTER)
    long_hard = _cluster_sum(long_.facet_mean, HARD_CLUSTER)
    short_axis = short_warm - short_hard
    long_axis = long_warm - long_hard
    pooled = math.sqrt(
        sum(long_.facet_stddev.get(f, 0.0) ** 2 for f in AFFECT_CLUSTER) / max(len(AFFECT_CLUSTER), 1)
    )
    if pooled == 0.0:
        return 0.0
    return (short_axis - long_axis) / pooled


def narrowing(dist: BaselineDistribution) -> float:
    """Entropy of the activation distribution. Lower = narrower."""
    total = sum(dist.facet_mean.values())
    if total <= 0.0:
        return 0.0
    probs = [v / total for v in dist.facet_mean.values() if v > 0]
    return -sum(p * math.log(p) for p in probs if p > 0)


def asymmetry(dist: BaselineDistribution) -> float:
    """Cognitive vs affect activation ratio. High = he's working through it
    cognitively while affect has gone flat."""
    cog = _cluster_sum(dist.facet_mean, COGNITIVE_CLUSTER)
    aff = _cluster_sum(dist.facet_mean, AFFECT_CLUSTER)
    if aff <= 0.001:
        return 0.0
    return cog / aff


# ── Perception step ──────────────────────────────────────────────────

async def perceive(
    runtime: CompanionRuntime,
    *,
    user_id: str,
) -> CandidateNotice | None:
    """Run once per night (or on demand). Returns at most one candidate
    concerning shift; None when nothing crosses the bar.

    Trust gate: if 30d baseline has fewer than ``TRUST_GATE_MIN_TURNS``
    turns, this function always returns None — Becca won't act like she
    knows him yet.
    """
    if not user_id:
        return None

    short = await load_baseline(runtime, user_id=user_id, window_days=7)
    medium = await load_baseline(runtime, user_id=user_id, window_days=30)
    long_ = await load_baseline(runtime, user_id=user_id, window_days=180)

    if short is None or medium is None or long_ is None:
        return None

    if medium.turn_count < TRUST_GATE_MIN_TURNS:
        return None

    if short.turn_count < MIN_TURNS_FOR_SHORT_WINDOW:
        return None

    # Texture measures
    season_disp = displacement(medium, long_)
    week_disp = displacement(short, medium)
    novel_week = week_disp - season_disp

    textures: list[tuple[str, float]] = []
    if novel_week < DISPLACEMENT_SIGMA_THRESHOLD:
        textures.append(("displacement", round(novel_week, 3)))

    medium_narrow = narrowing(medium)
    if medium_narrow > 0:
        ratio_narrow = narrowing(short) / medium_narrow
        if ratio_narrow < NARROWING_RATIO_THRESHOLD:
            textures.append(("narrowing", round(ratio_narrow, 3)))

    medium_asym = asymmetry(medium)
    if medium_asym > 0:
        ratio_asym = asymmetry(short) / medium_asym
        if ratio_asym > ASYMMETRY_RATIO_THRESHOLD:
            textures.append(("asymmetry", round(ratio_asym, 3)))

    if len(textures) < REQUIRED_TEXTURES:
        return None

    # The hard facets driving this shift
    hard_active: list[str] = []
    for facet in HARD_CLUSTER:
        s = short.facet_mean.get(facet, 0.0)
        m = medium.facet_mean.get(facet, 0.0)
        sd = medium.facet_stddev.get(facet, 0.0)
        if s > m + sd:
            hard_active.append(facet)

    # Memory cluster context — Sprint F ships the placeholder summary;
    # Sprint F+ wires personality_memory_associations.query_*.
    contextual_summary = ""
    contextual_ids: list[str] = []

    confidence = min(1.0, len(textures) / 3.0)

    return CandidateNotice(
        user_id=user_id,
        companion_id=runtime.companion_id,
        textures=textures,
        active_hard_facets=hard_active,
        contextual_memory_ids=contextual_ids,
        contextual_memory_summary=contextual_summary,
        confidence=confidence,
        detected_at=time.time(),
    )


__all__ = [
    "BaselineDistribution",
    "CandidateNotice",
    "WINDOWS_DAYS",
    "TRUST_GATE_MIN_TURNS",
    "build_baseline",
    "load_baseline",
    "rebuild_all_baselines",
    "displacement",
    "narrowing",
    "asymmetry",
    "perceive",
]
