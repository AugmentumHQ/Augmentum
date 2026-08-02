"""PAD projector — project facet activations onto a 3D affect coordinate.

Sprint 6, Aletheia × Augmentum arc Piece 3.

Augmentum's affect model is CAPS-style: hundreds of named facets fire
per turn with intensities, baselines tracked per (user, companion,
window). That richness is great for retrieval-time pattern matching
but unwieldy for downstream decision-making.

PAD (Pleasure / Arousal / Dominance) is the standard 3D summary
coordinate from affect psychology. We project the live facet
activations onto PAD via:

  valence    = mean(warm-cluster intensity) − mean(hard-cluster intensity)
  arousal    = activation_density / 7d_baseline_density, normalized
  dominance  = role.active − role.passive (from CompanionState)

Decays toward baseline at ~0.05/min naturally — old activations fall
out of the recency window and the projection cools.

Pure function. Reads from DB but never writes. Caller decides cadence;
typically the Observatory endpoint pulls once per page render, the
activity_selector pulls once per tick.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


# Facet clusters — match perception.py's existing cluster definitions
# so the warm/hard distinction is consistent across the substrate.
WARM_FACETS: frozenset[str] = frozenset({
    "warm", "playful", "delighted", "curious", "openhanded", "alert",
})
HARD_FACETS: frozenset[str] = frozenset({
    "unsure", "not_okay", "tired", "withholding", "frustrated", "still",
})

# Window over which to gather recent facet activations for PAD
# computation. Short enough that the projection responds to current
# state; long enough to dampen single-turn noise.
RECENT_WINDOW_MINUTES: int = 60


@dataclass(frozen=True, slots=True)
class PAD:
    """Project of current affect onto Pleasure / Arousal / Dominance."""
    valence: float        # [-1.0, 1.0]
    arousal: float        # [0.0, 1.0]
    dominance: float      # [-1.0, 1.0]
    sample_count: int     # how many activations contributed; 0 → defaults

    def as_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "sample_count": self.sample_count,
        }

    @classmethod
    def neutral(cls) -> PAD:
        """The default reading when no signal exists. Slightly
        positive-curious because that's Becca's documented baseline
        (personality doc §13: 'settled-curious')."""
        return cls(valence=0.1, arousal=0.4, dominance=0.0, sample_count=0)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


async def project_pad(
    backend: SQLiteBackend,
    *,
    user_id: str,
    companion_id: str = "becca",
    role_active: float = 0.0,
    role_passive: float = 1.0,
) -> PAD:
    """Compute the current PAD coordinate for ``user_id``.

    ``role_active`` / ``role_passive`` come from CompanionState (the
    role vector). Caller passes them in to keep this function pure.

    No-data path: returns :func:`PAD.neutral` when there are no recent
    activations and no baseline. Doesn't pretend to know affect when
    we genuinely don't.
    """
    if not user_id:
        return PAD.neutral()

    # 1. Pull recent facet activations (last RECENT_WINDOW_MINUTES)
    try:
        cur = await backend.conn.execute(
            f"""
            SELECT facet, intensity FROM personality_facet_activations
            WHERE user_id = ? AND companion_id = ?
              AND activated_at > datetime('now', '-{RECENT_WINDOW_MINUTES} minutes')
            """,
            (user_id, companion_id),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        return PAD.neutral()

    if not rows:
        # No recent signal — fall back to role-derived dominance only
        return PAD(
            valence=0.1,
            arousal=0.4,
            dominance=_clamp(role_active - role_passive, -1.0, 1.0),
            sample_count=0,
        )

    # 2. Valence — warm vs hard cluster intensities
    warm_sum = sum(float(r[1] or 0.0) for r in rows if (r[0] or "") in WARM_FACETS)
    hard_sum = sum(float(r[1] or 0.0) for r in rows if (r[0] or "") in HARD_FACETS)
    warm_count = sum(1 for r in rows if (r[0] or "") in WARM_FACETS)
    hard_count = sum(1 for r in rows if (r[0] or "") in HARD_FACETS)

    if warm_count + hard_count == 0:
        # Activations exist but none are affect-cluster — neutral valence
        valence = 0.0
    else:
        # Mean intensity within each cluster, then difference
        warm_mean = warm_sum / max(warm_count, 1)
        hard_mean = hard_sum / max(hard_count, 1)
        # Normalize: difference of means in [0, 1] each → diff in [-1, 1]
        valence = _clamp(warm_mean - hard_mean, -1.0, 1.0)

    # 3. Arousal — activation density vs baseline
    try:
        cur = await backend.conn.execute(
            """
            SELECT activation_density FROM companion_affect_baselines
            WHERE user_id = ? AND companion_id = ? AND window_days = 7
            """,
            (user_id, companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
        baseline_density = float(row[0]) if row and row[0] else 0.0
    except Exception:
        baseline_density = 0.0

    # Current density = facets per minute in the window
    current_density = len(rows) / max(RECENT_WINDOW_MINUTES, 1)

    if baseline_density > 0:
        ratio = current_density / baseline_density
        # Map ratio to arousal: 1.0× baseline → 0.5, 2× → 0.75, 0× → 0.0
        # Use a sigmoid-flavored mapping.
        arousal = _clamp(1.0 - math.exp(-ratio), 0.0, 1.0)
    else:
        # No baseline yet — use raw density with a gentle prior
        arousal = _clamp(0.3 + current_density * 2.0, 0.0, 1.0)

    # 4. Dominance — directly from role vector
    dominance = _clamp(role_active - role_passive, -1.0, 1.0)

    return PAD(
        valence=valence,
        arousal=arousal,
        dominance=dominance,
        sample_count=len(rows),
    )


__all__ = ["PAD", "project_pad", "WARM_FACETS", "HARD_FACETS"]
