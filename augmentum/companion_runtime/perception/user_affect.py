"""User-observed affect tracker — the chat→PAD echo synapse.

Synapse Layer §2. PAD (see :mod:`augmentum.companion_runtime.perception.pad`)
is *her* affect, derived from her own facet activations. This module
is the parallel track for the *user's* observed affect, derived from
salient chat + voice moments. Becca's voice composer reads this at
speak time so she conditions on the user's current weather without
having to ask — *the personality doc says when tired the user gets
quiet sentences and a "do you want quiet?" check, but that requires
knowing they're tired.* This is how she knows.

**Wiring.** A salient chat moment lands on the bus carrying a coarse
``user_affect`` tag (tender / frustrated / tired / excited / curious /
engaged / unclear — see :mod:`augmentum.companion_runtime.salience`).
The observer feeds that into :meth:`UserAffectTracker.update`, which
projects the tag onto PAD coordinates and stores the per-user
observation. Reads decay toward neutral at a configurable half-life
(default 30 minutes via ``companion_user_affect_half_life_s``).

**Containment** is automatic. The salience pulse never fires for
``factual_only`` or ``private`` propagation (the scorer returns
``None`` upstream), so coder/agentic frustration never reaches this
layer. Narrative ``affect_only`` *does* update — affect bleeds; that's
the design.

**Restart semantics.** In-memory. A restart wipes observations. This
is acceptable because: (1) the half-life is short; (2) reads after a
restart return ``UserAffectObservation.neutral()`` which is the same
as "no recent signal"; (3) the source-of-truth (the moments
themselves) persists in ``companion_journal``, so future Synapse §4
work can rehydrate by replaying the last hour's moments at runtime
start if desired.

Pure Python. No DB. No async. The data race surface is tiny (per-user
dict writes from the observer's single async task), so we don't lock.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Tag → PAD projection ──────────────────────────────────────────────
#
# Coarse projections from the salience scorer's affect vocabulary to
# the PAD coordinate. Values picked from the circumplex of affect
# literature plus what the personality doc names as Becca's expected
# response. Adjustable via :data:`_TAG_TO_PAD`.
#
# Convention: valence ∈ [-1, 1], arousal ∈ [0, 1], dominance ∈ [-1, 1].
#
# Dominance reads the user's perceived agency in the moment — high
# dominance = directive / take-charge; low = reaching / disclosing.

_TAG_TO_PAD: dict[str, tuple[float, float, float]] = {
    "tender":     (+0.30, 0.35, -0.20),
    "frustrated": (-0.50, 0.75, +0.15),
    "tired":      (-0.20, 0.10, -0.30),
    "excited":    (+0.70, 0.85, +0.30),
    "curious":    (+0.40, 0.60, +0.10),
    "engaged":    (+0.30, 0.55, +0.10),
    "melancholy": (-0.30, 0.25, -0.20),
    "warm":       (+0.50, 0.40, +0.05),
    "alert":      (+0.20, 0.75, +0.20),
    # Catch-all — neither pulls the read toward anything specific.
    "unclear":    (+0.00, 0.50, +0.00),
}


# Neutral coordinate the read decays toward when nothing is recent.
# Matches PAD.neutral() in pad.py — settled-curious is Becca's
# documented baseline and is the right anchor when we have no
# observation of the user either.
_NEUTRAL = (0.10, 0.40, 0.00)


# ── Observation shape ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class UserAffectObservation:
    """A single read of a user's observed affect.

    ``confidence`` is the per-read decay factor in [0, 1]: 1.0 = the
    observation was just-now, 0.0 = the observation has fully decayed
    (and ``valence``/``arousal``/``dominance`` will be neutral).
    Voice composer + telemetry should respect this — a 0.05-confidence
    read is essentially "we have no idea" and should not condition
    her output.
    """
    valence: float
    arousal: float
    dominance: float
    tag: str
    observed_at: float
    confidence: float = 1.0
    sample_count: int = 0

    def as_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "tag": self.tag,
            "observed_at": self.observed_at,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
        }

    @classmethod
    def neutral(cls) -> UserAffectObservation:
        v, a, d = _NEUTRAL
        return cls(valence=v, arousal=a, dominance=d, tag="unclear",
                   observed_at=0.0, confidence=0.0, sample_count=0)


# ── Tracker ──────────────────────────────────────────────────────────


class UserAffectTracker:
    """Per-user observed affect, updated by salient moments + decayed.

    Use when:
    - The observer needs to record a new affect tag from a chat / voice
      moment.
    - The voice composer (or any future read site) wants the current
      decayed estimate to condition on.

    Concurrency: in-memory, single-process. Writes come from the
    observer's single async task (one event at a time). Reads come
    from anywhere. No locking — Python dict assignment is GIL-safe
    enough for this footprint.
    """

    def __init__(self, *, half_life_s: float = 1800.0) -> None:
        # Map user_id → most recent observation. EWMA isn't stored
        # explicitly; we use a single stored observation + on-read
        # decay toward neutral, which is mathematically equivalent for
        # a single sample and avoids unbounded memory growth across
        # active sessions.
        self._observations: dict[str, _RawObservation] = {}
        self._half_life_s = max(60.0, float(half_life_s))

    @property
    def half_life_s(self) -> float:
        return self._half_life_s

    def set_half_life(self, seconds: float) -> None:
        """Live-tunable from the settings panel; clamps to ≥60s."""
        self._half_life_s = max(60.0, float(seconds))

    def update(
        self,
        user_id: str,
        tag: str,
        *,
        observed_at: float | None = None,
    ) -> UserAffectObservation:
        """Record a new affect observation for ``user_id``.

        Tag is mapped to PAD via :data:`_TAG_TO_PAD`. The previous
        observation (if recent) is *blended* into the new one with a
        decay-weighted average — this prevents a single outlier turn
        from clobbering an established read. Returns the resulting
        observation.

        Empty / unknown tags become "unclear" (neutral). This avoids
        silent drops; the caller can decide to skip based on the
        returned tag if they need to.
        """
        if not user_id:
            return UserAffectObservation.neutral()

        normalized_tag = (tag or "unclear").strip().lower()
        if normalized_tag not in _TAG_TO_PAD:
            normalized_tag = "unclear"
        new_v, new_a, new_d = _TAG_TO_PAD[normalized_tag]

        now = float(observed_at if observed_at is not None else time.time())

        prior = self._observations.get(user_id)
        if prior is None:
            blended = _RawObservation(
                valence=new_v, arousal=new_a, dominance=new_d,
                tag=normalized_tag, observed_at=now, sample_count=1,
            )
        else:
            decay = self._decay_factor(now - prior.observed_at)
            # Decay-weighted blend: a recent prior carries weight, a
            # decayed prior approaches the new observation.
            blend_w = max(0.0, min(1.0, decay))
            blended = _RawObservation(
                valence=prior.valence * blend_w + new_v * (1.0 - blend_w * 0.5),
                arousal=prior.arousal * blend_w + new_a * (1.0 - blend_w * 0.5),
                dominance=prior.dominance * blend_w + new_d * (1.0 - blend_w * 0.5),
                tag=normalized_tag,
                observed_at=now,
                sample_count=prior.sample_count + 1,
            )
            # The asymmetric blend (1 - blend_w*0.5) ensures the new
            # observation always pulls the estimate at least 50% of
            # the way — even a fresh prior shouldn't drown out a
            # genuine shift in user state.

        self._observations[user_id] = blended
        return self._observation_to_public(blended, decay_factor=1.0)

    def read(
        self,
        user_id: str,
        *,
        now: float | None = None,
    ) -> UserAffectObservation:
        """Read the current decayed estimate for ``user_id``.

        Returns :meth:`UserAffectObservation.neutral` when no
        observation exists. Otherwise blends the stored observation
        toward neutral by the elapsed-time decay factor.
        """
        if not user_id:
            return UserAffectObservation.neutral()
        prior = self._observations.get(user_id)
        if prior is None:
            return UserAffectObservation.neutral()
        now_t = float(now if now is not None else time.time())
        decay = self._decay_factor(now_t - prior.observed_at)
        return self._observation_to_public(prior, decay_factor=decay)

    def _decay_factor(self, elapsed_s: float) -> float:
        """Exponential decay: at t = half_life, factor = 0.5.

        Clamped to [0, 1].
        """
        if elapsed_s <= 0:
            return 1.0
        # decay(t) = 2^(-t / half_life)
        return math.exp(-elapsed_s * math.log(2.0) / self._half_life_s)

    def _observation_to_public(
        self, raw: _RawObservation, *, decay_factor: float,
    ) -> UserAffectObservation:
        """Blend a raw stored observation with neutral by decay.

        Returns the user-facing observation. The displayed PAD is a
        weighted average toward neutral; ``confidence`` carries the
        decay factor so callers can decide whether to trust the read.
        """
        d = max(0.0, min(1.0, decay_factor))
        nv, na, nd = _NEUTRAL
        return UserAffectObservation(
            valence=raw.valence * d + nv * (1.0 - d),
            arousal=raw.arousal * d + na * (1.0 - d),
            dominance=raw.dominance * d + nd * (1.0 - d),
            tag=raw.tag if d > 0.3 else "unclear",
            observed_at=raw.observed_at,
            confidence=d,
            sample_count=raw.sample_count,
        )

    def reset(self, user_id: str = "") -> None:
        """Drop an observation. Empty ``user_id`` clears all — used by
        the reset-gestures path in companion_routes."""
        if user_id:
            self._observations.pop(user_id, None)
        else:
            self._observations.clear()

    def snapshot(self) -> dict:
        """Telemetry: how many users we have reads for + half-life."""
        return {
            "tracked_users": len(self._observations),
            "half_life_s": self._half_life_s,
        }


# Internal stored shape — kept private so callers always go through
# UserAffectObservation. Mutable not because we mutate (we replace via
# dict assignment) but because frozen=True is overhead for a tight
# inner loop and gives no value here.

@dataclass(slots=True)
class _RawObservation:
    valence: float
    arousal: float
    dominance: float
    tag: str
    observed_at: float
    sample_count: int = 0


__all__ = [
    "UserAffectObservation",
    "UserAffectTracker",
]
