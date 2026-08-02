"""Reward signal channels — convert user reactions to berry movements.

Four channels per the growth-loop spec §4 (Reward signal sources):

  * **Explicit** — thumbs / save / share / dismiss / "stop". Highest
    weight, cleanest signal.
  * **Implicit** — engagement time, follow-up, repeat pattern. Medium
    weight, noisy.
  * **Affect** — BOM-substrate-derived emotional state during/after.
    Wired in Phase 5 when the substrate is queryable; placeholder here.
  * **Counterfactual** — anticipated answer was used without the user
    asking. Highest moat value; lowest signal density. Phase 5.

Phase 1 ships explicit + implicit only. Each entry point:

  1. Maps the signal → a berry delta (positive = earn, negative = veto).
  2. Calls the appropriate :class:`Economy` method, writing through to
     the transaction log with the source growth_log_id.
  3. Returns the resulting balance for the caller to surface.

Saturation curves, novelty bonuses, deferred banking for artifacts —
all Phase 5. This file is the dispatch shape, not the calibration
engine. Adjust the constants when empirical numbers land.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.companion.growth.economy import EarnResult, Economy
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Calibration constants ────────────────────────────────────────────
#
# Berry deltas for each explicit signal kind. These are starting
# placeholders — the spec §"Calibration brittleness" calls out that
# these need empirical tuning in Phase 5. Negative values are penalties
# applied via Economy.veto so lifetime totals stay accurate.

_EXPLICIT_DELTAS: dict[str, float] = {
    "thumbs_up":        +20.0,
    "thumbs_down":      -20.0,
    "save":             +30.0,
    "share":            +50.0,
    "more_like_this":   +25.0,
    "dismiss":          -3.0,
    "stop":             -15.0,
    "explicit_thanks":  +10.0,
}

# Implicit signals are noisier — lower magnitude.
_IMPLICIT_DELTAS: dict[str, float] = {
    "engaged_short":    +2.0,    # <30s engagement
    "engaged_medium":   +5.0,    # 30s..2min
    "engaged_long":     +10.0,   # 2min+
    "followed_up":      +8.0,
    "repeat_pattern":   +5.0,    # similar action requested again later
    "ignored":          0.0,     # no penalty; no signal
}


@dataclass(slots=True)
class RewardOutcome:
    """Result of applying a reward signal."""

    ok: bool
    delta: float
    berries_after: float
    reason: str = ""


async def apply_explicit(
    economy: Economy,
    *,
    signal: str,
    growth_log_id: str | None = None,
    evidence_ref: str = "",
) -> RewardOutcome:
    """Apply an explicit user signal (thumbs / save / dismiss / etc.)."""
    delta = _EXPLICIT_DELTAS.get(signal)
    if delta is None:
        return RewardOutcome(
            ok=False, delta=0.0, berries_after=0.0,
            reason=f"unknown_explicit_signal:{signal}",
        )
    return await _apply_delta(
        economy, delta=delta,
        signal_kind="explicit",
        reason=f"explicit:{signal}",
        growth_log_id=growth_log_id,
        evidence_ref=evidence_ref,
    )


async def apply_implicit(
    economy: Economy,
    *,
    signal: str,
    growth_log_id: str | None = None,
    evidence_ref: str = "",
) -> RewardOutcome:
    """Apply an implicit signal (engagement timer, follow-up, etc.)."""
    delta = _IMPLICIT_DELTAS.get(signal)
    if delta is None:
        return RewardOutcome(
            ok=False, delta=0.0, berries_after=0.0,
            reason=f"unknown_implicit_signal:{signal}",
        )
    return await _apply_delta(
        economy, delta=delta,
        signal_kind="implicit",
        reason=f"implicit:{signal}",
        growth_log_id=growth_log_id,
        evidence_ref=evidence_ref,
    )


async def apply_restraint_credit(
    economy: Economy,
    *,
    amount: float = 5.0,
    growth_log_id: str | None = None,
    evidence_ref: str = "",
) -> RewardOutcome:
    """Reward held-back action when noticed (catalog category M).

    The antidote to reward-hacking. Default 5 berries — small but real.
    """
    return await _apply_delta(
        economy, delta=amount,
        signal_kind="restraint",
        reason="restraint_credit",
        growth_log_id=growth_log_id,
        evidence_ref=evidence_ref,
    )


async def _apply_delta(
    economy: Economy,
    *,
    delta: float,
    signal_kind: str,
    reason: str,
    growth_log_id: str | None,
    evidence_ref: str,
) -> RewardOutcome:
    if delta == 0:
        # Real "no signal" — record nothing, don't churn the tx log.
        account = await economy.snapshot()
        return RewardOutcome(
            ok=True, delta=0.0,
            berries_after=account.berries,
            reason="no_op",
        )

    result: EarnResult
    if delta > 0:
        result = await economy.earn_berries(
            delta, signal_kind=signal_kind, reason=reason,
            growth_log_id=growth_log_id, evidence_ref=evidence_ref,
        )
    else:
        # Negative deltas use the veto path so they don't decrement
        # lifetime totals — vetoes are evidence about miscalibration,
        # not removal of trust that was genuinely earned.
        result = await economy.veto(
            abs(delta), reason=reason, evidence_ref=evidence_ref,
        )
    return RewardOutcome(
        ok=result.ok, delta=result.delta,
        berries_after=result.berries_after,
        reason=result.reason,
    )
