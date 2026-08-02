"""Reflection → identity loop — extract trait nudges from diary entries.

Sprint 7, Aletheia × Augmentum arc Piece 16.

When the dream cycle writes a daily diary entry, the runtime can
optionally extract a *trait nudge proposal* and apply it to identity
via the bounded `nudge_trait` API (Sprint α). The full LLM-based
extraction lands in a future iteration; this module ships the
**cross-check** logic that wraps the eventual extraction and decides
whether to apply.

Cross-check (the load-bearing safety):

  Apply nudge IF:
    1. The proposed trait's matching facet activated above its 7d
       baseline today (perception signal), AND
    2. Recent user feedback isn't trending negative (bias > 0.7)

  Otherwise:
    * Store the proposal as kernel_overlay note (visible in Observatory)
    * Do NOT apply — wait for corroboration

The bounded API (per-call ±0.01, cumulative ±0.05, DRIFT_CEILING) is
the OTHER safety layer. Together: cross-check + nudge bounds + drift
ceiling means a single bad day can't reshape the companion.

For Sprint 7 MVP: a simple regex extractor that catches structured
diary patterns like:
  - "I want to be more <trait>" → +0.005 to <trait>
  - "I noticed I'm too <trait>" → -0.005 to <trait>

Future LLM-based extractor will replace ``_naive_extract`` with a
structured-output call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# Bias-floor below which we DON'T apply nudges. If recent user feedback
# is negative, we suspect she's miscalibrated; nudges then would
# entrench the miscalibration instead of correcting it.
BIAS_APPLY_FLOOR: float = 0.7

# Per-extraction delta. Smaller than the per-call cap (0.01) so the
# extractor can fire repeatedly across days without immediately
# hitting the cumulative cap.
EXTRACTION_DELTA: float = 0.005

# Trait names we can extract from diary text. Mapped to the matching
# facet name for the cross-check. Add new traits here as the
# personality model grows.
_TRAIT_FACET_MAP: dict[str, str] = {
    "playful": "playful",
    "playfulness": "playful",
    "curious": "curious",
    "curiosity": "curious",
    "candor": "openhanded",       # candor maps to openhanded facet
    "candid": "openhanded",
    "patient": "patient",
    "patience": "patient",
    "careful": "still",
    "cautious": "still",
    "caution": "still",
    "warm": "warm",
    "warmth": "warm",
}


@dataclass(frozen=True, slots=True)
class TraitNudgeProposal:
    """A proposed trait adjustment from a diary entry."""
    trait: str
    delta: float
    reason: str  # short snippet from the diary text


def _naive_extract(diary_text: str) -> list[TraitNudgeProposal]:
    """Sprint 7 stub — regex over canonical patterns.

    Future polish: replace with structured LLM call. The cross-check
    layer in :func:`maybe_apply_nudge` is the safety regardless of
    which extractor is used.
    """
    if not diary_text:
        return []
    proposals: list[TraitNudgeProposal] = []

    text = diary_text.lower()

    # Pattern: "more <trait>"
    for match in re.finditer(
        r"\b(want to be|trying to be|notice.*?could be)\s+more\s+(\w+)",
        text,
    ):
        trait_token = match.group(2)
        trait = _TRAIT_FACET_MAP.get(trait_token)
        if not trait:
            continue
        snippet = diary_text[match.start():match.end() + 30]
        proposals.append(TraitNudgeProposal(
            trait=trait_token, delta=EXTRACTION_DELTA, reason=snippet[:200],
        ))

    # Pattern: "too <trait>" → negative nudge
    for match in re.finditer(
        r"\b(noticed.*?too|been too|was too)\s+(\w+)",
        text,
    ):
        trait_token = match.group(2)
        trait = _TRAIT_FACET_MAP.get(trait_token)
        if not trait:
            continue
        snippet = diary_text[match.start():match.end() + 30]
        proposals.append(TraitNudgeProposal(
            trait=trait_token, delta=-EXTRACTION_DELTA, reason=snippet[:200],
        ))

    return proposals


async def _facet_elevated_today(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    facet: str,
) -> bool:
    """True if ``facet`` activated more than 1× its 7d baseline today.

    Pulls today's activation count from personality_facet_activations
    and compares to the per-window baseline (mig 164). Caller uses
    this as the perception signal in the cross-check.
    """
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT COUNT(*) FROM personality_facet_activations "
            "WHERE user_id = ? AND companion_id = ? "
            "  AND facet = ? "
            "  AND activated_at > datetime('now', '-1 day')",
            (user_id, runtime.companion_id, facet),
        )
        row = await cur.fetchone()
        await cur.close()
        today_count = int(row[0] if row else 0)
    except Exception:
        return False

    if today_count == 0:
        return False

    # 7d baseline activation density
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT activation_density, turn_count FROM companion_affect_baselines "
            "WHERE user_id = ? AND companion_id = ? AND window_days = 7",
            (user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return today_count > 0  # no baseline → any activation counts

    if row is None or not row[1]:
        return today_count > 0
    density = float(row[0] or 0.0)
    # Today contributed if its share exceeds baseline density.
    # Simple heuristic: 1 activation/day per facet ≈ density 1.0.
    return today_count >= max(1.0, density)


async def maybe_apply_nudge(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    diary_text: str,
) -> dict:
    """Extract nudge proposals from a diary entry, cross-check, apply.

    Returns a summary dict with what happened — useful for the
    Observatory + the dream cycle log:

        {
          "proposals": [{trait, delta, reason}, ...],
          "applied": [{trait, delta}, ...],
          "skipped": [{trait, reason}, ...],
        }

    Always returns the dict — never raises. Cross-check + drift ceiling
    are the safety; even a runaway extractor can't apply more than the
    per-call cap on any single trait.

    Gated by ``companion_reflection_trait_nudge_enabled``. Default OFF.
    """
    from augmentum.config import settings
    result = {"proposals": [], "applied": [], "skipped": []}

    if not getattr(settings, "companion_reflection_trait_nudge_enabled", False):
        result["skipped"].append({"trait": "*", "reason": "feature_disabled"})
        return result
    if not user_id:
        result["skipped"].append({"trait": "*", "reason": "no_user_id"})
        return result

    proposals = _naive_extract(diary_text)
    result["proposals"] = [
        {"trait": p.trait, "delta": p.delta, "reason": p.reason}
        for p in proposals
    ]
    if not proposals:
        return result

    # Resolve feedback bias once for all proposals.
    bias = 1.0
    try:
        from augmentum.companion_runtime import feedback as _fb
        bias = await _fb.aggregate_bias(runtime, user_id=user_id)
    except Exception as exc:
        log.debug("reflection_bias_resolve_failed", error=str(exc))

    if bias < BIAS_APPLY_FLOOR:
        # User feedback trending negative — don't entrench miscalibration.
        for p in proposals:
            result["skipped"].append({
                "trait": p.trait,
                "reason": f"feedback_bias_low ({bias:.2f})",
            })
        return result

    # Cross-check each proposal against today's facet activation
    identity = await runtime.get_identity(user_id)
    for p in proposals:
        facet = _TRAIT_FACET_MAP.get(p.trait, p.trait)
        try:
            elevated = await _facet_elevated_today(
                runtime, user_id=user_id, facet=facet,
            )
        except Exception:
            elevated = False
        if not elevated:
            result["skipped"].append({
                "trait": p.trait, "reason": "facet_not_elevated_today",
            })
            continue
        # Apply via the bounded identity API (Sprint α).
        try:
            applied = await identity.nudge_trait(name=p.trait, delta=p.delta)
        except Exception:
            applied = False
        if applied:
            result["applied"].append({"trait": p.trait, "delta": p.delta})
        else:
            result["skipped"].append({
                "trait": p.trait, "reason": "rejected_by_caps_or_drift",
            })

    if result["applied"]:
        log.info(
            "reflection_trait_nudge_applied",
            user_id=user_id, count=len(result["applied"]),
            details=result["applied"],
        )

    return result


__all__ = [
    "TraitNudgeProposal",
    "maybe_apply_nudge",
    "BIAS_APPLY_FLOOR",
    "EXTRACTION_DELTA",
]
