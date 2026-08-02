"""Judgment gate (L3) — the decision that prevents the echo machine.

Given one fused :class:`Insight`, decide HOW (or whether) to surface it:
``SILENT`` / ``FILE_FOR_PULL`` / ``SPEAK`` / ``ACT_WITH_CONSENT``. The whole point
of the pipeline lives here — the same data either becomes a nag or a moment purely
by this decision (the proactive-assistant research finding, design spec §1, §4).

The core (``decide_delivery``) is PURE: every input — the insight, the regret
multiplier, remaining interruption budget, whether she's already in conversation —
is passed in. So the full decision matrix is unit-testable with no DB, no clock,
no live engine. Adapters that read the regret multiplier from ``feedback.py`` and
the budget from ``budget.py`` live in the wiring brick, not here.

Doctrine encoded:
  * **Pull-first.** The default for anything worth keeping is the Today digest, not
    an interruption.
  * **Regret gates the costly act.** Filing for pull is cheap → governed by raw
    strength. Interrupting is expensive → governed by strength × the user's regret
    multiplier (``feedback.aggregate_bias``: <1 = they dismiss her, so the bar
    rises; >1 = they engage, so it relaxes).
  * **In-conversation is the cheapest channel.** When she's already talking with
    them, mentioning a worthwhile insight costs no interruption — the bar drops and
    no budget is spent.
  * **Interrupting spends budget.** Unsolicited SPEAK requires budget remaining;
    exhausted → it downgrades to pull. Structural anti-nag.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.companion_runtime.perception.insight import (
    ACT_WITH_CONSENT,
    FILE_FOR_PULL,
    SILENT,
    SPEAK,
    DeliveryDecision,
    Insight,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class JudgmentConfig:
    """Tunable thresholds for the gate. Defaults bias hard toward restraint.

    ``pull_floor`` — below this base score (value × confidence), an insight isn't
        even worth the digest; it's filed for recall only (SILENT).
    ``convo_bar`` — in-conversation, an effective score at/above this is worth
        mentioning (no interruption cost, no budget spent).
    ``push_bar`` — the effective score an UNSOLICITED interruption must clear, on
        top of being time-critical and within budget. Highest bar by design.
    """

    pull_floor: float = 0.30
    convo_bar: float = 0.45
    push_bar: float = 0.65


_DEFAULT_CONFIG = JudgmentConfig()


def config_from_settings(settings: object) -> JudgmentConfig:
    """Read threshold overrides from the settings object, falling back to the
    restraint-biased defaults. Never raises."""
    def _f(name: str, default: float) -> float:
        try:
            return float(getattr(settings, name, default))
        except (TypeError, ValueError):
            return default
    return JudgmentConfig(
        pull_floor=_f("companion_judgment_pull_floor", _DEFAULT_CONFIG.pull_floor),
        convo_bar=_f("companion_judgment_convo_bar", _DEFAULT_CONFIG.convo_bar),
        push_bar=_f("companion_judgment_push_bar", _DEFAULT_CONFIG.push_bar),
    )


def decide_delivery(
    insight: Insight,
    *,
    regret_multiplier: float = 1.0,
    budget_remaining: int = 0,
    in_conversation: bool = False,
    now: float | None = None,
    config: JudgmentConfig | None = None,
) -> DeliveryDecision:
    """Decide how to surface ``insight``. Pure — all context is passed in.

    ``regret_multiplier`` is ``feedback.aggregate_bias`` in [0.5, 2.0]: <1 means
    the user dismisses her output (raise the interrupt bar), >1 means they engage.
    ``budget_remaining`` is the interruption budget left this window.
    ``in_conversation`` True when she's already mid-exchange (no interruption cost).
    """
    cfg = config or _DEFAULT_CONFIG

    # Expired insights are dead — decay beat delivery. Filed for recall, not shown.
    if insight.expires_at is not None and now is not None and now > insight.expires_at:
        return DeliveryDecision(SILENT, "expired before it earned delivery")

    # A consequential action never auto-fires — it's proposed via the gated-offer
    # confirm (bounded autonomy). The gate routes it; the offer layer asks.
    if insight.suggested_action and insight.stakes != "trivial_reversible":
        return DeliveryDecision(
            ACT_WITH_CONSENT,
            f"proposes a {insight.stakes} action — route through gated confirm",
        )

    base = insight.base_score
    # Too weak to even occupy a line in the digest → file for recall only.
    if base < cfg.pull_floor:
        return DeliveryDecision(SILENT, f"base {base:.2f} < pull_floor {cfg.pull_floor:.2f}")

    # Effective score governs the COSTLY channels (speak/interrupt). Regret only
    # bites here — filing for pull stays cheap and regret-independent.
    effective = base * max(0.0, regret_multiplier)

    # In-conversation: cheapest possible delivery. Worth mentioning → say it, no
    # budget spent, no time-critical requirement (the cost of speaking is ~zero
    # when they're already here).
    if in_conversation and effective >= cfg.convo_bar:
        return DeliveryDecision(
            SPEAK, f"in-conversation, effective {effective:.2f} ≥ convo_bar {cfg.convo_bar:.2f}",
        )

    # Unsolicited interruption — the highest bar: time-critical AND clears push_bar
    # AND budget remains. Any miss → it falls back to the pull surface.
    if insight.time_critical and effective >= cfg.push_bar and budget_remaining > 0:
        return DeliveryDecision(
            SPEAK,
            f"time-critical, effective {effective:.2f} ≥ push_bar {cfg.push_bar:.2f}, "
            f"budget {budget_remaining}",
            spent_budget=True,
        )

    # Default: worth keeping, not worth interrupting → glanceable on pull.
    why = "pull-first default"
    if insight.time_critical and budget_remaining <= 0:
        why = "time-critical but interruption budget exhausted → pull"
    elif insight.time_critical and effective < cfg.push_bar:
        why = f"time-critical but effective {effective:.2f} < push_bar {cfg.push_bar:.2f} → pull"
    return DeliveryDecision(FILE_FOR_PULL, why)
