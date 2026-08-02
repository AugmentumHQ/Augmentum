"""Pipeline orchestrator — fuse → judge → route, the loop end to end.

Ties the layers together: gather candidate insights (L1+L2 ``fusion.fuse``), run
each through the judgment gate (L3 ``judgment.decide_delivery``) threading ONE
interruption budget across the batch, then route each decision to a sink. Every
dependency is injected (regret multiplier, budget, sink) so the whole loop is
unit-testable with no runtime, DB, or model — the live adapter that resolves the
real regret/budget/snapshot is a thin wrapper the runtime adds in the wiring brick.

Routing contract — a :class:`PerceptionSink` does the surface-specific work:
  * SILENT          → nothing (the insight stays recallable via memory, not here)
  * FILE_FOR_PULL   → ``sink.file_for_pull`` (lands in the Today digest)
  * SPEAK           → ``sink.speak`` (in-conversation mention or budgeted interrupt)
  * ACT_WITH_CONSENT→ ``sink.propose_action`` (the gated-offer confirm)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from augmentum.companion_runtime.perception.fusion import FusionContext, fuse
from augmentum.companion_runtime.perception.insight import (
    ACT_WITH_CONSENT,
    FILE_FOR_PULL,
    SPEAK,
    DeliveryDecision,
    Insight,
)
from augmentum.companion_runtime.perception.judgment import (
    JudgmentConfig,
    decide_delivery,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@runtime_checkable
class PerceptionSink(Protocol):
    """Where delivered insights go. Implementations are surface-specific (Today
    digest, the becca speak path, the gated-offer layer). All async, best-effort —
    the orchestrator never lets a sink error abort the batch."""

    async def file_for_pull(self, insight: Insight, decision: DeliveryDecision) -> None: ...
    async def speak(self, insight: Insight, decision: DeliveryDecision) -> None: ...
    async def propose_action(self, insight: Insight, decision: DeliveryDecision) -> None: ...


def run_perception(
    ctx: FusionContext,
    *,
    regret_multiplier: float = 1.0,
    budget_remaining: int = 0,
    config: JudgmentConfig | None = None,
) -> list[tuple[Insight, DeliveryDecision]]:
    """Pure core: fuse candidates, then decide each — threading ONE budget across
    the batch so the strongest insights claim interruptions first and a single pass
    can't over-spend. Returns (insight, decision) pairs in strength order."""
    insights = fuse(ctx)  # already sorted strongest-first, deduped by shape
    out: list[tuple[Insight, DeliveryDecision]] = []
    remaining = max(0, int(budget_remaining))
    for insight in insights:
        decision = decide_delivery(
            insight,
            regret_multiplier=regret_multiplier,
            budget_remaining=remaining,
            in_conversation=ctx.in_conversation,
            now=ctx.now,
            config=config,
        )
        if decision.spent_budget:
            remaining -= 1
        out.append((insight, decision))
    return out


async def _route(sink: PerceptionSink, insight: Insight, decision: DeliveryDecision) -> bool:
    """Send one decision to its sink method. Returns True if a delivery happened
    (i.e. an interruption-budget unit should actually be charged). Never raises."""
    try:
        if decision.channel == FILE_FOR_PULL:
            await sink.file_for_pull(insight, decision)
        elif decision.channel == SPEAK:
            await sink.speak(insight, decision)
        elif decision.channel == ACT_WITH_CONSENT:
            await sink.propose_action(insight, decision)
        else:  # SILENT — nothing to deliver
            return False
    except Exception:  # noqa: BLE001 — a sink failure can't abort the batch
        log.warning("perception_sink_failed", channel=decision.channel, exc_info=True)
        return False
    return True


async def dispatch(
    decisions: list[tuple[Insight, DeliveryDecision]],
    *,
    sink: PerceptionSink,
    budget_store: Any | None = None,
    user_id: str = "",
    now: float = 0.0,
) -> dict[str, int]:
    """Route decisions to the sink and charge the interruption budget for SPEAK
    deliveries that actually spent it. Charging happens AFTER a successful delivery
    so a sink failure doesn't burn budget on a non-event. Returns a per-channel
    delivered count (for logging/Observatory)."""
    counts = {FILE_FOR_PULL: 0, SPEAK: 0, ACT_WITH_CONSENT: 0, "silent": 0}
    for insight, decision in decisions:
        delivered = await _route(sink, insight, decision)
        if not delivered:
            counts["silent"] += 1
            continue
        counts[decision.channel] = counts.get(decision.channel, 0) + 1
        if decision.spent_budget and budget_store is not None and user_id:
            try:
                budget_store.spend(user_id, now)
            except Exception:  # noqa: BLE001 — budget accounting is best-effort
                log.debug("budget_spend_failed", exc_info=True)
    if counts[SPEAK] or counts[ACT_WITH_CONSENT]:
        log.info("perception_dispatched", **{k: v for k, v in counts.items() if v})
    return counts


async def perceive_and_dispatch(
    ctx: FusionContext,
    *,
    regret_multiplier: float,
    budget_remaining: int,
    sink: PerceptionSink,
    budget_store: Any | None = None,
    config: JudgmentConfig | None = None,
) -> dict[str, int]:
    """One full pass: fuse → judge → route. The single entry the live adapter calls
    once it has resolved the regret multiplier (``feedback.aggregate_bias``), the
    remaining budget (``budget.BUDGET.remaining``), and a real sink."""
    decisions = run_perception(
        ctx, regret_multiplier=regret_multiplier,
        budget_remaining=budget_remaining, config=config,
    )
    return await dispatch(
        decisions, sink=sink, budget_store=budget_store,
        user_id=ctx.user_id, now=ctx.now,
    )
