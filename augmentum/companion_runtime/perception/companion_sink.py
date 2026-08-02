"""CompanionPerceptionSink — deliver judged insights through the EXISTING surfaces.

The doctrine is "feed data into the judgment machinery that already runs, don't
build a second brain." So the sink doesn't invent a surfacer — it routes the
gate's decisions onto the proven proactive plumbing:

  * **FILE_FOR_PULL** → enqueue to ``companion_initiative_queue`` (status pending),
    NOT surfaced. It sits for the Today digest / next natural re-engagement — the
    glanceable pull surface, no interruption.
  * **SPEAK** → enqueue + publish ``initiative.surfaced`` (the existing immediate
    surface path the runtime already delivers). The bus event is a POINTER
    (id/kind/score); the human content lives in the queued row's payload — exactly
    how ``behavior/initiative.py`` works, so the existing consumer reads it the
    same way.
  * **ACT_WITH_CONSENT** → enqueue + surface with a ``proposes_action`` marker so
    the consumer routes it to the gated-offer confirm (bounded autonomy).

This keeps the perception pipeline additive: it produces insights and decisions;
delivery reuses the queue + bus that already reach the user.
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.behavior import initiative
from augmentum.companion_runtime.perception.insight import DeliveryDecision, Insight
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Queue ``kind`` for perception-sourced proposals — distinct from initiative's own
# kinds (revisit_thread/share_creation/…) so they're attributable and the consumer
# can render them as "she noticed" rather than "she's revisiting".
PERCEIVED_KIND = "perceived"
SURFACE_TOPIC = "initiative.surfaced"


class CompanionPerceptionSink:
    """Routes :class:`DeliveryDecision`s onto the initiative queue + bus.

    ``runtime`` is a ``CompanionRuntime`` (needs ``.backend``, ``.companion_id``,
    ``.owner_user_id``, ``.bus``). Best-effort throughout — a delivery failure is
    logged and swallowed so one bad insight can't break a perception pass."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    def _payload(self, insight: Insight, decision: DeliveryDecision, **extra: Any) -> dict:
        return {
            "source": "perception",
            "summary": insight.summary,
            "shape": insight.shape,
            "source_kind": insight.kind,
            "evidence": list(insight.evidence)[:8],
            "time_critical": insight.time_critical,
            "suggested_action": insight.suggested_action,
            "channel": decision.channel,
            "reason": decision.reason,
            **extra,
        }

    async def _enqueue(self, insight: Insight, payload: dict) -> int | None:
        proposal = initiative.Proposal(
            kind=PERCEIVED_KIND, payload=payload, score=round(insight.base_score, 3),
        )
        return await initiative.enqueue(self.runtime, proposal)

    async def _surface(self, row_id: int, insight: Insight, **extra: Any) -> None:
        try:
            await self.runtime.bus.publish_topic(
                SURFACE_TOPIC,
                {
                    "id": row_id, "kind": PERCEIVED_KIND,
                    "score": round(insight.base_score, 3),
                    "summary": insight.summary, **extra,
                },
                source_companion_id=self.runtime.companion_id,
            )
        except Exception:  # noqa: BLE001 — surfacing is best-effort
            log.warning("perception_surface_failed", exc_info=True)

    async def file_for_pull(self, insight: Insight, decision: DeliveryDecision) -> None:
        # Enqueue only — it waits in the queue for the digest / re-engagement,
        # never interrupts.
        await self._enqueue(insight, self._payload(insight, decision))

    async def speak(self, insight: Insight, decision: DeliveryDecision) -> None:
        row_id = await self._enqueue(insight, self._payload(insight, decision))
        if row_id is not None:
            await self._surface(row_id, insight)

    async def propose_action(self, insight: Insight, decision: DeliveryDecision) -> None:
        # Surface with the action marker; the consumer routes to the gated-offer
        # confirm (the proposal never auto-fires).
        payload = self._payload(insight, decision, proposes_action=True)
        row_id = await self._enqueue(insight, payload)
        if row_id is not None:
            await self._surface(
                row_id, insight,
                proposes_action=True, suggested_action=insight.suggested_action,
            )
