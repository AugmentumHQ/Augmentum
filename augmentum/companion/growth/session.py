"""CompanionGrowthSession — coder-loop-shaped agent state for the growth loop.

Modeled on :class:`augmentum.coder.state.CoderState`: one object per
active session, holds plan + act log + economy deltas + outcome.
One concurrent session per (user_id, agent_id); the caller is
responsible for enforcing that (Phase 1: lock lives at the route layer).

Phase 1 scope (this implementation):

* ``plan()`` — drafts a plan from an :class:`ActionRequest` and writes the
  growth-log row at ``in_progress``.
* ``act()`` — debits mana, dispatches to the action handler, appends the
  result as an act-log step.
* ``verify()`` — Phase 1 is Tier 0 (silent reads) only: the verifier just
  marks the session done. Tier 1+ verification (dry-run, hold-out replay,
  rollback) lands in Phase 2+.
* ``archive()`` — finalizes the log row, bumps backlog attempt counters.

Lifecycle: ``await session.run()`` does plan → act → verify → archive in
order and returns the finalized :class:`GrowthLogEntry`. Callers can also
drive the phases individually if they need to inspect intermediate
state (used by tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from augmentum.companion.growth.actions import (
    ACTIONS,
    ActionContext,
    ActionRequest,
    ActionResult,
)
from augmentum.companion.growth.economy import Economy
from augmentum.companion.growth.store import (
    DEFAULT_AGENT_ID,
    BacklogItem,
    GrowthLogEntry,
    GrowthStore,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class SessionConfig:
    """Knobs for one session run.

    Defaults are intentionally generous so a manually-fired Phase 1
    session doesn't get blocked on caps — the trigger-condition layer
    (Phase 3) tightens these before autonomous firing.
    """

    max_steps: int = 5
    budget_mana: float = 50.0
    budget_berries: float = 0.0


@dataclass(slots=True)
class CompanionGrowthSession:
    """In-memory state for one growth-loop session.

    Constructed by the route layer (or the future autonomous trigger).
    Drive the lifecycle via :meth:`run` or the per-phase methods.
    """

    store: GrowthStore
    economy: Economy
    user_id: str
    agent_id: str = DEFAULT_AGENT_ID
    backlog: BacklogItem | None = None
    config: SessionConfig = field(default_factory=SessionConfig)

    # Injected dependencies — handlers that need these read them from the
    # ActionContext built inside ``act()``.
    memory_store: Any = None

    # Populated as the session runs.
    log_entry: GrowthLogEntry | None = None
    plan_dict: dict[str, Any] = field(default_factory=dict)
    act_results: list[ActionResult] = field(default_factory=list)
    mana_spent: float = 0.0
    berries_spent: float = 0.0
    berries_earned: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def plan(
        self,
        *,
        ad_hoc_request: ActionRequest | None = None,
    ) -> dict[str, Any]:
        """Draft the plan and open the growth-log row.

        Supports two modes:
          * Backlog-driven — ``self.backlog`` is set; the plan target +
            type come from the backlog item.
          * Ad-hoc — caller passes ``ad_hoc_request`` directly; backlog
            stays None.

        Either way the plan dict is persisted as ``plan_json`` on the
        growth-log row.
        """
        if ad_hoc_request is None and self.backlog is None:
            raise ValueError(
                "CompanionGrowthSession.plan requires either a backlog "
                "item or an ad_hoc_request",
            )

        if ad_hoc_request is not None:
            item_type = ad_hoc_request.action_type
            target_ref = ad_hoc_request.target_ref
            rationale = ad_hoc_request.rationale
        else:
            assert self.backlog is not None  # narrowed by check above
            item_type = self.backlog.item_type
            target_ref = self.backlog.target_ref
            rationale = self.backlog.rationale

        self.plan_dict = {
            "action_type": item_type,
            "target_ref": target_ref,
            "rationale": rationale,
            "max_steps": self.config.max_steps,
            "budget_mana": self.config.budget_mana,
            "budget_berries": self.config.budget_berries,
        }

        backlog_id = self.backlog.id if self.backlog else None
        self.log_entry = await self.store.start_session(
            user_id=self.user_id,
            agent_id=self.agent_id,
            backlog_id=backlog_id,
            plan=self.plan_dict,
            snapshot_ref="",  # Tier 0: nothing to roll back
        )
        if self.backlog is not None:
            await self.store.set_backlog_state(
                self.backlog.id,
                user_id=self.user_id, agent_id=self.agent_id,
                state="in_progress",
            )
        return self.plan_dict

    async def act(self) -> list[ActionResult]:
        """Dispatch the action handler.

        Phase 1 runs a single step (Recall is one-shot). Multi-step act
        loops (re-attempt, sub-action, mid-session consult) land in
        Phase 2+. The loop boundary already exists here — extending it
        is a matter of returning a continuation hint from the action
        result.
        """
        if self.log_entry is None:
            raise RuntimeError("CompanionGrowthSession.act called before plan")

        action_type = self.plan_dict.get("action_type", "")
        handler = ACTIONS.get(action_type)
        if handler is None:
            log.warning(
                "growth_session.unknown_action_type",
                action_type=action_type, user_id=self.user_id,
            )
            return []

        for step_idx in range(self.config.max_steps):
            # Debit before dispatch. Phase 1 cost model: action declares
            # ``mana_cost`` on the handler; debit is enforced here.
            mana_cost = float(getattr(handler, "mana_cost", 1.0))
            if mana_cost > 0:
                debit = await self.economy.debit_mana(
                    mana_cost,
                    growth_log_id=self.log_entry.id,
                    reason=f"action:{action_type}:step_{step_idx}",
                )
                if not debit.ok:
                    await self.store.append_act_step(
                        self.log_entry.id,
                        user_id=self.user_id, agent_id=self.agent_id,
                        step={
                            "step": step_idx,
                            "action_type": action_type,
                            "ok": False,
                            "error": debit.reason,
                            "mana_cost": mana_cost,
                            "mana_after": debit.mana_after,
                        },
                    )
                    break
                self.mana_spent += debit.debited

            ctx = ActionContext(
                user_id=self.user_id,
                agent_id=self.agent_id,
                growth_log_id=self.log_entry.id,
                target_ref=self.plan_dict.get("target_ref", ""),
                rationale=self.plan_dict.get("rationale", ""),
                memory_store=self.memory_store,
                growth_store=self.store,
            )
            result = await handler.run(ctx)
            self.act_results.append(result)

            await self.store.append_act_step(
                self.log_entry.id,
                user_id=self.user_id, agent_id=self.agent_id,
                step={
                    "step": step_idx,
                    "action_type": action_type,
                    "ok": result.ok,
                    "payload": result.payload,
                    "surface_event": result.surface_event,
                    "ledger_delta": result.ledger_delta,
                    "mana_cost": mana_cost,
                },
            )

            if not result.continue_loop:
                break

        return self.act_results

    async def verify(self) -> bool:
        """Phase-1 verify: Tier 0 means we trust the read-only action.

        Returns True (verified). Tier 1+ verifier work lands later.
        """
        return True

    async def archive(self, *, verified: bool = True) -> GrowthLogEntry:
        """Finalize the log row and update backlog counters."""
        if self.log_entry is None:
            raise RuntimeError("CompanionGrowthSession.archive called before plan")

        ok = verified and all(r.ok for r in self.act_results) if self.act_results else False
        outcome = "completed" if ok else "aborted"

        # Aggregate ledger deltas from action results.
        ledger_delta: dict[str, Any] = {}
        for r in self.act_results:
            for k, v in (r.ledger_delta or {}).items():
                ledger_delta[k] = ledger_delta.get(k, 0) + v if isinstance(v, (int, float)) else v

        await self.store.finalize_session(
            self.log_entry.id,
            user_id=self.user_id, agent_id=self.agent_id,
            outcome=outcome,
            tier=0,  # Phase 1 ships Tier 0 actions only
            approval_state="n/a",
            ledger_delta=ledger_delta,
            mana_spent=self.mana_spent,
            berries_spent=self.berries_spent,
            berries_earned=self.berries_earned,
        )

        if self.backlog is not None:
            await self.store.update_backlog_attempt(
                self.backlog.id,
                user_id=self.user_id, agent_id=self.agent_id,
                success=ok,
            )
            await self.store.set_backlog_state(
                self.backlog.id,
                user_id=self.user_id, agent_id=self.agent_id,
                state="pending" if not ok else "pending",
                # Backlog items stay pending after a successful attempt —
                # the next pass picks them again with success_count++.
                # Shelving happens elsewhere (saturation / decay logic,
                # Phase 5).
            )

        refreshed = await self.store.get_session(
            self.log_entry.id,
            user_id=self.user_id, agent_id=self.agent_id,
        )
        return refreshed or self.log_entry

    async def run(
        self, *, ad_hoc_request: ActionRequest | None = None,
    ) -> GrowthLogEntry:
        """Drive plan → act → verify → archive in one call."""
        await self.plan(ad_hoc_request=ad_hoc_request)
        await self.act()
        verified = await self.verify()
        return await self.archive(verified=verified)
