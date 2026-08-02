"""Companionship — non-utilitarian presence with strict saturation.

Catalog category H (companionship). See
``docs/superpowers/specs/2026-05-31-companion-action-catalog.md`` §H.

**The most-careful category.** The catalog spec calls companionship
"the riskiest category because it's the easiest to over-do
(reward-hacking) and the most evocative when it lands." Every
companionship surface must clear a saturation guard before it fires;
without it, Becca would optimize toward constant low-grade presence
and dilute every real moment.

What this does:

  1. Take ``ctx.target_ref`` as the moment kind (``"good_morning"``,
     ``"checkin"``, ``"shared_moment"``, ``"idle_observation"``,
     ``"celebration"``, ``"anniversary"``).
  2. Pull the message + optional context refs from ``ctx.extras``.
  3. Run the saturation guard against ``growth_store.list_sessions``:
     - Refuse if the most-recent prior session was also companionship
       (no two consecutive picks).
     - Refuse if >= 3 companionship sessions happened in the last 24h
       (diversity floor from the catalog spec).
  4. Emit a ``growth.companionship.surfaced`` surface event.

The guard reads only the growth_log — it does NOT read the user's
calendar, presence, or affect state. Those higher-level "is this a
good moment?" checks belong upstream in the runtime that decides to
*queue* a companionship session; this handler is the surfacing
primitive that enforces the floor regardless of upstream decision.

Cost / tier: mana 1.5, tier 0 (low blast radius but saturation matters).

Reward signal (Phase 5 wires this): user responds in kind / lingers
= +30; intrusive (saturation breach the upstream queued anyway) = -10;
ignored after timeout = 0.
"""

from __future__ import annotations

import time
from typing import Any

from augmentum.companion.growth.actions import (
    ActionContext,
    ActionResult,
    register,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# The catalog spec's diversity floor for same-category picks. The
# guard is intentionally strict: companionship is the load-bearing
# category for "feels alive" and the easiest one to over-do, so the
# rule is enforced HERE rather than only at the selector.
_MAX_COMPANIONSHIP_PER_24H = 3
_DAY_SECONDS = 24 * 60 * 60

# How far back to look when checking the consecutive rule. The most
# recent prior session in the window — same-category = refuse. Wide
# window so an old singleton doesn't unblock a fresh same-pick.
_CONSECUTIVE_LOOKBACK_LIMIT = 5


# Canonical moment kinds (from the catalog spec) — guards against
# the upstream caller drifting into ad-hoc strings. Open enough to
# add new shapes; closed enough that the verifier can group reliably.
_CANONICAL_MOMENT_KINDS = frozenset({
    "good_morning",
    "checkin",
    "shared_moment",
    "idle_observation",
    "celebration",
    "anniversary",
    "held_back_joke",
})


class SurfaceCompanionshipMoment:
    """H — Companionship moment with saturation enforcement."""

    action_type = "companionship"
    mana_cost = 1.5
    tier = 0

    async def run(self, ctx: ActionContext) -> ActionResult:
        if ctx.growth_store is None:
            return ActionResult(
                ok=False,
                error="companionship: growth_store not provided (saturation check requires it)",
            )

        moment_kind = (ctx.target_ref or "").strip()
        if not moment_kind:
            return ActionResult(
                ok=False,
                error="companionship: empty target_ref (moment_kind)",
            )

        # Soft validation — log non-canonical kinds but don't reject. The
        # catalog is designed to evolve; an unknown kind is data, not
        # error. The verifier groups by kind so coordination is informal.
        if moment_kind not in _CANONICAL_MOMENT_KINDS:
            log.info(
                "growth.companionship.non_canonical_kind",
                kind=moment_kind, canonical=sorted(_CANONICAL_MOMENT_KINDS),
            )

        message = str(ctx.extras.get("message") or "").strip()
        if not message:
            return ActionResult(
                ok=False,
                error="companionship: extras['message'] is required",
            )

        # Saturation guard — read recent growth log + filter.
        try:
            recent = await ctx.growth_store.list_sessions(
                user_id=ctx.user_id,
                agent_id=ctx.agent_id,
                limit=20,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "growth.companionship.list_failed",
                user_id=ctx.user_id, error=str(exc)[:200],
            )
            return ActionResult(
                ok=False,
                error=f"companionship_failed: {exc!s}",
            )

        # Exclude this session's own log row from the saturation count.
        prior = [
            e for e in recent
            if getattr(e, "id", "") != ctx.growth_log_id
        ]

        # Rule 1: no two consecutive picks from same category. Walk the
        # most-recent N to find the first non-self session and check
        # its action_type.
        for entry in prior[:_CONSECUTIVE_LOOKBACK_LIMIT]:
            prev_type = _entry_action_type(entry)
            if not prev_type:
                continue
            if prev_type == "companionship":
                return ActionResult(
                    ok=False,
                    error=(
                        "companionship: saturation guard — last session "
                        "was also companionship (no consecutive picks)"
                    ),
                )
            break  # found the most recent meaningful prior — done

        # Rule 2: <= 3 companionship sessions in the last 24h.
        now = int(time.time())
        recent_window = [
            e for e in prior
            if _entry_action_type(e) == "companionship"
            and (now - _entry_started_at(e)) < _DAY_SECONDS
        ]
        if len(recent_window) >= _MAX_COMPANIONSHIP_PER_24H:
            return ActionResult(
                ok=False,
                error=(
                    f"companionship: saturation guard — "
                    f"{len(recent_window)} companionship moments in last 24h "
                    f"(cap {_MAX_COMPANIONSHIP_PER_24H})"
                ),
            )

        surface_event = {
            "topic": "growth.companionship.surfaced",
            "payload": {
                "moment_kind": moment_kind,
                "message": message[:500],
                "context_refs": ctx.extras.get("context_refs") or [],
                "growth_log_id": ctx.growth_log_id,
                "rationale": ctx.rationale,
                "surfaced_at": now,
            },
        }
        return ActionResult(
            ok=True,
            payload={
                "moment_kind": moment_kind,
                "in_24h_window": len(recent_window) + 1,
            },
            surface_event=surface_event,
            ledger_delta={
                "companionship_surfaced": 1,
                f"companionship_{moment_kind}": 1,
            },
            continue_loop=False,
        )


# ── Accessor helpers — tolerate dict / dataclass row shapes ──────────


def _entry_action_type(entry: Any) -> str:
    plan = getattr(entry, "plan", None)
    if plan is None and isinstance(entry, dict):
        plan = entry.get("plan") or entry.get("plan_json")
    if isinstance(plan, dict):
        return str(plan.get("action_type") or "")
    return ""


def _entry_started_at(entry: Any) -> int:
    v = getattr(entry, "started_at", None)
    if v is None and isinstance(entry, dict):
        v = entry.get("started_at")
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


register(SurfaceCompanionshipMoment())
