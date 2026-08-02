"""Narrate-growth action — surface a brief on recent self-improvement work.

Catalog category K (self-improvement narrated). See
``docs/superpowers/specs/2026-05-31-companion-action-catalog.md`` §K.

What this does:

  1. Read the last N growth-log entries via the injected ``growth_store``.
  2. Group by ``action_type``, count outcomes (completed / aborted), tally
     mana spent.
  3. Compose a compact narration payload — what she's been practicing,
     where she landed, what she's planning next.
  4. Return an ActionResult whose ``surface_event`` is the narration. The
     UI / chat layer can render this inline; the spec frames it as the
     channel through which the user sees her growth is real.

The narration itself is structured data, not prose — leaving prose
composition to the surface that displays it (chat may want her voice,
the Becca panel may want a table). Keeps the action's blast radius
narrow and the tests deterministic.

Cost / tier: mana 1.5, tier 1 (one-shot surface event).

Reward signal (Phase 5 wires this): user engages with the narration
(reads, asks follow-up) = +20; ignored = 0; user explicitly approves a
referenced Tier-2 change = +30.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from augmentum.companion.growth.actions import (
    ActionContext,
    ActionResult,
    register,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Default look-back window. Larger windows tell a richer story but
# duplicate older narrations — the catalog spec's diversity floor
# (no more than 3 same-category picks per 24h) makes this a safe
# default. The user can override via ``ctx.extras["window_count"]``.
_DEFAULT_WINDOW = 20

# Floor on how many prior sessions the action needs to be worth
# emitting. With fewer entries the narration would be noise ("I did
# one recall, that's it"). Below this floor the action returns
# ok=False with a soft reason — the session caller can treat that as
# "skip, try again next cycle."
_MIN_ENTRIES_FOR_NARRATION = 3


class NarrateRecentGrowth:
    """K — Self-improvement narrated.

    Composes a structured digest of the last ``window_count``
    growth-log entries. No LLM call; the digest is data the chat or
    panel surface can render in its own voice.
    """

    action_type = "narrate_growth"
    mana_cost = 1.5
    tier = 1

    async def run(self, ctx: ActionContext) -> ActionResult:
        if ctx.growth_store is None:
            return ActionResult(
                ok=False,
                error="narrate_growth: growth_store not provided on ActionContext",
            )

        window = int(ctx.extras.get("window_count") or _DEFAULT_WINDOW)
        window = max(_MIN_ENTRIES_FOR_NARRATION, min(window, 200))

        try:
            entries = await ctx.growth_store.list_sessions(
                user_id=ctx.user_id,
                agent_id=ctx.agent_id,
                limit=window,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "growth.narrate.list_failed",
                user_id=ctx.user_id, error=str(exc)[:200],
            )
            return ActionResult(
                ok=False, error=f"narrate_failed: {exc!s}",
            )

        # Exclude THIS session's own log row from the digest — the act
        # step we're inside hasn't archived yet, but we shouldn't count
        # the in-progress shell as part of "what I've been doing."
        prior = [e for e in entries if getattr(e, "id", "") != ctx.growth_log_id]
        if len(prior) < _MIN_ENTRIES_FOR_NARRATION:
            return ActionResult(
                ok=False,
                error=(
                    f"narrate_growth: only {len(prior)} prior sessions "
                    f"(need {_MIN_ENTRIES_FOR_NARRATION})"
                ),
            )

        per_type: Counter[str] = Counter()
        per_outcome: Counter[str] = Counter()
        mana_total = 0.0
        berries_total = 0.0
        last_completed_ts = 0
        for entry in prior:
            action_type = _entry_action_type(entry) or "unknown"
            per_type[action_type] += 1
            per_outcome[_entry_outcome(entry) or "unknown"] += 1
            mana_total += _entry_float(entry, "mana_spent")
            berries_total += _entry_float(entry, "berries_earned")
            if _entry_outcome(entry) == "completed":
                ts = _entry_int(entry, "started_at")
                if ts > last_completed_ts:
                    last_completed_ts = ts

        # The narration payload: structured, surface-agnostic, easy to
        # render either as a table (Becca panel) or as prose (chat).
        surface_event = {
            "topic": "growth.narrate.surfaced",
            "payload": {
                "window_count": len(prior),
                "action_type_counts": dict(per_type.most_common()),
                "outcome_counts": dict(per_outcome),
                "mana_spent_total": round(mana_total, 2),
                "berries_earned_total": round(berries_total, 2),
                "last_completed_at": last_completed_ts or None,
                "rationale": ctx.rationale,
                "surfaced_at": int(time.time()),
            },
        }
        return ActionResult(
            ok=True,
            payload={
                "window_count": len(prior),
                "distinct_action_types": len(per_type),
            },
            surface_event=surface_event,
            ledger_delta={"narration_surfaced": 1},
            continue_loop=False,
        )


# ── Small accessor helpers — tolerate dict / dataclass / row shapes ──


def _entry_action_type(entry: Any) -> str:
    """Pull the action_type out of a log entry.

    The growth_log row stores ``plan_json`` as the plan dict (which
    carries ``action_type``). On the dataclass shape ``plan`` is the
    parsed dict; on raw rows ``plan_json`` may be a string. Cover both.
    """
    plan = getattr(entry, "plan", None)
    if plan is None and isinstance(entry, dict):
        plan = entry.get("plan") or entry.get("plan_json")
    if isinstance(plan, dict):
        return str(plan.get("action_type") or "")
    return ""


def _entry_outcome(entry: Any) -> str:
    v = getattr(entry, "outcome", None)
    if v is None and isinstance(entry, dict):
        v = entry.get("outcome")
    return str(v or "")


def _entry_float(entry: Any, key: str) -> float:
    v = getattr(entry, key, None)
    if v is None and isinstance(entry, dict):
        v = entry.get(key)
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _entry_int(entry: Any, key: str) -> int:
    v = getattr(entry, key, None)
    if v is None and isinstance(entry, dict):
        v = entry.get(key)
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


register(NarrateRecentGrowth())
