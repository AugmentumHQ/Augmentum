"""Discovery action — surface a memory the user likely forgot they had.

Catalog category B (discovery). See
``docs/superpowers/specs/2026-05-31-companion-action-catalog.md`` §B.

What this does:

  1. Take ``ctx.target_ref`` as the query topic.
  2. Recall a wider candidate set from memory (k=15 vs recall's 5).
  3. Pick the OLDEST candidate that beats a configurable age floor —
     the inverse of the recall action's "skip trivially recent" filter.
     Recall surfaces *connected* memories; discovery surfaces *forgotten*
     ones the user might want to be reminded of.
  4. Return an ActionResult with a discovery surface event.

The age floor defaults to 30 days — old enough that the user has
probably stopped thinking about the memory; recent enough that the
embedding similarity is still meaningful. ``ctx.extras["min_age_days"]``
overrides.

Cost / tier: mana 3.0, tier 1 (surface event with content).

Reward signal (Phase 5 wires this): engagement (opened / saved / "tell me more")
= +30; dismissed = -5 (mild penalty — bad-fit discoveries waste user attention).
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


# Default age floor — memories younger than this are recall territory,
# not discovery. The catalog spec frames Discovery as "found something
# the user didn't know to look for," which implies they've stopped
# actively thinking about it.
_DEFAULT_MIN_AGE_SECONDS = 30 * 24 * 60 * 60  # 30 days

# How many candidates to pull before filtering for age. Wider than
# recall's 5 because the age floor will drop most of them.
_CANDIDATE_LIMIT = 15


class DiscoverForgottenMemory:
    """B — Discovery (memory-substrate variant).

    Phase 1 covers the memory leg only. Future variants extend to
    file_index / browse_history / knowledge packs (catalog spec lists
    those as examples B's cross-modal cousins).
    """

    action_type = "discovery_surface"
    mana_cost = 3.0
    tier = 1

    async def run(self, ctx: ActionContext) -> ActionResult:
        if not ctx.target_ref:
            return ActionResult(ok=False, error="discovery: empty target_ref")

        memory_store = ctx.memory_store
        if memory_store is None:
            return ActionResult(
                ok=False,
                error="discovery: memory_store not provided on ActionContext",
            )

        min_age_days = ctx.extras.get("min_age_days")
        try:
            min_age_seconds = (
                int(min_age_days) * 24 * 60 * 60
                if min_age_days is not None else _DEFAULT_MIN_AGE_SECONDS
            )
        except (TypeError, ValueError):
            min_age_seconds = _DEFAULT_MIN_AGE_SECONDS

        try:
            hits = await memory_store.recall(
                query=ctx.target_ref,
                user_id=ctx.user_id,
                limit=_CANDIDATE_LIMIT,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "growth.discovery.memory_failed",
                user_id=ctx.user_id, error=str(exc)[:200],
            )
            return ActionResult(ok=False, error=f"discovery_failed: {exc!s}")

        chosen = _pick_oldest_above_floor(hits, min_age_seconds)
        if chosen is None:
            return ActionResult(
                ok=False,
                error="discovery: no candidate older than age floor",
            )

        surface_event = {
            "topic": "growth.discovery.surfaced",
            "payload": {
                "memory_id": _attr(chosen, ("id", "memory_id", "uuid")),
                "snippet": _snippet(chosen),
                "scope": _attr(chosen, ("scope", "session_id", "tier")),
                "age_seconds": _age_seconds(chosen),
                "target_ref": ctx.target_ref,
                "rationale": ctx.rationale,
                "surfaced_at": int(time.time()),
            },
        }
        return ActionResult(
            ok=True,
            payload={
                "candidate_count": len(hits) if hasattr(hits, "__len__") else 0,
                "memory_id": _attr(chosen, ("id", "memory_id", "uuid")),
            },
            surface_event=surface_event,
            ledger_delta={"discovery_surfaced": 1},
            continue_loop=False,
        )


# ── Helpers — same memory-shape tolerance as recall.py ───────────────


def _pick_oldest_above_floor(hits: Any, min_age_seconds: int) -> Any | None:
    """Pick the OLDEST hit that's at least ``min_age_seconds`` old.

    The inverse of recall's "skip recent" — we want the candidate the
    user has most plausibly forgotten about.
    """
    if not hits:
        return None
    now = int(time.time())
    eligible: list[tuple[int, Any]] = []
    for hit in hits:
        memory = hit[0] if (isinstance(hit, tuple) and hit) else hit
        if memory is None:
            continue
        created = _created_at(memory)
        if created is None:
            continue
        age = now - created
        if age < min_age_seconds:
            continue
        eligible.append((age, memory))
    if not eligible:
        return None
    # Most ancient candidate wins — that's the "I forgot I had this" hit.
    eligible.sort(key=lambda pair: pair[0], reverse=True)
    return eligible[0][1]


def _created_at(memory: Any) -> int | None:
    for attr in ("created_at", "ts", "event_time", "recorded_at"):
        v = getattr(memory, attr, None)
        if v is None and isinstance(memory, dict):
            v = memory.get(attr)
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return None


def _age_seconds(memory: Any) -> int:
    created = _created_at(memory)
    return int(time.time()) - created if created else 0


def _attr(memory: Any, names: tuple[str, ...]) -> str:
    for n in names:
        v = getattr(memory, n, None)
        if v is None and isinstance(memory, dict):
            v = memory.get(n)
        if v:
            return str(v)
    return ""


def _snippet(memory: Any) -> str:
    for n in ("text", "content", "body", "snippet", "summary"):
        v = getattr(memory, n, None)
        if v is None and isinstance(memory, dict):
            v = memory.get(n)
        if v:
            return str(v)[:500]
    return ""


register(DiscoverForgottenMemory())
