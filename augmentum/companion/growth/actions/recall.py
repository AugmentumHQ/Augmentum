"""Recall action — surface a past memory connected to the current target.

Catalog category E (recall). Phase 1 anchor action — see
``docs/superpowers/specs/2026-05-31-companion-action-catalog.md`` §E.

What this does:

  1. Take ``ctx.target_ref`` as the query (e.g. a topic the user has just
     touched in chat / browse / journal).
  2. Hit ``ctx.memory_store.recall()`` to retrieve a small set of
     candidate memories.
  3. Pick the top hit that ISN'T trivially recent — Recall's value is
     surfacing things the user has forgotten or hasn't connected, not
     re-showing what they wrote 5 minutes ago.
  4. Return an ActionResult with the recall framed as a surface event
     payload. The session writes this to ``act_log_json``; future
     phases pipe it to a websocket fanout for the UI.

Cost / tier: mana 2.0, tier 0 (passive read). No berry cost.

Reward signal (Phase 5 wires this): explicit thumbs / save / "tell me
more" follow-up = +20; dismissed = 0 (no penalty — recalls are cheap).
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


# Memories younger than this are considered "trivially recent" and
# skipped — recall is about surfacing the not-obvious, not duplicating
# the conversation context.
_RECENT_THRESHOLD_SECONDS = 6 * 60 * 60  # 6h


class RecallSurfaceConnection:
    """Phase 1 Recall handler — surface one past memory tied to a target."""

    action_type = "recall_connect"
    mana_cost = 2.0
    tier = 0

    async def run(self, ctx: ActionContext) -> ActionResult:
        if not ctx.target_ref:
            return ActionResult(ok=False, error="recall: empty target_ref")

        memory_store = ctx.memory_store
        if memory_store is None:
            return ActionResult(
                ok=False,
                error="recall: memory_store not provided on ActionContext",
            )

        try:
            hits = await memory_store.recall(
                query=ctx.target_ref,
                user_id=ctx.user_id,
                limit=5,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning(
                "growth.recall.memory_failed",
                user_id=ctx.user_id, error=str(exc)[:200],
            )
            return ActionResult(ok=False, error=f"recall_failed: {exc!s}")

        chosen = _pick_recall_candidate(hits)
        if chosen is None:
            return ActionResult(
                ok=False, error="recall: no eligible memory found",
            )

        # Frame the recall as a surface-event payload. The shape mirrors
        # other surface events emitted by the companion runtime so a
        # future websocket fanout can route it without translation.
        surface_event = {
            "topic": "growth.recall.surfaced",
            "payload": {
                "memory_id": _memory_id(chosen),
                "snippet": _memory_snippet(chosen),
                "scope": _memory_scope(chosen),
                "target_ref": ctx.target_ref,
                "rationale": ctx.rationale,
                "surfaced_at": int(time.time()),
            },
        }
        return ActionResult(
            ok=True,
            payload={
                "hit_count": len(hits) if hasattr(hits, "__len__") else 0,
                "memory_id": _memory_id(chosen),
            },
            surface_event=surface_event,
            ledger_delta={"recall_surfaced": 1},
            continue_loop=False,
        )


def _pick_recall_candidate(hits: Any) -> Any | None:
    """Pick the first hit that isn't trivially recent.

    Hits may be a list of Memory objects or (Memory, score) tuples or
    raw dicts depending on the store implementation. We tolerate all
    three to keep the action loosely coupled to the store's evolution.
    """
    if not hits:
        return None
    now = int(time.time())
    for hit in hits:
        memory = _unwrap_memory(hit)
        if memory is None:
            continue
        created = _memory_created_at(memory)
        if created and (now - created) < _RECENT_THRESHOLD_SECONDS:
            continue
        return memory
    return None


def _unwrap_memory(hit: Any) -> Any | None:
    if isinstance(hit, tuple) and hit:
        return hit[0]
    return hit


def _memory_id(memory: Any) -> str:
    for attr in ("id", "memory_id", "uuid"):
        v = getattr(memory, attr, None) or (
            memory.get(attr) if isinstance(memory, dict) else None
        )
        if v:
            return str(v)
    return ""


def _memory_snippet(memory: Any) -> str:
    for attr in ("text", "content", "body", "snippet", "summary"):
        v = getattr(memory, attr, None) or (
            memory.get(attr) if isinstance(memory, dict) else None
        )
        if v:
            return str(v)[:500]
    return ""


def _memory_scope(memory: Any) -> str:
    for attr in ("scope", "session_id", "tier"):
        v = getattr(memory, attr, None) or (
            memory.get(attr) if isinstance(memory, dict) else None
        )
        if v:
            return str(v)
    return ""


def _memory_created_at(memory: Any) -> int | None:
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


register(RecallSurfaceConnection())
