"""Memory hygiene verbs — forget (recall-then-confirm) and tier control.

Wiring program Phase 2 (2026-06-12). The memory substrate has had
soft-delete (``store.forget`` sets ``valid_until``) and a full tier
economy with audit trail since they shipped — but only the UI could
reach them. "Forget what I said about my sister" out loud went
nowhere.

Design law for forget: **never fuzzy-delete.** The verb recalls
first, speaks the actual fact it found, and parks a confirm via the
clarify machinery — the user's "yes" fills the slot next turn (same
ride weather.today's location uses). Only an assenting answer
deletes. A fuzzy match silently removing the wrong fact is a trust
catastrophe in a way a wrong volume step is not; speaking the fact
makes the blast radius visible before anything happens. Hard-forget
only — preference CHANGES keep flowing through the extractor's
supersede path; one verb must not carry three semantics.

Tier control is reversible (audit trail + revert route exist), so it
confirms nothing and just speaks what it did.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)

# Assent/denial vocabularies for the confirm slot. The router fills
# ``confirm`` with the user's raw answer; anything that isn't clearly
# assent KEEPS the memory — deletion never wins a tie. ("forget it"
# is deliberately absent from assent: colloquially it means
# "nevermind" as often as "yes, forget".)
_ASSENT_WORDS = (
    "yes", "yeah", "yep", "yup", "sure", "please", "do it",
    "go ahead", "confirm", "correct", "right", "ok", "okay",
)
_DENIAL_WORDS = (
    "no", "nope", "don't", "dont", "keep", "wait", "nevermind",
    "never mind", "cancel", "stop", "hold on",
)


def _reads_assent(answer: str) -> bool:
    low = f" {answer.strip().lower()} "
    if any(f" {w} " in low or low.strip() == w for w in _DENIAL_WORDS):
        return False
    return any(f" {w} " in low or low.strip() == w for w in _ASSENT_WORDS)


def age_phrase(created_at: str) -> str:
    """ISO timestamp → spoken-friendly age ('3 weeks ago')."""
    try:
        raw = (created_at or "").replace("Z", "+00:00")
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        days = max(0, (datetime.now(UTC) - ts).days)
    except (TypeError, ValueError):
        return ""
    if days < 2:
        return "today"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    if days < 365:
        return f"{days // 30} months ago"
    return "over a year ago"


def near_duplicate(a: str, b: str) -> bool:
    """Token-set containment ≥ 0.75 — catches the retry/double-trigger
    class (LLM tool-call retries, repeated explicit saves), NOT
    same-subject-new-value preference changes. Value changes are the
    extractor's supersede lane and, eventually, the corroboration
    economy's — a looser threshold here would silently merge distinct
    facts, which is data loss wearing a hygiene costume."""
    ta = {w for w in re.findall(r"[a-z0-9']+", (a or "").lower()) if len(w) > 2}
    tb = {w for w in re.findall(r"[a-z0-9']+", (b or "").lower()) if len(w) > 2}
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.75


def _store(session: SessionContext):
    return getattr(session.app_state, "memory_store", None) if session.app_state else None


# ---------------------------------------------------------------------------
# memory.forget
# ---------------------------------------------------------------------------

async def _memory_forget(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I'm not sure whose memory to touch.",
        )
    store = _store(session)
    if store is None:
        return ActionResult(
            short_circuit=True,
            speak="The memory store isn't available.",
        )

    # Confirm turn — the parked clarify filled memory_id last turn and
    # the user's answer arrives in ``confirm``.
    memory_id = str(args.get("memory_id") or "").strip()
    confirm = str(args.get("confirm") or "").strip()
    if memory_id and confirm:
        if not _reads_assent(confirm):
            log.info("memory_forget_kept", user_id=session.user_id)
            return ActionResult(
                short_circuit=True,
                speak="Kept it.",
                digest="forget cancelled — memory kept",
            )
        ok = await store.forget(memory_id, user_id=session.user_id)
        log.info(
            "memory_forget_confirmed",
            user_id=session.user_id, memory_id=memory_id, ok=bool(ok),
        )
        if not ok:
            return ActionResult(
                short_circuit=True,
                speak="I couldn't find that one anymore — it may already be gone.",
            )
        return ActionResult(
            short_circuit=True,
            speak="Forgotten.",
            toast="Memory forgotten",
            digest="one memory forgotten at the user's request",
        )

    query = str(args.get("query") or "").strip()
    if not query:
        return ActionResult(
            short_circuit=True,
            speak="What should I forget?",
            clarify={"missing": ["query"], "args": {}},
        )

    try:
        hits = await store.recall(query, user_id=session.user_id, limit=3)
    except Exception as exc:  # noqa: BLE001
        log.warning("memory_forget_recall_failed", error=str(exc))
        return ActionResult(
            short_circuit=True,
            speak="I couldn't search memory right now.",
        )
    if not hits:
        return ActionResult(
            short_circuit=True,
            speak=f"I don't have anything saved about {query[:60]}.",
        )

    top = hits[0]
    stub = (getattr(top, "content", "") or "")[:140]
    others = (
        " I have a couple of related ones too — this is the closest."
        if len(hits) > 1 else ""
    )
    return ActionResult(
        short_circuit=True,
        speak=f'I have: "{stub}" — forget that for good?{others}',
        clarify={
            "missing": ["confirm"],
            "args": {"memory_id": top.id, "query": query},
        },
    )


register_action(
    id="memory.forget",
    summary=(
        "Delete a saved memory at the user's request — recalls the "
        "matching fact, SPEAKS it back, and only deletes after the "
        "user confirms. Call for 'forget what I said about X', "
        "'delete that memory', 'stop remembering my address'. "
        "Siblings: changing a preference is just memory.save (the new "
        "value supersedes); keeping but de-emphasizing is memory.tier."
    ),
    examples=[
        "forget what I told you about my sister",
        "delete that memory about my old job",
        "stop remembering my lucky number",
        "forget my address",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": (
                "What to forget, in the user's words — used to recall "
                "the matching saved fact."
            ),
        },
        "memory_id": {
            "type": "string",
            "description": (
                "Internal — set by the confirm flow. Leave empty on "
                "the first call."
            ),
        },
        "confirm": {
            "type": "string",
            "description": (
                "Internal — the user's answer to the 'forget that?' "
                "question. Leave empty on the first call."
            ),
        },
    },
    fanout=_TIER3_ONLY,
    handler=_memory_forget,
    delivery="verbal",
    stakes="disruptive",
)


# ---------------------------------------------------------------------------
# memory.tier
# ---------------------------------------------------------------------------

_LEVELS = {
    "long_term": "core",
    "normal": "active",
    "archive": "archive",
}

_LEVEL_SPOKEN = {
    "long_term": "Keeping that close — it's long-term now.",
    "normal": "Moved it back to normal recall.",
    "archive": "Tucked that into the archive — still there if you ask.",
}


async def _memory_tier(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I'm not sure whose memory to touch.",
        )
    store = _store(session)
    if store is None:
        return ActionResult(
            short_circuit=True,
            speak="The memory store isn't available.",
        )
    level = str(args.get("level") or "").strip().lower()
    if level not in _LEVELS:
        return ActionResult(
            short_circuit=True,
            speak="Keep it long-term, normal, or archived?",
            clarify={"missing": ["level"], "args": dict(args)},
        )
    query = str(args.get("query") or "").strip()
    if not query:
        return ActionResult(
            short_circuit=True,
            speak="Which memory do you mean?",
            clarify={"missing": ["query"], "args": dict(args)},
        )

    try:
        hits = await store.recall(query, user_id=session.user_id, limit=1)
    except Exception as exc:  # noqa: BLE001
        log.warning("memory_tier_recall_failed", error=str(exc))
        return ActionResult(
            short_circuit=True,
            speak="I couldn't search memory right now.",
        )
    if not hits:
        return ActionResult(
            short_circuit=True,
            speak=(
                f"I don't have anything saved about {query[:60]} yet — "
                "want me to remember it first?"
            ),
        )

    top = hits[0]
    tier = _LEVELS[level]
    ok = await store.update_tier(
        top.id, tier, user_id=session.user_id, source="manual",
    )
    if not ok:
        return ActionResult(
            short_circuit=True,
            speak="I couldn't move that one.",
        )
    # Promoting out of provisional must also clear the 7-day TTL or
    # the promoted fact still dies on schedule (mirrors the tier
    # route's handling).
    if tier != "provisional":
        try:
            conn = getattr(store, "_conn", None)
            if conn is not None:
                await conn.execute(
                    "UPDATE memories SET provisional_expires_at = NULL "
                    "WHERE id = ? AND user_id = ?",
                    (top.id, session.user_id),
                )
                await conn.commit()
        except Exception:  # noqa: BLE001
            log.warning("memory_tier_ttl_clear_failed", exc_info=True)
    log.info(
        "memory_tier_verb",
        user_id=session.user_id, memory_id=top.id, tier=tier,
    )
    return ActionResult(
        short_circuit=True,
        speak=_LEVEL_SPOKEN[level],
        digest=f"memory tier → {tier}: {(top.content or '')[:60]}",
    )


register_action(
    id="memory.tier",
    summary=(
        "Silently change how prominently a saved memory is kept — "
        "long_term ('always remember this, it matters'), normal, or "
        "archive ('you don't need to keep that handy'). Reversible; "
        "no confirmation needed. Sibling: actually DELETING a memory "
        "is memory.forget."
    ),
    examples=[
        "remember that long-term, it's important",
        "keep that one close",
        "you don't need to keep that handy",
        "archive what I said about the old project",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": "Which saved memory, in the user's words.",
        },
        "level": {
            "type": "string",
            "enum": list(_LEVELS),
            "description": "long_term | normal | archive.",
        },
    },
    fanout=_TIER3_ONLY,
    handler=_memory_tier,
    delivery="verbal",
)
