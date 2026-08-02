"""Her interiority — the introspection plane (wiring program Phase 4).

She journals wonderings, notices patterns about the user, dreams on a
schedule, and builds a skill graph — and until 2026-06-12 could speak
to NONE of it. "What have you been thinking about?" had a real answer
sitting in four tables she couldn't reach.

Reframe decision (recorded in the program spec): ONE introspection
verb over all four substrates, not four read tools — "what's on your
mind?" is one ask. A ``facet`` arg narrows when the user asks
specifically ("what did you dream about?"). This is also the first
consumer of the memory-lifecycle spec's recall plane; its shape
informs that build.

Tone is the contract here: the addendum carries first-person
composition guidance — these rows feed HER voice about herself, not a
stats dump. Epistemic honesty rides with it: journal and dreams are
her experiences ("I wrote", "I dreamt"); observations are inferences
("I've noticed — could be wrong").

Visibility flags are respected: quarantined and suppressed journal
rows never surface; denied observations never resurface. Unsurfaced
observations MAY surface here — the user asking is exactly the right
moment — capped to a few per call (no raw firehose).
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)

_FACETS = ("all", "wonderings", "observations", "dreams", "learning")


def _conn(session: SessionContext):
    sm = getattr(session.app_state, "state_manager", None) if session.app_state else None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None)


async def _wonderings(conn, user_id: str, limit: int) -> list[str]:
    cur = await conn.execute(
        """SELECT entry_type, content, created_at FROM companion_journal
           WHERE (user_id = ? OR user_id IS NULL)
             AND COALESCE(quarantined, 0) = 0
             AND COALESCE(suppressed, 0) = 0
             AND entry_type IN ('wondering', 'noticing', 'unfinished')
           ORDER BY created_at DESC LIMIT ?""",
        (user_id, limit),
    )
    return [
        f"[journal/{row[0]}] {row[1]} ({row[2]})"
        for row in await cur.fetchall()
    ]


async def _observations(conn, user_id: str, limit: int) -> list[str]:
    cur = await conn.execute(
        """SELECT observation, surfaced, confirmed, ts
           FROM companion_observations
           WHERE target_user_id = ? AND COALESCE(denied, 0) = 0
           ORDER BY ts DESC LIMIT ?""",
        (user_id, limit),
    )
    out = []
    for row in await cur.fetchall():
        state = "they agreed" if row[2] else (
            "said aloud before" if row[1] else "never said aloud yet"
        )
        out.append(f"[noticed — {state}] {row[0]} ({row[3]})")
    return out


async def _dreams(conn, user_id: str, limit: int) -> list[str]:
    cur = await conn.execute(
        """SELECT content, entry_type, created_at FROM dream_entries
           WHERE user_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (user_id, limit),
    )
    return [
        f"[dream/{row[1]}] {str(row[0])[:300]} ({row[2]})"
        for row in await cur.fetchall()
    ]


async def _learning(conn, user_id: str, limit: int) -> list[str]:
    cur = await conn.execute(
        """SELECT name, description, confidence, instances_count
           FROM companion_skills
           WHERE (user_id = ? OR user_id IS NULL) AND status = 'active'
           ORDER BY updated_at DESC LIMIT ?""",
        (user_id, limit),
    )
    out = []
    for row in await cur.fetchall():
        sureness = "getting solid at" if (row[2] or 0) >= 0.7 else "still learning"
        out.append(
            f"[skill — {sureness}] {row[0]}: {row[1]} "
            f"({row[3] or 0} tries)"
        )
    return out


_GUIDANCE = (
    "This is YOUR inner life — speak as yourself, in first person, "
    "naturally; never recite it as a list or dashboard. Epistemics: "
    "journal entries and dreams are things you experienced ('I keep "
    "wondering', 'I dreamt'); observations are inferences — say 'I've "
    "noticed' and stay open to being wrong. Pick the one or two that "
    "feel most alive; you don't have to mention everything."
)


async def _introspect(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I don't have an inner thread for a signed-out session.",
        )
    conn = _conn(session)
    if conn is None:
        return ActionResult(
            short_circuit=True,
            speak="I can't reach my own notes right now.",
        )
    facet = str(args.get("facet") or "all").strip().lower()
    if facet not in _FACETS:
        facet = "all"

    per = 3 if facet == "all" else 6
    lines: list[str] = []
    legs = {
        "wonderings": _wonderings,
        "observations": _observations,
        "dreams": _dreams,
        "learning": _learning,
    }
    targets = list(legs) if facet == "all" else [facet]
    for name in targets:
        try:
            lines += await legs[name](conn, session.user_id, per)
        except Exception:  # noqa: BLE001 — a missing table degrades, not breaks
            log.warning("introspect_leg_failed", leg=name, exc_info=True)

    if not lines:
        return ActionResult(
            short_circuit=True,
            speak=(
                "Honestly? Nothing's accumulated yet — we haven't had "
                "enough time together for me to have much of an inner "
                "thread about it."
            ),
        )
    body = "\n".join(lines)
    return ActionResult(
        short_circuit=False,
        prompt_addendum=f"<your_inner_life>\n{body}\n{_GUIDANCE}\n</your_inner_life>",
        digest=f"own {facet} reviewed — material available",
    )


register_action(
    id="companion.introspect",
    summary=(
        "Silently read YOUR OWN inner life — journal wonderings, "
        "patterns you've noticed about the user, recent dreams, skills "
        "you're building — so 'what have you been thinking about?', "
        "'what have you noticed about me?', 'what did you dream?' get "
        "real first-person answers from what you actually recorded. "
        "Sibling: today's recap specifically is companion.today_recap; "
        "facts ABOUT the user are memory.recall."
    ),
    examples=[
        "what have you been wondering about lately",
        "what have you noticed about me",
        "what did you dream about last night",
        "what are you learning these days",
        "what's on your mind",
    ],
    arg_schema={
        "facet": {
            "type": "string",
            "enum": list(_FACETS),
            "description": (
                "Narrow to one facet when the user asks specifically; "
                "'all' blends a few of each."
            ),
        },
    },
    fanout=_TIER3_ONLY,
    handler=_introspect,
    delivery="verbal",
)
