"""app.act — press the app's own buttons by intent.

The long-tail companion to the verb registry (spec conversation
2026-06-10, the Apple App Intents / Windows App Actions pattern):
surfaces register outcome-shaped palette commands with agent metadata,
the client syncs the live catalog (``augmentum/intent/app_menu.py``),
and this ONE verb matches the user's ask against that closed,
stakes-capped menu and fires the matching client handler via the
``palette.run`` surface channel.

Division of labor with real verbs: verbs keep the head of the
distribution (frequent, argument-carrying, headless-capable,
data-returning); app.act covers arg-less context-bound "the thing on
screen right now" actions for the cost of a registration line each.

By construction this verb is client-coupled — a live entry means the
user's browser evaluated its ``when`` guard truthy moments ago, so a
connected surface is a given, and artifact delivery is the only
register that makes sense (the button's own UI feedback IS the ack).
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


async def _app_act(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        log.warning("app_act_no_user")
        return ActionResult(
            short_circuit=True,
            speak="I can't press buttons for a signed-out session.",
        )

    from augmentum.intent.app_menu import MENU, match_intent

    want = (args.get("intent") or "").strip() or (text or "").strip()
    if not want:
        return ActionResult(
            short_circuit=True,
            speak="What should I do?",
        )

    candidates = MENU.catalog(session.user_id)
    if not candidates:
        return ActionResult(
            short_circuit=True,
            speak=(
                "I don't have any quick actions available for what's "
                "on screen right now."
            ),
        )

    entry = await match_intent(
        want, candidates,
        app_state=session.app_state,
        user_id=session.user_id,
        session_id=session.session_id,
    )

    if entry is None:
        # Honest miss — name the nearest live options instead of
        # guessing (closed-world discipline; matcher already biased
        # toward none).
        names = ", ".join(e["description"] for e in candidates[:3])
        return ActionResult(
            short_circuit=True,
            speak=(
                f"I don't have a button for that. What I can do from "
                f"here: {names}."
            ),
        )

    speak = entry["speak"] or f"Done — {entry['description'].lower()}."
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "palette.run",
            "payload": {
                "command_id": entry["id"],
                "label": entry["description"],
            },
        },
        speak=speak,
        toast=f"→ {entry['description'][:60]}",
    )


register_action(
    id="app.act",
    summary=(
        "Press one of the app's own quick actions by describing the "
        "outcome — favorite the current station, toggle shuffle, and "
        "other context actions registered by the surface the user is "
        "looking at. Use for one-tap asks that no dedicated tool "
        "covers; the app matches the request to a live button and "
        "presses it."
    ),
    examples=[
        "add this to my favorites",
        "favorite this station",
        "I love this one, save it",
        "toggle shuffle",
        "do the thing on screen for me",
    ],
    arg_schema={
        "intent": {
            "type": "string",
            "description": (
                "What the user wants done, in plain words — e.g. 'add "
                "the current station to favorites'. The app finds the "
                "matching on-screen action."
            ),
        },
    },
    required=["intent"],
    surfaces=("becca", "chat"),
    fanout=_TIER3_ONLY,
    handler=_app_act,
    delivery="artifact",
)
