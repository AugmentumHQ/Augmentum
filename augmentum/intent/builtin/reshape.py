"""Reshape verb — "change the app for me," from any surface.

The one verb that makes "make it darker", "denser panels", "switch to light"
work the SAME whether typed in chat, spoken, or asked of the companion — because
every surface dispatches this one action and renders the one ``present()`` payload
it emits (``reshape.result``). It runs the surface-reshape engine
(`selfedit.surfaces`): classify the ask against the live surface catalog →
apply reversibly → verify by oracle-tier → record → present.

Sovereign by default (the model that classifies is the user's model list; Claude
is opt-in via the source layer). User-initiated only (pull, not push). Writes are
scoped to ``session.user_id``; refuses the anon row. Enablement is gated upstream
(the ``selfedit_enabled`` master switch / route), consistent with the API path.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Reachable both as an LLM tool (tier 3) and via fast regex for the common
# phrasings (tier 1) so "make it dark" lands instantly.
_FANOUT = ActionFanout(tier1=True, tier2=False, tier3=True)

_PATTERNS = [
    r"\bmake it (?:darker|lighter|dark|light)\b",
    r"\b(?:switch|change) (?:to )?(?:dark|light) (?:mode|theme)\b",
    r"\b(?:make|set) (?:the )?(?:panels?|layout|ui) (?:more )?(?:denser|dense|compact|comfortable|roomier)\b",
    r"\bchange (?:the )?theme\b",
    r"\b(?:rearrange|reshape|adjust) (?:the )?(?:app|ui|layout|interface)\b",
]


def _conn(app_state: Any) -> Any:
    backend = getattr(getattr(app_state, "state_manager", None), "backend", None)
    return getattr(backend, "conn", None) if backend else None


async def _reshape(text: str, session: SessionContext, args: dict[str, Any]) -> ActionResult:
    if not session.user_id:
        return ActionResult(short_circuit=True, fulfilled=False,
                            speak="I can only reshape things for a signed-in user.")
    app_state = session.app_state
    ask = str(args.get("ask") or text or "").strip()
    if not ask:
        return ActionResult(short_circuit=True, fulfilled=False,
                            speak="Tell me what to change.")

    # Build the classifier from the user's chosen model SOURCE (model list by
    # default; Claude opt-in). resolve_invoke falls back to the model list.
    from augmentum.selfedit.surfaces import (
        DEFAULT_SOURCE,
        SourceContext,
        build_model_classifier,
        handle_reshape_ask,
        resolve_invoke,
    )

    ctx = SourceContext(
        user_id=session.user_id,
        provider_registry=getattr(app_state, "provider_registry", None),
        settings_store=getattr(app_state, "settings", None))
    source_id = str(args.get("source") or DEFAULT_SOURCE)
    invoke = await resolve_invoke(source_id, ctx)
    if invoke is None:
        return ActionResult(short_circuit=True, fulfilled=False,
                            speak="I can't reach a model to understand that change right now.")

    try:
        pres = await handle_reshape_ask(
            ask, session.user_id, classify=build_model_classifier(invoke),
            conn=_conn(app_state), surface_hint=session.mode or "")
    except Exception as exc:  # noqa: BLE001 — never crash the turn; report honestly
        log.warning("reshape_verb_failed", error=repr(exc))
        return ActionResult(short_circuit=True, fulfilled=False,
                            speak="Something went wrong making that change.")

    # One payload, every surface: chat renders the card+chips, voice speaks +
    # listens for keep/undo, the Workshop shows the row — all from present().
    fulfilled = pres.status not in ("unmapped", "failed")
    return ActionResult(
        short_circuit=True,
        fulfilled=fulfilled,
        speak=pres.speech,
        toast=pres.headline,
        surface_emit={"channel": "reshape.result", "payload": pres.to_dict()})


register_action(
    id="surface.reshape",
    summary=(
        "Change Augmentum's own appearance/behavior for this user on request — "
        "theme, layout density, which panels show, and other per-user adaptations. "
        "Use when the user asks to change the app/UI itself: 'make it darker', "
        "'denser panels', 'switch to light theme', 'rearrange the layout'. NOT for "
        "content (notes, media, search) — only for reshaping the app surface."
    ),
    examples=[
        "make it darker", "switch to light theme", "make the panels denser",
        "use a more compact layout", "rearrange the app for me",
    ],
    patterns=_PATTERNS,
    surfaces=["becca", "chat"],
    fanout=_FANOUT,
    stakes="trivial_reversible",
    handler=_reshape,
    delivery="verbal",
)
