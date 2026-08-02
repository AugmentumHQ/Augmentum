"""companion.today_recap — open Today + speak the in-voice reflection.

The Today surface (``docs/.../today``) is Becca's daily journal-to-
the-user — a short prose entry in her own voice summarizing what she
observed today (notes the user wrote, media they engaged with,
dispatches she made on their behalf). It's the most relational-compute
piece in the product: a daily artifact she made for the user, not a
metrics dashboard.

This primitive lets the user verbally invoke it. The inferrer reads
the current day's reflection (``companion_runtime.today.get_today``)
into ``content_text`` so the handler can speak a short excerpt
alongside opening the surface. When there's no reflection yet (early
in the day, or presence mode silent), the handler still opens the
surface so the user sees the hint state.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _infer_today_args(
    partial_args: dict[str, Any],
    session: SessionContext,
    runtime: Any,
) -> dict[str, Any]:
    """Pull today's reflection content for the spoken excerpt."""
    args = dict(partial_args)
    if runtime is None or not session.user_id:
        return args
    try:
        from augmentum.companion_runtime import today as _today
        reflection = await _today.get_today(runtime, user_id=session.user_id)
    except Exception as exc:  # noqa: BLE001 — degrade to no excerpt
        log.debug("architect_today_get_failed", error=str(exc)[:200])
        return args
    if reflection is None:
        return args
    args["today_excerpt"] = (reflection.content_text or "").strip()
    args["today_date"] = reflection.date_local or ""
    return args


def _short_excerpt(text: str, max_chars: int = 180) -> str:
    """Pick a TTS-friendly leading slice — one or two sentences."""
    if not text:
        return ""
    # Prefer sentence boundary close to the limit; fall back to a hard cap.
    cut = text[: max_chars + 40]
    # Find the latest sentence-ending punctuation within the soft window.
    for end in (". ", "? ", "! "):
        idx = cut.rfind(end, 0, max_chars + 1)
        if idx > 40:  # avoid microscopic fragments
            return cut[: idx + 1].strip()
    return text[:max_chars].rstrip() + "…"


async def _companion_today_handler(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't show today's reflection for a signed-out session.",
        )

    excerpt = _short_excerpt((args.get("today_excerpt") or "").strip())

    log.info(
        "architect_companion_today",
        user_id=session.user_id,
        has_excerpt=bool(excerpt),
        date=args.get("today_date") or "",
    )

    # Always open the surface; the spoken line varies by whether we
    # have content. When the reflection isn't generated yet, the
    # Today panel surfaces its own hint UI — the spoken line just
    # acknowledges that.
    if excerpt:
        speak = excerpt
    else:
        speak = (
            "I haven't written today's reflection yet — opening the "
            "Today panel so you can see what I've got so far."
        )

    return ActionResult(
        short_circuit=True,
        speak=speak,
        surface_emit={
            "channel": "navigate.open_surface",
            "payload": {"surface": "today"},
        },
    )


register_action(
    id="companion.today_recap",
    summary=(
        "Open the Today reflection surface — the companion's in-voice "
        "journal of what they observed today — and speak a short "
        "excerpt. When no reflection is ready yet, opens the panel "
        "anyway so the user sees the live-state hint."
    ),
    examples=[
        "what did we do today",
        "what's today's reflection",
        "what did you write today",
        "read me today's journal",
        "open today",
        "show me today's notes",
        "what's on your mind today",
        "today recap",
        "recap today",
        "summarize my day",
        "daily summary",
    ],
    # Direct-phrasing regex — sub-100ms hits for the most common short
    # asks. Templates below cover the wordier inquiry forms. These
    # patterns intentionally do NOT overlap with grove.play_matching
    # ("play today's hits") or notes.start_capture — anchored on
    # "today" + a recap/summary word.
    patterns=[
        # "today recap", "today's recap", "recap (of|for) today"
        r"\btoday'?s? recap\b",
        r"\brecap (?:of |for )?today\b",
        # "summarize my day", "summarize today", "summary of today",
        # "daily summary"
        r"\bsummari[sz]e (?:my day|today)\b",
        r"\bsummary of today\b",
        r"\bdaily summary\b",
        # "what did I do today" / "what have I done today" — first-
        # person variants the inquiry-form templates miss.
        r"\bwhat (?:did|have) I (?:do|done|been (?:doing|up to))\s+today\b",
    ],
    handler=_companion_today_handler,
    delivery="artifact",
    surfaces=["becca", "chat"],
    stakes="trivial_reversible",
    arg_inferrer=_infer_today_args,
    templates=[
        # Direct imperative forms
        "(open|show|read) [me] [(the|your)] today [(reflection|journal|recap|note)]",
        "(read|show) [me] today's (reflection|journal|recap|note|entry)",
        # Possessive / inquiry forms — "what did we do today" matches the
        # spirit even though "what did we" is WH-question-shaped. The
        # template lint will warn (acceptable: this primitive IS the
        # right home for that conversational ask, not the LLM).
        "what did (we|you) (do|work on|talk about) today",
        "what's [(on your mind|today's)] (reflection|journal|recap|today)",
    ],
)
