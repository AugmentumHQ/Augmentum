"""Trail verbs — "take me there".

Headless-agency P2 (spec 2026-06-10): while the companion works
invisibly (web searches, page reads via the native loop), her trail
records every position she visits (``ReferentCache.trail``, appended
by ``companion_runtime/native_loop.py``). This verb is the user's
window in: it jumps THEIR screen to HER latest position.

Symmetric inverse of the presence layer — presence is her window into
the user's attention; the trail is the user's window into hers.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


async def _take_me_there(
    _text: str, session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    trail = list(getattr(session.referents, "trail", None) or [])
    if not trail:
        return ActionResult(
            short_circuit=True,
            speak=(
                "I haven't gone anywhere yet this conversation — "
                "ask me to look something up first."
            ),
        )

    head = trail[-1]
    kind = head.get("kind") or ""
    label = head.get("label") or ""
    ref = head.get("ref") or ""

    if kind == "page" and ref:
        # She read a page — open the browse panel at that exact URL.
        return ActionResult(
            short_circuit=True,
            surface_emit={
                "channel": "browse.open_url",
                "payload": {"url": ref, "query": label},
            },
            speak=f"Here — this is the page I was reading: {label}.",
            toast=f"→ {label[:60]}",
        )
    if kind == "coder_run" and ref:
        # She delegated a build — jump to the workspace she's working in. The
        # run's brief opens on its own when it completes; this is "show me
        # where you're building" in the meantime.
        return ActionResult(
            short_circuit=True,
            surface_emit={
                "channel": "coder.open_workspace",
                "payload": {"workspace_id": ref},
            },
            speak=f"Here — this is the workspace I'm building in: {label}.",
            toast=f"→ {label[:60]}",
        )
    if kind == "search" and label:
        # She ran a search — re-run it visibly in the browse panel.
        return ActionResult(
            short_circuit=True,
            surface_emit={
                "channel": "browse.search",
                "payload": {"query": label, "category": ""},
            },
            speak=f"Pulling up what I searched: {label}.",
            toast=f"→ search: {label[:50]}",
        )

    # Unknown trail kind — name it honestly rather than guessing a surface.
    return ActionResult(
        short_circuit=True,
        speak=(
            f"The last place I went was {label or kind or 'somewhere'} — "
            "I don't have a screen view for that one yet."
        ),
    )


register_action(
    id="companion.take_me_there",
    summary=(
        "Open the user's screen at the companion's latest working "
        "position — the page she read or search she ran while "
        "answering. Use when the user says 'take me there', 'show me "
        "where you found that', 'pull that up', or 'let me see it' "
        "after she gathered information."
    ),
    examples=[
        "take me there", "show me where you found that",
        "pull that page up", "let me see it", "open what you found",
    ],
    surfaces=("becca", "chat"),
    fanout=_TIER3_ONLY,
    handler=_take_me_there,
    delivery="artifact",
)
