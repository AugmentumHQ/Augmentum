"""Navigation actions — open / focus / switch surfaces.

These don't address Becca's reasoning so much as the application's
chrome. They short-circuit to a surface event the frontend router
turns into a toolbar-button-equivalent action. Most ship without a
spoken reply; users opening Browse don't want a TTS "okay, opening
browse" delaying their next ask.

Surface IDs match the frontend route map in
``ui/scripts/intent-action-router.js`` — keep them in sync.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionResult, SessionContext
from augmentum.intent.registry import register_action

# Canonical surface identifiers. The frontend router maps each to the
# appropriate opener (toggleBrowse, openCoder, etc.). Adding a surface
# means: append the id here, add a route case in the FE router, then
# (optionally) register a dedicated alias action below for natural
# phrasing.
_SURFACES = [
    "browse", "notes", "files", "coder", "grove",
    "today", "observatory", "settings", "discovery",
    "studio", "library", "marketplace", "agent", "voice",
    "youtube",
]
_SURFACE_PATTERN = "|".join(_SURFACES)


async def _navigate_open(
    _text: str, _session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    surface = args.get("surface", "").lower()
    if surface not in _SURFACES:
        return None
    # Surface a small toast so the user knows she heard them — without
    # a spoken ack, a successful navigate looks identical to "nothing
    # happened" (the panel slides in silently).
    pretty = {
        "today": "Today", "observatory": "Observatory",
        "youtube": "YouTube",
    }.get(surface, surface.capitalize())
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "navigate.open_surface",
            "payload": {"surface": surface},
        },
        toast=f"Opening {pretty}",
    )


register_action(
    id="navigate.open_surface",
    summary="Open or focus a top-level surface (browse, notes, coder, etc.).",
    examples=[
        "open browse", "go to notes", "show me my files",
        "open the coder", "pull up settings", "open grove",
        "show me today", "open observatory",
    ],
    patterns=[
        # "open <surface>", "go to <surface>", "show <surface>", etc.
        # Capture the surface name into ``args.surface`` so the handler
        # can route. The trailing optional words ("panel", "tab", "the
        # ... section") absorb chatty phrasing without dropping the
        # match.
        rf"\b(?:open|go to|show(?: me)?|pull up|switch to)\s+"
        rf"(?:(?:the|my)\s+)?(?P<surface>{_SURFACE_PATTERN})"
        rf"(?:\s+(?:panel|tab|surface|section|view))?\b",
    ],
    arg_schema={
        "surface": {"type": "string", "enum": _SURFACES},
    },
    required=["surface"],
    handler=_navigate_open,
    delivery="artifact",
)


async def _navigate_back(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    return ActionResult(
        short_circuit=True,
        surface_emit={"channel": "navigate.back", "payload": {}},
        toast="Closed",
    )


register_action(
    id="navigate.back",
    summary="Navigate back — close the active overlay or pop the view stack.",
    examples=["go back", "back", "close that", "close this"],
    patterns=[
        r"^(?:go )?back$",
        r"\bclose (?:that|this|it)\b",
    ],
    handler=_navigate_back,
    delivery="artifact",
)
