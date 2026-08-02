"""Chat session verbs — new session, history browse.

LLM-orchestrated capabilities (Tier 3 only). The model picks these
verbs based on intent, not regex on phrases like "new chat." That
switchboard pattern misfires on conversational uses ("I started a
new chat with my doctor yesterday") and forces awkward phrasings
the user wouldn't naturally say.

Currently ships:

  * ``chat.new``     — start a fresh session and switch focus to it.
  * ``chat.history`` — open the chats history surface.

Switching to a session by topic ("switch to my coder chat") is
deferred — needs session-by-title resolution against ``ui_sessions``
and a session-picker UI surface.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action


_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


async def _chat_new(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    # Front-end router maps ``chat.new`` to the existing
    # ``augmentum:new-session`` CustomEvent that chat/index.js already
    # listens for (same event the Ctrl+Shift+S shortcut fires). Zero
    # new wiring on the chat side — the verb just gives Becca a route
    # into the same path.
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "chat.new",
            "payload": {},
        },
        toast="New chat",
    )


register_action(
    id="chat.new",
    summary=(
        "Start a fresh chat session for the user, separate from the "
        "current conversation. Use when the user wants to begin a new "
        "topic, reset context, or set aside the current thread."
    ),
    examples=[
        "new chat", "start a new chat", "open a new conversation",
        "fresh chat please", "let's start over",
    ],
    fanout=_TIER3_ONLY,
    handler=_chat_new,
    delivery="artifact",
)


async def _chat_history(
    _text: str, _session: SessionContext, _args: dict[str, Any],
) -> ActionResult:
    # No dedicated history panel — the sidebar lists sessions, and the
    # Settings panel has a Chats tab. Surfaces Settings as the canonical
    # "let me find old chats" entrypoint.
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "navigate.open_surface",
            "payload": {"surface": "settings"},
        },
        toast="Chat history",
    )


register_action(
    id="chat.history",
    summary=(
        "Open the chat history surface where the user can browse past "
        "sessions. Use when the user wants to find or revisit an "
        "earlier conversation."
    ),
    examples=[
        "show me my chat history", "show recent chats",
        "find an old chat", "where are my past conversations",
    ],
    fanout=_TIER3_ONLY,
    handler=_chat_history,
    delivery="artifact",
)
