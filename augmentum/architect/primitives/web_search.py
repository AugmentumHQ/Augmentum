"""web.search — architect-callable web search ON THE USER'S SCREEN.

This is the SCREEN twin of the headless ``web_search`` tool, and it is
the EXCEPTION, not the default. It fires only when the user explicitly
wants to LOOK at the result list themselves — "show me the results",
"open a search for X", "pull up", "let me look". It dispatches to the
browse panel (nothing is read back into the model's loop), so the user
scans results naturally.

The far more common request — "search for X", "look up Y", "google Z",
"find info on W" — means "find out and tell me." Those belong to the
headless ``web_search`` tool: it gathers silently and the model answers
in its own words (headless-first doctrine; spec
2026-06-10-companion-headless-agency-design). Routing those here was
the observed failure: she'd open a browser and guess a query instead of
answering. Keep this verb's examples explicitly SHOW-shaped so the
relevance ranker doesn't pull it in for plain look-ups.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import (
    ActionFanout,
    ActionResult,
    SessionContext,
)
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Tier-3-only: the LLM picks this verb based on intent understanding +
# context, not by matching open-slot regex templates. Previous Tier-1
# templates ate phrases like "search the web and tell me about the
# latest news" → query="the web and tell me about the latest news",
# producing nonsense SearXNG queries that returned nothing. The LLM
# handles "search X and Y" by understanding the request shape and
# calling this verb with query="latest news", routing the rest through
# its own response. See [[no-regex-switchboard]] for the design rule.
_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


async def _web_search_handler(
    text: str,
    session: SessionContext,
    args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't search for a signed-out session.",
        )

    query = (args.get("query") or "").strip()
    if not query:
        # Open the panel anyway (folded in from the retired search.web
        # twin) and park the ask so the answer fills the slot.
        return ActionResult(
            short_circuit=True,
            speak="What should I search for?",
            surface_emit={
                "channel": "navigate.open_surface",
                "payload": {"surface": "browse"},
            },
            clarify={"missing": ["query"]},
        )

    # Truncate label for the spoken ack — keep it short.
    short = query[:60].rstrip()
    if len(query) > 60:
        short += "…"

    log.info(
        "architect_web_search",
        user_id=session.user_id, query=query[:80],
    )

    return ActionResult(
        short_circuit=True,
        speak=f"Searching for {short}.",
        surface_emit={
            "channel": "browse.search",
            "payload": {
                "query": query,
                # Empty category = "all results, no filter". The
                # browse panel honors operator-set defaults if any.
                "category": "",
            },
        },
    )


register_action(
    id="web.search",
    summary=(
        "Open web results ON THEIR SCREEN so they can look themselves. "
        "This is the EXCEPTION, not the default: use it ONLY when they "
        "explicitly want to SEE / scan / browse the result list "
        "('show me the results', 'open a search', 'pull up', 'let me "
        "look'). For a normal 'search X' / 'look up X' / 'google X' "
        "where they want the ANSWER, use the web_search tool instead — "
        "gather silently and tell them in your own words."
    ),
    examples=[
        "show me search results for cookie recipes",
        "pull up web results on bezier curves",
        "open a web search for boba shops nearby",
        "let me look at the results myself",
        "show me what comes up for that",
    ],
    handler=_web_search_handler,
    delivery="artifact",
    arg_schema={
        "query": {
            "type": "string",
            "description": (
                "The keyword query to run on their screen — design it for "
                "a search engine, don't echo their sentence. Drop filler "
                "and the 'show me / pull up' framing, keep the salient "
                "keywords; \"exact phrase\" and site:domain.com operators "
                "work here too (same SearXNG backend as web_search)."
            ),
        },
    },
    required=["query"],
    surfaces=["becca", "chat"],
    stakes="trivial_reversible",
    fanout=_TIER3_ONLY,
)
