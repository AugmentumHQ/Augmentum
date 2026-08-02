"""browse.find — find a recently-visited article matching a query.

User: "find that article about X", "open the article about Y",
"pull up the page about Z", "show me that thing I was reading
about W". Imperative-only.

The substrate moat: ``browse_history`` carries the per-user visit
record. The inferrer substring-matches the user's query against the
domain + URL path tokens, returns the freshest matching entry. The
handler emits ``browse.open_url`` so the browse panel opens the
article directly.

When nothing matches, the handler falls back to opening a search
for the query rather than refusing — "find that article about X"
becomes a search for X, which is usually what the user wanted
when their memory is hazy.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import (
    ActionFanout,
    ActionResult,
    SessionContext,
)

# Tier-3-only: LLM picks based on intent + context, not open-slot regex.
# See [[no-regex-switchboard]].
_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def _conn_from_runtime(runtime: Any) -> Any:
    if runtime is None:
        return None
    sm = getattr(runtime, "state_manager", None)
    if sm is None:
        app_state = getattr(runtime, "_app_state", None)
        if app_state is not None:
            sm = getattr(app_state, "state_manager", None)
    if sm is None:
        return None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None) if backend else None


async def _infer_browse_find_args(
    partial_args: dict[str, Any],
    session: SessionContext,
    runtime: Any,
) -> dict[str, Any]:
    """Substring-match the user's query against browse_history.

    Searches both ``domain`` and the URL path (token-split on /-_)
    so "find that article about bezier curves" matches a URL like
    ``en.wikipedia.org/wiki/Bezier_curve``.
    """
    from augmentum.architect.inference import query_browse_history

    args = dict(partial_args)
    query = (args.get("query") or "").strip().lower()
    if not query:
        return args

    conn = await _conn_from_runtime(runtime)
    if conn is None or not session.user_id:
        return args

    history = await query_browse_history(conn, session.user_id, limit=30)
    if not history:
        return args

    # Split query into significant tokens (drop the obvious filler).
    fillers = {"the", "a", "an", "about", "on", "that", "thing", "article", "page"}
    query_tokens = [t for t in query.split() if t and t not in fillers]
    if not query_tokens:
        return args

    best = None
    best_score = 0
    for row in history:
        url = (row.get("url") or "").lower()
        domain = (row.get("domain") or "").lower()
        if not url:
            continue
        # Token-decompose the path for substring matching.
        path_tokens = url.replace("/", " ").replace("-", " ").replace("_", " ").split()
        path_str = " ".join(path_tokens)
        # Score = number of query tokens that appear in URL/domain.
        score = sum(1 for t in query_tokens if t in path_str or t in domain)
        if score > best_score:
            best_score = score
            best = row

    # Require at least one matching token — otherwise the freshest
    # entry would always win even when totally unrelated.
    if best and best_score >= 1:
        args["resolved_url"] = best.get("url") or ""
        args["resolved_title"] = best.get("domain") or ""
        args["match_score"] = best_score

    return args


async def _browse_find_handler(
    text: str,
    session: SessionContext,
    args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't browse for a signed-out session.",
        )

    query = (args.get("query") or "").strip()
    if not query:
        return ActionResult(
            short_circuit=True,
            speak="What article should I find?",
        )

    resolved_url = (args.get("resolved_url") or "").strip()
    resolved_title = (args.get("resolved_title") or "").strip()

    if resolved_url:
        # Found a match in browse_history — open it directly.
        log.info(
            "architect_browse_find_hit",
            user_id=session.user_id, query=query[:80],
            url=resolved_url[:120], score=args.get("match_score", 0),
        )
        short = query[:50]
        return ActionResult(
            short_circuit=True,
            speak=f"Opening {resolved_title or short}.",
            surface_emit={
                "channel": "browse.open_url",
                "payload": {
                    "url": resolved_url,
                    "query": query,
                },
            },
        )

    # No history match — fall back to a fresh web search.
    log.info(
        "architect_browse_find_fallback_search",
        user_id=session.user_id, query=query[:80],
    )
    return ActionResult(
        short_circuit=True,
        speak=f"I don't see that in your history. Searching for {query[:50]}.",
        surface_emit={
            "channel": "browse.search",
            "payload": {"query": query, "category": ""},
        },
    )


register_action(
    id="browse.find",
    summary=(
        "Find and open a previously-visited article matching the "
        "user's query. Substring-matches against browse_history; "
        "falls back to a fresh web search when nothing in history "
        "matches."
    ),
    examples=[
        "find that article about bezier curves",
        "open the article about react hooks",
        "pull up the page about jazz history",
        "show me that thing I was reading about miles davis",
        "find the page about python decorators",
    ],
    handler=_browse_find_handler,
    delivery="artifact",
    arg_schema={
        "query": {
            "type": "string",
            "description": "Topic / keyword to find in the user's browse history.",
        },
    },
    required=["query"],
    surfaces=["becca", "chat"],
    stakes="trivial_reversible",
    arg_inferrer=_infer_browse_find_args,
    fanout=_TIER3_ONLY,
)
