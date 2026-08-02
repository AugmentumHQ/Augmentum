"""files.find — find a file in the user's VFS by query.

Sibling of ``browse.find`` (which searches browse_history). This one
searches the unified file index — uploads, synced media library
entries, document chunks, image generations, knowledge-pack source
docs. Anything that's been registered with the VFS adapter layer is
reachable from voice.

Inferrer queries ``FileIndexService.search`` (FTS5 + optional vector
fallback) on the user's scoped rows. The handler emits ``files.open``
with the matched file_id when confident, otherwise falls back to
``navigate.open_surface`` for the Files panel with the query
pre-populated so the user can pick.

Surface scoping: becca + chat. The full-screen voice call modal isn't
the right place to open a file — the file viewer needs UI real estate
the modal hides. The voice call's LLM can still tool-call this and
get a graceful surface-filtered no-op; that's fine.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

# Tier-3-only: LLM picks based on intent + context. See [[no-regex-switchboard]].
_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)

log = get_logger(__name__)


# Filler tokens stripped before FTS query construction. The matcher
# captures the natural "find me my resume" phrasing into ``query``;
# the inferrer + this filter clean it to just the meaningful tokens.
_FILLER_TOKENS = frozenset({
    "the", "a", "an", "my", "me", "that", "this", "called",
    "named", "titled", "about", "on", "of", "with",
    "any", "some", "thing", "stuff", "file", "doc", "document",
})


def _clean_query(raw: str) -> str:
    tokens = [t for t in raw.lower().split() if t and t not in _FILLER_TOKENS]
    return " ".join(tokens)


# (The old single-result _search_file_index helper retired 2026-06-11
# — resolution now goes through augmentum/retrieval/fabric.py, which
# adds the confidence-gated act/offer/miss lifecycle this handler
# used to lack.)


async def _files_find_handler(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't search files for a signed-out session.",
        )

    query = (args.get("query") or "").strip()
    # Fallback: some Tier 1 hits (literal example matches like "find my
    # resume") match without a slot capture. Re-derive the query from
    # the raw transcript by stripping the imperative head + filler
    # tokens. Mirrors the same pattern in time_timer.py.
    if not query and text:
        cleaned = _clean_query(text)
        if cleaned:
            query = cleaned
    if not query:
        return ActionResult(
            short_circuit=True,
            speak="What file are you looking for?",
        )

    cleaned_query = _clean_query(query)

    # Resolve through the retrieval fabric (single index leg → the
    # media resolver's confidence gates apply). The old path took
    # hits[0] blind — a weak top hit opened the WRONG file confidently;
    # the fabric demotes that to an offer or an honest panel fallback.
    app_state = getattr(session, "app_state", None)
    resolution = None
    if app_state is not None and cleaned_query:
        try:
            from augmentum.retrieval.fabric import resolve as fabric_resolve
            resolution = await fabric_resolve(
                cleaned_query,
                user_id=session.user_id,
                app_state=app_state,
                sources=("index",),
            )
        except Exception:  # noqa: BLE001 — degrade to the panel fallback
            log.warning("files_find_fabric_failed", exc_info=True)

    if resolution is not None and resolution.outcome == "act" and resolution.item:
        top = resolution.item
        log.info(
            "architect_files_find_hit",
            user_id=session.user_id, file_id=top.id,
            title=top.title[:80], query=query[:80],
        )
        short = (top.title or query)[:60]
        return ActionResult(
            short_circuit=True,
            speak=f"Opening {short}.",
            surface_emit={
                "channel": "files.open",
                "payload": {
                    "file_id": top.id,
                    "title": top.title,
                    "kind": top.kind,
                    "source": top.source,
                },
            },
        )

    if resolution is not None and resolution.outcome == "offer" and resolution.candidates:
        # Close matches but no confident winner — name them and open
        # the panel pre-filtered so one tap finishes the job.
        names = ", ".join(c.title[:40] for c in resolution.candidates[:3])
        log.info(
            "architect_files_find_offer",
            user_id=session.user_id, query=query[:80],
            count=len(resolution.candidates),
        )
        return ActionResult(
            short_circuit=True,
            speak=f"A few close matches — {names}. I've pulled them up.",
            surface_emit={
                "channel": "files.search_open",
                "payload": {"query": cleaned_query or query},
            },
        )

    # No confident match — open the Files panel with the query so the
    # user can browse. Better than refusing.
    log.info(
        "architect_files_find_fallback_panel",
        user_id=session.user_id, query=query[:80],
    )
    return ActionResult(
        short_circuit=True,
        speak=f"I couldn't find {query[:50]} — opening your files so you can take a look.",
        surface_emit={
            "channel": "files.search_open",
            "payload": {"query": query},
        },
    )


register_action(
    id="files.find",
    summary=(
        "Find and OPEN one specific file of theirs on screen (uploads, "
        "synced media, documents, image generations, knowledge "
        "sources). FTS-searches the file index scoped to the user; "
        "falls back to opening the files panel with the query when no "
        "confident match exists. Sibling: to browse several results "
        "instead of opening one, search.local."
    ),
    examples=[
        "find my resume",
        "open the PDF about quantum computing",
        "find the document called proposal",
        "pull up that file about jazz history",
        "open the spreadsheet about budgets",
        "find that image of the sunset",
    ],
    handler=_files_find_handler,
    delivery="artifact",
    arg_schema={
        "query": {
            "type": "string",
            "description": "Keyword or title fragment to search for.",
        },
    },
    required=["query"],
    surfaces=["becca", "chat"],
    stakes="trivial_reversible",
    fanout=_TIER3_ONLY,
)
