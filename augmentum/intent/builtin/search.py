"""Search verbs — local / knowledge / web.

LLM-orchestrated capabilities (Tier 3 only). The model picks the right
search surface based on intent understanding plus context, NOT by
matching qualifier words ("my notes", "my files") in the transcript.
The switchboard pattern conflates user phrasing with intent;
"search my notes for fermentation" and "look in my fermentation
journal for the saison recipe" should land on the same verb even
though the surface phrasing differs.

Two search surfaces share the same "find something" mental model:

  * ``search.local``     — Files panel filtered to user content
  * ``search.knowledge`` — Browse panel scoped to knowledge packs
                            (Wikipedia / MDWiki / Stack Exchange / etc.)

Open-internet SCREEN search is ``web.search`` (architect primitive,
augmentum/architect/primitives/web_search.py) — a ``search.web``
twin that lived here was retired 2026-06-11 (see note below).
Headless web search (results returned INTO the model's loop, nothing
on screen) is the ``web_search`` / ``web`` chat tools.

Both verbs here use existing front-end channels (browse.search,
files.search_open) — no new router wiring required.

Surface IDs in dispatches MUST match the channels in
``ui/scripts/intent-action-router.js``.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


# ── Local files ─────────────────────────────────────────────────────

async def _search_local(
    _text: str, _session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    query = (args.get("query") or "").strip()
    if not query:
        return ActionResult(
            short_circuit=True,
            surface_emit={
                "channel": "navigate.open_surface",
                "payload": {"surface": "files"},
            },
            toast="Opening Files",
        )
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "files.search_open",
            "payload": {"query": query},
        },
        toast=f"Finding: {query}",
    )


register_action(
    id="search.local",
    summary=(
        "Search their own files on the user's screen (Files panel) — "
        "uploads, media library, documents, artifacts, anything stored "
        "on the Augmentum host. Use to BROWSE several results: their "
        "own content (notes they wrote, docs they uploaded, media they "
        "added) as opposed to web or knowledge-pack content. Siblings: "
        "to open ONE specific file directly, files.find; for reference "
        "corpora, search.knowledge."
    ),
    examples=[
        "find on my computer the foundation",
        "find in my files my dissertation",
        "search my files for budget spreadsheet",
        "find me some sci-fi in my library",
        "look for the foundation books",
        "find audiobooks about ancient rome",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": "Search terms scoped to local user content.",
        },
    },
    fanout=_TIER3_ONLY,
    handler=_search_local,
)


# ── Knowledge packs ─────────────────────────────────────────────────

async def _search_knowledge(
    _text: str, _session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    query = (args.get("query") or "").strip()
    if not query:
        return ActionResult(
            short_circuit=True,
            surface_emit={
                "channel": "navigate.open_surface",
                "payload": {"surface": "browse"},
            },
            toast="Opening Browse",
        )
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "browse.search",
            "payload": {"query": query, "category": "knowledge"},
        },
        toast=f"Looking up: {query}",
    )


register_action(
    id="search.knowledge",
    summary=(
        "Search offline reference packs on the user's screen — "
        "Wikipedia, MDWiki, Stack Exchange, DevDocs corpora they have "
        "attached. Use for factual/reference lookups when the user "
        "wants authoritative encyclopedic content rather than open-web "
        "results or their own files."
    ),
    examples=[
        "look up the battle of agincourt",
        "look up python list comprehensions",
        "search my notes for fermentation",
        "find in knowledge about quantum tunneling",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": "Topic or fact to look up in attached knowledge packs.",
        },
    },
    fanout=_TIER3_ONLY,
    handler=_search_knowledge,
)


# ── Web (open internet) ─────────────────────────────────────────────
# Retired 2026-06-11: ``search.web`` duplicated the architect primitive
# ``web.search`` byte-for-byte in effect (same browse.search surface
# emit, same payload). Two names for one capability fragmented model
# behavior across families — companion_eval caught different models
# reaching for different aliases. ``web.search`` (voice-manifest
# canonical) absorbed the empty-query open-Browse fallback. The
# local/knowledge verbs above keep the search.* namespace; open-web
# screen search is augmentum/architect/primitives/web_search.py.
