"""Coder knowledge-pack search — offline reference retrieval.

Until 2026-07-06 the coder had NO reach into knowledge packs: DevDocs /
Wikipedia ZIMs and augpacks were chat-injection-only, so every API or
stdlib reference lookup paid the open-web tax (doc_search → SearXNG →
rerank → hope), which is exactly where small models fail — bad queries,
SEO noise, spam loops. Pack retrieval is the opposite shape: curated
closed corpus, deterministic Xapian/FTS+vector hybrid, <50ms warm.

Design decisions (Matt, 2026-07-06):
- The tool DESCRIPTION lists the installed packs (name + curation
  date) so the model knows what's reachable without a probe call —
  and knows a language ISN'T covered without wasting an iteration.
  Tool schemas are rebuilt each turn, so the list stays current.
- With no packs installed the description says so and steers the
  model to doc_search; execute() degrades to a pointer, never an
  error loop.
"""
from __future__ import annotations

from augmentum.coder.tools import _CoderTool, _truncate
from augmentum.knowledge.runtime import get_pack_manager
from augmentum.tools.base import ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_MAX_RESULTS = 6
_SNIPPET_CHARS = 900


def _installed_reference_packs() -> list[dict]:
    """Active, searchable (non-language) packs, or [] when unavailable."""
    mgr = get_pack_manager()
    if mgr is None:
        return []
    try:
        return [
            p for p in mgr.installed
            if p.get("active") and p.get("pack_id")
        ]
    except Exception:
        log.warning("pack_search_listing_failed", exc_info=True)
        return []


def _pack_inventory_line() -> str:
    """One-line inventory for the tool description: name (curated date)."""
    packs = _installed_reference_packs()
    if not packs:
        return ""
    parts = []
    for p in packs[:20]:
        name = p.get("name") or p.get("pack_id")
        date = (p.get("build_date") or "").strip()
        parts.append(f"{name} (curated {date})" if date else str(name))
    return "; ".join(parts)


class PackSearchTool(_CoderTool):
    """Hybrid search over the locally installed knowledge packs."""

    @property
    def name(self) -> str:
        return "pack_search"

    @property
    def description(self) -> str:
        inventory = _pack_inventory_line()
        if not inventory:
            return (
                "Search locally installed offline knowledge packs "
                "(DevDocs / reference corpora). NO PACKS ARE CURRENTLY "
                "INSTALLED — do not call this tool; use doc_search for "
                "the live web instead. (The user can install packs from "
                "Settings → Knowledge.)"
            )
        return (
            "Search the locally installed offline knowledge packs — "
            "FIRST RESORT for API / stdlib / library reference lookups "
            "(faster and cleaner than the live web; no SEO spam). "
            f"Installed packs: {inventory}. "
            "If the language/library you need is not in that list, use "
            "doc_search instead — and prefer doc_search for anything "
            "time-sensitive (new releases, comparisons, news): packs "
            "are snapshots as of their curation date."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword query (3-8 terms): symbol/API name + what about it (e.g. 'asyncio TaskGroup cancellation').",
                },
            },
            "required": ["query"],
        }

    async def execute(self, *, query: str = "", **_kwargs) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="query is required", validation_error=True)

        mgr = get_pack_manager()
        packs = _installed_reference_packs()
        if mgr is None or not packs:
            return ToolResult(
                success=False,
                error=(
                    "No knowledge packs are installed (or the pack "
                    "subsystem is disabled). Use doc_search for the live "
                    "web instead. Do not retry pack_search this turn."
                ),
            )

        pack_ids = [str(p["pack_id"]) for p in packs]
        try:
            results = await mgr.search(
                query.strip(), pack_ids=pack_ids, limit=_MAX_RESULTS,
            )
        except Exception as exc:
            log.warning("pack_search_failed", query=query[:120], exc_info=True)
            return ToolResult(
                success=False,
                error=f"Pack search failed: {str(exc)[:200]}. Fall back to doc_search.",
            )

        if not results:
            return ToolResult(
                success=True,
                output=(
                    f"No pack results for '{query}'. The installed packs "
                    "may not cover this topic — try doc_search for the "
                    "live web rather than rephrasing the same pack query."
                ),
                metadata={"query": query, "results": 0},
            )

        lines = [f"Pack search: '{query}' — {len(results)} result(s)\n"]
        for i, r in enumerate(results):
            heading = " › ".join(x for x in (r.title, r.section) if x)
            lines.append(f"{i + 1}. {heading or '(untitled)'}  [{r.pack_id}]")
            if r.url:
                lines.append(f"   {r.url}")
            body = (r.content or "").strip()
            if body:
                lines.append("   " + body[:_SNIPPET_CHARS].replace("\n", "\n   "))
            lines.append("")

        return ToolResult(
            success=True,
            output=_truncate("\n".join(lines)),
            metadata={
                "query": query,
                "results": len(results),
                "packs_searched": len(pack_ids),
            },
        )


__all__ = ["PackSearchTool"]
