"""Iterative web research tool — the universal "look it up" primitive.

`web_search` returns one query's hits; `web_fetch` reads one page. Neither
*persists* when the first try comes back thin. `research` closes that gap:
given a question, it runs several queries (the caller's alternates plus a
deterministic broaden-on-empty), merges + de-dupes across them with domain
diversity, optionally deep-reads the strongest sources past their snippets,
and returns either a grounded evidence digest with citations OR an honest
"couldn't find it — here's what I tried" miss.

Deliberately model-agnostic and vertical-free: it has no weather/traffic/
finance special-casing. Anything addressable on the web is researchable the
same way, so a weak local model gets the same retry resilience a strong one
would improvise. Typed sources (e.g. weather.today) stay separate verbs the
agent calls opportunistically — baking a location-regex matcher in here would
just be the open-slot intent-switchboard anti-pattern in disguise.

Built on the existing WebSearchTool / WebFetchTool instances (engine-health
fallback, quality filtering, browse dispatch chain all reused), so it needs
no backend/LLM access of its own. Synthesis stays with the caller's loop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.web_fetch import WebFetchTool
    from augmentum.tools.web_search import WebSearchTool

log = get_logger(__name__)


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").replace("www.", "").lower()
    except Exception:
        return ""


def _broaden(query: str) -> str:
    """One deterministic reformulation toward a broader query.

    Used only when a query returned nothing — strips a trailing qualifier
    clause and quotes, and caps length, so an over-specific phrasing gets a
    second, looser shot before we conclude "not found". Returns "" when it
    can't produce a meaningfully different query (don't re-run the same one).
    """
    s = (query or "").strip().strip('"').strip()
    low0 = s.lower()
    for sep in (" - ", " — ", ", ", ": ", " (",):
        if sep in s:
            s = s.split(sep, 1)[0].strip()
            break
    words = s.split()
    if len(words) > 8:
        s = " ".join(words[:8])
    s = s.strip()
    return s if s and s.lower() != low0 else ""


class ResearchTool(Tool):
    """Research a question across the web, retrying with different queries
    and reading sources, until it has an answer or an honest miss."""

    @property
    def name(self) -> str:
        return "research"

    @property
    def description(self) -> str:
        return (
            "Research a question on the web and get back a synthesized "
            "evidence digest with citations — SILENTLY (nothing opens on "
            "the user's screen). Unlike web_search (one query) this runs "
            "several queries, retries with a broader phrasing when a query "
            "finds nothing, reads the strongest sources in full, and tells "
            "you honestly when the information isn't available. Pass a few "
            "'alt_queries' (different phrasings/angles) for hard or "
            "ambiguous questions. Use this as your default for any factual "
            "lookup that might need more than a single search."
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
                    "description": "The question or topic to research.",
                },
                "alt_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional alternate phrasings / sub-queries to try "
                        "(different angles, synonyms, or decomposed parts of "
                        "an intricate request). More angles = better recall."
                    ),
                },
                "fetch_pages": {
                    "type": "boolean",
                    "description": (
                        "Read the top sources in full (not just snippets). "
                        "Default true; set false for a faster snippet-only pass."
                    ),
                    "default": True,
                },
                "recency": {
                    "type": "string",
                    "description": (
                        "Hint for time-sensitive topics: one of day, week, "
                        "month, year. Biases toward news sources."
                    ),
                },
            },
            "required": ["query"],
        }

    @property
    def timeout(self) -> float:
        return 90.0

    def __init__(
        self,
        *,
        search_tool: WebSearchTool,
        fetch_tool: WebFetchTool,
    ) -> None:
        self._search = search_tool
        self._fetch = fetch_tool

    def validate_input(self, **kwargs) -> bool:
        q = kwargs.get("query", "")
        return isinstance(q, str) and len(q.strip()) > 0

    # ── Budgets (operator-tunable, with safe fallbacks) ──────────────

    def _budgets(self) -> tuple[bool, int, float, int]:
        try:
            from augmentum.config import settings
        except Exception:
            return True, 4, 60.0, 2
        enabled = bool(getattr(settings, "companion_research_enabled", True))
        max_q = int(getattr(settings, "companion_research_max_queries", 4) or 4)
        max_s = float(getattr(settings, "companion_research_max_seconds", 60.0) or 60.0)
        fetch_top = int(getattr(settings, "companion_research_fetch_top", 2) or 2)
        return enabled, max(1, min(max_q, 8)), max(10.0, max_s), max(0, min(fetch_top, 4))

    async def execute(
        self,
        *,
        query: str,
        alt_queries: list[str] | None = None,
        fetch_pages: bool = True,
        recency: str = "",
    ) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(success=False, error="Empty research query")

        enabled, max_queries, max_seconds, fetch_top = self._budgets()
        if not enabled:
            return ToolResult(
                success=False,
                error="Web research is disabled by the operator.",
            )

        started = time.monotonic()

        def _time_left() -> float:
            return max_seconds - (time.monotonic() - started)

        categories = "news" if str(recency).strip().lower() in {"day", "week"} else "general"

        # Build the query plan: the main query first, then caller-supplied
        # angles, de-duplicated, capped to the budget.
        plan: list[str] = []
        for q in [query, *(alt_queries or [])]:
            q = (q or "").strip()
            if q and q.lower() not in {p.lower() for p in plan}:
                plan.append(q)
            if len(plan) >= max_queries:
                break

        merged: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        per_domain: dict[str, int] = {}
        tried: list[str] = []
        rate_limited = False

        def _absorb(results: list[dict]) -> None:
            for r in results:
                url = (r.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                dom = r.get("source") or _domain(url)
                # Domain diversity: at most 2 hits per site so one chatty
                # domain can't crowd out other resources.
                if per_domain.get(dom, 0) >= 2:
                    continue
                seen_urls.add(url)
                per_domain[dom] = per_domain.get(dom, 0) + 1
                merged.append({
                    "title": r.get("title") or "Untitled",
                    "url": url,
                    "snippet": r.get("snippet") or "",
                    "source": dom,
                })

        async def _search_once(q: str) -> list[dict]:
            try:
                res = await self._search.execute(
                    query=q, num_results=6, categories=categories,
                )
            except Exception as exc:  # noqa: BLE001 — one query failing isn't fatal
                log.warning("research_search_failed", query=q[:80], error=str(exc)[:160])
                return []
            if res.metadata.get("rate_limited"):
                nonlocal rate_limited
                rate_limited = True
            return list(res.metadata.get("results") or [])

        # Gather pass: each planned query, with one broaden retry on empty.
        for q in plan:
            if _time_left() <= 2:
                break
            tried.append(q)
            results = await _search_once(q)
            if not results:
                broadened = _broaden(q)
                if broadened and _time_left() > 2:
                    tried.append(broadened)
                    results = await _search_once(broadened)
            _absorb(results)
            # Enough strong, diverse sources gathered — stop early.
            if len(merged) >= 6:
                break

        # Honest miss — nothing solid across every query we tried.
        if not merged:
            return ToolResult(
                success=True,  # a clean, relayable outcome, not a tool error
                output=self._miss_text(query, tried, rate_limited),
                metadata={
                    "miss": True,
                    "rate_limited": rate_limited,
                    "tried_queries": tried,
                    "num_sources": 0,
                },
            )

        # Deep-read the strongest sources past their snippets, time permitting.
        excerpts: list[tuple[dict, str]] = []
        if fetch_pages and fetch_top > 0:
            for r in merged[:fetch_top]:
                if _time_left() <= 4:
                    break
                try:
                    fr = await self._fetch.execute(url=r["url"], max_chars=2400)
                except Exception as exc:  # noqa: BLE001 — best-effort depth
                    log.debug("research_fetch_failed", url=r["url"][:80], error=str(exc)[:120])
                    continue
                if fr.success and fr.output:
                    excerpts.append((r, fr.output.strip()[:2200]))

        output = self._format_evidence(query, merged, excerpts)
        from augmentum.security.untrusted import wrap_untrusted
        return ToolResult(
            success=True,
            output=wrap_untrusted("web/research", output),
            metadata={
                "miss": False,
                "query": query,
                "tried_queries": tried,
                "num_sources": len(merged),
                "fetched": len(excerpts),
                "rate_limited": rate_limited,
                "citations": [
                    {"title": r["title"], "url": r["url"], "source": r["source"]}
                    for r in merged
                ],
            },
        )

    # ── Formatting ───────────────────────────────────────────────────

    def _format_evidence(
        self, query: str, merged: list[dict], excerpts: list[tuple[dict, str]],
    ) -> str:
        lines: list[str] = [
            f"Research on: {query}",
            f"Found {len(merged)} source(s).",
            "",
            "Sources:",
        ]
        for i, r in enumerate(merged, 1):
            lines.append(f"[{i}] {r['title']} — {r['source']}")
            lines.append(f"    {r['url']}")
            if r["snippet"]:
                lines.append(f"    {r['snippet']}")
        if excerpts:
            lines.append("")
            lines.append("Full-text excerpts from the strongest sources:")
            for r, text in excerpts:
                lines.append("")
                lines.append(f"--- {r['title']} ({r['source']}) ---")
                lines.append(text)
        lines.append("")
        lines.append(
            "Synthesize an answer from the sources above and cite them by "
            "number. If they disagree or don't actually cover the question, "
            "say so rather than guessing."
        )
        return "\n".join(lines)

    def _miss_text(self, query: str, tried: list[str], rate_limited: bool) -> str:
        tried_str = "; ".join(dict.fromkeys(tried)) or query
        if rate_limited:
            return (
                f"Couldn't research \"{query}\" right now — the search "
                "engines are rate-limited and returned nothing. Tried: "
                f"{tried_str}. Worth trying again in a few minutes."
            )
        return (
            f"Couldn't find solid information for \"{query}\". "
            f"Tried these queries with no usable results: {tried_str}. "
            "The information may not be readily available on the open web, "
            "or it may need a more specific phrasing or a different resource. "
            "Report this honestly rather than answering from memory."
        )
