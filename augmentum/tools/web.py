"""Unified web tool — auto-routes between fetch and search.

Detects URLs in the query and fetches them directly. Falls back to SearXNG
search for non-URL queries. After searching, auto-fetches the top result(s)
to provide actual page content instead of just snippets.

Source-aware pipeline:
  1. Pre-search:  topic detection adds site: hints to SearXNG queries
  2. Post-search: AVOID-tier domains filtered out before auto-fetch
  3. Post-fetch:  source quality annotation appended to fetched content
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.preferred_sources import (
    AVOID,
    describe_source,
    domain_quality,
    get_topic_sites,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.web_fetch import WebFetchTool
    from augmentum.tools.web_search import WebSearchTool

log = get_logger(__name__)

# Max site: operators to add to a single search query
_MAX_SITE_HINTS = 2

# Matches URLs or URL-like strings (domain.tld patterns)
_URL_RE = re.compile(
    r"https?://\S+"                      # explicit http(s)://
    r"|(?:www\.)\S+\.\w{2,}"            # www.example.com
    r"|(?:\S+\.(?:com|org|net|gov|edu|io|dev|co|info|me|app|ai)\b\S*)",  # bare domain
    re.IGNORECASE,
)
_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_QUERY_STOPWORDS = {
    "the", "and", "for", "but", "with", "from", "into", "onto", "this", "that",
    "these", "those", "what", "when", "where", "which", "while", "who", "whom",
    "how", "why", "can", "does", "did", "will", "would", "should", "could",
    "are", "was", "were", "been", "being", "have", "has", "had", "its",
    "tell", "show", "find", "get", "about", "some", "more", "most", "best",
    "good", "bad", "any", "all", "new", "old", "top", "info", "information",
    "help", "please", "thanks", "thank",
}


def _extract_urls(text: str) -> list[str]:
    """Extract URLs from text, normalizing bare domains to https://."""
    urls = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:!?)]}>\"'")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        urls.append(url)
    return urls


def _page_preview(url: str, metadata: dict | None, source_desc: str = "") -> dict:
    meta = metadata or {}
    return {
        "url": meta.get("url", url) or url,
        "title": meta.get("title", ""),
        "excerpt": meta.get("preview_excerpt", ""),
        "char_count": meta.get("char_count", 0),
        "content_type": meta.get("content_type", ""),
        "truncated": bool(meta.get("truncated")),
        "source": source_desc,
    }


def _tokenize_query(query: str) -> set[str]:
    if not query:
        return set()
    return {
        token.lower()
        for token in _QUERY_TOKEN_RE.findall(query.lower())
        if token.lower() not in _QUERY_STOPWORDS
    }


def _result_topical_score(query_tokens: set[str], result: dict) -> int:
    if not query_tokens:
        return 0
    title = str(result.get("title", "") or "").lower()
    snippet = str(result.get("snippet", "") or "").lower()
    url = str(result.get("url", "") or "").lower()
    domain = str(result.get("source", "") or "").lower()
    haystack = f"{title} {snippet} {url} {domain}"
    overlap = sum(1 for token in query_tokens if token in haystack)
    title_overlap = sum(1 for token in query_tokens if token in title)
    return overlap + title_overlap


def _rank_search_results_for_fetch(query: str, results: list[dict]) -> list[dict]:
    query_tokens = _tokenize_query(query)
    candidates = [result for result in results if result.get("url")]
    if not candidates:
        return []

    filtered = [
        result for result in candidates
        if domain_quality(result.get("url", "")) != AVOID
    ]
    if not filtered:
        filtered = candidates

    return sorted(
        filtered,
        key=lambda result: (
            -_result_topical_score(query_tokens, result),
            -domain_quality(result.get("url", "")),
            int(result.get("rank") or 999),
        ),
    )


def _render_search_results_text(results: list[dict]) -> str:
    lines: list[str] = []
    for index, result in enumerate(results, 1):
        title = result.get("title") or "Untitled"
        url = result.get("url") or ""
        snippet = result.get("snippet") or "No snippet available."
        lines.append(f"[{index}] {title}")
        lines.append(f"    URL: {url}")
        lines.append(f"    {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


class WebTool(Tool):
    """Unified web lookup — search or fetch based on input.

    - If the query contains a URL, fetches it directly.
    - If the query is plain text, searches via SearXNG then auto-fetches
      the top result for full content.
    - If a direct fetch fails, falls back to search.
    """

    @property
    def name(self) -> str:
        return "web"

    @property
    def description(self) -> str:
        return (
            "Look up information on the web and READ it, silently — "
            "accepts a URL to fetch directly, or a search query; a "
            "query searches and then auto-fetches the top result's "
            "full content (unlike web_search, which only returns the "
            "result list). Nothing opens on the user's screen. "
            "Use specific, detailed queries — e.g. 'Python 3.12 match statement syntax' "
            "not just 'python match'. For current events, include the year."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No results": "Try broader search terms, remove specific names or dates, or break the query into simpler parts.",
            "Connection refused": "The search service is temporarily unavailable. Answer from your own knowledge instead.",
            "Timeout": "The search took too long. Try a shorter, more specific query.",
            "403": "Access denied for that URL. Try searching for the topic instead of fetching the URL directly.",
        }

    @property
    def requires_services(self) -> list[str]:
        return ["searxng"]

    @property
    def produces(self) -> list[str]:
        return ["text"]

    @property
    def model_hint(self) -> str:
        return "Use this for ANY question about current events, facts you're unsure about, or when the user says 'search' or 'look up'."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A URL to fetch directly, or a search query. "
                        "Examples: 'https://example.com/article' or 'weather in NYC today'"
                    ),
                },
            },
            "required": ["query"],
        }

    @property
    def timeout(self) -> float:
        return 50.0

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def cache_ttl(self) -> float:
        return 300.0

    def health_check(self) -> bool:
        """Check if SearXNG search is likely reachable."""
        # Delegate to the search tool's health if available
        return getattr(self._search, "_base_url", "") != ""

    def __init__(
        self,
        search_tool: WebSearchTool,
        fetch_tool: WebFetchTool,
        *,
        auto_fetch_top: int = 1,
        fetch_max_chars: int = 12000,
    ) -> None:
        self._search = search_tool
        self._fetch = fetch_tool
        self._auto_fetch_top = auto_fetch_top
        self._fetch_max_chars = fetch_max_chars

    def validate_input(self, **kwargs) -> bool:
        query = kwargs.get("query", "")
        return isinstance(query, str) and len(query.strip()) > 0

    async def execute(self, *, query: str) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult(success=False, error="Empty query")

        urls = _extract_urls(query)

        if urls:
            # URL detected — fetch directly
            return await self._handle_url(urls[0], query)

        # No URL — search then auto-fetch
        return await self._handle_search(query)

    async def _handle_url(self, url: str, original_query: str) -> ToolResult:
        """Fetch a URL directly. Falls back to search if fetch fails."""
        log.info("web_tool_fetch", url=url)
        result = await self._fetch.execute(url=url, max_chars=self._fetch_max_chars)

        if result.success and result.output and result.output.strip():
            page = _page_preview(url, result.metadata)
            return ToolResult(
                success=True,
                output=f"Content from {url}:\n\n{result.output}\n\nSource: {url}",
                metadata={"mode": "fetch", "url": url, "urls": [url], "page": page, **(result.metadata or {})},
            )

        # Fetch failed or empty — fall back to search
        log.info("web_tool_fetch_fallback", url=url, error=result.error)

        # Use the non-URL part of the query for search, or the domain
        search_query = _strip_urls(original_query).strip()
        if not search_query:
            # Extract something searchable from the URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            search_query = parsed.netloc + " " + parsed.path.replace("/", " ")

        return await self._handle_search(
            search_query,
            fallback_note=f"(Could not fetch {url} directly — searched instead)\n\n",
        )

    async def _handle_search(
        self, query: str, *, fallback_note: str = "",
    ) -> ToolResult:
        """Search via SearXNG and auto-fetch top result(s).

        Pipeline:
          1. Pre-search: detect topic → add site: hints for preferred sources
          2. Search via SearXNG
          3. Post-search: filter AVOID domains, sort by quality
          4. Auto-fetch top result(s) with source annotation
        """
        # --- 1. Pre-search: topic-aware site hints ---
        search_query = _build_search_query(query)
        log.info("web_tool_search", query_chars=len(search_query), original_chars=len(query))
        log.debug("web_tool_search_query", query=search_query, original=query)

        search_result = await self._search.execute(
            query=search_query, num_results=8, categories="general",
        )

        if not search_result.success:
            return search_result

        # Extract URLs from search results for auto-fetch
        search_results = list(search_result.metadata.get("results", []))
        ranked_results = _rank_search_results_for_fetch(query, search_results)
        search_urls = [result.get("url", "") for result in ranked_results if result.get("url")]
        search_text = _render_search_results_text(ranked_results)

        if not search_urls:
            # No fetchable URLs — return search snippets as-is
            return ToolResult(
                success=True,
                output=f"{fallback_note}Search results for '{query}':\n\n{search_text or search_result.output}",
                metadata={
                    "mode": "search_only",
                    "query": query,
                    "results": ranked_results,
                    "urls": search_urls,
                },
            )

        # --- 2. Post-search: keep the fetch order aligned with the same
        # ranked evidence we expose to the model and the UI.
        ranked_urls = search_urls

        # --- 3. Auto-fetch top result(s) with source annotation ---
        fetched_parts: list[str] = []
        fetched_urls: list[str] = []
        fetched_pages: list[dict] = []
        for url in ranked_urls:
            if len(fetched_parts) >= self._auto_fetch_top:
                break
            try:
                fetch_result = await self._fetch.execute(
                    url=url, max_chars=self._fetch_max_chars,
                )
                if fetch_result.success and fetch_result.output.strip():
                    # Annotate with source quality so the LLM knows
                    # what it's reading (e.g. "[weather.gov] quality=excellent,
                    # categories: weather, freshness: realtime")
                    header = f"--- Content from {url} ---"
                    source_desc = describe_source(url)
                    if source_desc:
                        header += f"\n{source_desc}"
                    fetched_parts.append(
                        f"{header}\n{fetch_result.output}"
                    )
                    fetched_urls.append(url)
                    fetched_pages.append(
                        _page_preview(url, fetch_result.metadata, source_desc),
                    )
                else:
                    log.info(
                        "web_tool_autofetch_skip",
                        url=url,
                        reason=fetch_result.error or "empty content",
                    )
            except Exception:
                log.info("web_tool_autofetch_failed", url=url, exc_info=True)

        # Combine: search snippets + fetched content + verified sources
        sections: list[str] = []
        if fallback_note:
            sections.append(fallback_note.rstrip())
        sections.append(f"Search results for '{query}':\n\n{search_text or search_result.output}")
        if fetched_parts:
            sections.append("\n\n".join(fetched_parts))

        # Append verified source URLs so the LLM can cite them accurately
        all_urls = fetched_urls or search_urls[:5]
        if all_urls:
            source_lines = ["Sources:"]
            for i, url in enumerate(all_urls, 1):
                source_lines.append(f"  [{i}] {url}")
            sections.append("\n".join(source_lines))

        return ToolResult(
            success=True,
            output="\n\n".join(sections),
            metadata={
                "mode": "search_and_fetch",
                "query": query,
                "results": ranked_results,
                "fetched_pages": fetched_pages,
                "fetched_urls": fetched_urls,
                "urls": all_urls,
                "search_results": len(ranked_results),
            },
        )


def _build_search_query(query: str) -> str:
    """Enhance a search query with site: hints from the preferred sources registry.

    Gated by `web_search_topic_hints_enabled` (default OFF). When enabled and
    the query matches known topics (e.g. "weather", "python language"),
    appends up to 2 site: operators to steer SearXNG toward authoritative
    sources. This is additive — the original query is always preserved.

    Examples (when enabled):
        "weather in NYC"   → "weather in NYC site:weather.gov"
        "python async io"  → "python async io site:docs.python.org"
        "random question"  → "random question"  (no topic match)
    """
    from augmentum.config import settings
    if not getattr(settings, "web_search_topic_hints_enabled", False):
        return query

    topic_sites = get_topic_sites(query)
    if not topic_sites:
        return query

    # Add top site hints (SearXNG treats site: as a soft boost, not a hard filter)
    hints = topic_sites[:_MAX_SITE_HINTS]
    site_clause = " ".join(f"site:{s}" for s in hints)
    return f"{query} {site_clause}"


def _strip_urls(text: str) -> str:
    """Remove URLs from text."""
    return _URL_RE.sub("", text)


def _extract_result_urls(search_output: str) -> list[str]:
    """Extract URLs from formatted search results (URL: https://... lines)."""
    urls = []
    for line in search_output.splitlines():
        line = line.strip()
        if line.startswith("URL: "):
            url = line[5:].strip()
            if url.startswith(("http://", "https://")):
                urls.append(url)
    return urls
