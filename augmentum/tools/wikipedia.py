"""Wikipedia lookup tool — queries MediaWiki API directly via httpx."""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)

_API_URL = "https://en.wikipedia.org/w/api.php"
_MAX_EXTRACT_CHARS = 8000


class WikipediaTool(Tool):
    """Look up information on Wikipedia."""

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return (
            "Search Wikipedia and retrieve article summaries. "
            "Use for factual lookups, definitions, historical events, "
            "biographies, and general knowledge."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No results": "Try a more specific or alternative spelling. Wikipedia titles are case-sensitive.",
            "disambiguation": "The query matched multiple articles. Be more specific — add context like the person's profession or the topic's field.",
            "Timeout": "Wikipedia API is slow. Try a shorter query.",
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The topic or question to look up on Wikipedia",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of articles to return (max 5)",
                    "default": 1,
                },
                "full_article": {
                    "type": "boolean",
                    "description": "If true, fetch the full article extract instead of just the intro",
                    "default": False,
                },
            },
            "required": ["query"],
        }

    @property
    def timeout(self) -> float:
        return 15.0

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client

    def validate_input(self, **kwargs) -> bool:
        query = kwargs.get("query", "")
        return isinstance(query, str) and len(query.strip()) > 0

    async def _search(self, query: str, limit: int) -> list[str]:
        """Search Wikipedia and return matching page titles."""
        resp = await self._client.get(
            _API_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": min(limit, 5),
                "format": "json",
                "utf8": 1,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [r["title"] for r in data.get("query", {}).get("search", [])]

    async def _get_extracts(
        self, titles: list[str], *, intro_only: bool = True
    ) -> list[dict]:
        """Fetch article extracts for given titles."""
        params: dict = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": "|".join(titles),
            "format": "json",
            "utf8": 1,
            "exlimit": len(titles),
        }
        if intro_only:
            params["exintro"] = 1
        else:
            params["exchars"] = _MAX_EXTRACT_CHARS

        resp = await self._client.get(_API_URL, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        results = []
        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            extract = page.get("extract", "").strip()
            if extract:
                title = page.get("title", "")
                # Build canonical Wikipedia URL from title
                url_title = title.replace(" ", "_")
                url = f"https://en.wikipedia.org/wiki/{url_title}"
                results.append({
                    "title": title,
                    "extract": extract,
                    "pageid": page.get("pageid"),
                    "url": url,
                })
        return results

    async def execute(
        self,
        *,
        query: str,
        num_results: int = 1,
        full_article: bool = False,
    ) -> ToolResult:
        """Search Wikipedia and return article summaries."""
        if not query.strip():
            return ToolResult(success=False, error="Empty query")

        num_results = max(1, min(num_results, 5))

        try:
            titles = await self._search(query, num_results)
        except Exception as exc:
            log.warning("wikipedia_search_failed", query=query, error=str(exc))
            return ToolResult(
                success=False,
                error=f"Wikipedia search failed: {exc}",
            )

        if not titles:
            return ToolResult(
                success=True,
                output="No Wikipedia articles found for this query.",
                metadata={"query": query, "num_results": 0},
            )

        try:
            articles = await self._get_extracts(titles, intro_only=not full_article)
        except Exception as exc:
            log.warning("wikipedia_extract_failed", query=query, error=str(exc))
            return ToolResult(
                success=False,
                error=f"Wikipedia extract failed: {exc}",
            )

        if not articles:
            return ToolResult(
                success=True,
                output="No extractable content found for the matching articles.",
                metadata={"query": query, "num_results": 0},
            )

        lines: list[str] = []
        for i, article in enumerate(articles, 1):
            if num_results > 1:
                lines.append(f"--- [{i}] {article['title']} ---")
            else:
                lines.append(f"# {article['title']}")
            lines.append("")
            lines.append(article["extract"])
            lines.append("")

        # Append verified source URLs so the LLM can cite them accurately
        lines.append("Sources:")
        for i, article in enumerate(articles, 1):
            lines.append(f"  [{i}] {article['title']}: {article['url']}")

        output = "\n".join(lines).rstrip()

        # Wrap at source — article extracts are external content; the markers
        # travel with the result through every downstream tool loop.
        from augmentum.security.untrusted import wrap_untrusted
        output = wrap_untrusted("web/wikipedia", output)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "query": query,
                "num_results": len(articles),
                "titles": [a["title"] for a in articles],
                "urls": [a["url"] for a in articles],
            },
        )
