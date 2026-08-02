"""SearXNG web search tool."""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.engine_health import TRACKER, EngineHealthTracker
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import httpx

# Single network-error retry — one transient blip recovery. SearXNG already
# does per-engine retry + suspension internally (see settings_engines docs),
# so client-side exponential backoff fights its suspension state rather
# than helping. Jittered (±50%) so concurrent searches from the same user
# don't all retry on the exact same tick and synchronize against an
# upstream rate-limiter.
_NETWORK_RETRY_BACKOFF = 1.0
_NETWORK_RETRY_JITTER = 0.5

# Static last-resort fallback engine set, used only when the health
# tracker has every candidate marked suspended (so its dynamic set is
# empty). Chosen from observed suspension behavior (2026-06-10 live
# probes): Bing and DuckDuckGo answered every query even while Brave
# suspended after 2 rapid requests and Mojeek denied access outright.
# Wikipedia stays as the no-rate-limit knowledge anchor. The normal
# fallback path asks EngineHealthTracker.healthy_fallback_engines()
# first, which excludes engines observed suspended in recent responses.
_FALLBACK_ENGINES = "bing,duckduckgo,wikipedia"

# Trusted, harness-authored guidance appended AFTER the untrusted results block
# (outside the <<<UNTRUSTED>>> markers, so the model reads it as instruction, not
# attacker-controlled page text). Steers the model to deliberately READ the best
# source when a snippet is too thin — the intentional, quality-gated alternative
# to blindly auto-fetching every result and flooding context with junk. Points
# at web_fetch, which is now exposed in the chat tool loop.
_FETCH_NUDGE = (
    "The above are search previews, not full pages. If a result looks "
    "authoritative but its snippet doesn't fully answer the question "
    "(e.g. you need a table, day-by-day figures, or exact numbers), call "
    "web_fetch(url) on the best 1-2 to read the full page before answering. "
    "Skip fetching low-quality, off-topic, or redundant results."
)

# Same steer, phrased for the dead-end case: the query returned only pages
# already seen this turn. Re-searching the same thing won't help — read what
# you already found instead.
_DEDUP_FETCH_NUDGE = (
    "If one of the URLs you already found is authoritative but you haven't "
    "read it yet, call web_fetch(url) on it instead of searching again. "
    "Otherwise answer from what you've gathered."
)

log = get_logger(__name__)


def _shorten(text: str, limit: int = 220) -> str:
    """Collapse whitespace, strip SearXNG highlight HTML, then truncate.

    SearXNG often wraps query matches in <b>/<strong> tags and leaves
    HTML entities (&amp;, &#39;) in the snippet. Cleaning here means the
    LLM and downstream metadata.results consumers see plain prose.
    """
    if not text:
        return ""
    try:
        from augmentum.discovery.quality import _clean_search_snippet
        text = _clean_search_snippet(text)
    except Exception:
        text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").replace("www.", "")
    except Exception:
        return ""


class WebSearchTool(Tool):
    """Search the web using a SearXNG instance."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web SILENTLY and answer in your own words — the "
            "DEFAULT for any 'search X' / 'look up X' / 'google X' / "
            "'find info on X' / 'what's the latest on X' request. Most "
            "'search for…' asks mean 'find out and tell me', so prefer "
            "this: nothing opens on their screen, the results come back "
            "to you, and you speak the answer. Use web.search ONLY when "
            "they explicitly want to SEE the result list themselves "
            "('show me', 'open', 'pull up')."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def model_hint(self) -> str:
        # Appended to the description for sub-~14B models only (see
        # Tool.model_hint / passthrough _inject_tool_schemas). Small
        # models lean on the user's sentence; this keeps the query
        # keyword-shaped without a second model call.
        return (
            "QUERY TIP: pass 3-8 plain keywords (the key nouns/names), "
            "not the user's full sentence or a question — e.g. 'best "
            "noise cancelling headphones 2026', not 'what are the best "
            "headphones I can buy'."
        )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query YOU design for a keyword search "
                        "engine — NOT the user's sentence verbatim. SearXNG "
                        "matches keywords, not questions, so: drop "
                        "conversational filler ('can you', 'I want to know', "
                        "'tell me about', 'what is the'); keep the salient "
                        "nouns, names and entities; aim for ~3-8 strong "
                        "keywords. Use operators when they sharpen it: "
                        "\"exact phrase\" in quotes, site:domain.com to "
                        "target a known authoritative source, OR between "
                        "alternatives, -term to exclude, and add the year "
                        "for anything time-sensitive. "
                        "E.g. user 'hey can you find out the latest on the "
                        "mars sample return mission' → 'Mars Sample Return "
                        "mission news 2026'."
                    ),
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
                "categories": {
                    "type": "string",
                    "description": "SearXNG search categories (e.g. general, science, news)",
                    "default": "general",
                },
            },
            "required": ["query"],
        }

    @property
    def timeout(self) -> float:
        return 50.0

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str = "http://searxng:8080",
        tracker: EngineHealthTracker | None = None,
    ) -> None:
        self._client = http_client
        self._base_url = base_url.rstrip("/")
        # Process-wide by default: suspension state describes the SearXNG
        # instance, not a user. Injectable for tests.
        self._tracker = tracker or TRACKER

    def validate_input(self, **kwargs) -> bool:
        query = kwargs.get("query", "")
        return isinstance(query, str) and len(query.strip()) > 0

    async def _call_searxng(
        self,
        *,
        query: str,
        categories: str,
        engines: str | None = None,
    ) -> tuple[dict | None, Exception | None]:
        """One SearXNG round-trip. Returns (json_data, None) or (None, exc).

        Optionally constrains to a specific engine set via the SearXNG
        ``engines=`` query param, used by the fallback path when the
        full fan-out came back fully blocked.
        """
        params: dict[str, str] = {
            "q": query,
            "format": "json",
            "categories": categories,
        }
        if engines:
            params["engines"] = engines
        try:
            response = await self._client.get(
                f"{self._base_url}/search",
                params=params,
                timeout=15.0,
            )
            response.raise_for_status()
            return response.json(), None
        except Exception as exc:
            return None, exc

    async def execute(
        self,
        *,
        query: str,
        num_results: int = 5,
        categories: str = "general",
    ) -> ToolResult:
        """Execute a web search via SearXNG.

        Returns formatted search results with title, URL, and snippet for each hit.

        Branches on SearXNG's documented response contract (results +
        ``unresponsive_engines``) to distinguish genuine no-match from
        infrastructure failure, instead of treating every "0 results"
        identically. Falls back to a known-good engine subset on infra
        failure and to raw SearXNG output if the quality filter strips
        everything.
        """
        if not query.strip():
            return ToolResult(success=False, error="Empty search query")

        num_results = max(1, min(int(num_results), 20))

        # First call — full engine fan-out.
        data, exc = await self._call_searxng(query=query, categories=categories)

        # One retry for network errors only (transient blip protection).
        if exc is not None:
            backoff = _NETWORK_RETRY_BACKOFF * random.uniform(
                1.0 - _NETWORK_RETRY_JITTER, 1.0 + _NETWORK_RETRY_JITTER,
            )
            await asyncio.sleep(backoff)
            data, exc = await self._call_searxng(query=query, categories=categories)
            if exc is not None:
                log.warning("web_search_network_failed", query=query, error=str(exc))
                return ToolResult(
                    success=False,
                    error=f"Search failed: {exc}",
                )

        self._tracker.record_response(data)
        raw_results: list[dict] = list(data.get("results") or [])
        unresponsive: list = list(data.get("unresponsive_engines") or [])
        fallback_used = False

        # Infra-failure branch: zero results AND some engines reported
        # unresponsive → not a real no-match, the requested engines are
        # blocked/timing out. Reissue constrained to engines the health
        # tracker hasn't seen suspended recently; static set only when
        # the tracker has everything marked down.
        if not raw_results and unresponsive:
            fallback_engines = (
                self._tracker.healthy_fallback_engines() or _FALLBACK_ENGINES
            )
            log.info(
                "web_search_engines_unresponsive",
                query=query,
                unresponsive=unresponsive,
                action="retry_with_fallback_engines",
                fallback_engines=fallback_engines,
            )
            fb_data, fb_exc = await self._call_searxng(
                query=query, categories=categories, engines=fallback_engines,
            )
            if fb_exc is None and fb_data is not None:
                self._tracker.record_response(fb_data)
                raw_results = list(fb_data.get("results") or [])
                fallback_used = True
                if not raw_results:
                    log.warning(
                        "web_search_fallback_empty",
                        query=query,
                        fallback_unresponsive=list(fb_data.get("unresponsive_engines") or []),
                    )
            else:
                log.warning("web_search_fallback_failed", query=query, error=str(fb_exc))

        # Still nothing. Distinguish full infra blackout from honest
        # no-match: when engines were unresponsive AND the tracker has
        # fallback candidates suspended, the empty result is a rate-limit
        # artifact — say so, with a recovery estimate, instead of letting
        # the model tell the user "there's nothing about that topic".
        if not raw_results:
            retry_s = self._tracker.earliest_retry_seconds()
            if unresponsive and retry_s is not None:
                minutes = max(1, round(retry_s / 60))
                return ToolResult(
                    success=True,
                    output=(
                        "Search engines are rate-limited right now — no "
                        "results available for this query. Estimated "
                        f"recovery in about {minutes} minute"
                        f"{'s' if minutes != 1 else ''}; try again then."
                    ),
                    metadata={
                        "query": query,
                        "num_results": 0,
                        "rate_limited": True,
                        "retry_in_seconds": retry_s,
                        "engines_unavailable": self._tracker.suspended_summary(),
                        "unresponsive_engines": unresponsive,
                        "fallback_used": fallback_used,
                    },
                )
            return ToolResult(
                success=True,
                output=(
                    "No results found for this query. Try a different "
                    "phrasing — broaden it, narrow it, or use different "
                    "keywords — rather than concluding the topic has no "
                    "information."
                ),
                metadata={
                    "query": query,
                    "num_results": 0,
                    "unresponsive_engines": unresponsive,
                    "fallback_used": fallback_used,
                },
            )

        # Partial-success observability: log if some engines failed but
        # we still got results from the rest. SearXNG handled this
        # gracefully — we just want to see it in our logs.
        if unresponsive and not fallback_used:
            log.info(
                "web_search_partial",
                query=query,
                results=len(raw_results),
                unresponsive=unresponsive,
            )

        # Quality pipeline: reject junk, non-English, unfetchable; rank by reputation
        try:
            from augmentum.discovery.quality import filter_for_llm
            filtered = filter_for_llm(raw_results, query=query)
        except Exception:
            log.debug("web_search_filter_failed", exc_info=True)
            filtered = raw_results

        # Raw fallback: filter stripped everything (common for queries that
        # hit unfetchable platforms — Twitter, Pinterest, Reddit threads).
        # Better to hand the LLM imperfect snippets than silent emptiness.
        #
        # BUT only fall back when at least one raw result shares some
        # query tokens. Otherwise we're recovering high-rep generic
        # homepages (CNN, BBC, AP) that have zero relevance to the
        # actual query — that's confidently wrong, worse than empty.
        # The filter's relevance floor drops these intentionally; the
        # fallback used to put them right back. Now it only fires when
        # the raw results were rejected for fetchability reasons
        # (JS-walls, paywalls), not relevance reasons.
        quality_filtered = True
        if raw_results and not filtered:
            try:
                from augmentum.discovery.quality import _query_relevance_score, _tokenize_llm_query
                query_tokens = _tokenize_llm_query(query)
                any_relevant = any(
                    _query_relevance_score(
                        query_tokens,
                        title=(r.get("title") or ""),
                        snippet=(r.get("content") or r.get("snippet") or ""),
                        url=(r.get("url") or ""),
                    ) > 0
                    for r in raw_results
                ) if query_tokens else True  # all-stopwords query → no floor
            except Exception:
                log.debug("web_search_relevance_check_failed", exc_info=True)
                any_relevant = True  # fail-open to old behavior

            if any_relevant:
                log.info(
                    "web_search_filter_stripped_all",
                    query=query,
                    raw_count=len(raw_results),
                    action="raw_fallback",
                )
                filtered = raw_results
                quality_filtered = False
            else:
                log.info(
                    "web_search_zero_relevance",
                    query=query,
                    raw_count=len(raw_results),
                    sample_titles=[
                        (r.get("title") or "")[:80]
                        for r in raw_results[:3]
                    ],
                    action="return_no_results",
                )
                # filtered stays []; the "no results found" branch below fires

        # Deduplicate and limit. Cross-round dedup (turn_search_dedup): skip URLs
        # already returned earlier this turn so repeated rounds surface only NEW
        # pages; the filtered pool backfills with fresh ones.
        from augmentum.tools.turn_search_dedup import get_turn_dedup
        dedup = get_turn_dedup()
        seen_urls: set[str] = set()
        deduped: list[dict] = []
        skipped_dup = 0
        for result in filtered:
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            if dedup is not None and not dedup.mark("web", url):
                skipped_dup += 1
                continue
            deduped.append(result)
            if len(deduped) >= num_results:
                break

        if not deduped and skipped_dup:
            from augmentum.security.untrusted import wrap_untrusted
            return ToolResult(
                success=True,
                output=wrap_untrusted("web/search", (
                    f"All {skipped_dup} matching results were already returned earlier "
                    f"this turn — no new pages from this query."
                )) + "\n\n" + _DEDUP_FETCH_NUDGE,
                metadata={"query": query, "num_results": 0, "results": [], "urls": []},
            )

        # Format results into readable text. Snippets are cleaned of
        # SearXNG's <b> highlight tags / HTML entities here so the LLM
        # doesn't have to reason around markup it can't render.
        try:
            from augmentum.discovery.quality import _clean_search_snippet
        except Exception:
            _clean_search_snippet = lambda s: " ".join((s or "").split())  # noqa: E731

        lines: list[str] = []
        for i, result in enumerate(deduped, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = _clean_search_snippet(
                result.get("content") or "No snippet available."
            )
            lines.append(f"[{i}] {title}")
            lines.append(f"    URL: {url}")
            lines.append(f"    {snippet}")
            lines.append("")

        if skipped_dup:
            lines.append(
                f"(Note: {skipped_dup} result(s) already returned earlier this turn "
                f"were omitted. Refine the query or stop if you have enough.)"
            )
        raw_output = "\n".join(lines).rstrip()

        # Build Plan Phase 1.1: web search hits are the canonical
        # attacker-controlled surface — anyone can publish a page with
        # injected instructions. Wrap the formatted output in untrusted-
        # content markers so when this tool result is stitched into the
        # conversation, the model treats search hits as data, not
        # instructions. The policy preamble explaining the marker
        # convention is added by the recall path (memory/knowledge); on
        # tool-only turns the markers themselves still cue the model.
        from augmentum.security.untrusted import wrap_untrusted
        # Nudge rides OUTSIDE the untrusted markers — it's trusted harness
        # guidance, not page content, so the model must not treat it as data
        # (and an attacker page can't spoof it from inside a result snippet).
        output = wrap_untrusted("web/search", raw_output) + "\n\n" + _FETCH_NUDGE

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "query": query,
                "num_results": len(deduped),
                "categories": categories,
                "urls": [r.get("url", "") for r in deduped if r.get("url")],
                "unresponsive_engines": unresponsive,
                "fallback_used": fallback_used,
                "quality_filtered": quality_filtered,
                "results": [
                    {
                        "rank": i,
                        "title": _shorten(result.get("title", "Untitled"), 140),
                        "url": result.get("url", ""),
                        "snippet": _shorten(result.get("content", "No snippet available."), 240),
                        "source": _domain(result.get("url", "")),
                    }
                    for i, result in enumerate(deduped, 1)
                ],
            },
        )
