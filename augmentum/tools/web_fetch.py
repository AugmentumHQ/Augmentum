"""Web content fetching tool with content extraction.

Shares the browse_fetch dispatch chain so agent tools get the same
clean, structured content humans see in the browse tab: Wikipedia REST
API instead of JS-walled HTML, arXiv Atom API instead of PDF binary,
Reddit Atom feed instead of "prove you're human" page, PDF text via
pymupdf, .ipynb cells parsed, etc.

The chain runs only when an httpx client is wired in at construction
(production path). The no-arg constructor used by older tests preserves
the original trafilatura-only behavior so unit tests don't break.
"""

from __future__ import annotations

import asyncio
import html
import re
import types
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger
from augmentum.utils.safe_http import SafeHttpClient, SafeHttpError

if TYPE_CHECKING:
    import httpx

log = get_logger(__name__)


def _strip_html_tags(html: str) -> str:
    """Crude HTML tag stripping fallback when trafilatura is not available."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate_at_paragraph(text: str, max_chars: int) -> str:
    """Truncate text at the nearest paragraph boundary before max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rfind("\n\n")
    if cut > max_chars // 2:
        return text[:cut].rstrip()
    cut = text[:max_chars].rfind("\n")
    if cut > max_chars // 2:
        return text[:cut].rstrip()
    return text[:max_chars].rstrip() + "..."


def _extract_with_trafilatura(html: str) -> str | None:
    """Attempt content extraction via trafilatura (may not be installed)."""
    try:
        import trafilatura  # type: ignore[import-untyped]
    except ImportError:
        return None
    return trafilatura.extract(html)


def _extract_title(html_text: str) -> str:
    """Extract the page title from raw HTML for UI previews."""
    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    title = html.unescape(match.group(1))
    title = " ".join(title.split())
    return title[:160]


def _format_intercept_for_llm(result: dict, max_chars: int) -> ToolResult:
    """Shape a browse_fetch intercept dict for LLM consumption.

    Intercepts return rich payloads with html/text/title/author/date/etc.
    The LLM wants the text body with a small structured header so it can
    cite source, author, and publish date without re-parsing the body.
    """
    text = (result.get("text") or "").strip()
    if not text:
        # Some intercepts (video embeds) have no body — surface metadata.
        text = result.get("title") or "(no text content)"

    truncated = _truncate_at_paragraph(text, max_chars)

    header_parts: list[str] = []
    if result.get("title"):
        header_parts.append(f"Title: {result['title']}")
    if result.get("author"):
        header_parts.append(f"Author: {result['author']}")
    if result.get("date"):
        header_parts.append(f"Date: {result['date']}")
    if result.get("sitename"):
        header_parts.append(f"Site: {result['sitename']}")
    final_url = result.get("url") or ""
    if final_url:
        header_parts.append(f"URL: {final_url}")

    output = "\n".join(header_parts)
    if output:
        output += "\n\n"
    output += truncated

    return ToolResult(
        success=True,
        output=output,
        metadata={
            "url": final_url,
            "title": result.get("title", ""),
            "author": result.get("author", ""),
            "date": result.get("date", ""),
            "sitename": result.get("sitename", ""),
            "source": result.get("source", ""),
            "page_type": result.get("page_type", ""),
            "char_count": len(truncated),
            "truncated": len(text) > max_chars,
        },
    )


class WebFetchTool(Tool):
    """Fetch and extract content from a web page."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch and extract content from a web page"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FETCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return",
                    "default": 20000,
                },
            },
            "required": ["url"],
        }

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._safe_client = SafeHttpClient()
        # Shared httpx client used by browse_fetch's platform intercepts
        # for their per-platform JSON API calls. When None we fall back
        # to the trafilatura-only path (preserves older unit-test behavior).
        self._http_client = http_client

    def validate_input(self, **kwargs) -> bool:
        url = kwargs.get("url", "")
        return isinstance(url, str) and url.startswith(("http://", "https://"))

    async def execute(self, *, url: str, max_chars: int = 20000) -> ToolResult:
        """Fetch a URL, extract its content, and return truncated text."""
        if not url.startswith(("http://", "https://")):
            return ToolResult(success=False, error="URL must start with http:// or https://")

        max_chars = max(100, min(int(max_chars), 100_000))

        # No http_client → use the simple trafilatura-only path (test fallback).
        if self._http_client is None:
            return await self._fallback_simple_fetch(url, max_chars)

        # Lazy import to keep tool-load free of proxy-layer dependencies
        # at module import time (avoids load-order surprises in tests).
        try:
            from augmentum.proxy.browse_routes import (
                _canonicalize_url,
                _fetch_with_chrome_tls,
                _handle_file_type,
                _is_hostile_domain,
                _is_junk_content,
                _try_arxiv_api,
                _try_discourse_api,
                _try_github_api,
                _try_github_blob_api,
                _try_github_gist_api,
                _try_hf_api,
                _try_hn_api,
                _try_reddit_api,
                _try_render_feed,
                _try_stackexchange_api,
                _try_wikipedia_api,
            )
        except Exception as exc:
            log.warning("web_fetch_pipeline_import_failed", error=str(exc))
            return await self._fallback_simple_fetch(url, max_chars)

        url = _canonicalize_url(url)
        host = (urlparse(url).hostname or "").lower()
        if _is_hostile_domain(host):
            return ToolResult(
                success=False,
                error=(
                    f"This site ({host}) requires a real browser session and "
                    "doesn't render useful content to a server-side fetch. "
                    "Try a different URL — search for the topic on a "
                    "fetch-friendly source."
                ),
                metadata={"url": url, "hostname": host, "reason": "hostile_domain"},
            )

        # Synthesize a Request-shaped object so the existing intercept
        # functions (which expect FastAPI's Request) can pull our shared
        # httpx client off `.app.state.http_client` unchanged.
        request_shim = types.SimpleNamespace(
            app=types.SimpleNamespace(
                state=types.SimpleNamespace(http_client=self._http_client),
            )
        )

        # Platform intercepts — same order as browse_fetch. Each returns
        # a dict on match, None on miss; we format the dict for the LLM.
        intercepts = (
            _try_wikipedia_api,
            _try_reddit_api,
            _try_hn_api,
            _try_github_blob_api,
            _try_github_gist_api,
            _try_github_api,
            _try_hf_api,
            _try_arxiv_api,
            _try_stackexchange_api,
            _try_discourse_api,
        )
        for intercept in intercepts:
            try:
                result = await intercept(url, request_shim)
            except Exception as exc:
                log.debug(
                    "web_fetch_intercept_failed",
                    intercept=intercept.__name__,
                    url=url[:120],
                    error=str(exc),
                )
                continue
            if result:
                return _format_intercept_for_llm(result, max_chars)

        # Raw fetch with Chrome TLS fingerprint (better anti-bot than the
        # plain SafeHttpClient path), then run the post-fetch handlers.
        try:
            raw_html, fetch_meta = await _fetch_with_chrome_tls(url)
        except SafeHttpError as exc:
            return ToolResult(
                success=False,
                error=f"Fetch blocked: {exc}",
                metadata={"url": url, "reason": "ssrf_or_size"},
            )
        except Exception as exc:
            log.warning("web_fetch_failed", url=url, error=str(exc))
            return ToolResult(
                success=False,
                error=f"Fetch failed: {exc}",
                metadata={"url": url},
            )

        # RSS/Atom feed sniff (URLs that ARE feeds, not feeds linked from HTML)
        try:
            feed_result = _try_render_feed(url, raw_html, fetch_meta)
        except Exception:
            feed_result = None
        if feed_result:
            return _format_intercept_for_llm(feed_result, max_chars)

        # File-type handler — PDFs (pymupdf text), .ipynb cells, code files,
        # pretty-printed JSON/XML. Anything binary or rich gets routed here
        # before trafilatura tries to extract from garbage bytes.
        try:
            file_result = _handle_file_type(url, raw_html, fetch_meta)
        except Exception:
            file_result = None
        if file_result:
            return _format_intercept_for_llm(file_result, max_chars)

        # Article extraction via trafilatura, fall back to tag-strip.
        extracted: str | None = None
        try:
            extracted = await asyncio.to_thread(_extract_with_trafilatura, raw_html)
        except Exception:
            log.debug("trafilatura_extraction_failed", url=url, exc_info=True)
        if not extracted:
            extracted = _strip_html_tags(raw_html)

        # Junk-content detection — return a clear actionable error so the
        # LLM understands "this URL isn't going to work, try another" rather
        # than processing the wall text as if it were the article body.
        if extracted and _is_junk_content(extracted):
            log.info("web_fetch_junk_detected", url=url, preview=extracted[:80])
            return ToolResult(
                success=False,
                error=(
                    "This page returned a login wall, CAPTCHA, or error page "
                    "instead of real content. The URL may require a logged-in "
                    "session or render content client-side."
                ),
                metadata={"url": url, "reason": "junk_content"},
            )

        if not extracted:
            return ToolResult(
                success=True,
                output="(page returned no extractable text content)",
                metadata={"url": url, "char_count": 0},
            )

        truncated = _truncate_at_paragraph(extracted, max_chars)
        final_url = str(fetch_meta.get("url", url))
        title = _extract_title(raw_html)
        output = f"{truncated}\n\nSource: {final_url}"

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "url": final_url,
                "title": title,
                "char_count": len(truncated),
                "content_type": fetch_meta.get("content_type", ""),
                "truncated": len(extracted) > max_chars,
            },
        )

    async def _fallback_simple_fetch(self, url: str, max_chars: int) -> ToolResult:
        """Trafilatura-only fetch — used when no http_client is wired in.

        Preserves the original behavior so older unit tests that construct
        `WebFetchTool()` with no args don't break.
        """
        try:
            raw_html, fetch_meta = await self._safe_client.fetch(url)
        except SafeHttpError as exc:
            return ToolResult(success=False, error=f"Fetch blocked: {exc}")
        except Exception as exc:
            log.warning("web_fetch_failed", url=url, error=str(exc))
            return ToolResult(success=False, error=f"Fetch failed: {exc}")

        extracted: str | None = None
        try:
            extracted = await asyncio.to_thread(_extract_with_trafilatura, raw_html)
        except Exception:
            log.debug("trafilatura_extraction_failed", url=url, exc_info=True)

        if not extracted:
            extracted = _strip_html_tags(raw_html)

        if not extracted:
            return ToolResult(
                success=True,
                output="(page returned no extractable text content)",
                metadata={"url": url, "char_count": 0},
            )

        truncated = _truncate_at_paragraph(extracted, max_chars)
        final_url = str(fetch_meta.get("url", url))
        title = _extract_title(raw_html)
        preview_excerpt = _truncate_at_paragraph(truncated, 420)
        raw_output = f"{truncated}\n\nSource: {final_url}"

        # Build Plan Phase 1.1: fetched URL content is the single most
        # attacker-controlled surface in the system — anyone can host a
        # page with embedded prompt injection. Wrap the extracted body
        # in untrusted-content markers so the model treats it as data.
        from augmentum.security.untrusted import wrap_untrusted
        output = wrap_untrusted("web/fetch", raw_output)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "url": final_url,
                "title": title,
                "preview_excerpt": preview_excerpt,
                "char_count": len(truncated),
                "content_type": fetch_meta.get("content_type", ""),
                "truncated": len(extracted) > max_chars,
            },
        )
