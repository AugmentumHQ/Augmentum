"""Tests for direct URL fetch in search pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.search.direct_fetch import (
    extract_urls,
    fetch_urls_for_context,
    format_fetched_context,
    strip_urls,
)

# ---------------------------------------------------------------------------
# extract_urls
# ---------------------------------------------------------------------------


class TestExtractUrls:
    def test_single_url(self):
        urls = extract_urls("summarize https://example.com/article")
        assert urls == ["https://example.com/article"]

    def test_multiple_urls(self):
        urls = extract_urls("compare https://a.com and https://b.com")
        assert urls == ["https://a.com", "https://b.com"]

    def test_url_with_path(self):
        urls = extract_urls("read https://example.com/path/to/article?q=1&p=2")
        assert len(urls) == 1
        assert "q=1&p=2" in urls[0]

    def test_url_with_trailing_period(self):
        urls = extract_urls("Check this: https://example.com/article.")
        assert urls == ["https://example.com/article"]

    def test_url_with_trailing_comma(self):
        urls = extract_urls("See https://example.com, it's great")
        assert urls == ["https://example.com"]

    def test_url_with_trailing_question_mark(self):
        """Trailing ? after URL (not in query string) should be stripped."""
        urls = extract_urls("What is https://example.com?")
        assert urls == ["https://example.com"]

    def test_url_in_parentheses(self):
        urls = extract_urls("(see https://example.com/article)")
        assert urls == ["https://example.com/article"]

    def test_no_urls(self):
        urls = extract_urls("what is the capital of France")
        assert urls == []

    def test_deduplicates(self):
        urls = extract_urls("https://example.com and also https://example.com")
        assert urls == ["https://example.com"]

    def test_http_and_https(self):
        urls = extract_urls("http://old.com and https://new.com")
        assert len(urls) == 2

    def test_url_with_fragment(self):
        urls = extract_urls("see https://example.com/page#section")
        assert urls == ["https://example.com/page#section"]

    def test_complex_url(self):
        urls = extract_urls("look at https://arxiv.org/abs/2401.12345v2")
        assert urls == ["https://arxiv.org/abs/2401.12345v2"]


# ---------------------------------------------------------------------------
# strip_urls
# ---------------------------------------------------------------------------


class TestStripUrls:
    def test_strip_single_url(self):
        result = strip_urls("summarize https://example.com/article please")
        assert result == "summarize please"

    def test_strip_multiple_urls(self):
        result = strip_urls("compare https://a.com and https://b.com content")
        assert result == "compare and content"

    def test_no_urls(self):
        result = strip_urls("what is Python")
        assert result == "what is Python"

    def test_url_only(self):
        result = strip_urls("https://example.com/article")
        assert result == ""

    def test_preserves_surrounding_text(self):
        result = strip_urls("read this https://example.com then summarize")
        assert result == "read this then summarize"


# ---------------------------------------------------------------------------
# format_fetched_context
# ---------------------------------------------------------------------------


class TestFormatFetchedContext:
    def test_formats_successful_fetch(self):
        fetched = [{
            "url": "https://example.com/article",
            "content": "This is the article content.",
            "success": True,
            "error": None,
            "char_count": 27,
        }]
        result = format_fetched_context(fetched, credibility_enabled=False)
        assert "Directly fetched: https://example.com/article" in result
        assert "URL: https://example.com/article" in result
        assert "This is the article content." in result

    def test_skips_failed_fetch(self):
        fetched = [{
            "url": "https://example.com/404",
            "content": "",
            "success": False,
            "error": "404 Not Found",
            "char_count": 0,
        }]
        result = format_fetched_context(fetched, credibility_enabled=False)
        assert result == ""

    def test_includes_credibility_tag(self):
        fetched = [{
            "url": "https://nasa.gov/missions",
            "content": "NASA content here.",
            "success": True,
            "error": None,
            "char_count": 18,
        }]
        result = format_fetched_context(fetched, credibility_enabled=True)
        assert "[credibility:" in result
        assert "institutional" in result

    def test_empty_list(self):
        assert format_fetched_context([]) == ""

    def test_multiple_results(self):
        fetched = [
            {
                "url": "https://a.com",
                "content": "Content A",
                "success": True,
                "error": None,
                "char_count": 9,
            },
            {
                "url": "https://b.com",
                "content": "Content B",
                "success": True,
                "error": None,
                "char_count": 9,
            },
        ]
        result = format_fetched_context(fetched, credibility_enabled=False)
        assert "Content A" in result
        assert "Content B" in result


# ---------------------------------------------------------------------------
# fetch_urls_for_context
# ---------------------------------------------------------------------------


class TestFetchUrlsForContext:
    @pytest.fixture()
    def mock_registry(self):
        """Create a mock tool registry with a web_fetch tool."""
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=MagicMock(
            output="Fetched page content here.",
            success=True,
            error=None,
        ))

        registry = MagicMock()
        registry.get.return_value = mock_tool
        return registry

    @pytest.mark.asyncio()
    async def test_fetches_urls(self, mock_registry):
        results = await fetch_urls_for_context(
            ["https://example.com/article"],
            mock_registry,
        )
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["content"] == "Fetched page content here."

    @pytest.mark.asyncio()
    async def test_handles_fetch_failure(self):
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value=MagicMock(
            output=None,
            success=False,
            error="Connection refused",
        ))
        registry = MagicMock()
        registry.get.return_value = mock_tool

        results = await fetch_urls_for_context(
            ["https://unreachable.example.com"],
            registry,
        )
        assert len(results) == 1
        assert results[0]["success"] is False

    @pytest.mark.asyncio()
    async def test_no_registry(self):
        results = await fetch_urls_for_context(["https://example.com"], None)
        assert results == []

    @pytest.mark.asyncio()
    async def test_no_web_fetch_tool(self):
        registry = MagicMock()
        registry.get.return_value = None
        results = await fetch_urls_for_context(["https://example.com"], registry)
        assert results == []

    @pytest.mark.asyncio()
    async def test_caps_at_five_urls(self, mock_registry):
        urls = [f"https://example.com/{i}" for i in range(10)]
        results = await fetch_urls_for_context(urls, mock_registry)
        assert len(results) == 5

    @pytest.mark.asyncio()
    async def test_empty_urls(self, mock_registry):
        results = await fetch_urls_for_context([], mock_registry)
        assert results == []

    @pytest.mark.asyncio()
    async def test_timeout_handling(self):
        async def slow_fetch(**kwargs):
            import asyncio
            await asyncio.sleep(10)

        mock_tool = MagicMock()
        mock_tool.execute = slow_fetch
        registry = MagicMock()
        registry.get.return_value = mock_tool

        results = await fetch_urls_for_context(
            ["https://slow.example.com"],
            registry,
            timeout=0.1,
        )
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "timed out" in results[0]["error"]


# ---------------------------------------------------------------------------
# Integration: _needs_search with URLs
# ---------------------------------------------------------------------------


class TestNeedsSearchWithUrls:
    """Verify that _needs_search triggers for URL-containing queries."""

    def test_url_triggers_search(self):
        from augmentum.modes.analytical.engine import AnalyticalEngine

        assert AnalyticalEngine._needs_search(
            "summarize https://example.com/article"
        ) is True

    def test_url_only_triggers_search(self):
        from augmentum.modes.analytical.engine import AnalyticalEngine

        assert AnalyticalEngine._needs_search(
            "https://example.com/article"
        ) is True

    def test_no_url_no_keywords_no_search(self):
        from augmentum.modes.analytical.engine import AnalyticalEngine

        assert AnalyticalEngine._needs_search(
            "calculate 2 + 2"
        ) is False
