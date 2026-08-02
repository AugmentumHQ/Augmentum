"""Tests for the unified web tool (auto-routing between fetch and search)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.tools.base import ToolCategory, ToolResult
from augmentum.tools.web import (
    WebTool,
    _build_search_query,
    _extract_urls,
    _extract_result_urls,
    _strip_urls,
)


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------


class TestUrlDetection:
    def test_https_url(self):
        assert _extract_urls("check https://example.com/page") == ["https://example.com/page"]

    def test_http_url(self):
        assert _extract_urls("see http://example.org") == ["http://example.org"]

    def test_bare_domain_com(self):
        urls = _extract_urls("go to example.com/news")
        assert len(urls) == 1
        assert urls[0] == "https://example.com/news"

    def test_bare_domain_gov(self):
        urls = _extract_urls("check weather.gov")
        assert len(urls) == 1
        assert urls[0] == "https://weather.gov"

    def test_bare_domain_io(self):
        urls = _extract_urls("look at docs.python.io/guide")
        assert len(urls) == 1
        assert "docs.python.io" in urls[0]

    def test_www_prefix(self):
        urls = _extract_urls("visit www.example.com")
        assert len(urls) == 1
        assert urls[0] == "https://www.example.com"

    def test_no_url(self):
        assert _extract_urls("what is the weather today") == []

    def test_strips_trailing_punctuation(self):
        urls = _extract_urls("see https://example.com.")
        assert urls[0] == "https://example.com"

    def test_multiple_urls(self):
        urls = _extract_urls("compare https://a.com and https://b.com")
        assert len(urls) == 2

    def test_plain_words_not_matched(self):
        # "recommend" contains ".com" substring but shouldn't match
        assert _extract_urls("I recommend this book") == []


class TestStripUrls:
    def test_strips_url(self):
        result = _strip_urls("summarize https://example.com please")
        assert "example.com" not in result
        assert "summarize" in result
        assert "please" in result

    def test_no_urls(self):
        assert _strip_urls("just a query") == "just a query"


class TestExtractResultUrls:
    def test_extracts_from_search_output(self):
        output = (
            "[1] Title\n"
            "    URL: https://example.com/1\n"
            "    Snippet\n\n"
            "[2] Title 2\n"
            "    URL: https://example.com/2\n"
            "    Snippet 2\n"
        )
        urls = _extract_result_urls(output)
        assert urls == ["https://example.com/1", "https://example.com/2"]

    def test_no_urls(self):
        assert _extract_result_urls("No results found.") == []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def search_tool():
    tool = AsyncMock()
    tool.name = "web_search"
    tool.execute = AsyncMock(return_value=ToolResult(
        success=True,
        output=(
            "[1] Weather NYC\n"
            "    URL: https://weather.gov/nyc\n"
            "    Sunny 72F\n\n"
            "[2] NYC Forecast\n"
            "    URL: https://forecast.io/nyc\n"
            "    Clear skies\n"
        ),
        metadata={"query": "weather NYC", "num_results": 2},
    ))
    return tool


@pytest.fixture
def fetch_tool():
    tool = AsyncMock()
    tool.name = "web_fetch"
    tool.execute = AsyncMock(return_value=ToolResult(
        success=True,
        output="Full page content: The weather in NYC is sunny and 72F...",
        metadata={"url": "https://weather.gov/nyc", "char_count": 55},
    ))
    return tool


@pytest.fixture
def web_tool(search_tool, fetch_tool):
    return WebTool(search_tool=search_tool, fetch_tool=fetch_tool)


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


class TestRouting:
    @pytest.mark.asyncio
    async def test_url_goes_to_fetch(self, web_tool, fetch_tool, search_tool):
        """A query with a URL should fetch directly, not search."""
        result = await web_tool.execute(query="https://example.com/article")
        fetch_tool.execute.assert_awaited_once()
        search_tool.execute.assert_not_awaited()
        assert result.success
        assert "Content from https://example.com/article" in result.output
        assert result.metadata["mode"] == "fetch"

    @pytest.mark.asyncio
    async def test_bare_domain_goes_to_fetch(self, web_tool, fetch_tool, search_tool):
        """A bare domain like example.com should be detected and fetched."""
        result = await web_tool.execute(query="read example.com/news")
        fetch_tool.execute.assert_awaited()
        assert result.success

    @pytest.mark.asyncio
    async def test_plain_query_goes_to_search(self, web_tool, fetch_tool, search_tool):
        """A plain text query should search, then auto-fetch top result."""
        result = await web_tool.execute(query="weather in NYC today")
        search_tool.execute.assert_awaited_once()
        # Auto-fetch should also be called
        assert fetch_tool.execute.await_count >= 1
        assert result.success
        assert "Search results" in result.output
        assert result.metadata["mode"] == "search_and_fetch"

    @pytest.mark.asyncio
    async def test_empty_query_fails(self, web_tool):
        result = await web_tool.execute(query="")
        assert not result.success
        assert "Empty" in result.error

    @pytest.mark.asyncio
    async def test_whitespace_query_fails(self, web_tool):
        result = await web_tool.execute(query="   ")
        assert not result.success


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestFallback:
    @pytest.mark.asyncio
    async def test_fetch_failure_falls_back_to_search(
        self, search_tool, fetch_tool,
    ):
        """If direct fetch fails, should fall back to search."""
        fetch_tool.execute = AsyncMock(return_value=ToolResult(
            success=False, error="Connection refused",
        ))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="https://broken.com/page")
        # Should have tried fetch, then fallen back to search
        assert fetch_tool.execute.await_count >= 1
        assert search_tool.execute.await_count >= 1
        assert result.success
        assert "Could not fetch" in result.output

    @pytest.mark.asyncio
    async def test_fetch_empty_falls_back_to_search(
        self, search_tool, fetch_tool,
    ):
        """If fetch returns empty content, should fall back to search."""
        fetch_tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="",  # empty
        ))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="https://empty.com")
        assert search_tool.execute.await_count >= 1

    @pytest.mark.asyncio
    async def test_search_failure_propagates(self, fetch_tool):
        """If search fails, the error should propagate."""
        search_tool = AsyncMock()
        search_tool.execute = AsyncMock(return_value=ToolResult(
            success=False, error="SearXNG unavailable",
        ))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="weather today")
        assert not result.success
        assert "SearXNG" in result.error


# ---------------------------------------------------------------------------
# Auto-fetch after search
# ---------------------------------------------------------------------------


class TestAutoFetch:
    @pytest.mark.asyncio
    async def test_auto_fetches_top_result(self, web_tool, fetch_tool):
        """After search, should auto-fetch a result URL (preferred sources first)."""
        result = await web_tool.execute(query="weather NYC")
        # fetch_tool called with a URL from search results
        fetch_calls = fetch_tool.execute.call_args_list
        assert len(fetch_calls) >= 1
        # Should fetch one of the search result URLs (weather.gov preferred)
        fetched_url = fetch_calls[0].kwargs.get("url")
        assert fetched_url in ("https://weather.gov/nyc", "https://forecast.io/nyc")

    @pytest.mark.asyncio
    async def test_auto_fetch_count_configurable(self, search_tool, fetch_tool):
        """auto_fetch_top controls how many results get fetched."""
        web_tool = WebTool(
            search_tool=search_tool,
            fetch_tool=fetch_tool,
            auto_fetch_top=2,
        )
        await web_tool.execute(query="weather NYC")
        # Should fetch top 2 URLs
        assert fetch_tool.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_auto_fetch_failure_still_returns_snippets(
        self, search_tool,
    ):
        """If auto-fetch fails, search snippets are still returned."""
        fetch_tool = AsyncMock()
        fetch_tool.execute = AsyncMock(side_effect=Exception("Timeout"))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="weather NYC")
        assert result.success
        # Should still have search results even though fetch failed
        assert "Search results" in result.output
        assert "Weather NYC" in result.output

    @pytest.mark.asyncio
    async def test_no_search_urls_returns_snippets_only(self, fetch_tool):
        """If search results have no URLs, just return snippets."""
        search_tool = AsyncMock()
        search_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="No results found.",
            metadata={"query": "obscure topic", "num_results": 0},
        ))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="obscure topic")
        assert result.success
        assert result.metadata["mode"] == "search_only"
        # fetch should NOT be called since no URLs in results
        fetch_tool.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auto_fetch_skips_failed_tries_next(self):
        """If first URL returns 403/empty, should try the next URL."""
        # Use same-tier domains so quality sorting preserves original order
        search_tool = AsyncMock()
        search_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output=(
                "[1] Site A\n"
                "    URL: https://siteA.xyz/page\n"
                "    Snippet A\n\n"
                "[2] Site B\n"
                "    URL: https://siteB.xyz/page\n"
                "    Snippet B\n"
            ),
            metadata={"query": "test", "num_results": 2},
        ))
        fail_result = ToolResult(success=True, output="")  # empty content
        ok_result = ToolResult(
            success=True,
            output="Actual content here...",
            metadata={"url": "https://siteB.xyz/page", "char_count": 22},
        )
        fetch_tool = AsyncMock()
        fetch_tool.execute = AsyncMock(side_effect=[fail_result, ok_result])

        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)
        result = await web_tool.execute(query="test query")

        # Should have tried both URLs
        assert fetch_tool.execute.await_count == 2
        # Should have content from the second URL
        assert "Actual content" in result.output
        assert result.metadata["fetched_urls"] == ["https://siteB.xyz/page"]

    @pytest.mark.asyncio
    async def test_auto_fetch_skips_403_tries_next(self):
        """If first URL returns a fetch error (403), should try the next."""
        search_tool = AsyncMock()
        search_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output=(
                "[1] Site A\n"
                "    URL: https://siteA.xyz/page\n"
                "    Snippet A\n\n"
                "[2] Site B\n"
                "    URL: https://siteB.xyz/page\n"
                "    Snippet B\n"
            ),
            metadata={"query": "test", "num_results": 2},
        ))
        fail_result = ToolResult(success=False, error="HTTP 403 Forbidden")
        ok_result = ToolResult(
            success=True,
            output="Forecast data...",
            metadata={"url": "https://siteB.xyz/page", "char_count": 18},
        )
        fetch_tool = AsyncMock()
        fetch_tool.execute = AsyncMock(side_effect=[fail_result, ok_result])

        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)
        result = await web_tool.execute(query="test query")

        assert fetch_tool.execute.await_count == 2
        assert "Forecast data" in result.output

    @pytest.mark.asyncio
    async def test_auto_fetch_stops_after_enough_successes(self, search_tool):
        """Should stop fetching once auto_fetch_top successes are reached."""
        ok_result = ToolResult(
            success=True, output="Content...",
            metadata={"char_count": 10},
        )
        fetch_tool = AsyncMock()
        fetch_tool.execute = AsyncMock(return_value=ok_result)

        web_tool = WebTool(
            search_tool=search_tool, fetch_tool=fetch_tool, auto_fetch_top=1,
        )
        await web_tool.execute(query="weather NYC")

        # Should stop after 1 success, not try the second URL
        assert fetch_tool.execute.await_count == 1


# ---------------------------------------------------------------------------
# Tool properties
# ---------------------------------------------------------------------------


class TestToolProperties:
    def test_name(self, web_tool):
        assert web_tool.name == "web"

    def test_category(self, web_tool):
        assert web_tool.category == ToolCategory.SEARCH

    def test_schema_has_query(self, web_tool):
        schema = web_tool.input_schema
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_validate_input(self, web_tool):
        assert web_tool.validate_input(query="test")
        assert not web_tool.validate_input(query="")
        assert not web_tool.validate_input(query=123)


# ---------------------------------------------------------------------------
# Pre-search: topic-aware site hints
# ---------------------------------------------------------------------------


class TestBuildSearchQuery:
    def test_weather_adds_site_hint(self):
        q = _build_search_query("weather in NYC today")
        assert "site:weather.gov" in q
        # Original query preserved
        assert "weather in NYC today" in q

    def test_python_adds_site_hint(self):
        q = _build_search_query("python async generators")
        assert "site:docs.python.org" in q

    def test_no_topic_match_unchanged(self):
        q = _build_search_query("random unrelated query")
        assert q == "random unrelated query"
        assert "site:" not in q

    def test_max_two_site_hints(self):
        q = _build_search_query("python list comprehension")
        site_count = q.count("site:")
        assert site_count <= 2

    def test_medical_adds_site_hint(self):
        q = _build_search_query("symptoms of diabetes")
        assert "site:" in q
        # Should steer toward medical sources
        assert any(s in q for s in [
            "site:mayoclinic.org", "site:medlineplus.gov",
            "site:cdc.gov", "site:nih.gov",
        ])

    def test_tax_adds_irs(self):
        q = _build_search_query("how to file tax return")
        assert "site:irs.gov" in q

    def test_security_adds_site_hint(self):
        q = _build_search_query("owasp sql injection prevention")
        assert "site:owasp.org" in q


# ---------------------------------------------------------------------------
# Post-search: AVOID domain filtering
# ---------------------------------------------------------------------------


class TestAvoidFiltering:
    @pytest.mark.asyncio
    async def test_avoid_domains_filtered_from_autofetch(self):
        """AVOID-tier domains should be filtered out before auto-fetch."""
        search_tool = AsyncMock()
        search_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output=(
                "[1] AccuWeather NYC\n"
                "    URL: https://accuweather.com/nyc\n"
                "    Sunny\n\n"
                "[2] NWS NYC\n"
                "    URL: https://weather.gov/nyc\n"
                "    Clear 72F\n"
            ),
            metadata={"query": "weather NYC", "num_results": 2},
        ))
        fetch_tool = AsyncMock()
        fetch_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="NWS forecast content...",
            metadata={"url": "https://weather.gov/nyc", "char_count": 22},
        ))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="test query without topic match")
        # Should only fetch weather.gov (accuweather.com is AVOID)
        assert fetch_tool.execute.await_count == 1
        fetched_url = fetch_tool.execute.call_args.kwargs.get("url")
        assert fetched_url == "https://weather.gov/nyc"

    @pytest.mark.asyncio
    async def test_all_avoid_degrades_gracefully(self):
        """If ALL results are AVOID, should still try them (degrade gracefully)."""
        search_tool = AsyncMock()
        search_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output=(
                "[1] LinkedIn Profile\n"
                "    URL: https://linkedin.com/in/user\n"
                "    Profile\n"
            ),
            metadata={"query": "test", "num_results": 1},
        ))
        fetch_tool = AsyncMock()
        fetch_tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="Some content",
            metadata={"char_count": 12},
        ))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="obscure query no topic")
        # Should still try the AVOID URL rather than returning nothing
        assert fetch_tool.execute.await_count >= 1


# ---------------------------------------------------------------------------
# Post-fetch: source annotation
# ---------------------------------------------------------------------------


class TestSourceAnnotation:
    @pytest.mark.asyncio
    async def test_fetched_content_has_source_annotation(self):
        """Fetched content from known sources should include quality metadata."""
        search_tool = AsyncMock()
        search_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output=(
                "[1] Weather Forecast\n"
                "    URL: https://weather.gov/forecast\n"
                "    Current conditions\n"
            ),
            metadata={"query": "weather", "num_results": 1},
        ))
        fetch_tool = AsyncMock()
        fetch_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output="Temperature: 72F, Humidity: 45%...",
            metadata={"url": "https://weather.gov/forecast", "char_count": 35},
        ))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="unrelated query no hints")
        # Should include source quality annotation
        assert "excellent" in result.output
        assert "weather" in result.output

    @pytest.mark.asyncio
    async def test_unknown_source_no_annotation(self):
        """Unknown domains should not have a source annotation line."""
        search_tool = AsyncMock()
        search_tool.execute = AsyncMock(return_value=ToolResult(
            success=True,
            output=(
                "[1] Random\n"
                "    URL: https://unknownsite12345.xyz/page\n"
                "    Content\n"
            ),
            metadata={"query": "test", "num_results": 1},
        ))
        fetch_tool = AsyncMock()
        fetch_tool.execute = AsyncMock(return_value=ToolResult(
            success=True, output="Page content here.",
            metadata={"char_count": 18},
        ))
        web_tool = WebTool(search_tool=search_tool, fetch_tool=fetch_tool)

        result = await web_tool.execute(query="obscure query")
        # Should NOT have quality annotation for unknown sites
        assert "excellent" not in result.output
        assert "quality=" not in result.output
