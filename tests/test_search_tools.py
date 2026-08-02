"""Tests for search support modules — filter, preferred_sources, result_processing."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest

from augmentum.tools.base import Tool, ToolCategory
from augmentum.tools.preferred_sources import (
    AVOID,
    EXCELLENT,
    GOOD,
    UNKNOWN,
    SourceInfo,
    domain_quality,
    get_sources_by_category,
    get_topic_sites,
    sort_urls_by_quality,
)
from augmentum.tools.result_processing import truncate_tool_result


# ---------------------------------------------------------------------------
# Preferred sources tests
# ---------------------------------------------------------------------------


class TestDomainQuality:
    """Quality tier lookups for known and unknown domains."""

    def test_known_excellent_domain(self):
        quality = domain_quality("https://weather.gov/forecast")
        assert quality == EXCELLENT

    def test_unknown_domain_returns_zero(self):
        quality = domain_quality("https://totally-unknown-site-xyz123.com/page")
        assert quality == UNKNOWN

    def test_avoid_domain(self):
        quality = domain_quality("https://nws.noaa.gov/data")
        assert quality == AVOID


class TestGetSourcesByCategory:
    """Category-based source retrieval."""

    def test_weather_category_has_sources(self):
        sources = list(get_sources_by_category("weather"))
        assert len(sources) > 0
        # All returned sources should have the weather category
        for domain, info in sources:
            assert "weather" in info.categories

    def test_unknown_category_returns_empty(self):
        sources = list(get_sources_by_category("nonexistent_category_xyz"))
        assert len(sources) == 0


class TestGetTopicSites:
    """Topic-to-domain mapping for search hints."""

    def test_weather_topic_returns_domains(self):
        sites = get_topic_sites("weather forecast")
        assert len(sites) > 0
        # Should return domain strings
        assert all(isinstance(s, str) for s in sites)

    def test_empty_query_returns_empty(self):
        sites = get_topic_sites("")
        assert len(sites) == 0


class TestSortUrlsByQuality:
    """URL sorting by domain quality tier."""

    def test_excellent_sorted_first(self):
        urls = [
            "https://unknown-site.com/page",
            "https://weather.gov/forecast",
        ]
        sorted_urls = sort_urls_by_quality(urls)
        # weather.gov (EXCELLENT) should come first
        assert "weather.gov" in sorted_urls[0]

    def test_unknown_urls_preserved(self):
        urls = ["https://a.com", "https://b.com"]
        sorted_urls = sort_urls_by_quality(urls)
        assert len(sorted_urls) == 2


class TestSourceInfo:
    """SourceInfo data structure."""

    def test_construction_with_defaults(self):
        info = SourceInfo(quality=GOOD, categories=("tech",))
        assert info.quality == GOOD
        assert info.content_type == "article"
        assert info.freshness == "static"
        assert info.requires_js is False

    def test_frozen_dataclass(self):
        info = SourceInfo(quality=EXCELLENT, categories=("news",))
        with pytest.raises(AttributeError):
            info.quality = AVOID


# ---------------------------------------------------------------------------
# Result processing tests
# ---------------------------------------------------------------------------


class TestTruncateToolResult:
    """Smart truncation with head + tail preservation."""

    def test_short_text_unchanged(self):
        text = "Hello world"
        assert truncate_tool_result(text, max_chars=1000) == text

    def test_long_text_truncated(self):
        text = "x" * 10000
        result = truncate_tool_result(text, max_chars=1000)
        assert len(result) <= 1200  # approximate, allows for notice
        assert "truncated" in result

    def test_preserves_tail(self):
        text = "HEAD" + "x" * 10000 + "TAIL_MARKER"
        result = truncate_tool_result(text, max_chars=2000, tail_chars=500)
        assert "TAIL_MARKER" in result
        assert "HEAD" in result

    def test_empty_text_unchanged(self):
        assert truncate_tool_result("") == ""

    def test_none_text_unchanged(self):
        # truncate_tool_result should handle None gracefully
        assert truncate_tool_result(None) is None  # type: ignore[arg-type]

    def test_exact_max_unchanged(self):
        text = "x" * 4000
        result = truncate_tool_result(text, max_chars=4000)
        assert result == text


# ---------------------------------------------------------------------------
# Tool filter tests
# ---------------------------------------------------------------------------


class TestToolFilter:
    """Smart tool pre-filtering based on query patterns."""

    def test_filter_module_importable(self):
        from augmentum.tools.filter import filter_tools_for_query
        assert callable(filter_tools_for_query)

    def test_search_query_includes_web_tools(self):
        from augmentum.tools.filter import filter_tools_for_query

        web_tool = MagicMock(spec=Tool)
        web_tool.name = "web_search"
        web_tool.category = ToolCategory.SEARCH

        calc_tool = MagicMock(spec=Tool)
        calc_tool.name = "calculator"
        calc_tool.category = ToolCategory.VERIFY

        tools = [web_tool, calc_tool]
        filtered = filter_tools_for_query("What is the latest news?", tools)
        names = [t.name for t in filtered]
        assert "web_search" in names

    def test_math_query_includes_calculator(self):
        from augmentum.tools.filter import filter_tools_for_query

        calc_tool = MagicMock(spec=Tool)
        calc_tool.name = "calculator"
        calc_tool.category = ToolCategory.VERIFY

        web_tool = MagicMock(spec=Tool)
        web_tool.name = "web_search"
        web_tool.category = ToolCategory.SEARCH

        tools = [web_tool, calc_tool]
        filtered = filter_tools_for_query("Calculate 2 + 2", tools)
        names = [t.name for t in filtered]
        assert "calculator" in names

    def test_url_query_includes_fetch(self):
        from augmentum.tools.filter import filter_tools_for_query

        fetch_tool = MagicMock(spec=Tool)
        fetch_tool.name = "web_fetch"
        fetch_tool.category = ToolCategory.FETCH

        tools = [fetch_tool]
        filtered = filter_tools_for_query("Summarize https://example.com", tools)
        names = [t.name for t in filtered]
        assert "web_fetch" in names

    def test_date_query_includes_datetime(self):
        from augmentum.tools.filter import filter_tools_for_query

        dt_tool = MagicMock(spec=Tool)
        dt_tool.name = "datetime"
        dt_tool.category = ToolCategory.VERIFY

        tools = [dt_tool]
        filtered = filter_tools_for_query("What time is it?", tools)
        names = [t.name for t in filtered]
        assert "datetime" in names
