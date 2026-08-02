"""Tests for ToolRegistry — discovery, lookup, alias resolution, metrics."""

from __future__ import annotations

from unittest.mock import MagicMock

from augmentum.tools.base import Tool, ToolCategory
from augmentum.tools.registry import ToolMetrics, ToolRegistry


def _make_tool(name: str, category: ToolCategory = ToolCategory.SEARCH) -> MagicMock:
    """Create a minimal mock tool."""
    tool = MagicMock(spec=Tool)
    tool.name = name
    tool.category = category
    return tool


class TestToolRegistryCore:
    """Registration, retrieval, and listing."""

    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = _make_tool("web_search")
        reg.register(tool)
        assert reg.get("web_search") is tool

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None

    def test_list_tools_returns_all(self):
        reg = ToolRegistry()
        t1 = _make_tool("web_search", ToolCategory.SEARCH)
        t2 = _make_tool("calculator", ToolCategory.VERIFY)
        reg.register(t1)
        reg.register(t2)
        assert len(reg.list_tools()) == 2

    def test_list_tools_filters_by_category(self):
        reg = ToolRegistry()
        t1 = _make_tool("web_search", ToolCategory.SEARCH)
        t2 = _make_tool("calculator", ToolCategory.VERIFY)
        reg.register(t1)
        reg.register(t2)
        search_tools = reg.list_tools(category=ToolCategory.SEARCH)
        assert len(search_tools) == 1
        assert search_tools[0].name == "web_search"

    def test_unregister_existing(self):
        reg = ToolRegistry()
        tool = _make_tool("web_search")
        reg.register(tool)
        assert reg.unregister("web_search") is True
        assert reg.get("web_search") is None

    def test_unregister_missing_returns_false(self):
        reg = ToolRegistry()
        assert reg.unregister("nonexistent") is False

    def test_register_duplicate_warns(self):
        reg = ToolRegistry()
        t1 = _make_tool("web_search")
        t2 = _make_tool("web_search")
        reg.register(t1)
        reg.register(t2)
        # Second registration overwrites
        assert reg.get("web_search") is t2


class TestToolRegistryResolve:
    """Fuzzy name resolution (aliases, normalization, substring)."""

    def test_resolve_exact_match(self):
        reg = ToolRegistry()
        tool = _make_tool("web_search")
        reg.register(tool)
        assert reg.resolve("web_search") is tool

    def test_resolve_alias(self):
        reg = ToolRegistry()
        tool = _make_tool("web_search")
        reg.register(tool)
        assert reg.resolve("google") is tool

    def test_resolve_normalized_name(self):
        reg = ToolRegistry()
        tool = _make_tool("web_search")
        reg.register(tool)
        # Markdown-wrapped name
        assert reg.resolve("`web_search`") is tool

    def test_resolve_hyphenated_alias(self):
        reg = ToolRegistry()
        tool = _make_tool("web_search")
        reg.register(tool)
        assert reg.resolve("web-search") is tool

    def test_resolve_substring_match(self):
        reg = ToolRegistry()
        tool = _make_tool("web_search")
        reg.register(tool)
        # "search" is a substring of "web_search" and an alias
        resolved = reg.resolve("search")
        assert resolved is tool

    def test_resolve_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg.resolve("totally_unknown_tool_xyz") is None

    def test_resolve_python_alias(self):
        reg = ToolRegistry()
        tool = _make_tool("python_exec")
        reg.register(tool)
        assert reg.resolve("python") is tool
        assert reg.resolve("run_code") is tool

    def test_resolve_companion_catalog_keys(self):
        # The companion runtime advertises catalogue KEYS (files_read, code_run)
        # whose canonical registry ids differ (search_files, python_exec). These
        # must resolve so the native FC loop can expose them instead of logging
        # tool_resolve_failed every turn.
        reg = ToolRegistry()
        sf = _make_tool("search_files")
        px = _make_tool("python_exec")
        reg.register(sf)
        reg.register(px)
        assert reg.resolve("files_read") is sf
        assert reg.resolve("code_run") is px

    def test_resolve_calculator_alias(self):
        reg = ToolRegistry()
        tool = _make_tool("calculator")
        reg.register(tool)
        assert reg.resolve("calc") is tool
        assert reg.resolve("calculate") is tool


class TestToolRegistryPhase:
    """Phase-based tool filtering for UARF pipeline."""

    def test_get_for_phase_relevant(self):
        reg = ToolRegistry()
        reg.register(_make_tool("web_search", ToolCategory.SEARCH))
        reg.register(_make_tool("calculator", ToolCategory.VERIFY))
        tools = reg.get_for_phase("relevant")
        names = [t.name for t in tools]
        assert "web_search" in names
        # VERIFY not in RELEVANT phase
        assert "calculator" not in names

    def test_get_for_phase_apply(self):
        reg = ToolRegistry()
        reg.register(_make_tool("web_search", ToolCategory.SEARCH))
        reg.register(_make_tool("calculator", ToolCategory.VERIFY))
        tools = reg.get_for_phase("apply")
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "calculator" in names

    def test_get_for_phase_empty(self):
        reg = ToolRegistry()
        reg.register(_make_tool("web_search", ToolCategory.SEARCH))
        tools = reg.get_for_phase("assess")
        assert tools == []

    def test_get_for_phase_exclude(self):
        reg = ToolRegistry()
        reg.register(_make_tool("web_search", ToolCategory.SEARCH))
        reg.register(_make_tool("web_fetch", ToolCategory.FETCH))
        tools = reg.get_for_phase("relevant", exclude=frozenset({"web_search"}))
        names = [t.name for t in tools]
        assert "web_search" not in names
        assert "web_fetch" in names


class TestToolMetrics:
    """Lightweight call tracking."""

    def test_record_success(self):
        m = ToolMetrics()
        m.record("web_search", success=True, elapsed_ms=100.0)
        snap = m.snapshot()
        assert snap["web_search"]["calls"] == 1
        assert snap["web_search"]["successes"] == 1

    def test_record_failure(self):
        m = ToolMetrics()
        m.record("web_search", success=False, elapsed_ms=50.0)
        snap = m.snapshot()
        assert snap["web_search"]["failures"] == 1

    def test_record_cache_hit(self):
        m = ToolMetrics()
        m.record("calculator", success=True, elapsed_ms=1.0, cached=True)
        snap = m.snapshot()
        assert snap["calculator"]["cache_hits"] == 1
        # Cached should not count as success
        assert snap["calculator"]["successes"] == 0
