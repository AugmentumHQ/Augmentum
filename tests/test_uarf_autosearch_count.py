"""Regression test: UARF auto-search must count only real result blocks.

The cross-round search dedup (turn_search_dedup) makes web_search append a
human-facing "(N already shown)" note to its output. UARF's auto-search parses
that text to count results into ``search_result_count`` (which gates the search
retry at engine.py:1000). Url-less blocks — the dedup note + untrusted-wrapper
markers — must NOT be counted, or a thin-result turn could wrongly suppress its
retry.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from augmentum.modes.analytical.engine import AnalyticalEngine
from augmentum.tools.base import ToolResult


class _FakeSearchTool:
    """Mimics web_search.execute output: 2 real result blocks + a dedup note."""

    name = "web_search"

    async def execute(self, *, query="", num_results=5, **kwargs):
        output = (
            "[1] Alpha\n    URL: https://a.example/x\n    snippet a\n\n"
            "[2] Beta\n    URL: https://b.example/y\n    snippet b\n\n"
            "(Note: 3 result(s) already returned earlier this turn were omitted.)"
        )
        return ToolResult(success=True, output=output)


class _FakeRegistry:
    def __init__(self, tool):
        self._tool = tool

    def get(self, name):
        return self._tool if name == "web_search" else None


@pytest.mark.asyncio
async def test_autosearch_counts_only_url_blocks():
    engine = AnalyticalEngine(MagicMock())
    engine._tool_registry = _FakeRegistry(_FakeSearchTool())

    await engine._execute_auto_search(["weather"], results_per_query=5)

    # Two URL-bearing results; the "(Note: …)" block must not be counted.
    assert engine._state.search_result_count == 2


@pytest.mark.asyncio
async def test_autosearch_all_dedup_note_counts_zero():
    """A query that returns ONLY the 'all already shown' message → 0 results,
    so the retry can still fire (the note must not read as a phantom hit)."""

    class _AllDupTool:
        name = "web_search"

        async def execute(self, *, query="", num_results=5, **kwargs):
            return ToolResult(success=True, output=(
                "All 5 matching results were already returned earlier this turn "
                "— no new pages. Answer from what you already gathered."
            ))

    engine = AnalyticalEngine(MagicMock())
    engine._tool_registry = _FakeRegistry(_AllDupTool())

    await engine._execute_auto_search(["weather"], results_per_query=5)
    assert engine._state.search_result_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
