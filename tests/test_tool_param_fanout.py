"""Regression tests for LLM-supplied tool parameters that don't match the schema.

Origin (2026-07-26): Gemma 4 12B, using interleaved thinking, batched two
web searches into one call as ``query=["a", "b"]``. ``query`` is declared
``"type": "string"``, so ``execute`` hit ``query.strip()`` and raised
``AttributeError: 'list' object has no attribute 'strip'``.

The crash was the small half of the bug. The large half: a failed tool
was scored ``empty`` — the same bucket as "the search ran and found
nothing" — so the synthesis layer told the model the tools "returned no
useful results" and to answer from its own knowledge. The model then
walked back source-backed claims it had already cited. Tests here lock
BOTH halves: the call must succeed via fan-out, and a genuine failure
must be distinguishable from an empty result.
"""

from __future__ import annotations

import pytest

from augmentum.modes.passthrough.handler import (
    _assess_result_quality,
    _max_iterations,
    _quality_synthesis_addendum,
)
from augmentum.tools.base import Tool, ToolCategory, ToolResult, invoke_tool
from augmentum.tools.params import MAX_FANOUT


class _FakeSearch(Tool):
    name = "web_search"
    description = "fake"
    category = ToolCategory.SEARCH

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer"},
            },
            "required": ["query"],
        }

    async def execute(self, *, query: str, num_results: int = 5) -> ToolResult:
        # .strip() is the exact call that used to explode.
        self.calls.append((query.strip(), num_results))
        return ToolResult(success=True, output=f"[1] result for {query}")


class TestListForStringParam:
    @pytest.mark.asyncio
    async def test_multi_query_fans_out_instead_of_crashing(self):
        """The original failure: two queries in one call."""
        tool = _FakeSearch()
        res = await tool.invoke(
            {"query": ["mars sample return 2026", "esa budget 2026"]},
        )
        assert res.success
        assert len(tool.calls) == 2
        # Neither query is dropped or silently merged.
        assert tool.calls[0][0] == "mars sample return 2026"
        assert tool.calls[1][0] == "esa budget 2026"
        assert "mars sample return 2026" in res.output
        assert "esa budget 2026" in res.output

    @pytest.mark.asyncio
    async def test_single_element_list_unwraps_without_fanout(self):
        tool = _FakeSearch()
        res = await tool.invoke({"query": ["just one"]})
        assert res.success
        assert tool.calls == [("just one", 5)]

    @pytest.mark.asyncio
    async def test_accepts_up_to_the_limit(self):
        tool = _FakeSearch()
        res = await tool.invoke({"query": [f"q{i}" for i in range(MAX_FANOUT)]})
        assert res.success
        assert len(tool.calls) == MAX_FANOUT

    @pytest.mark.asyncio
    async def test_over_limit_runs_nothing_and_says_why(self):
        """Past the limit we refuse rather than run a subset.

        Running the first N of many would produce a confident answer built
        on queries the model never agreed to drop — an invisible failure.
        """
        tool = _FakeSearch()
        res = await tool.invoke({"query": [f"q{i}" for i in range(MAX_FANOUT + 5)]})
        assert not res.success
        assert res.failure_kind == "invalid_input"
        assert tool.calls == []
        assert str(MAX_FANOUT) in res.error

    @pytest.mark.asyncio
    async def test_scalar_coercion_still_applies(self):
        tool = _FakeSearch()
        await tool.invoke({"query": "x", "num_results": "7"})
        assert tool.calls[0][1] == 7


class TestFailureIsTyped:
    @pytest.mark.asyncio
    async def test_raising_tool_returns_typed_result_not_exception(self):
        class Boom(_FakeSearch):
            async def execute(self, *, query: str, num_results: int = 5):
                raise RuntimeError("searxng unreachable")

        res = await Boom().invoke({"query": "x"})
        assert not res.success
        assert res.failure_kind == "internal_error"

    @pytest.mark.asyncio
    async def test_duck_typed_tool_without_invoke_still_runs(self):
        """MCP-bridged tools and adapters aren't always Tool subclasses.

        A hardening pass must not introduce the very AttributeError class
        it was written to remove.
        """
        class DuckTool:
            name = "duck"

            async def execute(self, **kwargs):
                return ToolResult(success=True, output=f"ran {kwargs.get('q')}")

        res = await invoke_tool(DuckTool(), {"q": "hello"})
        assert res.success and "hello" in res.output


class TestCrashIsNotAnEmptyResult:
    def test_crash_and_empty_score_differently(self):
        assert _assess_result_quality("web_search", "", False, "internal_error") == "broken"
        # No failure_kind (an upstream "0 results" style failure) stays 'empty'.
        assert _assess_result_quality("web_search", "", False, "") == "empty"

    def test_broken_guidance_does_not_tell_model_to_doubt_sources(self):
        """The regression that made the model retract cited material."""
        empty = _quality_synthesis_addendum({"web_search": "empty"})
        assert "your own knowledge" in empty

        broken = _quality_synthesis_addendum({"web_search": "broken"})
        assert "internal error" in broken.lower()
        # Must NOT recycle the "answer from your own knowledge" framing,
        # and must explicitly protect already-sourced claims.
        assert "do not revise" in broken.lower()
        assert "retry" in broken.lower()

    def test_broken_takes_precedence_over_empty(self):
        mixed = _quality_synthesis_addendum(
            {"web_search": "broken", "wikipedia": "empty"},
        )
        assert "internal error" in mixed.lower()


class TestIterationBudget:
    def test_user_limit_overrides_install_default(self):
        assert _max_iterations(object(), 40) == 40

    def test_zero_means_unlimited_but_bounded(self):
        from augmentum.modes.passthrough.handler import _ITERATION_CEILING

        assert _max_iterations(object(), 0) == _ITERATION_CEILING
        assert _max_iterations(object(), 9999) == _ITERATION_CEILING

    def test_request_override_wins_over_user_preference(self):
        class Req:
            tool_max_iterations = 7

        assert _max_iterations(Req(), 40) == 7

    def test_unset_falls_through_to_install_default(self):
        from augmentum.config import settings

        assert _max_iterations(object(), None) == (
            settings.passthrough_tool_max_iterations
        )
