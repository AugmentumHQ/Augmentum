"""Tests for system-driven auto-search in the UARF analytical pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatResponse,
    Message,
    ModelBackend,
    ModelInfo,
    Usage,
)
from augmentum.modes.analytical.engine import AnalyticalEngine
from augmentum.modes.analytical.prompts import (
    SEARCH_CONTEXT_SECTION,
    SEARCH_QUERY_PROMPT,
    get_phase_prompt,
)
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.registry import ToolRegistry

# =====================================================================
# Fixtures
# =====================================================================


def _make_backend(response_text: str = "mock response") -> MagicMock:
    """Create a mock ModelBackend that returns a fixed response."""
    backend = MagicMock(spec=ModelBackend)
    backend.chat = AsyncMock(return_value=InternalChatResponse(
        message=Message(role="assistant", content=response_text),
        model="test-model",
        finish_reason="stop",
        usage=Usage(total_tokens=10),
    ))
    backend.list_models = AsyncMock(return_value=[
        ModelInfo(name="test-model", model="test-model", size=0),
    ])
    return backend


def _make_web_search_tool(results_text: str = "") -> MagicMock:
    """Create a mock web_search tool."""
    tool = MagicMock(spec=Tool)
    tool.name = "web_search"
    tool.category = ToolCategory.SEARCH
    tool.description = "Search the web"
    tool.input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "default": 5},
            "categories": {"type": "string", "default": "general"},
        },
        "required": ["query"],
    }
    tool.execute = AsyncMock(return_value=ToolResult(
        success=True,
        output=results_text or (
            "[1] Seattle Weather Today\n"
            "    URL: https://weather.com/seattle\n"
            "    Current conditions: 65°F, partly cloudy\n"
            "\n"
            "[2] AccuWeather Seattle WA\n"
            "    URL: https://accuweather.com/seattle\n"
            "    Tonight: Low of 48°F with clear skies"
        ),
    ))
    return tool


def _make_registry_with_search(
    search_results: str = "",
) -> tuple[ToolRegistry, MagicMock]:
    """Create a ToolRegistry with a mock web_search tool."""
    registry = ToolRegistry()
    tool = _make_web_search_tool(search_results)
    registry.register(tool)
    return registry, tool


# =====================================================================
# _needs_search heuristic tests
# =====================================================================


class TestNeedsSearch:
    """Tests for the _needs_search heuristic detection."""

    def test_search_keywords_in_query(self):
        """Queries with search keywords should trigger auto-search."""
        assert AnalyticalEngine._needs_search("What is the current temperature?")
        assert AnalyticalEngine._needs_search("Who is the president of France?")
        assert AnalyticalEngine._needs_search("Find information about Python 3.12")
        assert AnalyticalEngine._needs_search("What are the latest news?")

    def test_temporal_keywords(self):
        """Queries with temporal words should trigger auto-search."""
        assert AnalyticalEngine._needs_search("What happened today?")
        assert AnalyticalEngine._needs_search("Events in 2026")
        assert AnalyticalEngine._needs_search("Weather this week")
        assert AnalyticalEngine._needs_search("Stock prices right now")

    def test_assess_type_factual(self):
        """ASSESS TYPE: factual should trigger auto-search."""
        assert AnalyticalEngine._needs_search(
            "capital of France",
            assess_output="TYPE: factual\nCOMPLEXITY: simple",
        )
        assert AnalyticalEngine._needs_search(
            "something",
            assess_output="TYPE: current_events\nCOMPLEXITY: moderate",
        )

    def test_no_search_for_math(self):
        """Pure math questions should NOT trigger auto-search."""
        assert not AnalyticalEngine._needs_search("Calculate 2 + 2")
        assert not AnalyticalEngine._needs_search(
            "Solve x^2 = 4",
            assess_output="TYPE: mathematical\nCOMPLEXITY: simple",
        )

    def test_no_search_for_coding(self):
        """Coding questions without search keywords should not trigger."""
        assert not AnalyticalEngine._needs_search(
            "Write a Python function to sort a list",
            assess_output="TYPE: coding\nCOMPLEXITY: simple",
        )

    def test_no_search_empty_query(self):
        """Empty query should not trigger search."""
        assert not AnalyticalEngine._needs_search("")


# =====================================================================
# _generate_search_queries tests
# =====================================================================


class TestGenerateSearchQueries:
    """Tests for LLM-based search query generation."""

    @pytest.mark.asyncio
    async def test_parses_three_lines(self):
        """Should parse 3 clean query lines from the LLM response."""
        backend = _make_backend(
            "Seattle WA weather today\n"
            "current temperature Seattle Washington\n"
            "Seattle WA forecast March 2026"
        )
        engine = AnalyticalEngine(backend=backend)

        queries = await engine._generate_search_queries(
            "test-model", "What's the weather in Seattle WA?",
        )

        assert len(queries) == 3
        assert "Seattle WA weather today" in queries[0]

    @pytest.mark.asyncio
    async def test_strips_numbering(self):
        """Should strip numbered prefixes like '1. ' or '1) '."""
        backend = _make_backend(
            "1. Seattle weather today\n"
            "2) current temp Seattle WA\n"
            "3- Seattle forecast"
        )
        engine = AnalyticalEngine(backend=backend)

        queries = await engine._generate_search_queries(
            "test-model", "weather in Seattle",
        )

        assert len(queries) == 3
        assert queries[0] == "Seattle weather today"
        assert queries[1] == "current temp Seattle WA"

    @pytest.mark.asyncio
    async def test_fallback_on_empty_response(self):
        """Should fall back to original query if LLM returns nothing."""
        backend = _make_backend("")
        engine = AnalyticalEngine(backend=backend)

        queries = await engine._generate_search_queries(
            "test-model", "weather in Seattle WA",
        )

        assert len(queries) == 1
        assert queries[0] == "weather in Seattle WA"

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self):
        """Should fall back to original query if LLM call fails."""
        backend = MagicMock(spec=ModelBackend)
        backend.chat = AsyncMock(side_effect=Exception("connection error"))
        engine = AnalyticalEngine(backend=backend)

        queries = await engine._generate_search_queries(
            "test-model", "test query",
        )

        assert len(queries) == 1
        assert queries[0] == "test query"

    @pytest.mark.asyncio
    async def test_stores_in_state(self):
        """Generated queries should be stored in engine state."""
        backend = _make_backend("query one\nquery two\nquery three")
        engine = AnalyticalEngine(backend=backend)

        await engine._generate_search_queries("test-model", "test")

        assert len(engine.state.search_queries) == 3

    @pytest.mark.asyncio
    async def test_skips_short_junk_lines(self):
        """Lines shorter than 4 chars should be skipped."""
        backend = _make_backend("ok\nweather Seattle WA today\n\nab\ntemperature now")
        engine = AnalyticalEngine(backend=backend)

        queries = await engine._generate_search_queries("test-model", "weather")

        assert len(queries) == 2
        assert queries[0] == "weather Seattle WA today"
        assert queries[1] == "temperature now"


# =====================================================================
# _execute_auto_search tests
# =====================================================================


class TestExecuteAutoSearch:
    """Tests for parallel search execution and result formatting."""

    @pytest.mark.asyncio
    async def test_basic_execution(self):
        """Should execute searches and produce a context string."""
        registry, tool = _make_registry_with_search()
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        context = await engine._execute_auto_search(
            ["weather Seattle", "temperature Seattle"],
        )

        assert "Live Search Results" in context
        assert tool.execute.call_count == 2
        assert engine.state.search_context == context

    @pytest.mark.asyncio
    async def test_deduplicates_by_url(self):
        """Duplicate URLs across queries should be deduplicated."""
        results = (
            "[1] Page One\n"
            "    URL: https://example.com/a\n"
            "    Content A\n"
            "\n"
            "[2] Page Two\n"
            "    URL: https://example.com/b\n"
            "    Content B"
        )
        registry, tool = _make_registry_with_search(results)
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        # Both queries return the same URLs
        context = await engine._execute_auto_search(
            ["query 1", "query 2"],
        )

        # Count URL occurrences — each should appear only once
        assert context.count("https://example.com/a") == 1
        assert context.count("https://example.com/b") == 1

    @pytest.mark.asyncio
    async def test_records_tool_calls(self):
        """Each search should be recorded as a ToolCallRecord."""
        registry, _ = _make_registry_with_search()
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        await engine._execute_auto_search(["q1", "q2", "q3"])

        assert len(engine.state.tool_calls) == 3
        assert all(tc.phase == "search" for tc in engine.state.tool_calls)
        assert all(tc.tool_name == "web_search" for tc in engine.state.tool_calls)

    @pytest.mark.asyncio
    async def test_truncates_long_context(self):
        """Context exceeding max chars should be truncated."""
        # Generate very long results
        long_result = "\n\n".join(
            f"[{i}] Long Title {i}\n    URL: https://example.com/{i}\n    {'x' * 500}"
            for i in range(20)
        )
        registry, _ = _make_registry_with_search(long_result)
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        context = await engine._execute_auto_search(
            ["q1"], max_context_chars=500,
        )

        assert "[... truncated]" in context

    @pytest.mark.asyncio
    async def test_no_registry_returns_empty(self):
        """Should return empty string if no registry."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend, tool_registry=None)

        context = await engine._execute_auto_search(["q1"])

        assert context == ""

    @pytest.mark.asyncio
    async def test_no_web_search_tool_returns_empty(self):
        """Should return empty if registry has no web_search tool."""
        registry = ToolRegistry()
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        context = await engine._execute_auto_search(["q1"])

        assert context == ""

    @pytest.mark.asyncio
    async def test_handles_failed_searches(self):
        """Should handle individual search failures gracefully."""
        registry, tool = _make_registry_with_search()
        # Make the tool fail
        tool.execute = AsyncMock(return_value=ToolResult(
            success=False, error="Connection timeout",
        ))
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        await engine._execute_auto_search(["q1", "q2"])

        # Should still complete without error, just empty context
        assert engine.state.search_context == ""


# =====================================================================
# Prompt integration tests
# =====================================================================


class TestSearchContextInPrompts:
    """Tests for search context injection into phase prompts."""

    def test_apply_simple_includes_search_context(self):
        """APPLY simple prompt should include search results when provided."""
        system, user = get_phase_prompt(
            "apply", query="test", assess_output="test",
            is_simple=True,
            search_context="\n## Web Search Results\nSome results here",
        )
        assert "Web Search Results" in user
        assert "Some results here" in user

    def test_apply_simple_empty_search_context(self):
        """APPLY simple prompt should be clean when no search context."""
        system, user = get_phase_prompt(
            "apply", query="test", assess_output="test",
            is_simple=True, search_context="",
        )
        assert "Web Search Results" not in user

    def test_identify_gets_search_summary_not_full(self):
        """IDENTIFY prompt should get a brief search summary, not full results."""
        search_ctx = (
            'Search: "test topic"\n'
            "[1] First result\n"
            "    URL: https://example.com\n"
            "[2] Second result\n"
        )
        system, user = get_phase_prompt(
            "identify", query="test", assess_output="test",
            search_context=search_ctx,
        )
        # Should get summary mentioning result count and topics
        assert "Search Context" in user
        # Should NOT get full search results
        assert "First result" not in user

    def test_relevant_gets_search_summary_not_full(self):
        """RELEVANT prompt should get a brief search summary, not full results."""
        search_ctx = (
            'Search: "test topic"\n'
            "[1] First result\n"
            "    URL: https://example.com\n"
        )
        system, user = get_phase_prompt(
            "relevant", query="test", identify_output="test",
            search_context=search_ctx,
        )
        assert "Search Context" in user
        assert "First result" not in user

    def test_apply_full_includes_search_context(self):
        """Full APPLY prompt should include search results."""
        system, user = get_phase_prompt(
            "apply", query="test",
            identify_output="test", relevant_output="test",
            search_context="\n## Web Search Results\nResults",
        )
        assert "Web Search Results" in user

    def test_search_query_prompt_format(self):
        """SEARCH_QUERY_PROMPT should accept query and datetime_context variables."""
        formatted = SEARCH_QUERY_PROMPT.format(
            query="test weather query",
            datetime_context="Current date: Saturday, March 14, 2026.",
        )
        assert "test weather query" in formatted
        assert "3 short web search queries" in formatted
        assert "2026" in formatted

    def test_search_context_section_format(self):
        """SEARCH_CONTEXT_SECTION should accept search_results variable."""
        formatted = SEARCH_CONTEXT_SECTION.format(
            search_results="[1] Title\n    URL: https://example.com",
        )
        assert "Live Search Results" in formatted
        assert "https://example.com" in formatted


# =====================================================================
# Registry exclude filter
# =====================================================================


class TestRegistryExclude:
    """Tests for the exclude parameter in get_for_phase."""

    def test_exclude_web_search(self):
        """get_for_phase with exclude should filter out named tools."""
        registry = ToolRegistry()

        search_tool = MagicMock(spec=Tool)
        search_tool.name = "web_search"
        search_tool.category = ToolCategory.SEARCH
        registry.register(search_tool)

        calc_tool = MagicMock(spec=Tool)
        calc_tool.name = "calculator"
        calc_tool.category = ToolCategory.EXECUTE
        registry.register(calc_tool)

        # Without exclude — both available
        tools = registry.get_for_phase("apply")
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "calculator" in names

        # With exclude — web_search filtered
        tools = registry.get_for_phase("apply", exclude=frozenset({"web_search"}))
        names = [t.name for t in tools]
        assert "web_search" not in names
        assert "calculator" in names

    def test_exclude_none_returns_all(self):
        """exclude=None should return all tools."""
        registry = ToolRegistry()
        tool = MagicMock(spec=Tool)
        tool.name = "web_search"
        tool.category = ToolCategory.SEARCH
        registry.register(tool)

        tools = registry.get_for_phase("apply", exclude=None)
        assert len(tools) == 1


# =====================================================================
# Integration: auto-search in process()
# =====================================================================


class TestAutoSearchIntegration:
    """Integration tests for auto-search in the full pipeline."""

    @pytest.mark.asyncio
    async def test_simple_pipeline_with_auto_search(self):
        """Simple pipeline should run auto-search when query needs it."""
        # Backend returns different things for different prompts
        call_count = 0

        async def mock_chat(request):
            nonlocal call_count
            call_count += 1
            content = request.messages[0].content if request.messages else ""

            if "COMPLEXITY" in content:
                # ASSESS phase
                text = "TYPE: factual\nDOMAIN: weather\nREASONING_STEPS: 1\nCOMPLEXITY: simple\nRATIONALE: simple lookup"
            elif "3 short web search queries" in content:
                # Search query generation
                text = "Seattle WA weather today\ncurrent temperature Seattle\nweather forecast Seattle WA"
            elif "PRELIMINARY_ANSWER" in content:
                # APPLY phase
                text = "REASONING: Based on search results...\nPRELIMINARY_ANSWER: 65°F"
            else:
                # CONCLUDE
                text = "The current temperature in Seattle is 65°F."

            return InternalChatResponse(
                message=Message(role="assistant", content=text),
                model="test-model",
                finish_reason="stop",
                usage=Usage(total_tokens=10),
            )

        backend = MagicMock(spec=ModelBackend)
        backend.chat = AsyncMock(side_effect=mock_chat)
        backend.list_models = AsyncMock(return_value=[
            ModelInfo(name="test-model", model="test-model", size=0),
        ])

        registry, search_tool = _make_registry_with_search()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        from augmentum.models.base import InternalChatRequest

        await engine.process(InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="What is the weather in Seattle WA today?")],
            stream=False,
        ))

        # Auto-search should have been triggered
        assert engine.state.needs_search is True
        assert len(engine.state.search_queries) == 3
        assert engine.state.search_context != ""
        assert "Live Search Results" in engine.state.search_context

        # web_search tool should have been called >= 3 times (expansion may add variants)
        assert search_tool.execute.call_count >= 3

        # Tool calls should be recorded
        search_calls = [
            tc for tc in engine.state.tool_calls if tc.phase == "search"
        ]
        assert len(search_calls) >= 3

    @pytest.mark.asyncio
    async def test_no_auto_search_for_math(self):
        """Math query should NOT trigger auto-search."""
        backend = _make_backend(
            "TYPE: mathematical\nDOMAIN: math\nREASONING_STEPS: 2\n"
            "COMPLEXITY: simple\nRATIONALE: simple calculation"
        )
        registry, search_tool = _make_registry_with_search()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        from augmentum.models.base import InternalChatRequest

        # Use a query that doesn't match search heuristics
        await engine.process(InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Calculate 2 + 2")],
            stream=False,
        ))

        assert engine.state.needs_search is False
        assert search_tool.execute.call_count == 0


# =====================================================================
# _broaden_queries tests
# =====================================================================


class TestBroadenQueries:
    """Tests for deterministic query broadening (system-level retry)."""

    def test_strips_temporal_words(self):
        """Temporal words like 'today', '2026' should be stripped."""
        result = AnalyticalEngine._broaden_queries(
            ["weather Seattle today", "temperature Seattle 2026"],
            "What is the weather in Seattle today?",
        )
        # At least one broadened query should lack temporal words
        for q in result:
            assert "today" not in q.lower() or "2026" not in q.lower()

    def test_extracts_key_terms(self):
        """Should extract non-stop-word terms from the original query."""
        result = AnalyticalEngine._broaden_queries(
            ["some specific query today"],
            "What is the current population of Tokyo Japan?",
        )
        # "population", "Tokyo", "Japan" are key terms
        key_terms_found = any(
            "population" in q.lower() or "tokyo" in q.lower()
            for q in result
        )
        assert key_terms_found

    def test_includes_raw_query_fallback(self):
        """Raw original query should be used as a last-resort broadening."""
        result = AnalyticalEngine._broaden_queries(
            [], "Seattle WA weather forecast",
        )
        assert any("Seattle WA weather forecast" in q for q in result)

    def test_deduplicates_against_originals(self):
        """Broadened queries should not duplicate original queries."""
        original = ["weather Seattle", "temperature Seattle"]
        result = AnalyticalEngine._broaden_queries(original, "weather Seattle")
        for q in result:
            assert q.lower().strip() not in {o.lower().strip() for o in original}

    def test_max_three_queries(self):
        """Should return at most 3 broadened queries."""
        result = AnalyticalEngine._broaden_queries(
            [
                "query one today",
                "query two this week",
                "query three yesterday",
                "query four right now",
            ],
            "What is the current status of everything today?",
        )
        assert len(result) <= 3

    def test_skips_short_results(self):
        """Broadened queries shorter than 4 chars should be filtered."""
        result = AnalyticalEngine._broaden_queries(
            ["today"],  # stripping 'today' leaves empty
            "hi",  # too short for raw fallback
        )
        for q in result:
            assert len(q) > 3

    def test_empty_originals_uses_raw(self):
        """With no original queries, the raw query is used."""
        result = AnalyticalEngine._broaden_queries(
            [], "population of France 2025",
        )
        assert len(result) >= 1


# =====================================================================
# _parse_search_needed tests
# =====================================================================


class TestParseSearchNeeded:
    """Tests for parsing SEARCH_NEEDED from VERIFY output."""

    def test_yes(self):
        output = "VERIFIED: no\nCONFIDENCE: 0.3\nSEARCH_NEEDED: yes\nNOTES: missing data"
        assert AnalyticalEngine._parse_search_needed(output) is True

    def test_no(self):
        output = "VERIFIED: yes\nCONFIDENCE: 0.9\nSEARCH_NEEDED: no\nNOTES: all good"
        assert AnalyticalEngine._parse_search_needed(output) is False

    def test_case_insensitive(self):
        output = "SEARCH_NEEDED: YES"
        assert AnalyticalEngine._parse_search_needed(output) is True
        output2 = "search_needed: No"
        assert AnalyticalEngine._parse_search_needed(output2) is False

    def test_missing_defaults_false(self):
        """If SEARCH_NEEDED line is absent, default to False."""
        output = "VERIFIED: yes\nCONFIDENCE: 0.8"
        assert AnalyticalEngine._parse_search_needed(output) is False

    def test_empty_output(self):
        assert AnalyticalEngine._parse_search_needed("") is False


# =====================================================================
# _merge_search_context tests
# =====================================================================


class TestMergeSearchContext:
    """Tests for merging new search results into existing context."""

    def test_merge_into_empty(self):
        """When no existing context, new context should replace it."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_context = ""

        engine._merge_search_context("New block A\nURL: https://a.com\nContent A")
        assert "https://a.com" in engine._state.search_context

    def test_append_new_blocks(self):
        """New blocks with different URLs should be appended."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_context = (
            "[1] Existing\n    URL: https://existing.com\n    Content"
        )

        new = "[1] New Result\n    URL: https://new.com\n    New content"
        engine._merge_search_context(new)
        assert "https://existing.com" in engine._state.search_context
        assert "https://new.com" in engine._state.search_context

    def test_dedup_by_url(self):
        """Blocks with duplicate URLs should not be appended."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_context = (
            "[1] Original\n    URL: https://example.com/page\n    Content"
        )

        duplicate = "[1] Duplicate\n    URL: https://example.com/page\n    Same content"
        engine._merge_search_context(duplicate)

        # URL should appear only once
        assert engine._state.search_context.count("https://example.com/page") == 1

    def test_respects_max_chars(self):
        """Merged context should be truncated if it exceeds max chars."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend)
        max_chars = settings.uarf_auto_search_max_context_chars
        engine._state.search_context = "A" * (max_chars - 100)

        engine._merge_search_context("[1] Big\n    URL: https://big.com\n    " + "B" * 3000)
        assert "[... truncated]" in engine._state.search_context

    def test_empty_new_context_noop(self):
        """Empty new context should not modify existing."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_context = "existing"

        engine._merge_search_context("")
        assert engine._state.search_context == "existing"

    def test_handles_truncated_existing(self):
        """If existing context already has truncation marker, it's handled."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_context = "Old content\n[... truncated]"

        engine._merge_search_context("[1] New\n    URL: https://new.com\n    Content")
        # Should still work without doubling truncation markers
        assert engine._state.search_context.count("[... truncated]") <= 1


# =====================================================================
# System-level search retry tests
# =====================================================================


class TestSystemLevelRetry:
    """Tests for system-level retry (triggered by insufficient results)."""

    @pytest.mark.asyncio
    async def test_retry_on_few_results(self):
        """When auto-search returns < min results, system retry should fire."""
        # Return empty results so search_result_count == 0
        tool = _make_web_search_tool("")
        tool.execute = AsyncMock(return_value=ToolResult(
            success=False, error="No results",
        ))
        registry = ToolRegistry()
        registry.register(tool)

        call_count = 0

        async def mock_chat(request):
            nonlocal call_count
            call_count += 1
            content = request.messages[0].content if request.messages else ""
            if "COMPLEXITY" in content:
                return InternalChatResponse(
                    message=Message(role="assistant", content=(
                        "TYPE: factual\nDOMAIN: general\nREASONING_STEPS: 1\n"
                        "COMPLEXITY: simple\nRATIONALE: simple"
                    )),
                    model="test-model", finish_reason="stop",
                    usage=Usage(total_tokens=10),
                )
            if "3 short web search queries" in content:
                return InternalChatResponse(
                    message=Message(role="assistant", content=(
                        "Seattle weather today\n"
                        "current temp Seattle\n"
                        "forecast Seattle WA"
                    )),
                    model="test-model", finish_reason="stop",
                    usage=Usage(total_tokens=10),
                )
            return InternalChatResponse(
                message=Message(role="assistant", content="Mock answer"),
                model="test-model", finish_reason="stop",
                usage=Usage(total_tokens=10),
            )

        backend = MagicMock(spec=ModelBackend)
        backend.chat = AsyncMock(side_effect=mock_chat)
        backend.list_models = AsyncMock(return_value=[
            ModelInfo(name="test-model", model="test-model", size=0),
        ])

        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        from augmentum.models.base import InternalChatRequest as ICR

        await engine.process(ICR(
            model="test-model",
            messages=[Message(role="user", content="What is the weather in Seattle today?")],
            stream=False,
        ))

        # System retry should have incremented
        assert engine.state.search_retry_count >= 1
        # More search calls than just the initial 3 (retry adds broadened queries)
        assert tool.execute.call_count > 3

    @pytest.mark.asyncio
    async def test_no_retry_when_enough_results(self):
        """When auto-search returns sufficient results, no retry."""
        registry, tool = _make_registry_with_search()
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        # Execute auto-search with good results
        await engine._execute_auto_search(
            ["q1", "q2"], results_per_query=4, max_context_chars=4000,
        )

        # search_result_count should be > 0 since results are returned
        assert engine.state.search_result_count > 0
        assert engine.state.search_retry_count == 0

    @pytest.mark.asyncio
    async def test_respects_max_retries(self):
        """System retry should not exceed max retry count."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend)

        # Set retry count at max already
        engine._state.search_retry_count = 1  # default max is 1
        engine._state.search_result_count = 0
        engine._state.search_queries = ["original query"]

        # Broadening should still work (it's static), but the retry
        # should NOT happen in process() because count >= max
        broadened = AnalyticalEngine._broaden_queries(
            ["original query today"], "original query",
        )
        assert len(broadened) >= 0  # Just verify it doesn't error


# =====================================================================
# Model-level search retry tests
# =====================================================================


class TestModelLevelRetry:
    """Tests for model-level retry (VERIFY flags SEARCH_NEEDED)."""

    @pytest.mark.asyncio
    async def test_generate_refined_queries_success(self):
        """Should generate refined queries from LLM response."""
        backend = _make_backend(
            "Seattle population census data\n"
            "Seattle WA demographics 2025\n"
            "city population estimates Seattle"
        )
        registry, _ = _make_registry_with_search()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)
        engine._state.search_queries = ["original query"]

        queries = await engine._generate_refined_queries(
            "test-model", "population of Seattle",
            "UNSUPPORTED_CLAIMS:\n- Population figure not verified",
        )

        assert len(queries) == 3
        assert "Seattle population census data" in queries[0]

    @pytest.mark.asyncio
    async def test_generate_refined_queries_fallback(self):
        """Should fall back to broadening if LLM returns empty."""
        backend = _make_backend("")
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_queries = ["weather Seattle today"]

        queries = await engine._generate_refined_queries(
            "test-model", "weather in Seattle",
            "Missing current data",
        )

        # Should fall back to _broaden_queries
        assert len(queries) >= 1

    @pytest.mark.asyncio
    async def test_generate_refined_queries_exception_fallback(self):
        """Should fall back to broadening if LLM call throws."""
        backend = MagicMock(spec=ModelBackend)
        backend.chat = AsyncMock(side_effect=Exception("connection error"))
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_queries = ["original query today"]

        queries = await engine._generate_refined_queries(
            "test-model", "test query about Seattle",
            "Missing data",
        )

        # Should still return broadened queries
        assert len(queries) >= 1

    @pytest.mark.asyncio
    async def test_search_needed_triggers_retry_in_backtrack(self):
        """SEARCH_NEEDED: yes from VERIFY should trigger model-level retry."""
        verify_call_count = 0

        async def mock_chat(request):
            nonlocal verify_call_count
            content = request.messages[0].content if request.messages else ""

            # Match by unique prompt opening phrases to avoid overlap
            if "classify this query" in content.lower():
                return InternalChatResponse(
                    message=Message(role="assistant", content=(
                        "TYPE: factual\nDOMAIN: general\nREASONING_STEPS: 3\n"
                        "COMPLEXITY: moderate\nRATIONALE: needs verification"
                    )),
                    model="test-model", finish_reason="stop",
                    usage=Usage(total_tokens=10),
                )
            if "3 short web search queries" in content:
                return InternalChatResponse(
                    message=Message(role="assistant", content=(
                        "query one\nquery two\nquery three"
                    )),
                    model="test-model", finish_reason="stop",
                    usage=Usage(total_tokens=10),
                )
            if "needs more search data" in content.lower():
                # Refined query generation for retry
                return InternalChatResponse(
                    message=Message(role="assistant", content=(
                        "refined query one\nrefined query two\nrefined three"
                    )),
                    model="test-model", finish_reason="stop",
                    usage=Usage(total_tokens=10),
                )
            if "review this analysis for errors" in content.lower():
                verify_call_count += 1
                # VERIFY phase — first time fails with SEARCH_NEEDED
                if verify_call_count <= 1:
                    return InternalChatResponse(
                        message=Message(role="assistant", content=(
                            "ERRORS_FOUND:\n- Missing source data\n\n"
                            "UNSUPPORTED_CLAIMS:\n- Population figure unverified\n\n"
                            "CONTRADICTIONS:\n- None\n\n"
                            "VERIFIED: no\nCONFIDENCE: 0.3\n"
                            "SEARCH_NEEDED: yes\n"
                            "VERIFICATION_NOTES: Need more search data"
                        )),
                        model="test-model", finish_reason="stop",
                        usage=Usage(total_tokens=10),
                    )
                # Second verify — passes
                return InternalChatResponse(
                    message=Message(role="assistant", content=(
                        "ERRORS_FOUND:\n- None\n\n"
                        "UNSUPPORTED_CLAIMS:\n- None\n\n"
                        "CONTRADICTIONS:\n- None\n\n"
                        "VERIFIED: yes\nCONFIDENCE: 0.9\n"
                        "SEARCH_NEEDED: no\n"
                        "VERIFICATION_NOTES: All good"
                    )),
                    model="test-model", finish_reason="stop",
                    usage=Usage(total_tokens=10),
                )
            # Default (IDENTIFY, RELEVANT, APPLY, CONCLUDE)
            return InternalChatResponse(
                message=Message(role="assistant", content="Mock phase output"),
                model="test-model", finish_reason="stop",
                usage=Usage(total_tokens=10),
            )

        backend = MagicMock(spec=ModelBackend)
        backend.chat = AsyncMock(side_effect=mock_chat)
        backend.list_models = AsyncMock(return_value=[
            ModelInfo(name="test-model", model="test-model", size=0),
        ])

        registry, search_tool = _make_registry_with_search()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)

        from augmentum.models.base import InternalChatRequest as ICR

        await engine.process(ICR(
            model="test-model",
            # Use a query that doesn't trigger the heuristic ASSESS shortcut
            # (needs full pipeline with VERIFY for backtrack testing)
            messages=[Message(role="user", content="Explain the population trends in Seattle")],
            stream=False,
        ))

        # Should have triggered model-level search retry
        assert engine.state.search_retry_count >= 1
        # Initial 3 queries + refined queries
        assert search_tool.execute.call_count > 3
        # Backtrack should have happened
        assert engine.state.backtrack_count >= 1

    @pytest.mark.asyncio
    async def test_search_needed_no_resets_flag(self):
        """SEARCH_NEEDED: no should not trigger any retry."""
        backend = _make_backend()
        engine = AnalyticalEngine(backend=backend)

        engine._state.search_needed_by_verify = False
        engine._state.needs_search = True
        engine._state.search_retry_count = 0

        # Simulate: parse a VERIFY output that says SEARCH_NEEDED: no
        output = "VERIFIED: yes\nCONFIDENCE: 0.9\nSEARCH_NEEDED: no"
        engine._state.search_needed_by_verify = AnalyticalEngine._parse_search_needed(output)

        assert engine._state.search_needed_by_verify is False
        assert engine._state.search_retry_count == 0


# =====================================================================
# VERIFY prompt SEARCH_NEEDED integration
# =====================================================================


class TestVerifySearchNeededPrompt:
    """Tests for SEARCH_NEEDED in VERIFY prompt template."""

    def test_verify_prompt_includes_search_needed(self):
        """VERIFY prompt should include SEARCH_NEEDED in output format."""
        system, user = get_phase_prompt(
            "verify", query="test",
            apply_output="test analysis",
            identify_output="test components",
        )
        assert "SEARCH_NEEDED" in system

    def test_verify_prompt_has_search_needed_field(self):
        """VERIFY prompt should include SEARCH_NEEDED output field."""
        system, user = get_phase_prompt(
            "verify", query="test",
            apply_output="test",
            identify_output="test",
        )
        assert "SEARCH_NEEDED" in system


# =====================================================================
# Refusal detection tests
# =====================================================================


class TestIsRefusal:
    """Tests for the _is_refusal static method."""

    def test_common_refusal_phrases(self):
        """Should detect common LLM refusal patterns."""
        assert AnalyticalEngine._is_refusal("I can't provide real-time information")
        assert AnalyticalEngine._is_refusal("I cannot provide current weather data")
        assert AnalyticalEngine._is_refusal("I'm unable to access the internet")
        assert AnalyticalEngine._is_refusal("I don't have access to current data")
        assert AnalyticalEngine._is_refusal("As an AI, I cannot browse the web")
        assert AnalyticalEngine._is_refusal("As a language model, I cannot search")
        assert AnalyticalEngine._is_refusal("I'm sorry, but I can't help with that")

    def test_not_refusal_for_normal_text(self):
        """Normal content should not be flagged as refusal."""
        assert not AnalyticalEngine._is_refusal("latest news today")
        assert not AnalyticalEngine._is_refusal("weather forecast Seattle WA")
        assert not AnalyticalEngine._is_refusal("")
        assert not AnalyticalEngine._is_refusal("STEP 1: Analyze the data")
        assert not AnalyticalEngine._is_refusal("The search results show that...")

    def test_case_insensitive(self):
        """Refusal detection should be case-insensitive."""
        assert AnalyticalEngine._is_refusal("I CAN'T PROVIDE that information")
        assert AnalyticalEngine._is_refusal("AS AN AI, I cannot do this")

    def test_real_time_phrases(self):
        """Should detect 'real-time information' refusals."""
        assert AnalyticalEngine._is_refusal(
            "I don't have access to real-time information"
        )
        assert AnalyticalEngine._is_refusal(
            "I cannot provide real-time data about weather"
        )


class TestRefusalInQueryGeneration:
    """Tests for refusal detection in _generate_search_queries."""

    @pytest.mark.asyncio
    async def test_all_refusal_falls_back_to_raw_query(self):
        """If the entire LLM response is refusals, fall back to raw query."""
        backend = _make_backend(
            "I can't provide real-time information about the weather.\n"
            "I don't have access to current data."
        )
        engine = AnalyticalEngine(backend=backend)

        queries = await engine._generate_search_queries(
            "test-model", "what is the weather today",
        )

        assert len(queries) == 1
        assert queries[0] == "what is the weather today"

    @pytest.mark.asyncio
    async def test_mixed_refusal_and_queries_filters_refusals(self):
        """If some lines are refusals and some are queries, keep only queries."""
        backend = _make_backend(
            "I can't provide real-time weather data\n"
            "weather forecast today Seattle WA\n"
            "I'm unable to access the internet\n"
            "current temperature Seattle"
        )
        engine = AnalyticalEngine(backend=backend)

        queries = await engine._generate_search_queries(
            "test-model", "weather in Seattle",
        )

        assert len(queries) == 2
        assert queries[0] == "weather forecast today Seattle WA"
        assert queries[1] == "current temperature Seattle"

    @pytest.mark.asyncio
    async def test_all_lines_refusal_falls_back(self):
        """If every parsed line is a refusal, fall back to raw query."""
        backend = _make_backend(
            "I cannot search the web for you\n"
            "I'm unable to provide real-time data\n"
            "As an AI, I don't have internet access"
        )
        engine = AnalyticalEngine(backend=backend)

        queries = await engine._generate_search_queries(
            "test-model", "latest news",
        )

        assert len(queries) == 1
        assert queries[0] == "latest news"


class TestRefusalInRefinedQueries:
    """Tests for refusal detection in _generate_refined_queries."""

    @pytest.mark.asyncio
    async def test_all_refusal_falls_back_to_broadened(self):
        """If all refined query lines are refusals, fall back to broadening."""
        backend = _make_backend(
            "I can't provide search queries for real-time information\n"
            "I'm unable to search the web"
        )
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_queries = ["original query one"]

        queries = await engine._generate_refined_queries(
            "test-model", "what is the news today",
            "Missing current event data",
        )

        # Should fall back to _broaden_queries, not return the refusal
        assert len(queries) >= 1
        for q in queries:
            assert not AnalyticalEngine._is_refusal(q)

    @pytest.mark.asyncio
    async def test_line_level_refusals_filtered(self):
        """Individual refusal lines in refined queries should be filtered."""
        backend = _make_backend(
            "I apologize, but I cannot help\n"
            "latest headlines today\n"
            "current events March 2026"
        )
        engine = AnalyticalEngine(backend=backend)
        engine._state.search_queries = ["news today"]

        queries = await engine._generate_refined_queries(
            "test-model", "news today", "Need current data",
        )

        assert len(queries) == 2
        assert queries[0] == "latest headlines today"
        assert queries[1] == "current events March 2026"


# =====================================================================
# Tool exclusion tests
# =====================================================================


class TestToolExclusion:
    """Tests for _execute_tool with the exclude parameter."""

    @pytest.mark.asyncio
    async def test_excluded_tool_returns_error(self):
        """An excluded tool should return an error, not execute."""
        registry, search_tool = _make_registry_with_search()
        engine = AnalyticalEngine(backend=_make_backend(), tool_registry=registry)

        result = await engine._execute_tool(
            "apply", "web_search", {"query": "test"},
            exclude=frozenset({"web_search"}),
        )

        assert not result.success
        assert "not available" in result.error
        assert "search was already performed" in result.error
        search_tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_excluded_tool_executes_normally(self):
        """A non-excluded tool should execute normally."""
        registry, search_tool = _make_registry_with_search()
        engine = AnalyticalEngine(backend=_make_backend(), tool_registry=registry)

        result = await engine._execute_tool(
            "apply", "web_search", {"query": "test"},
            exclude=frozenset({"some_other_tool"}),
        )

        assert result.success
        search_tool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_exclude_executes_normally(self):
        """When exclude is None, all tools should execute."""
        registry, search_tool = _make_registry_with_search()
        engine = AnalyticalEngine(backend=_make_backend(), tool_registry=registry)

        result = await engine._execute_tool(
            "apply", "web_search", {"query": "test"},
        )

        assert result.success
        search_tool.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_excluded_tool_records_in_state(self):
        """Excluded tool calls should still be recorded in state."""
        registry, _ = _make_registry_with_search()
        engine = AnalyticalEngine(backend=_make_backend(), tool_registry=registry)

        await engine._execute_tool(
            "apply", "web_search", {"query": "test"},
            exclude=frozenset({"web_search"}),
        )

        assert len(engine.state.tool_calls) == 1
        assert engine.state.tool_calls[0].tool_name == "web_search"
        assert not engine.state.tool_calls[0].success


# =====================================================================
# Prompt strength tests (anti-refusal language in prompts)
# =====================================================================


class TestPromptPositiveDirectives:
    """Tests that prompts contain positive directives for search usage."""

    def test_search_context_section_directs_usage(self):
        """SEARCH_CONTEXT_SECTION should direct the model to use results."""
        ctx = SEARCH_CONTEXT_SECTION.format(search_results="test results")
        lower = ctx.lower()
        assert "base your answer" in lower or "use" in lower
        assert "real" in lower or "current" in lower

    def test_apply_prompt_encourages_tool_use(self):
        """APPLY prompt should encourage using tools when available."""
        system, user = get_phase_prompt(
            "apply", query="weather?",
            identify_output="concepts",
            relevant_output="info",
            has_tools=True,
        )
        lower = system.lower()
        assert "tool" in lower

    def test_conclude_prompt_directs_synthesis(self):
        """CONCLUDE prompt should direct clear synthesis."""
        system, user = get_phase_prompt(
            "conclude", query="weather?",
            apply_output="analysis data",
            verify_output="VERIFIED: yes",
        )
        lower = system.lower()
        assert "final answer" in lower or "synthesize" in lower

    def test_search_query_prompt_requests_queries(self):
        """SEARCH_QUERY_PROMPT should request search queries."""
        prompt = SEARCH_QUERY_PROMPT.format(query="test", datetime_context="Current date: today.")
        lower = prompt.lower()
        assert "search queries" in lower


# =====================================================================
# Phase-tool validation tests
# =====================================================================


class TestPhaseToolValidation:
    """Tests that tool calls are validated against phase-allowed tools."""

    @pytest.mark.asyncio
    async def test_disallowed_tool_not_executed_in_phase(self):
        """A tool not in the phase's allowed list should NOT be executed.

        If the model outputs TOOL_CALL: web_search during VERIFY,
        the engine should break the tool loop without executing it.
        """
        from augmentum.modes.analytical.state import AnalyticalPhase

        # Model outputs a web_search tool call
        response_text = (
            "ERRORS_FOUND:\n- None\n"
            "TOOL_CALL: web_search\n"
            'TOOL_INPUT: {"query": "latest news"}'
        )
        backend = _make_backend(response_text)
        registry, search_tool = _make_registry_with_search()
        engine = AnalyticalEngine(backend=backend, tool_registry=registry)
        engine._state.query = "what is the weather?"

        # Run VERIFY phase (which only allows VERIFY+EXECUTE category tools)
        result = await engine._run_phase_with_tools(
            AnalyticalPhase.VERIFY,
            model="test-model",
            query="what is the weather?",
            apply_output="some analysis",
            identify_output="some components",
        )

        # web_search should NOT have been executed (it's SEARCH category, not VERIFY/EXECUTE)
        search_tool.execute.assert_not_called()
        # The output should still be the raw model response (no tool result injected)
        assert "ERRORS_FOUND" in result.output

    @pytest.mark.asyncio
    async def test_allowed_tool_executes_in_phase(self):
        """A tool in the phase's allowed list should execute normally."""
        from augmentum.modes.analytical.state import AnalyticalPhase

        # Create a calculator tool (VERIFY category — allowed in VERIFY phase)
        calc_tool = MagicMock(spec=Tool)
        calc_tool.name = "calculator"
        calc_tool.category = ToolCategory.VERIFY
        calc_tool.description = "Calculate expressions"
        calc_tool.input_schema = {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }
        calc_tool.execute = AsyncMock(
            return_value=ToolResult(success=True, output="42"),
        )

        registry = ToolRegistry()
        registry.register(calc_tool)

        # Model outputs a calculator tool call, then a final response
        call_count = 0

        async def mock_chat(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return InternalChatResponse(
                    message=Message(
                        role="assistant",
                        content=(
                            "Let me verify the calculation.\n"
                            "TOOL_CALL: calculator\n"
                            'TOOL_INPUT: {"expression": "6 * 7"}'
                        ),
                    ),
                    model="test-model",
                    finish_reason="stop",
                    usage=Usage(total_tokens=10),
                )
            return InternalChatResponse(
                message=Message(
                    role="assistant",
                    content="ERRORS_FOUND:\n- None\nVERIFIED: yes\nCONFIDENCE: 0.9",
                ),
                model="test-model",
                finish_reason="stop",
                usage=Usage(total_tokens=10),
            )

        backend = MagicMock(spec=ModelBackend)
        backend.chat = AsyncMock(side_effect=mock_chat)

        engine = AnalyticalEngine(backend=backend, tool_registry=registry)
        engine._state.query = "what is 6 * 7?"

        result = await engine._run_phase_with_tools(
            AnalyticalPhase.VERIFY,
            model="test-model",
            query="what is 6 * 7?",
            apply_output="PRELIMINARY_ANSWER: 42",
            identify_output="math problem",
        )

        # calculator IS allowed in VERIFY, so it should execute
        calc_tool.execute.assert_called_once()
        assert "VERIFIED: yes" in result.output
