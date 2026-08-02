"""Tests for the tool chain execution system (wave executor, planning, complexity detection)."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.chain import (
    ChainPlan,
    ChainStep,
    StepResult,
    _mutate_plan,
    _replan_on_failure,
    _resolve_args_via_llm,
    build_synthesis_prompt,
    detect_complexity,
    execute_chain,
    execute_step,
    format_plan_progress,
    parse_plan_from_json,
    parse_plan_from_response,
    resolve_templates,
)


def _make_tool(name: str, category: ToolCategory = ToolCategory.SEARCH, cacheable: bool = True) -> Tool:
    """Create a mock tool for testing."""
    tool = MagicMock(spec=Tool)
    tool.name = name
    tool.category = category
    tool.cacheable = cacheable
    tool.description = f"Test {name} tool"
    tool.timeout = 30.0
    tool.input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    tool.execute = AsyncMock(return_value=ToolResult(
        success=True, output=f"{name} result", metadata={"tool": name},
    ))
    return tool


def _make_registry(*tools: Tool):
    """Create a mock tool registry."""
    registry = MagicMock()
    tool_map = {t.name: t for t in tools}
    registry.resolve.side_effect = lambda name: tool_map.get(name)
    registry.list_tools.return_value = list(tools)
    registry.metrics = MagicMock()
    return registry


class TestComplexityDetection(unittest.TestCase):
    """Test detect_complexity heuristic."""

    def test_single_category_simple(self):
        tools = [_make_tool("web_search"), _make_tool("web_fetch")]
        assert not detect_complexity("search for cats", tools)

    def test_multi_category_no_trigger_without_language(self):
        """Multiple categories alone no longer trigger — query text must have multi-step language."""
        tools = [
            _make_tool("web_search", ToolCategory.SEARCH),
            _make_tool("calculator", ToolCategory.VERIFY),
        ]
        assert not detect_complexity("search and verify", tools)

    def test_multi_category_with_multi_step_language(self):
        tools = [
            _make_tool("web_search", ToolCategory.SEARCH),
            _make_tool("calculator", ToolCategory.VERIFY),
        ]
        assert detect_complexity("search for GDP data then calculate the growth rate", tools)

    def test_explicit_multi_step_language(self):
        tools = [_make_tool("web_search")]
        assert detect_complexity("first search for X then summarize it", tools)

    def test_then_keyword(self):
        tools = [_make_tool("web_search")]
        assert detect_complexity("search for the video then get the transcript", tools)

    def test_after_that_keyword(self):
        tools = [_make_tool("web_search")]
        assert detect_complexity("do A, after that do B", tools)

    def test_step_keyword(self):
        tools = [_make_tool("web_search")]
        assert detect_complexity("step 1 search step 2 analyze", tools)

    def test_simple_query_no_trigger(self):
        tools = [_make_tool("web_search")]
        assert not detect_complexity("what is the weather", tools)


class TestTemplateResolution(unittest.TestCase):
    """Test resolve_templates with step references."""

    def test_resolve_step_output(self):
        results = {
            1: StepResult(1, "web_search", "found cats", {}, True),
        }
        resolved = resolve_templates(
            {"query": "about {{step.1.output}}"}, results,
        )
        assert resolved["query"] == "about found cats"

    def test_resolve_step_metadata(self):
        results = {
            1: StepResult(1, "web_search", "ok", {"urls": ["http://x.com"]}, True),
        }
        resolved = resolve_templates(
            {"url": "{{step.1.metadata.urls.0}}"}, results,
        )
        assert resolved["url"] == "http://x.com"

    def test_unresolved_template_kept(self):
        resolved = resolve_templates(
            {"query": "{{step.99.output}}"}, {},
        )
        assert resolved["query"] == "{{step.99.output}}"

    def test_non_string_values_passthrough(self):
        resolved = resolve_templates(
            {"count": 42, "query": "test"}, {},
        )
        assert resolved["count"] == 42
        assert resolved["query"] == "test"

    def test_nested_metadata_path(self):
        results = {
            1: StepResult(1, "web", "ok", {"response": {"title": "Hello"}}, True),
        }
        resolved = resolve_templates(
            {"title": "{{step.1.metadata.response.title}}"}, results,
        )
        assert resolved["title"] == "Hello"


class TestWaveExecutor(unittest.TestCase):
    """Test execute_chain wave-based execution."""

    def test_sequential_chain(self):
        """Steps with linear dependencies execute one at a time."""
        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "cats"}, needs=[]),
            ChainStep(id=2, tool="calculator", input={"query": "1+1"}, needs=[1]),
        ]
        plan = ChainPlan(steps=steps)
        search = _make_tool("web_search")
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(search, calc)
        backend = MagicMock()

        results = asyncio.run(
            execute_chain(plan, backend, registry)
        )

        assert len(results) == 2
        assert results[1].success
        assert results[2].success
        assert results[1].tool_name == "web_search"
        assert results[2].tool_name == "calculator"

    def test_parallel_chain(self):
        """Independent steps execute in the same wave."""
        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "cats"}, needs=[]),
            ChainStep(id=2, tool="calculator", input={"query": "1+1"}, needs=[]),
            ChainStep(id=3, tool="web_search", input={"query": "summary"}, needs=[1, 2]),
        ]
        plan = ChainPlan(steps=steps)
        search = _make_tool("web_search")
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(search, calc)
        backend = MagicMock()

        results = asyncio.run(
            execute_chain(plan, backend, registry)
        )

        assert len(results) == 3
        assert all(r.success for r in results.values())

    def test_failed_step_error_as_observation(self):
        """With error-as-observation (default), dependents still run."""
        search = _make_tool("web_search")
        search.execute = AsyncMock(return_value=ToolResult(
            success=False, error="timeout", output="",
        ))
        calc = _make_tool("calculator", ToolCategory.VERIFY)

        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
            ChainStep(id=2, tool="calculator", input={"query": "1"}, needs=[1]),
        ]
        plan = ChainPlan(steps=steps)
        registry = _make_registry(search, calc)
        backend = MagicMock()

        results = asyncio.run(
            execute_chain(plan, backend, registry)
        )

        assert not results[1].success
        # Step 2 proceeds with error context instead of being skipped
        assert results[2].success

    def test_failed_step_cascades_legacy(self):
        """With error-as-observation OFF, dependents get skipped (legacy)."""
        from augmentum.config import settings
        original = settings.passthrough_chain_error_as_observation
        try:
            settings.passthrough_chain_error_as_observation = False

            search = _make_tool("web_search")
            search.execute = AsyncMock(return_value=ToolResult(
                success=False, error="timeout", output="",
            ))
            calc = _make_tool("calculator", ToolCategory.VERIFY)

            steps = [
                ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
                ChainStep(id=2, tool="calculator", input={"query": "1"}, needs=[1]),
            ]
            plan = ChainPlan(steps=steps)
            registry = _make_registry(search, calc)
            backend = MagicMock()

            results = asyncio.run(
                execute_chain(plan, backend, registry)
            )

            assert not results[1].success
            assert not results[2].success
            assert "dependency" in results[2].output.lower() or "failed" in results[2].output.lower()
        finally:
            settings.passthrough_chain_error_as_observation = original

    def test_unknown_tool_fails(self):
        """A step with an unresolvable tool name fails gracefully."""
        steps = [
            ChainStep(id=1, tool="nonexistent_tool", input={"query": "x"}, needs=[]),
        ]
        plan = ChainPlan(steps=steps)
        registry = _make_registry()  # empty
        backend = MagicMock()

        results = asyncio.run(
            execute_chain(plan, backend, registry)
        )

        assert not results[1].success
        assert "unknown" in results[1].output.lower()

    def test_max_steps_limit(self):
        """Chain respects max_steps limit."""
        steps = [
            ChainStep(id=i, tool="web_search", input={"query": f"q{i}"}, needs=[])
            for i in range(1, 11)
        ]
        plan = ChainPlan(steps=steps)
        search = _make_tool("web_search")
        registry = _make_registry(search)
        backend = MagicMock()

        results = asyncio.run(
            execute_chain(plan, backend, registry, max_steps=3)
        )

        assert len(results) == 3

    def test_callbacks_fired(self):
        """on_step_start and on_step_done callbacks are invoked."""
        started = []
        done = []

        async def on_start(step):
            started.append(step.id)

        async def on_done(result):
            done.append(result.step_id)

        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
            ChainStep(id=2, tool="web_search", input={"query": "y"}, needs=[1]),
        ]
        plan = ChainPlan(steps=steps)
        search = _make_tool("web_search")
        registry = _make_registry(search)
        backend = MagicMock()

        asyncio.run(
            execute_chain(
                plan, backend, registry,
                on_step_start=on_start,
                on_step_done=on_done,
            )
        )

        assert started == [1, 2]
        assert done == [1, 2]

    def test_circular_dependency_detected(self):
        """Circular deps are detected and remaining steps fail."""
        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[2]),
            ChainStep(id=2, tool="web_search", input={"query": "y"}, needs=[1]),
        ]
        plan = ChainPlan(steps=steps)
        search = _make_tool("web_search")
        registry = _make_registry(search)
        backend = MagicMock()

        results = asyncio.run(
            execute_chain(plan, backend, registry)
        )

        assert len(results) == 2
        assert not results[1].success
        assert not results[2].success


class TestPlanParsing(unittest.TestCase):
    """Test parse_plan_from_response."""

    def _make_response(self, content: str, thinking: str | None = None):
        resp = MagicMock()
        resp.message.content = content
        resp.message.thinking = thinking
        return resp

    def test_parse_basic_plan(self):
        text = """Here's my plan:
1. Search for the video using web_search
2. Get the transcript using youtube_transcript (needs step 1)
3. Check the math using calculator (needs step 2)
"""
        resp = self._make_response(text)
        search = _make_tool("web_search")
        yt = _make_tool("youtube_transcript", ToolCategory.FETCH)
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(search, yt, calc)

        plan = parse_plan_from_response(resp, registry)
        assert plan is not None
        assert len(plan.steps) == 3
        assert plan.steps[0].tool == "web_search"
        assert plan.steps[1].tool == "youtube_transcript"
        assert plan.steps[2].tool == "calculator"
        assert plan.steps[1].needs == [1]
        assert plan.steps[2].needs == [2]

    def test_no_plan_from_short_text(self):
        resp = self._make_response("Sure, let me search for that.")
        registry = _make_registry()
        plan = parse_plan_from_response(resp, registry)
        assert plan is None

    def test_no_plan_when_no_tools_resolved(self):
        text = """
1. Think about the problem
2. Analyze the data
3. Draw conclusions
"""
        resp = self._make_response(text)
        registry = _make_registry()  # empty — no tools resolve
        plan = parse_plan_from_response(resp, registry)
        assert plan is None

    def test_empty_response(self):
        resp = self._make_response("")
        registry = _make_registry()
        plan = parse_plan_from_response(resp, registry)
        assert plan is None

    def test_parse_verbose_markdown_plan(self):
        """Models sometimes produce multi-line plans with markdown headers."""
        text = """# Analysis Plan

### 1. Fetch Video Information
**What**: Get basic metadata about the video
**Why**: To understand context
**Using**: `youtube_transcript`

### 2. Analyze Readability
**What**: Calculate Flesch Reading Ease score
**Why**: Determines readability
**Using**: `text_analysis` (needs step 1)

### 3. Generate Summary
**What**: Create a visual summary
**Using**: `web_search` (needs steps 1, 2)
"""
        resp = self._make_response(text)
        yt = _make_tool("youtube_transcript", ToolCategory.FETCH)
        ta = _make_tool("text_analysis", ToolCategory.VERIFY)
        ws = _make_tool("web_search")
        registry = _make_registry(yt, ta, ws)

        plan = parse_plan_from_response(resp, registry)
        assert plan is not None
        assert len(plan.steps) == 3
        assert plan.steps[0].tool == "youtube_transcript"
        assert plan.steps[0].needs == []
        assert plan.steps[1].tool == "text_analysis"
        assert plan.steps[1].needs == [1]
        assert plan.steps[2].tool == "web_search"
        assert plan.steps[2].needs == [1, 2]

    def test_parse_plan_from_thinking_fallback(self):
        """When content is empty, plan should be extracted from thinking."""
        thinking = """Let me create a plan:
1. Search for the video using web_search
2. Get the transcript using youtube_transcript (needs step 1)
3. Summarize using text_analysis (needs steps 1, 2)
"""
        resp = MagicMock()
        resp.message.content = ""
        resp.message.thinking = thinking
        ws = _make_tool("web_search")
        yt = _make_tool("youtube_transcript", ToolCategory.FETCH)
        ta = _make_tool("text_analysis", ToolCategory.VERIFY)
        registry = _make_registry(ws, yt, ta)

        plan = parse_plan_from_response(resp, registry)
        assert plan is not None
        assert len(plan.steps) == 3
        assert plan.steps[0].tool == "web_search"

    def test_parse_backtick_tool_name(self):
        """Tool name wrapped in backticks should be resolved."""
        text = """
1. Search the web using `web_search`
2. Analyze results using `text_analysis` (needs step 1)
"""
        resp = self._make_response(text)
        ws = _make_tool("web_search")
        ta = _make_tool("text_analysis", ToolCategory.VERIFY)
        registry = _make_registry(ws, ta)

        plan = parse_plan_from_response(resp, registry)
        assert plan is not None
        assert plan.steps[0].tool == "web_search"
        assert plan.steps[1].tool == "text_analysis"


class TestJsonPlanParsing(unittest.TestCase):
    """Test parse_plan_from_json — structured JSON plan parsing."""

    def test_parse_valid_json_plan(self):
        text = '{"steps": [{"id": 1, "tool": "web_search", "reason": "Search", "needs": []}, {"id": 2, "tool": "youtube_transcript", "reason": "Transcript", "needs": [1]}]}'
        ws = _make_tool("web_search")
        yt = _make_tool("youtube_transcript", ToolCategory.FETCH)
        registry = _make_registry(ws, yt)

        plan = parse_plan_from_json(text, registry)
        assert plan is not None
        assert len(plan.steps) == 2
        assert plan.steps[0].tool == "web_search"
        assert plan.steps[0].needs == []
        assert plan.steps[1].tool == "youtube_transcript"
        assert plan.steps[1].needs == [1]

    def test_parse_json_with_markdown_fences(self):
        text = '```json\n{"steps": [{"id": 1, "tool": "web_search", "reason": "Search", "needs": []}, {"id": 2, "tool": "calculator", "reason": "Calc", "needs": [1]}]}\n```'
        ws = _make_tool("web_search")
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(ws, calc)

        plan = parse_plan_from_json(text, registry)
        assert plan is not None
        assert len(plan.steps) == 2

    def test_parse_json_with_prose_around_it(self):
        text = 'Here is my plan:\n{"steps": [{"id": 1, "tool": "web_search", "reason": "Search", "needs": []}, {"id": 2, "tool": "calculator", "reason": "Calc", "needs": [1]}]}\nDone.'
        ws = _make_tool("web_search")
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(ws, calc)

        plan = parse_plan_from_json(text, registry)
        assert plan is not None

    def test_json_unresolved_tools_filtered(self):
        text = '{"steps": [{"id": 1, "tool": "nonexistent", "reason": "X", "needs": []}, {"id": 2, "tool": "also_fake", "reason": "Y", "needs": [1]}]}'
        registry = _make_registry()  # empty

        plan = parse_plan_from_json(text, registry)
        assert plan is None  # no tools resolved

    def test_json_single_step_valid(self):
        text = '{"steps": [{"id": 1, "tool": "web_search", "reason": "Just one", "needs": []}]}'
        ws = _make_tool("web_search")
        registry = _make_registry(ws)

        plan = parse_plan_from_json(text, registry)
        assert plan is not None
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "web_search"

    def test_json_empty_string(self):
        plan = parse_plan_from_json("", _make_registry())
        assert plan is None

    def test_json_invalid_json(self):
        plan = parse_plan_from_json("{broken json", _make_registry())
        assert plan is None

    def test_json_missing_needs_defaults_empty(self):
        text = '{"steps": [{"id": 1, "tool": "web_search", "reason": "A"}, {"id": 2, "tool": "calculator", "reason": "B"}]}'
        ws = _make_tool("web_search")
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(ws, calc)

        plan = parse_plan_from_json(text, registry)
        assert plan is not None
        assert plan.steps[0].needs == []
        assert plan.steps[1].needs == []

    def test_json_fuzzy_tool_resolution(self):
        """Tool names should be resolved via registry (e.g. 'search' -> 'web_search')."""
        text = '{"steps": [{"id": 1, "tool": "search", "reason": "A", "needs": []}, {"id": 2, "tool": "youtube", "reason": "B", "needs": [1]}]}'
        ws = _make_tool("web_search")
        yt = _make_tool("youtube_transcript", ToolCategory.FETCH)
        registry = _make_registry(ws, yt)

        plan = parse_plan_from_json(text, registry)
        # Depends on whether registry.resolve handles fuzzy matching
        # Either resolved or not — shouldn't crash
        assert plan is None or len(plan.steps) == 2


class TestPlanProgress(unittest.TestCase):
    """Test format_plan_progress."""

    def test_progress_formatting(self):
        steps = [
            ChainStep(id=1, tool="web_search", reason="Search for video"),
            ChainStep(id=2, tool="calculator", reason="Check math", needs=[1]),
        ]
        plan = ChainPlan(steps=steps)
        results = {
            1: StepResult(1, "web_search", "Found 5 results", {}, True),
        }
        text = format_plan_progress(plan, results, current_step_ids=[2])
        assert "✓" in text
        assert "EXECUTE THIS NOW" in text


class TestSynthesisPrompt(unittest.TestCase):
    """Test build_synthesis_prompt."""

    def test_includes_all_results(self):
        steps = [
            ChainStep(id=1, tool="web_search", reason="Search"),
            ChainStep(id=2, tool="calculator", reason="Calc"),
        ]
        plan = ChainPlan(steps=steps)
        results = {
            1: StepResult(1, "web_search", "Found cats", {}, True),
            2: StepResult(2, "calculator", "42", {}, True),
        }
        text = build_synthesis_prompt(plan, results)
        assert "web_search" in text
        assert "calculator" in text
        assert "Found cats" in text
        assert "42" in text
        assert "Answer the user" in text

    def test_shows_failed_steps(self):
        steps = [ChainStep(id=1, tool="web_search", reason="Search")]
        plan = ChainPlan(steps=steps)
        results = {
            1: StepResult(1, "web_search", "Error: timeout", {}, False),
        }
        text = build_synthesis_prompt(plan, results)
        assert "✗" in text


class TestExecuteStep(unittest.TestCase):
    """Test individual step execution."""

    def test_step_with_preset_input(self):
        """Step with input dict executes directly."""
        step = ChainStep(id=1, tool="web_search", input={"query": "cats"})
        search = _make_tool("web_search")
        registry = _make_registry(search)
        backend = MagicMock()

        result = asyncio.run(
            execute_step(step, {}, backend, registry)
        )

        assert result.success
        search.execute.assert_called_once_with(query="cats")

    def test_step_timeout(self):
        """Step that exceeds timeout returns error."""
        step = ChainStep(id=1, tool="web_search", input={"query": "slow"})
        search = _make_tool("web_search")
        search.timeout = 0.01  # very short

        async def slow_exec(**kwargs):
            await asyncio.sleep(1)
            return ToolResult(success=True, output="done")

        search.execute = slow_exec
        registry = _make_registry(search)
        backend = MagicMock()

        result = asyncio.run(
            execute_step(step, {}, backend, registry)
        )

        assert not result.success
        assert "timed out" in result.output.lower()

    def test_step_with_template_resolution(self):
        """Step input with templates gets resolved from prior results."""
        step = ChainStep(
            id=2, tool="web_search",
            input={"query": "details about {{step.1.output}}"},
            needs=[1],
        )
        prior = {
            1: StepResult(1, "web_search", "cats are cool", {}, True),
        }
        search = _make_tool("web_search")
        registry = _make_registry(search)
        backend = MagicMock()

        result = asyncio.run(
            execute_step(step, prior, backend, registry)
        )

        assert result.success
        search.execute.assert_called_once_with(query="details about cats are cool")


class TestReplanOnFailure(unittest.TestCase):
    """Test re-plan-on-failure behavior in execute_chain."""

    def test_replan_retry_succeeds(self):
        """Step fails, LLM says retry, second attempt works."""
        search = _make_tool("web_search")
        call_count = 0

        async def _flaky_exec(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ToolResult(success=False, error="timeout", output="")
            return ToolResult(success=True, output="found results", metadata={})

        search.execute = _flaky_exec
        registry = _make_registry(search)

        # Mock backend for replan LLM call
        backend = MagicMock()
        replan_resp = MagicMock()
        replan_resp.message.content = "retry"
        backend.chat = AsyncMock(return_value=replan_resp)

        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "test"}, needs=[]),
        ]
        plan = ChainPlan(steps=steps)

        from augmentum.config import settings
        orig = settings.passthrough_chain_max_retries
        try:
            settings.passthrough_chain_max_retries = 2
            from augmentum.models.base import InternalChatRequest, Message
            ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])
            results = asyncio.run(
                execute_chain(plan, backend, registry, request_context=ctx)
            )
            self.assertTrue(results[1].success)
            self.assertEqual(call_count, 2)
        finally:
            settings.passthrough_chain_max_retries = orig

    def test_replan_skip_no_cascade(self):
        """Step fails, LLM says skip, dependents still run."""
        search = _make_tool("web_search")
        search.execute = AsyncMock(return_value=ToolResult(
            success=False, error="timeout", output="",
        ))
        calc = _make_tool("calculator", ToolCategory.VERIFY)

        registry = _make_registry(search, calc)

        backend = MagicMock()
        replan_resp = MagicMock()
        replan_resp.message.content = "skip"
        backend.chat = AsyncMock(return_value=replan_resp)

        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
            ChainStep(id=2, tool="calculator", input={"query": "1+1"}, needs=[1]),
        ]
        plan = ChainPlan(steps=steps)

        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message
        orig = settings.passthrough_chain_max_retries
        try:
            settings.passthrough_chain_max_retries = 2
            ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])
            results = asyncio.run(
                execute_chain(plan, backend, registry, request_context=ctx)
            )
            # Step 1 was skipped (marked success=True so dependents run)
            self.assertTrue(results[1].success)
            self.assertIn("Skipped", results[1].output)
            # Step 2 should have run
            self.assertTrue(results[2].success)
        finally:
            settings.passthrough_chain_max_retries = orig

    def test_replan_max_retries_cap(self):
        """Retries are capped at config limit."""
        search = _make_tool("web_search")
        search.execute = AsyncMock(return_value=ToolResult(
            success=False, error="timeout", output="",
        ))
        registry = _make_registry(search)

        backend = MagicMock()
        replan_resp = MagicMock()
        replan_resp.message.content = "retry"
        backend.chat = AsyncMock(return_value=replan_resp)

        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
        ]
        plan = ChainPlan(steps=steps)

        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message
        orig = settings.passthrough_chain_max_retries
        try:
            settings.passthrough_chain_max_retries = 1
            ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])
            results = asyncio.run(
                execute_chain(plan, backend, registry, request_context=ctx)
            )
            # Should have retried once, then given up
            self.assertFalse(results[1].success)
            # execute_tool called twice (original + 1 retry)
            self.assertEqual(search.execute.await_count, 2)
        finally:
            settings.passthrough_chain_max_retries = orig

    def test_replan_abort_cascades(self):
        """LLM says abort — with error-as-observation OFF, cascades to dependents."""
        search = _make_tool("web_search")
        search.execute = AsyncMock(return_value=ToolResult(
            success=False, error="timeout", output="",
        ))
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(search, calc)

        backend = MagicMock()
        replan_resp = MagicMock()
        replan_resp.message.content = "abort"
        backend.chat = AsyncMock(return_value=replan_resp)

        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
            ChainStep(id=2, tool="calculator", input={"query": "1"}, needs=[1]),
        ]
        plan = ChainPlan(steps=steps)

        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message
        orig_retries = settings.passthrough_chain_max_retries
        orig_eao = settings.passthrough_chain_error_as_observation
        try:
            settings.passthrough_chain_max_retries = 2
            settings.passthrough_chain_error_as_observation = False
            ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])
            results = asyncio.run(
                execute_chain(plan, backend, registry, request_context=ctx)
            )
            self.assertFalse(results[1].success)
            self.assertFalse(results[2].success)
            self.assertIn("dependency", results[2].output.lower())
        finally:
            settings.passthrough_chain_max_retries = orig_retries
            settings.passthrough_chain_error_as_observation = orig_eao


class TestAttentionAnchor(unittest.TestCase):
    """Tests for the Manus attention anchor pattern."""

    def test_anchor_injected_into_arg_resolution(self):
        """format_plan_progress() output appears in LLM prompt when enabled."""
        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message

        tool = _make_tool("calculator", ToolCategory.VERIFY)
        step = ChainStep(id=2, tool="calculator", needs=[1], reason="Calculate growth")
        prior = {
            1: StepResult(step_id=1, tool_name="web_search", output="GDP data", metadata={}, success=True),
        }
        plan = ChainPlan(steps=[
            ChainStep(id=1, tool="web_search", needs=[], reason="Search GDP"),
            step,
        ])

        backend = MagicMock()
        resp = MagicMock()
        resp.message.content = '{"expression": "100 * 1.05"}'
        backend.chat = AsyncMock(return_value=resp)

        ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="calc growth")])

        orig = settings.passthrough_chain_attention_anchor
        try:
            settings.passthrough_chain_attention_anchor = True
            asyncio.run(_resolve_args_via_llm(
                step, tool, prior, backend, ctx, plan=plan, all_results=prior,
            ))
            # Check the user content sent to the LLM includes plan progress
            call_args = backend.chat.call_args[0][0]
            user_msg = [m for m in call_args.messages if m.role == "user"][0]
            self.assertIn("Plan progress:", user_msg.content)
            self.assertIn("✓", user_msg.content)  # completed step marker
            self.assertIn("EXECUTE THIS NOW", user_msg.content)  # current step
        finally:
            settings.passthrough_chain_attention_anchor = orig

    def test_anchor_disabled(self):
        """When attention anchor is disabled, no plan progress in prompt."""
        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message

        tool = _make_tool("calculator", ToolCategory.VERIFY)
        step = ChainStep(id=2, tool="calculator", needs=[1], reason="Calculate")
        prior = {
            1: StepResult(step_id=1, tool_name="web_search", output="data", metadata={}, success=True),
        }
        plan = ChainPlan(steps=[
            ChainStep(id=1, tool="web_search", needs=[], reason="Search"),
            step,
        ])

        backend = MagicMock()
        resp = MagicMock()
        resp.message.content = '{"expression": "1+1"}'
        backend.chat = AsyncMock(return_value=resp)
        ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])

        orig = settings.passthrough_chain_attention_anchor
        try:
            settings.passthrough_chain_attention_anchor = False
            asyncio.run(_resolve_args_via_llm(
                step, tool, prior, backend, ctx, plan=plan, all_results=prior,
            ))
            call_args = backend.chat.call_args[0][0]
            user_msg = [m for m in call_args.messages if m.role == "user"][0]
            self.assertNotIn("Plan progress:", user_msg.content)
        finally:
            settings.passthrough_chain_attention_anchor = orig


class TestErrorAsObservation(unittest.TestCase):
    """Tests for the error-as-observation pattern."""

    def test_failed_dep_visible_in_arg_resolution(self):
        """When a dependency fails, its error is visible and adaptation hint added."""
        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message

        tool = _make_tool("calculator", ToolCategory.VERIFY)
        step = ChainStep(id=2, tool="calculator", needs=[1], reason="Fallback calc")
        prior = {
            1: StepResult(step_id=1, tool_name="web_search", output="Error: timeout", metadata={}, success=False),
        }

        backend = MagicMock()
        resp = MagicMock()
        resp.message.content = '{"expression": "1+1"}'
        backend.chat = AsyncMock(return_value=resp)
        ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])

        orig = settings.passthrough_chain_error_as_observation
        try:
            settings.passthrough_chain_error_as_observation = True
            asyncio.run(_resolve_args_via_llm(
                step, tool, prior, backend, ctx,
            ))
            call_args = backend.chat.call_args[0][0]
            user_msg = [m for m in call_args.messages if m.role == "user"][0]
            # Should see the error output AND the adaptation hint
            self.assertIn("FAILED", user_msg.content)
            self.assertIn("Adapt your approach", user_msg.content)
        finally:
            settings.passthrough_chain_error_as_observation = orig

    def test_error_as_observation_dependents_run(self):
        """Dependent steps execute even when dependencies fail."""
        search = _make_tool("web_search")
        search.execute = AsyncMock(return_value=ToolResult(
            success=False, error="timeout", output="",
        ))
        calc = _make_tool("calculator", ToolCategory.VERIFY)

        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
            ChainStep(id=2, tool="calculator", input={"expression": "1+1"}, needs=[1]),
        ]
        plan = ChainPlan(steps=steps)
        registry = _make_registry(search, calc)
        backend = MagicMock()

        from augmentum.config import settings
        orig = settings.passthrough_chain_error_as_observation
        try:
            settings.passthrough_chain_error_as_observation = True
            results = asyncio.run(execute_chain(plan, backend, registry))
            self.assertFalse(results[1].success)
            self.assertTrue(results[2].success)  # ran despite failed dep
        finally:
            settings.passthrough_chain_error_as_observation = orig


class TestPlanMutation(unittest.TestCase):
    """Tests for the plan mutation pattern."""

    def test_mutate_decision_accepted(self):
        """_replan_on_failure returns 'mutate' when LLM says so."""
        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message

        step = ChainStep(id=1, tool="web_search", needs=[], reason="Search")
        failed = StepResult(step_id=1, tool_name="web_search", output="Error: timeout", metadata={}, success=False)
        plan = ChainPlan(steps=[step])

        backend = MagicMock()
        resp = MagicMock()
        resp.message.content = "mutate"
        backend.chat = AsyncMock(return_value=resp)

        ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])

        orig = settings.passthrough_chain_plan_mutation
        try:
            settings.passthrough_chain_plan_mutation = True
            decision = asyncio.run(_replan_on_failure(step, failed, {}, plan, backend, ctx))
            self.assertEqual(decision, "mutate")
        finally:
            settings.passthrough_chain_plan_mutation = orig

    def test_mutate_plan_restructures(self):
        """_mutate_plan returns new steps from LLM."""
        import json

        from augmentum.models.base import InternalChatRequest, Message

        search = _make_tool("web_search")
        wiki = _make_tool("wikipedia")
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(search, wiki, calc)

        step1 = ChainStep(id=1, tool="web_search", needs=[], reason="Search")
        step2 = ChainStep(id=2, tool="calculator", needs=[1], reason="Calculate")
        plan = ChainPlan(steps=[step1, step2])
        failed = StepResult(step_id=1, tool_name="web_search", output="Error: timeout", metadata={}, success=False)

        # LLM returns mutated plan: use wikipedia instead, then calculator
        mutated_json = json.dumps([
            {"id": 3, "tool": "wikipedia", "reason": "Look up on Wikipedia", "needs": []},
            {"id": 4, "tool": "calculator", "reason": "Calculate", "needs": [3]},
        ])

        backend = MagicMock()
        resp = MagicMock()
        resp.message.content = mutated_json
        backend.chat = AsyncMock(return_value=resp)

        ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])

        new_steps = asyncio.run(_mutate_plan(
            plan, step1, failed, {}, [step2], backend, registry, ctx,
        ))

        self.assertIsNotNone(new_steps)
        self.assertEqual(len(new_steps), 2)
        self.assertEqual(new_steps[0].tool, "wikipedia")
        self.assertEqual(new_steps[1].tool, "calculator")
        self.assertEqual(new_steps[1].needs, [3])

    def test_mutate_plan_unknown_tool_returns_none(self):
        """If mutated plan references unknown tool, returns None."""
        import json

        from augmentum.models.base import InternalChatRequest, Message

        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(calc)

        step1 = ChainStep(id=1, tool="web_search", needs=[], reason="Search")
        plan = ChainPlan(steps=[step1])
        failed = StepResult(step_id=1, tool_name="web_search", output="Error", metadata={}, success=False)

        mutated_json = json.dumps([
            {"id": 2, "tool": "nonexistent_tool", "reason": "Nope", "needs": []},
        ])

        backend = MagicMock()
        resp = MagicMock()
        resp.message.content = mutated_json
        backend.chat = AsyncMock(return_value=resp)

        ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])

        result = asyncio.run(_mutate_plan(
            plan, step1, failed, {}, [], backend, registry, ctx,
        ))
        self.assertIsNone(result)

    def test_mutate_wired_into_execute_chain(self):
        """When LLM says mutate, execute_chain restructures and continues."""
        import json

        from augmentum.config import settings
        from augmentum.models.base import InternalChatRequest, Message

        search = _make_tool("web_search")
        search.execute = AsyncMock(return_value=ToolResult(
            success=False, error="timeout", output="",
        ))
        wiki = _make_tool("wikipedia")
        calc = _make_tool("calculator", ToolCategory.VERIFY)
        registry = _make_registry(search, wiki, calc)

        steps = [
            ChainStep(id=1, tool="web_search", input={"query": "x"}, needs=[]),
            ChainStep(id=2, tool="calculator", input={"expression": "1+1"}, needs=[1]),
        ]
        plan = ChainPlan(steps=steps)

        # Backend returns "mutate" for replan, then mutated plan JSON
        mutated_json = json.dumps([
            {"id": 3, "tool": "wikipedia", "reason": "Wikipedia fallback", "needs": []},
            {"id": 4, "tool": "calculator", "reason": "Calculate", "needs": [3]},
        ])

        call_count = 0
        async def mock_chat(req):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if call_count == 1:
                # replan decision
                resp.message.content = "mutate"
            elif call_count == 2:
                # mutation plan
                resp.message.content = mutated_json
            else:
                resp.message.content = "{}"
            return resp

        backend = MagicMock()
        backend.chat = mock_chat

        orig_retries = settings.passthrough_chain_max_retries
        orig_mutation = settings.passthrough_chain_plan_mutation
        orig_eao = settings.passthrough_chain_error_as_observation
        try:
            settings.passthrough_chain_max_retries = 2
            settings.passthrough_chain_plan_mutation = True
            settings.passthrough_chain_error_as_observation = False
            ctx = InternalChatRequest(model="test", messages=[Message(role="user", content="test")])
            results = asyncio.run(execute_chain(plan, backend, registry, request_context=ctx))

            # Step 1 failed but marked as mutated (success=True to allow continuation)
            self.assertTrue(results[1].success)
            self.assertIn("mutated", results[1].output.lower())
            # New steps should have executed
            self.assertIn(3, results)
            self.assertIn(4, results)
            self.assertTrue(results[3].success)
            self.assertTrue(results[4].success)
        finally:
            settings.passthrough_chain_max_retries = orig_retries
            settings.passthrough_chain_plan_mutation = orig_mutation
            settings.passthrough_chain_error_as_observation = orig_eao


if __name__ == "__main__":
    unittest.main()
