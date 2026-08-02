"""Tests for reasoning flow execution: resolver, variables, executor."""

from __future__ import annotations

import pytest

from augmentum.reasoning.models import FlowStep, ReasoningFlow
from augmentum.reasoning.variables import (
    DEFAULT_USER_TEMPLATE,
    StepContext,
    build_user_message,
    resolve_variables,
)


# ---------------------------------------------------------------------------
# Variable substitution tests
# ---------------------------------------------------------------------------


class TestStepContext:
    def test_initial_state(self):
        ctx = StepContext(query="What is Python?", model="qwen2.5:72b")
        assert ctx.query == "What is Python?"
        assert ctx.model == "qwen2.5:72b"
        assert ctx.previous_output == ""
        assert ctx.all_outputs == ""
        assert ctx.complexity == ""

    def test_record_step(self):
        ctx = StepContext(query="test")
        ctx.record_step("Assess", "COMPLEXITY: simple")
        assert ctx.previous_output == "COMPLEXITY: simple"
        assert ctx.get_step_output("Assess") == "COMPLEXITY: simple"

    def test_multiple_steps(self):
        ctx = StepContext(query="test")
        ctx.record_step("Step1", "output1")
        ctx.record_step("Step2", "output2")
        assert ctx.previous_output == "output2"
        assert ctx.get_step_output("Step1") == "output1"
        assert ctx.get_step_output("Step2") == "output2"
        assert "Step1" in ctx.all_outputs
        assert "Step2" in ctx.all_outputs

    def test_missing_step(self):
        ctx = StepContext(query="test")
        assert ctx.get_step_output("nonexistent") == ""


class TestResolveVariables:
    def test_simple_query(self):
        ctx = StepContext(query="What is 2+2?")
        result = resolve_variables("Answer: {query}", ctx)
        assert result == "Answer: What is 2+2?"

    def test_model_variable(self):
        ctx = StepContext(query="test", model="llama3.1")
        result = resolve_variables("Model: {model}", ctx)
        assert result == "Model: llama3.1"

    def test_complexity_variable(self):
        ctx = StepContext(query="test")
        ctx.complexity = "complex"
        result = resolve_variables("Level: {complexity}", ctx)
        assert result == "Level: complex"

    def test_previous_output(self):
        ctx = StepContext(query="test")
        ctx.record_step("Prior", "some analysis")
        result = resolve_variables("Based on: {previous_output}", ctx)
        assert result == "Based on: some analysis"

    def test_step_reference(self):
        ctx = StepContext(query="test")
        ctx.record_step("Assess", "COMPLEXITY: moderate")
        result = resolve_variables("Assessment: {step:Assess}", ctx)
        assert result == "Assessment: COMPLEXITY: moderate"

    def test_multiple_step_references(self):
        ctx = StepContext(query="test")
        ctx.record_step("A", "alpha")
        ctx.record_step("B", "beta")
        result = resolve_variables("{step:A} and {step:B}", ctx)
        assert result == "alpha and beta"

    def test_all_outputs(self):
        ctx = StepContext(query="test")
        ctx.record_step("S1", "out1")
        ctx.record_step("S2", "out2")
        result = resolve_variables("{all_outputs}", ctx)
        assert "S1" in result
        assert "out1" in result
        assert "S2" in result
        assert "out2" in result

    def test_search_results(self):
        ctx = StepContext(query="test")
        ctx.search_results = "Result 1: ...\nResult 2: ..."
        result = resolve_variables("{search_results}", ctx)
        assert "Result 1" in result

    def test_tools_variable(self):
        ctx = StepContext(query="test")
        result = resolve_variables("{tools}", ctx, tools_section="## Tools\n- calculator")
        assert "calculator" in result

    def test_no_variables(self):
        ctx = StepContext(query="test")
        result = resolve_variables("No variables here", ctx)
        assert result == "No variables here"

    def test_empty_template(self):
        ctx = StepContext(query="test")
        result = resolve_variables("", ctx)
        assert result == ""

    def test_unknown_step_reference(self):
        ctx = StepContext(query="test")
        result = resolve_variables("{step:Missing}", ctx)
        assert result == ""


class TestBuildUserMessage:
    def test_default_template(self):
        ctx = StepContext(query="What is AI?")
        msg = build_user_message("", ctx)
        assert "What is AI?" in msg

    def test_custom_template(self):
        ctx = StepContext(query="test query")
        msg = build_user_message("Question: {query}", ctx)
        assert msg == "Question: test query"

    def test_with_conversation(self):
        ctx = StepContext(query="test")
        ctx.conversation = "User: Hi\nAssistant: Hello"
        msg = build_user_message("", ctx)
        assert "Hi" in msg

    def test_collapses_blank_lines(self):
        ctx = StepContext(query="test")
        msg = build_user_message("{query}\n\n\n\n{previous_output}", ctx)
        assert "\n\n\n" not in msg


# ---------------------------------------------------------------------------
# Step filtering tests
# ---------------------------------------------------------------------------


class TestStepFiltering:
    def test_no_gate_always_runs(self):
        from augmentum.reasoning.executor import filter_steps_by_complexity

        steps = [FlowStep(name="S1", complexity_gate=[], enabled=True)]
        assert len(filter_steps_by_complexity(steps, "simple")) == 1
        assert len(filter_steps_by_complexity(steps, "complex")) == 1

    def test_gate_filters(self):
        from augmentum.reasoning.executor import filter_steps_by_complexity

        steps = [
            FlowStep(name="Always", complexity_gate=[], enabled=True),
            FlowStep(name="Complex Only", complexity_gate=["complex"], enabled=True),
            FlowStep(name="Moderate+", complexity_gate=["moderate", "complex"], enabled=True),
        ]
        simple = filter_steps_by_complexity(steps, "simple")
        assert len(simple) == 1
        assert simple[0].name == "Always"

        moderate = filter_steps_by_complexity(steps, "moderate")
        assert len(moderate) == 2
        names = [s.name for s in moderate]
        assert "Always" in names
        assert "Moderate+" in names

        complex_ = filter_steps_by_complexity(steps, "complex")
        assert len(complex_) == 3

    def test_disabled_steps_excluded(self):
        from augmentum.reasoning.executor import filter_steps_by_complexity

        steps = [
            FlowStep(name="Enabled", enabled=True),
            FlowStep(name="Disabled", enabled=False),
        ]
        assert len(filter_steps_by_complexity(steps, "moderate")) == 1


# ---------------------------------------------------------------------------
# Resolver tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def resolver_store():
    import aiosqlite
    from pathlib import Path
    from augmentum.reasoning.store import FlowStore

    db = await aiosqlite.connect(":memory:")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP, description TEXT DEFAULT '')"
    )
    migration_path = Path(__file__).parent.parent / "augmentum" / "state" / "migrations" / "011_reasoning_flows.sql"
    await db.executescript(migration_path.read_text())

    store = FlowStore(db)
    await store.seed_builtins()
    yield store
    await db.close()


@pytest.mark.asyncio
class TestResolver:
    async def test_default_resolution(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        flow = await resolve_flow(resolver_store)
        assert flow is not None
        assert flow.name == "Quick Answer"

    async def test_keyword_resolution(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        flow = await resolve_flow(resolver_store, query="can you debug this code error?")
        assert flow is not None
        assert flow.name == "Code Review"

    async def test_math_keyword(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        flow = await resolve_flow(resolver_store, query="calculate the derivative of x^2")
        assert flow is not None
        assert flow.name == "Math & Science"

    async def test_debate_keyword(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        flow = await resolve_flow(resolver_store, query="should we ban AI?")
        assert flow is not None
        assert flow.name == "Debate"

    async def test_explicit_flow_id(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        # Get the Code Review flow ID
        flows = await resolver_store.list_flows()
        code_review = next((f for f, _ in flows if f.name == "Code Review"), None)
        assert code_review is not None

        # Explicit selection overrides everything
        flow = await resolve_flow(
            resolver_store,
            query="what is the weather?",  # would normally match differently
            explicit_flow_id=code_review.id,
        )
        assert flow is not None
        assert flow.name == "Code Review"

    async def test_model_pinning(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        # Create a custom flow pinned to a model
        custom = ReasoningFlow(
            name="Pinned Flow",
            auto_select=True,
            pinned_models=["deepseek-coder"],
            steps=[FlowStep(name="S", role="respond", stream_to_user=True)],
        )
        await resolver_store.create_flow(custom)

        flow = await resolve_flow(resolver_store, model="deepseek-coder:33b")
        assert flow is not None
        assert flow.name == "Pinned Flow"

    async def test_no_store_returns_none(self):
        from augmentum.reasoning.resolver import resolve_flow

        flow = await resolve_flow(None)
        assert flow is None

    async def test_no_keyword_match_falls_to_standard(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        # No keywords match — falls to Standard via auto routing
        flow = await resolve_flow(resolver_store, query="what is the meaning of life?")
        assert flow is not None
        assert flow.name == "Quick Answer"

    async def test_user_default_skips_keyword_matching(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        # Set Quick Answer as default (not Auto Routing)
        flows = await resolver_store.list_flows()
        quick = next((f for f, _ in flows if f.name == "Quick Answer"), None)
        assert quick is not None
        await resolver_store.set_default(quick.id)

        # Query with debug/error keywords that would normally match Code Review
        flow = await resolve_flow(resolver_store, query="can you debug this code error?")
        assert flow is not None
        assert flow.name == "Quick Answer"  # user's default wins

    async def test_auto_routing_default_uses_keywords(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        # Ensure Auto Routing is default
        flows = await resolver_store.list_flows()
        ar = next((f for f, _ in flows if f.name == "Auto Routing"), None)
        assert ar is not None
        await resolver_store.set_default(ar.id)

        # Keywords should route to Code Review
        flow = await resolve_flow(resolver_store, query="can you debug this code error?")
        assert flow is not None
        assert flow.name == "Code Review"

    async def test_auto_routing_no_match_falls_to_standard(self, resolver_store):
        from augmentum.reasoning.resolver import resolve_flow

        # Auto Routing is default, generic query
        flow = await resolve_flow(resolver_store, query="hello how are you?")
        assert flow is not None
        assert flow.name == "Quick Answer"


# ---------------------------------------------------------------------------
# Tool resolution tests
# ---------------------------------------------------------------------------


class TestToolResolution:
    def test_resolve_by_name(self):
        from unittest.mock import MagicMock
        from augmentum.reasoning.executor import _resolve_tools_for_step

        registry = MagicMock()
        tool1 = MagicMock()
        tool1.name = "calculator"
        registry.get.return_value = tool1
        registry.get_for_phase.return_value = []

        step = FlowStep(tool_names=["calculator"])
        tools = _resolve_tools_for_step(step, registry)
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    def test_resolve_by_category(self):
        from unittest.mock import MagicMock
        from augmentum.reasoning.executor import _resolve_tools_for_step

        registry = MagicMock()
        tool1 = MagicMock()
        tool1.name = "web_search"
        registry.get.return_value = None
        registry.get_for_phase.return_value = [tool1]

        step = FlowStep(tool_categories=["search"])
        tools = _resolve_tools_for_step(step, registry)
        assert len(tools) == 1

    def test_no_registry(self):
        from augmentum.reasoning.executor import _resolve_tools_for_step

        step = FlowStep(tool_names=["calculator"])
        tools = _resolve_tools_for_step(step, None)
        assert tools == []

    def test_dedup(self):
        from unittest.mock import MagicMock
        from augmentum.reasoning.executor import _resolve_tools_for_step

        registry = MagicMock()
        tool1 = MagicMock()
        tool1.name = "calculator"
        registry.get.return_value = tool1
        registry.get_for_phase.return_value = [tool1]

        step = FlowStep(tool_names=["calculator"], tool_categories=["verify"])
        tools = _resolve_tools_for_step(step, registry)
        assert len(tools) == 1  # deduped

    def test_exclude(self):
        from unittest.mock import MagicMock
        from augmentum.reasoning.executor import _resolve_tools_for_step

        registry = MagicMock()
        tool1 = MagicMock()
        tool1.name = "web_search"
        registry.get.return_value = tool1
        registry.get_for_phase.return_value = []

        step = FlowStep(tool_names=["web_search"])
        tools = _resolve_tools_for_step(step, registry, exclude=frozenset({"web_search"}))
        assert len(tools) == 0
