"""Tests for augmentum/reasoning/ — models, variables, resolver, store."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.reasoning.models import (
    FlowCreateRequest,
    FlowStep,
    ReasoningFlow,
    VALID_COMPLEXITIES,
    VALID_ROLES,
)
from augmentum.reasoning.variables import (
    StepContext,
    build_user_message,
    resolve_variables,
)


class TestFlowStep:
    """Verify FlowStep model construction and validation."""

    def test_default_construction(self):
        step = FlowStep()
        assert step.role == "analyze"
        assert step.enabled is True
        assert step.output_cap == 800

    def test_valid_roles(self):
        for role in VALID_ROLES:
            step = FlowStep(role=role)
            assert step.role == role

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="Invalid step role"):
            FlowStep(role="invalid_role")

    def test_valid_complexity_gate(self):
        step = FlowStep(complexity_gate=["simple", "complex"])
        assert step.complexity_gate == ["simple", "complex"]

    def test_invalid_complexity_gate_raises(self):
        with pytest.raises(ValueError, match="Invalid complexity gate"):
            FlowStep(complexity_gate=["ultra"])

    def test_long_name_raises(self):
        with pytest.raises(ValueError, match="at most 200"):
            FlowStep(name="x" * 201)

    def test_huge_prompt_raises(self):
        with pytest.raises(ValueError, match="at most 100,000"):
            FlowStep(system_prompt="x" * 100_001)

    def test_empty_role_allowed(self):
        step = FlowStep(role="")
        assert step.role == ""


class TestReasoningFlow:
    """Verify ReasoningFlow model construction."""

    def test_default_construction(self):
        flow = ReasoningFlow()
        assert flow.name == ""
        assert flow.version == 1
        assert flow.is_default is False
        assert flow.is_builtin is False
        assert flow.auto_select is True
        assert flow.autonomy_level == 2

    def test_with_steps(self):
        flow = ReasoningFlow(
            name="Test Flow",
            steps=[FlowStep(name="Step 1"), FlowStep(name="Step 2")],
        )
        assert len(flow.steps) == 2
        assert flow.steps[0].name == "Step 1"

    def test_trigger_keywords(self):
        flow = ReasoningFlow(trigger_keywords=["code", "debug"])
        assert "code" in flow.trigger_keywords


class TestFlowCreateRequest:
    """Verify create request validation."""

    def test_requires_name(self):
        with pytest.raises(Exception):
            FlowCreateRequest(name="")

    def test_valid_creation(self):
        req = FlowCreateRequest(name="My Flow")
        assert req.name == "My Flow"
        assert req.auto_select is True


class TestStepContext:
    """Verify variable context accumulation."""

    def test_initial_state(self):
        ctx = StepContext(query="What is Python?")
        assert ctx.query == "What is Python?"
        assert ctx.previous_output == ""

    def test_record_step(self):
        ctx = StepContext(query="test")
        ctx.record_step("Analyze", "Analysis result")
        assert ctx.previous_output == "Analysis result"
        assert ctx.get_step_output("Analyze") == "Analysis result"

    def test_all_outputs_skips_classify(self):
        ctx = StepContext(query="test")
        ctx.record_step("Classify", "TYPE: code")
        ctx.record_step("Analyze", "Real analysis")
        all_out = ctx.all_outputs
        assert "Classify" not in all_out
        assert "Analyze" in all_out

    def test_all_outputs_empty_when_no_steps(self):
        ctx = StepContext(query="test")
        assert ctx.all_outputs == ""


class TestResolveVariables:
    """Verify template variable substitution."""

    def test_query_substitution(self):
        ctx = StepContext(query="Hello world")
        result = resolve_variables("Question: {query}", ctx)
        assert result == "Question: Hello world"

    def test_previous_output(self):
        ctx = StepContext(query="test")
        ctx.record_step("step1", "output1")
        result = resolve_variables("Previous: {previous_output}", ctx)
        assert result == "Previous: output1"

    def test_step_reference(self):
        ctx = StepContext(query="test")
        ctx.record_step("Analyze", "analysis here")
        result = resolve_variables("Result: {step:Analyze}", ctx)
        assert result == "Result: analysis here"

    def test_model_variable(self):
        ctx = StepContext(query="test", model="llama3.1:8b")
        result = resolve_variables("Model: {model}", ctx)
        assert result == "Model: llama3.1:8b"

    def test_empty_template_returns_empty(self):
        ctx = StepContext(query="test")
        assert resolve_variables("", ctx) == ""

    def test_no_variables_unchanged(self):
        ctx = StepContext(query="test")
        result = resolve_variables("Plain text here", ctx)
        assert result == "Plain text here"


class TestBuildUserMessage:
    """Verify user message construction."""

    def test_default_template(self):
        ctx = StepContext(query="What is Python?")
        result = build_user_message("", ctx)
        assert "What is Python?" in result

    def test_custom_template(self):
        ctx = StepContext(query="test query")
        result = build_user_message("Custom: {query}", ctx)
        assert result == "Custom: test query"

    def test_collapses_excessive_newlines(self):
        ctx = StepContext(query="test")
        result = build_user_message("a\n\n\n\n\nb", ctx)
        assert "\n\n\n" not in result
