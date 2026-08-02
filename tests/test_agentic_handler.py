"""Tests for agentic mode handler, planner, and working memory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.modes.agentic.planner import (
    PLAN_SYSTEM_PROMPT,
    mark_current_step,
    parse_plan,
    plan_to_context,
    update_plan_step,
)
from augmentum.modes.agentic.working_memory import WorkingMemory


class TestParsePlan:
    """Plan parsing from LLM output."""

    def test_parse_standard_plan(self):
        raw = (
            "## Task: Research climate change\n\n"
            "- [ ] 1. Search for recent climate data\n"
            "- [ ] 2. Analyze trends\n"
            "- [ ] 3. Summarize findings\n"
        )
        title, steps = parse_plan(raw)
        assert title == "Research climate change"
        assert len(steps) == 3
        assert "Search" in steps[0]
        assert "Analyze" in steps[1]
        assert "Summarize" in steps[2]

    def test_parse_plan_fallback_numbered(self):
        raw = "1. First step\n2. Second step\n3. Third step"
        title, steps = parse_plan(raw)
        assert len(steps) == 3
        assert "First" in steps[0]

    def test_parse_plan_empty_returns_empty(self):
        title, steps = parse_plan("")
        assert title == ""
        assert steps == []

    def test_parse_plan_title_fallback_from_first_step(self):
        raw = "- [ ] 1. Do something important and meaningful"
        title, steps = parse_plan(raw)
        assert len(steps) == 1
        # Title falls back to first step (truncated)
        assert "something" in title.lower()


class TestUpdatePlanStep:
    """Plan step marking."""

    def test_mark_step_complete(self):
        plan = "- [ ] 1. Step one\n- [ ] 2. Step two\n- [ ] 3. Step three"
        updated = update_plan_step(plan, 0)
        assert "[x]" in updated.split("\n")[0]
        assert "[ ]" in updated.split("\n")[1]

    def test_mark_step_with_note(self):
        plan = "- [ ] 1. Step one\n- [ ] 2. Step two"
        updated = update_plan_step(plan, 0, note="done in 3s")
        assert "(done in 3s)" in updated.split("\n")[0]

    def test_mark_already_complete_no_change(self):
        plan = "- [x] 1. Step one\n- [ ] 2. Step two"
        updated = update_plan_step(plan, 0)
        # Already complete, should stay the same
        assert updated == plan


class TestMarkCurrentStep:
    """CURRENT marker for attention anchoring."""

    def test_mark_adds_current_marker(self):
        plan = "- [ ] 1. Step one\n- [ ] 2. Step two\n- [ ] 3. Step three"
        result = mark_current_step(plan, 1)
        lines = result.split("\n")
        assert "CURRENT" not in lines[0]
        assert "CURRENT" in lines[1]
        assert "CURRENT" not in lines[2]

    def test_mark_removes_previous_marker(self):
        plan = "- [ ] 1. Step one <- CURRENT\n- [ ] 2. Step two"
        # The function strips "← CURRENT" (not "<-"), so use the real marker
        plan_with_marker = "- [ ] 1. Step one \u2190 CURRENT\n- [ ] 2. Step two"
        result = mark_current_step(plan_with_marker, 1)
        lines = result.split("\n")
        assert "CURRENT" not in lines[0]
        assert "CURRENT" in lines[1]

    def test_mark_skips_completed_steps(self):
        plan = "- [x] 1. Step one\n- [ ] 2. Step two"
        result = mark_current_step(plan, 1)
        # Step 1 is complete, so CURRENT should not appear on it
        assert "CURRENT" not in result.split("\n")[0]


class TestPlanToContext:
    """Plan formatting for context injection."""

    def test_non_empty_plan(self):
        plan = "- [ ] 1. Step one"
        ctx = plan_to_context(plan)
        assert "Current Task Plan" in ctx
        assert "Step one" in ctx

    def test_empty_plan_returns_empty(self):
        assert plan_to_context("") == ""
        assert plan_to_context("   ") == ""


class TestPlanSystemPrompt:
    """Plan system prompt constant."""

    def test_prompt_has_instructions(self):
        assert "planner" in PLAN_SYSTEM_PROMPT.lower()
        assert "checklist" in PLAN_SYSTEM_PROMPT.lower()


class TestWorkingMemory:
    """Cross-step working memory accumulation."""

    def test_initial_state(self):
        wm = WorkingMemory(goal="Write a report")
        assert wm.goal == "Write a report"
        assert wm.total_tool_calls == 0
        assert wm.total_llm_calls == 0
        assert wm.all_step_names == []
        assert wm.last_output == ""

    def test_record_generative_output(self):
        wm = WorkingMemory(goal="test")
        wm.record_generative_output("Draft", "Here is the draft content.")
        assert wm.total_llm_calls == 1
        assert "Draft" in wm.all_step_names
        assert wm.last_output == "Here is the draft content."

    def test_record_chain_results(self):
        wm = WorkingMemory(goal="test")
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.output = "Search result content"
        mock_result.tool_name = "web_search"
        wm.record_chain_results("Research", {0: mock_result})
        assert wm.total_tool_calls == 1
        assert "Research" in wm.all_step_names

    def test_build_context_for_step(self):
        wm = WorkingMemory(goal="test")
        wm.record_generative_output("Plan", "Step 1: Research\nStep 2: Draft")
        wm.record_generative_output("Research", "Found some data about climate.")
        context = wm.build_context_for_step("Draft")
        assert "Prior Results" in context
        assert "Plan" in context
        assert "Research" in context

    def test_build_context_empty_memory(self):
        wm = WorkingMemory(goal="test")
        context = wm.build_context_for_step("Draft")
        assert context == ""

    def test_get_step_output(self):
        wm = WorkingMemory(goal="test")
        wm.record_generative_output("Draft", "Draft content here")
        assert wm.get_step_output("Draft") == "Draft content here"
        assert wm.get_step_output("Nonexistent") == ""

    def test_record_artifact(self):
        wm = WorkingMemory(goal="test")
        wm.record_artifact({"display_name": "report.pdf", "format": "pdf"})
        assert len(wm._artifacts) == 1

    def test_to_synthesis_context(self):
        wm = WorkingMemory(goal="Write a report about AI")
        wm.record_generative_output("Draft", "AI is transforming...")
        wm.record_artifact({"display_name": "report.pdf", "format": "pdf"})
        ctx = wm.to_synthesis_context()
        assert "Write a report about AI" in ctx
        assert "Draft" in ctx
        assert "report.pdf" in ctx

    def test_format_for_plan_context(self):
        wm = WorkingMemory(goal="test")
        wm.record_generative_output("Step 1", "Done step 1")
        wm.record_generative_output("Step 2", "Done step 2")
        formatted = wm.format_for_plan_context()
        assert "Completed" in formatted
        assert "Step 1" in formatted
        assert "Step 2" in formatted

    def test_format_for_plan_context_empty(self):
        wm = WorkingMemory(goal="test")
        assert wm.format_for_plan_context() == ""
