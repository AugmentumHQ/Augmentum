"""Tests for analytical mode handler, engine, state, and prompts."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.modes.analytical.state import (
    AnalyticalPhase,
    AnalyticalResult,
    AnalyticalState,
    PhaseResult,
    ToolCallRecord,
)


class TestAnalyticalPhaseEnum:
    """UARF phase enum values."""

    def test_all_phases_exist(self):
        assert AnalyticalPhase.ASSESS.value == "assess"
        assert AnalyticalPhase.IDENTIFY.value == "identify"
        assert AnalyticalPhase.GATHER.value == "gather"
        assert AnalyticalPhase.RELEVANT.value == "relevant"
        assert AnalyticalPhase.APPLY.value == "apply"
        assert AnalyticalPhase.VERIFY.value == "verify"
        assert AnalyticalPhase.CONCLUDE.value == "conclude"
        assert AnalyticalPhase.RESPOND.value == "respond"


class TestAnalyticalState:
    """State tracking for the UARF pipeline."""

    def test_initial_state(self):
        state = AnalyticalState(query="What is gravity?")
        assert state.query == "What is gravity?"
        assert state.complexity == "moderate"
        assert state.current_phase == AnalyticalPhase.ASSESS
        assert state.backtrack_count == 0

    def test_state_tracks_phase_results(self):
        state = AnalyticalState(query="test")
        result = PhaseResult(
            phase=AnalyticalPhase.ASSESS,
            output="TYPE: factual\nCOMPLEXITY: simple",
            confidence=0.9,
        )
        state.phase_results["assess"] = result
        assert "assess" in state.phase_results
        assert state.phase_results["assess"].confidence == 0.9

    def test_state_search_fields_default_empty(self):
        state = AnalyticalState()
        assert state.needs_search is False
        assert state.search_queries == []
        assert state.search_context == ""

    def test_state_tool_calls_list(self):
        state = AnalyticalState()
        record = ToolCallRecord(
            phase="apply",
            tool_name="calculator",
            input_data={"expression": "2+2"},
            output="4",
            success=True,
        )
        state.tool_calls.append(record)
        assert len(state.tool_calls) == 1
        assert state.tool_calls[0].tool_name == "calculator"


class TestPhaseResult:
    """Phase result dataclass behavior."""

    def test_default_confidence(self):
        result = PhaseResult(phase=AnalyticalPhase.ASSESS, output="test")
        assert result.confidence == 0.0
        assert result.needs_backtrack is False

    def test_backtrack_flag(self):
        result = PhaseResult(
            phase=AnalyticalPhase.VERIFY,
            output="contradictions found",
            needs_backtrack=True,
            backtrack_reason="Missing data",
        )
        assert result.needs_backtrack is True
        assert result.backtrack_reason == "Missing data"


class TestAnalyticalResult:
    """Final pipeline result."""

    def test_result_fields(self):
        result = AnalyticalResult(
            conclusion="The answer is 42",
            phase_results={},
            complexity="simple",
            total_tokens=150,
        )
        assert result.conclusion == "The answer is 42"
        assert result.complexity == "simple"


class TestPhasePrompts:
    """Phase prompt generation."""

    def test_get_phase_prompt_assess(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt
        system, user = get_phase_prompt("assess", query="What is gravity?")
        assert "classify" in system.lower() or "complexity" in system.lower()
        # Query now lives in the shared prefix (system-side) so phases can
        # share KV cache across the UARF pipeline.
        assert "gravity" in system

    def test_get_phase_prompt_apply(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt
        system, user = get_phase_prompt(
            "apply",
            query="What is 2+2?",
            assess_output="TYPE: mathematical\nCOMPLEXITY: simple",
        )
        assert len(system) > 0
        assert "2+2" in system  # query in shared prefix, not user

    def test_get_phase_prompt_verify(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt
        system, user = get_phase_prompt(
            "verify",
            query="test",
            apply_output="The answer is 4",
        )
        assert len(system) > 0

    def test_get_phase_prompt_conclude(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt
        system, user = get_phase_prompt(
            "conclude",
            query="What is gravity?",
            apply_output="Gravity is a force...",
            verify_output="Looks correct.",
        )
        assert len(system) > 0

    def test_scope_search_context_apply_gets_full(self):
        from augmentum.modes.analytical.prompts import scope_search_context
        ctx = "Full search results here"
        assert scope_search_context(ctx, "apply") == ctx

    def test_scope_search_context_assess_gets_nothing(self):
        from augmentum.modes.analytical.prompts import scope_search_context
        ctx = "Full search results here"
        assert scope_search_context(ctx, "assess") == ""


class TestAnalyticalEngineImport:
    """Engine construction and constants."""

    def test_engine_constants(self):
        from augmentum.modes.analytical.engine import (
            _MAX_PHASE_RETRIES,
            _CONFIDENCE_THRESHOLD,
            _MAX_TOOL_CALLS_PER_PHASE,
        )
        assert _MAX_PHASE_RETRIES == 2
        assert _CONFIDENCE_THRESHOLD == 0.5
        assert _MAX_TOOL_CALLS_PER_PHASE == 3

    def test_get_max_phase_retries(self):
        from augmentum.modes.analytical.engine import _get_max_phase_retries
        result = _get_max_phase_retries()
        assert isinstance(result, int)
        assert result >= 1

    def test_get_confidence_threshold(self):
        from augmentum.modes.analytical.engine import _get_confidence_threshold
        result = _get_confidence_threshold()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
