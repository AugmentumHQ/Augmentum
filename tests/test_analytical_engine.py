"""Tests for the UARF analytical engine — phases, routing, backtracking, handler."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)
from augmentum.modes.analytical.engine import AnalyticalEngine
from augmentum.modes.analytical.handler import AnalyticalHandler
from augmentum.modes.analytical.prompts import get_phase_prompt
from augmentum.modes.analytical.state import AnalyticalPhase, AnalyticalState, PhaseResult

# --- Mock Backend ---


@dataclass
class _CallRecord:
    """Tracks a single call to the mock backend."""

    system_prompt: str
    user_content: str


class MockAnalyticalBackend(ModelBackend):
    """Mock backend that returns canned responses based on phase prompts.

    Inspects the system prompt to determine which phase is being called
    and returns an appropriate canned response.
    """

    def __init__(
        self,
        *,
        assess_complexity: str = "moderate",
        verify_pass: bool = True,
        verify_confidence: float = 0.85,
        conclude_text: str = "Based on the analysis, the answer is 42.",
    ) -> None:
        self.assess_complexity = assess_complexity
        self.verify_pass = verify_pass
        self.verify_confidence = verify_confidence
        self.conclude_text = conclude_text
        self.calls: list[_CallRecord] = []
        self._call_count = 0

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        system_prompt = ""
        user_content = ""
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_content = msg.content

        self.calls.append(_CallRecord(system_prompt=system_prompt, user_content=user_content))
        self._call_count += 1

        content = self._generate_response(system_prompt)

        return InternalChatResponse(
            message=Message(role="assistant", content=content),
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=50, completion_tokens=100, total_tokens=150),
        )

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        # For streaming, call chat to get the full response, then chunk it
        response = await self.chat(request)
        content = response.message.content

        # Yield the content in small chunks
        chunk_size = 10
        for i in range(0, len(content), chunk_size):
            chunk_text = content[i : i + chunk_size]
            yield InternalStreamChunk(
                content_delta=chunk_text,
                role="assistant" if i == 0 else None,
                model=request.model,
            )

        yield InternalStreamChunk(
            content_delta="",
            finish_reason="stop",
            model=request.model,
            done=True,
        )

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails()

    def _generate_response(self, system_prompt: str) -> str:
        """Generate a canned response based on which phase prompt is detected."""
        prompt_lower = system_prompt.lower()

        if "classify this query" in prompt_lower:
            return (
                "TYPE: analytical\n"
                "DOMAIN: general\n"
                "REASONING_STEPS: 3\n"
                f"COMPLEXITY: {self.assess_complexity}\n"
                f"RATIONALE: This query requires {self.assess_complexity} analysis."
            )

        if "decompose this query" in prompt_lower:
            return (
                "KEY_CONCEPTS:\n"
                "- Concept A\n"
                "- Concept B\n\n"
                "UNKNOWNS:\n"
                "- The relationship between A and B\n\n"
                "CONSTRAINTS:\n"
                "- Must be logically consistent\n\n"
                "ASSUMPTIONS:\n"
                "- Standard conditions apply\n\n"
                "SUB_PROBLEMS:\n"
                "- Determine A\n"
                "- Determine B\n"
            )

        if "analyze what this query requires" in prompt_lower:
            # GATHER phase (merged IDENTIFY+RELEVANT for moderate)
            return (
                "KEY_CONCEPTS:\n"
                "- Concept A\n"
                "- Concept B\n\n"
                "RELEVANT_KNOWLEDGE:\n"
                "- A is related to B through mechanism X\n\n"
                "INFORMATION_GAPS:\n"
                "- None identified\n"
            )

        if "gather the specific evidence" in prompt_lower:
            return (
                "RELEVANT_KNOWLEDGE:\n"
                "- A is related to B through mechanism X\n"
                "- B has property Y\n\n"
                "APPLICABLE_METHODS:\n"
                "- Method 1: Direct analysis\n"
                "- Method 2: Comparative analysis\n\n"
                "INFORMATION_GAPS:\n"
                "- None identified\n"
            )

        if "solve this query" in prompt_lower or "answer this query" in prompt_lower:
            return (
                "STEP 1: Analyze concept A\n"
                "A has the following properties...\n\n"
                "STEP 2: Analyze concept B\n"
                "B relates to A through...\n\n"
                "PRELIMINARY_ANSWER: The answer is 42."
            )

        if "review this analysis for errors" in prompt_lower:
            verified = "yes" if self.verify_pass else "no"
            errors = "None" if self.verify_pass else "- Logical error in step 2"
            notes = "appears sound" if self.verify_pass else "has issues"
            return (
                "ERRORS_FOUND:\n"
                f"- {errors}\n\n"
                "UNSUPPORTED_CLAIMS:\n"
                "- None\n\n"
                "CONTRADICTIONS:\n"
                "- None\n\n"
                f"VERIFIED: {verified}\n"
                f"CONFIDENCE: {self.verify_confidence}\n"
                f"VERIFICATION_NOTES: Analysis {notes}."
            )

        if "write the final answer" in prompt_lower or "synthesize" in prompt_lower:
            return self.conclude_text

        if "answer this question directly" in prompt_lower:
            # RESPOND phase (merged APPLY+CONCLUDE for simple queries)
            return self.conclude_text

        # Fallback
        return "Unknown phase response."


class MockFailThenPassBackend(MockAnalyticalBackend):
    """Backend that fails verification on first attempt, passes on second."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._verify_call_count = 0

    def _generate_response(self, system_prompt: str) -> str:
        prompt_lower = system_prompt.lower()

        if "review this analysis for errors" in prompt_lower:
            self._verify_call_count += 1
            if self._verify_call_count == 1:
                # First verification fails
                return (
                    "ERRORS_FOUND:\n"
                    "- Logical error in step 2\n\n"
                    "UNSUPPORTED_CLAIMS:\n"
                    "- Claim about B is not supported\n\n"
                    "CONTRADICTIONS:\n"
                    "- None\n\n"
                    "VERIFIED: no\n"
                    "CONFIDENCE: 0.3\n"
                    "VERIFICATION_NOTES: Analysis has significant issues."
                )
            # Subsequent verifications pass
            return (
                "ERRORS_FOUND:\n"
                "- None\n\n"
                "UNSUPPORTED_CLAIMS:\n"
                "- None\n\n"
                "CONTRADICTIONS:\n"
                "- None\n\n"
                "VERIFIED: yes\n"
                "CONFIDENCE: 0.9\n"
                "VERIFICATION_NOTES: Revised analysis is sound."
            )

        return super()._generate_response(system_prompt)


class MockAlwaysFailVerifyBackend(MockAnalyticalBackend):
    """Backend that always fails verification — used to test max backtrack limit."""

    def _generate_response(self, system_prompt: str) -> str:
        prompt_lower = system_prompt.lower()

        if "review this analysis for errors" in prompt_lower:
            return (
                "ERRORS_FOUND:\n"
                "- Critical error found\n\n"
                "UNSUPPORTED_CLAIMS:\n"
                "- Multiple unsupported claims\n\n"
                "CONTRADICTIONS:\n"
                "- Contradiction between step 1 and step 3\n\n"
                "VERIFIED: no\n"
                "CONFIDENCE: 0.2\n"
                "VERIFICATION_NOTES: Analysis has critical issues."
            )

        return super()._generate_response(system_prompt)


# --- Helpers ---


def make_request(
    user_content: str,
    system_content: str = "",
    model: str = "llama3.1:8b",
) -> InternalChatRequest:
    messages = []
    if system_content:
        messages.append(Message(role="system", content=system_content))
    messages.append(Message(role="user", content=user_content))
    return InternalChatRequest(model=model, messages=messages)


# === State Tests ===


class TestAnalyticalState:
    def test_default_state(self):
        """Default state should have reasonable defaults."""
        state = AnalyticalState()
        assert state.query == ""
        assert state.complexity == "moderate"
        assert state.current_phase == AnalyticalPhase.ASSESS
        assert state.phase_results == {}
        assert state.backtrack_count == 0
        assert state.max_backtracks == 3
        assert state.facts_identified == []
        assert state.assumptions == []
        assert state.sub_tasks == []

    def test_phase_enum_values(self):
        """All phase enum values should be correct."""
        assert AnalyticalPhase.ASSESS.value == "assess"
        assert AnalyticalPhase.IDENTIFY.value == "identify"
        assert AnalyticalPhase.RELEVANT.value == "relevant"
        assert AnalyticalPhase.APPLY.value == "apply"
        assert AnalyticalPhase.VERIFY.value == "verify"
        assert AnalyticalPhase.CONCLUDE.value == "conclude"

    def test_phase_result_defaults(self):
        """PhaseResult should have reasonable defaults."""
        result = PhaseResult(phase=AnalyticalPhase.ASSESS, output="test")
        assert result.confidence == 0.0
        assert result.needs_backtrack is False
        assert result.backtrack_reason == ""
        assert result.tokens_used == 0


# === Prompt Tests ===


class TestPrompts:
    def test_assess_prompt_contains_query(self):
        """ASSESS system prompt should include the query (shared prefix moved
        the query out of user content into system for prefix-cache reuse)."""
        system, user = get_phase_prompt("assess", query="What is 2+2?")
        assert "What is 2+2?" in system
        assert "COMPLEXITY:" in system

    def test_identify_prompt_includes_assess_output(self):
        """IDENTIFY user content should include previous ASSESS output;
        query still appears but now in the shared system prefix."""
        system, user = get_phase_prompt(
            "identify",
            query="test query",
            assess_output="COMPLEXITY: moderate",
        )
        assert "COMPLEXITY: moderate" in user
        assert "test query" in system

    def test_verify_prompt_includes_apply_output(self):
        """VERIFY user content should include APPLY output, system has format."""
        system, user = get_phase_prompt(
            "verify",
            query="test",
            apply_output="PRELIMINARY_ANSWER: 42",
            identify_output="KEY_CONCEPTS: A",
        )
        assert "PRELIMINARY_ANSWER: 42" in user
        assert "VERIFIED:" in system
        assert "CONFIDENCE:" in system

    def test_simple_apply_prompt_used(self):
        """Simple queries should use the shorter APPLY prompt."""
        simple_sys, simple_user = get_phase_prompt("apply", query="test", is_simple=True)
        full_sys, full_user = get_phase_prompt(
            "apply", query="test", is_simple=False,
            relevant_output="some relevant info",
        )
        # Simple user content should not contain RELEVANT phase references
        assert "Evidence & Methods" not in simple_user
        assert "Evidence & Methods" in full_user

    def test_simple_conclude_prompt_used(self):
        """Simple queries should use the shorter CONCLUDE prompt."""
        simple_sys, simple_user = get_phase_prompt("conclude", query="test", is_simple=True)
        full_sys, full_user = get_phase_prompt(
            "conclude", query="test", is_simple=False,
            verify_output="VERIFIED: yes",
        )
        # Full user content includes verification section
        assert "Verification" not in simple_user
        assert "Verification" in full_user

    def test_backtrack_context_injected(self):
        """Backtrack context should be included in APPLY user content."""
        system, user = get_phase_prompt(
            "apply",
            query="test",
            backtrack_context="Previous attempt had errors in step 2.",
        )
        assert "Previous attempt had errors in step 2." in user

    def test_unknown_phase_returns_empty(self):
        """Unknown phase names should return empty tuple of strings."""
        result = get_phase_prompt("nonexistent", query="test")
        assert result == ("", "")


# === Engine Tests ===


class TestAnalyticalEngine:
    @pytest.mark.asyncio
    async def test_assess_extracts_complexity_simple(self):
        """ASSESS phase should correctly extract 'simple' complexity."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)
        request = make_request("What is 2+2?")

        result = await engine.process(request)

        assert result.complexity == "simple"
        assert "assess" in result.phase_results

    @pytest.mark.asyncio
    async def test_assess_extracts_complexity_moderate(self):
        """ASSESS phase should correctly extract 'moderate' complexity."""
        backend = MockAnalyticalBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        request = make_request("Compare the economic policies of two countries.")

        result = await engine.process(request)

        assert result.complexity == "moderate"

    @pytest.mark.asyncio
    async def test_assess_extracts_complexity_complex(self):
        """ASSESS phase should correctly extract 'complex' complexity."""
        backend = MockAnalyticalBackend(assess_complexity="complex")
        engine = AnalyticalEngine(backend)
        request = make_request("Derive the proof for Fermat's Last Theorem.")

        result = await engine.process(request)

        assert result.complexity == "complex"

    @pytest.mark.asyncio
    async def test_simple_query_takes_shortcut(self):
        """Simple queries should run ASSESS + merged RESPOND (populates apply+conclude)."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)
        request = make_request("What is 2+2?")

        result = await engine.process(request)

        # RESPOND merges APPLY+CONCLUDE into one call, but populates both keys
        assert "assess" in result.phase_results
        assert "apply" in result.phase_results
        assert "conclude" in result.phase_results
        assert "respond" in result.phase_results
        # Should NOT have these phases
        assert "identify" not in result.phase_results
        assert "relevant" not in result.phase_results
        assert "verify" not in result.phase_results

    @pytest.mark.asyncio
    async def test_simple_query_backend_call_count(self):
        """Simple queries should make exactly 2 backend calls (ASSESS + RESPOND)."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)
        # Use a query that doesn't trigger the heuristic ASSESS shortcut
        request = make_request("Is 2+2 equal to 4?")

        await engine.process(request)

        assert len(backend.calls) == 2

    @pytest.mark.asyncio
    async def test_moderate_query_runs_moderate_pipeline(self):
        """Moderate queries should run GATHER-based pipeline (5 phases stored)."""
        backend = MockAnalyticalBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        request = make_request("Compare X and Y in detail.")

        result = await engine.process(request)

        # Moderate: ASSESS → GATHER → APPLY → VERIFY → CONCLUDE
        # GATHER output is also copied to "identify" slot for compatibility
        assert "assess" in result.phase_results
        assert "gather" in result.phase_results
        assert "identify" in result.phase_results  # copied from gather
        assert "apply" in result.phase_results
        assert "verify" in result.phase_results
        assert "conclude" in result.phase_results

    @pytest.mark.asyncio
    async def test_moderate_query_backend_call_count(self):
        """Moderate queries should make exactly 4 backend calls (ASSESS, GATHER, APPLY, VERIFY)."""
        backend = MockAnalyticalBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        request = make_request("Compare X and Y.")

        await engine.process(request)

        # 4 real LLM calls: ASSESS, GATHER, APPLY, VERIFY
        # (IDENTIFY is copied from GATHER, CONCLUDE is copied from APPLY)
        assert len(backend.calls) == 4

    @pytest.mark.asyncio
    async def test_complex_query_runs_full_pipeline(self):
        """Complex queries should also run the full 6-phase pipeline."""
        backend = MockAnalyticalBackend(assess_complexity="complex")
        engine = AnalyticalEngine(backend)
        request = make_request("Derive a proof step by step.")

        result = await engine.process(request)

        assert len(result.phase_results) == 6
        assert result.complexity == "complex"

    @pytest.mark.asyncio
    async def test_verify_failure_triggers_backtrack(self):
        """When VERIFY fails, APPLY should be re-run with backtrack context."""
        backend = MockFailThenPassBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        request = make_request("Analyze this complex problem.")

        result = await engine.process(request)

        # Should have backtracked once
        assert result.backtrack_count == 1
        # Moderate: 4 calls (ASSESS, GATHER, APPLY, VERIFY) + 2 retried (APPLY + VERIFY) = 6
        assert len(backend.calls) == 6

    @pytest.mark.asyncio
    async def test_verify_failure_backtrack_includes_reason(self):
        """Backtracked APPLY should receive the verification failure reason."""
        backend = MockFailThenPassBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        request = make_request("Analyze this.")

        await engine.process(request)

        # Find the second APPLY call (the retried one)
        apply_calls = [
            c for c in backend.calls
            if "solve this query" in c.system_prompt.lower()
            or "answer this query" in c.system_prompt.lower()
        ]
        assert len(apply_calls) == 2
        # The second call should contain backtrack context in user content
        assert "Revision Guidance" in apply_calls[1].user_content

    @pytest.mark.asyncio
    async def test_max_backtrack_limit_respected(self):
        """Engine should stop backtracking after reaching max_backtracks."""
        backend = MockAlwaysFailVerifyBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        engine._state.max_backtracks = 2
        request = make_request("This will keep failing.")

        result = await engine.process(request)

        # Should not exceed max backtracks
        assert result.backtrack_count <= 2
        # Still produces a result (with the last attempt)
        assert "conclude" in result.phase_results

    @pytest.mark.asyncio
    async def test_phase_results_stored_in_state(self):
        """All phase results should be stored in engine state."""
        backend = MockAnalyticalBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        request = make_request("Test query")

        await engine.process(request)

        # Check state has all phases
        assert engine.state.query == "Test query"
        assert engine.state.complexity == "moderate"
        assert len(engine.state.phase_results) == 6

    @pytest.mark.asyncio
    async def test_phase_results_have_outputs(self):
        """Each phase result should have non-empty output."""
        backend = MockAnalyticalBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        request = make_request("Test query")

        result = await engine.process(request)

        for phase_name, phase_result in result.phase_results.items():
            assert phase_result.output, f"Phase {phase_name} has empty output"

    @pytest.mark.asyncio
    async def test_total_tokens_accumulated(self):
        """Total tokens should be the sum of all phase tokens."""
        backend = MockAnalyticalBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        request = make_request("Test")

        result = await engine.process(request)

        # Each call uses 150 tokens, moderate = 4 calls = 600
        assert result.total_tokens == 600

    @pytest.mark.asyncio
    async def test_conclusion_comes_from_conclude_phase(self):
        """The result conclusion should come from RESPOND (merged APPLY+CONCLUDE)."""
        backend = MockAnalyticalBackend(
            assess_complexity="simple",
            conclude_text="The final answer is 7.",
        )
        engine = AnalyticalEngine(backend)
        request = make_request("What is 3+4?")

        result = await engine.process(request)

        assert result.conclusion == "The final answer is 7."

    @pytest.mark.asyncio
    async def test_uses_model_from_request(self):
        """Engine should use the model specified in the original request."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)
        request = make_request("Test", model="mistral:7b")

        await engine.process(request)

        # ASSESS + RESPOND = 2 backend calls for simple queries
        assert len(backend.calls) == 2

    @pytest.mark.asyncio
    async def test_query_extracted_from_last_user_message(self):
        """Engine should extract the query from the last user message."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="Be helpful."),
                Message(role="user", content="First question"),
                Message(role="assistant", content="Answer 1"),
                Message(role="user", content="Second question"),
            ],
        )

        await engine.process(request)

        assert engine.state.query == "Second question"


# === Complexity Parsing Tests ===


class TestComplexityParsing:
    def test_parse_simple(self):
        assert AnalyticalEngine._parse_complexity("COMPLEXITY: simple") == "simple"

    def test_parse_moderate(self):
        assert AnalyticalEngine._parse_complexity("COMPLEXITY: moderate") == "moderate"

    def test_parse_complex(self):
        assert AnalyticalEngine._parse_complexity("COMPLEXITY: complex") == "complex"

    def test_parse_case_insensitive(self):
        assert AnalyticalEngine._parse_complexity("COMPLEXITY: SIMPLE") == "simple"
        assert AnalyticalEngine._parse_complexity("complexity: Complex") == "complex"

    def test_parse_embedded_in_text(self):
        text = "This is a moderate query.\nCOMPLEXITY: moderate\nSome more text."
        assert AnalyticalEngine._parse_complexity(text) == "moderate"

    def test_parse_missing_defaults_to_moderate(self):
        assert AnalyticalEngine._parse_complexity("No complexity here.") == "moderate"

    def test_parse_empty_defaults_to_moderate(self):
        assert AnalyticalEngine._parse_complexity("") == "moderate"


# === Confidence Parsing Tests ===


class TestConfidenceParsing:
    def test_parse_confidence_value(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 0.85") == 0.85

    def test_parse_confidence_one(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 1.0") == 1.0

    def test_parse_confidence_zero(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 0.0") == 0.0

    def test_parse_confidence_clamped_high(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 1.5") == 1.0

    def test_parse_confidence_missing(self):
        assert AnalyticalEngine._parse_confidence("No confidence here.") == 0.0

    def test_parse_confidence_embedded(self):
        text = "Some text\nCONFIDENCE: 0.72\nMore text"
        assert AnalyticalEngine._parse_confidence(text) == 0.72


# === Verified Parsing Tests ===


class TestVerifiedParsing:
    def test_parse_verified_yes(self):
        assert AnalyticalEngine._parse_verified("VERIFIED: yes") is True

    def test_parse_verified_no(self):
        assert AnalyticalEngine._parse_verified("VERIFIED: no") is False

    def test_parse_verified_case_insensitive(self):
        assert AnalyticalEngine._parse_verified("VERIFIED: YES") is True
        assert AnalyticalEngine._parse_verified("VERIFIED: No") is False

    def test_parse_verified_missing(self):
        assert AnalyticalEngine._parse_verified("No verified line.") is False


# === Handler Tests ===


class TestAnalyticalHandler:
    @pytest.mark.asyncio
    async def test_handle_returns_conclude_output(self):
        """Handler should return the CONCLUDE phase output as the response."""
        backend = MockAnalyticalBackend(
            assess_complexity="simple",
            conclude_text="The answer is 42.",
        )
        handler = AnalyticalHandler(backend)
        request = make_request("What is the answer?")

        response = await handler.handle(request)

        assert response.message.content == "The answer is 42."
        assert response.message.role == "assistant"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_handle_preserves_model(self):
        """Handler should preserve the model name in the response."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        handler = AnalyticalHandler(backend)
        request = make_request("Test", model="mistral:7b")

        response = await handler.handle(request)

        assert response.model == "mistral:7b"

    @pytest.mark.asyncio
    async def test_handle_includes_usage(self):
        """Handler should include token usage in the response."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        handler = AnalyticalHandler(backend)
        request = make_request("Test")

        response = await handler.handle(request)

        assert response.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_stream_emits_phase_indicators(self):
        """Streaming should emit structured phase metadata chunks."""
        backend = MockAnalyticalBackend(assess_complexity="moderate")
        handler = AnalyticalHandler(backend)
        request = make_request("Analyze this.")

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Should contain augmentum phase metadata chunks
        meta_chunks = [c for c in chunks if c.augmentum]
        assert len(meta_chunks) > 0

        # Should have phase status updates (moderate: ASSESS, GATHER, APPLY, VERIFY, CONCLUDE)
        phases_seen = {c.augmentum["phase"] for c in meta_chunks}
        assert "ASSESS" in phases_seen
        assert "CONCLUDE" in phases_seen

        # Should have running and complete statuses
        # (filter to status chunks — content delta chunks don't carry phase_status)
        status_chunks = [c for c in meta_chunks if "phase_status" in c.augmentum]
        statuses_seen = {c.augmentum["phase_status"] for c in status_chunks}
        assert "running" in statuses_seen
        assert "complete" in statuses_seen

    @pytest.mark.asyncio
    async def test_stream_emits_conclude_content(self):
        """Streaming should include the CONCLUDE phase content."""
        backend = MockAnalyticalBackend(
            assess_complexity="simple",
            conclude_text="Streamed final answer.",
        )
        handler = AnalyticalHandler(backend)
        request = make_request("What is this?")

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        content = "".join(c.content_delta for c in chunks)

        assert "Streamed final answer." in content

    @pytest.mark.asyncio
    async def test_stream_simple_uses_short_pipeline(self):
        """Simple streaming should use the short pipeline (no VERIFY)."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        handler = AnalyticalHandler(backend)
        request = make_request("Simple question?")

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        meta_chunks = [c for c in chunks if c.augmentum]
        phases_seen = {c.augmentum["phase"] for c in meta_chunks}

        # Simple pipeline: ASSESS, RESPOND (merged APPLY+CONCLUDE)
        assert "ASSESS" in phases_seen
        assert "RESPOND" in phases_seen or "CONCLUDE" in phases_seen
        # Complexity should be detected
        complexity_chunks = [c for c in meta_chunks if c.augmentum.get("complexity")]
        assert any(c.augmentum["complexity"] == "simple" for c in complexity_chunks)


# === Handler Factory Integration Tests ===


class TestHandlerFactory:
    def test_analytical_mode_returns_handler(self):
        """Handler factory should return AnalyticalHandler for analytical mode."""
        from unittest.mock import MagicMock

        from augmentum.classifier.router import Mode
        from augmentum.proxy.handler_factory import get_handler_for_mode

        backend = MockAnalyticalBackend()
        app_state = MagicMock()
        app_state.narrative_engines = {}

        handler = get_handler_for_mode(
            Mode.ANALYTICAL,
            backend,
            "ses_test",
            app_state,
        )

        assert isinstance(handler, AnalyticalHandler)


# === Verification Issues Extraction Tests ===


class TestVerificationIssues:
    def test_extracts_errors(self):
        output = (
            "ERRORS_FOUND:\n"
            "- Error in step 2\n"
            "- Error in step 4\n\n"
            "UNSUPPORTED_CLAIMS:\n"
            "- None\n\n"
            "CONTRADICTIONS:\n"
            "- None\n"
        )
        issues = AnalyticalEngine._extract_verification_issues(output)
        assert "Error in step 2" in issues
        assert "Error in step 4" in issues

    def test_extracts_all_sections(self):
        output = (
            "ERRORS_FOUND:\n"
            "- Logic error\n\n"
            "UNSUPPORTED_CLAIMS:\n"
            "- Claim X is not supported\n\n"
            "CONTRADICTIONS:\n"
            "- Step 1 contradicts step 3\n"
        )
        issues = AnalyticalEngine._extract_verification_issues(output)
        assert "Logic error" in issues
        assert "Claim X is not supported" in issues
        assert "Step 1 contradicts step 3" in issues

    def test_skips_none_sections(self):
        output = (
            "ERRORS_FOUND:\n"
            "- None\n\n"
            "UNSUPPORTED_CLAIMS:\n"
            "- None\n\n"
            "CONTRADICTIONS:\n"
            "- None\n"
        )
        issues = AnalyticalEngine._extract_verification_issues(output)
        assert issues == "Verification failed with low confidence."

    def test_fallback_message_when_no_sections(self):
        issues = AnalyticalEngine._extract_verification_issues("No structured output.")
        assert issues == "Verification failed with low confidence."


# === Edge Cases ===


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_user_message(self):
        """Engine should handle requests with no user message gracefully."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)
        request = InternalChatRequest(
            model="test",
            messages=[Message(role="system", content="Be helpful.")],
        )

        result = await engine.process(request)

        # Should still produce a result (query will be empty)
        assert result.conclusion is not None

    @pytest.mark.asyncio
    async def test_multiple_user_messages_uses_last(self):
        """Engine should use the last user message as the query."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="user", content="First"),
                Message(role="assistant", content="Response"),
                Message(role="user", content="Last question"),
            ],
        )

        await engine.process(request)

        assert engine.state.query == "Last question"

    @pytest.mark.asyncio
    async def test_backtrack_count_zero_when_verify_passes(self):
        """Backtrack count should be 0 when verification passes on first try."""
        backend = MockAnalyticalBackend(
            assess_complexity="moderate",
            verify_pass=True,
            verify_confidence=0.9,
        )
        engine = AnalyticalEngine(backend)
        request = make_request("Analyze this.")

        result = await engine.process(request)

        assert result.backtrack_count == 0

    @pytest.mark.asyncio
    async def test_max_backtracks_set_to_zero(self):
        """With max_backtracks=0, engine should never retry."""
        backend = MockAlwaysFailVerifyBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)
        engine._state.max_backtracks = 0
        request = make_request("Test")

        result = await engine.process(request)

        # Only 1 attempt at APPLY+VERIFY (no retries)
        assert result.backtrack_count == 0
        # Moderate: 4 calls: ASSESS + GATHER + APPLY + VERIFY
        assert len(backend.calls) == 4


# =====================================================================
# Conversation context tests
# =====================================================================


class TestBuildConversationContext:
    """Tests for _build_conversation_context extraction and formatting."""

    def test_no_prior_messages(self):
        """Single user message returns empty string — nothing to contextualize."""
        request = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="What is 2+2?")],
            stream=False,
        )
        result = AnalyticalEngine._build_conversation_context(request)
        assert result == ""

    def test_single_prior_turn(self):
        """One prior user+assistant pair is formatted correctly."""
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="user", content="What is the GDP of France?"),
                Message(role="assistant", content="France's GDP is about $3 trillion."),
                Message(role="user", content="How does that compare to Germany?"),
            ],
            stream=False,
        )
        result = AnalyticalEngine._build_conversation_context(request)
        assert "GDP of France" in result
        assert "$3 trillion" in result
        # Current query should NOT be in the context
        assert "Germany" not in result

    def test_multiple_prior_turns(self):
        """Multiple prior turns are included in chronological order."""
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="user", content="First question"),
                Message(role="assistant", content="First answer"),
                Message(role="user", content="Second question"),
                Message(role="assistant", content="Second answer"),
                Message(role="user", content="Third question"),
            ],
            stream=False,
        )
        result = AnalyticalEngine._build_conversation_context(request)
        assert "First question" in result
        assert "First answer" in result
        assert "Second question" in result
        assert "Second answer" in result
        # Current query excluded
        assert "Third question" not in result
        # Chronological order: first before second
        assert result.index("First question") < result.index("Second question")

    def test_max_turns_limit(self):
        """Only the most recent N turns are included."""
        messages = []
        for i in range(10):
            messages.append(Message(role="user", content=f"Question {i}"))
            messages.append(Message(role="assistant", content=f"Answer {i}"))
        messages.append(Message(role="user", content="Final question"))

        request = InternalChatRequest(model="test", messages=messages, stream=False)
        result = AnalyticalEngine._build_conversation_context(request, max_turns=3)
        # Should have turns 7, 8, 9 (most recent 3)
        assert "Question 7" in result
        assert "Answer 7" in result
        assert "Question 9" in result
        assert "Answer 9" in result
        # Older turns excluded
        assert "Question 6" not in result
        assert "Answer 6" not in result

    def test_max_chars_truncation(self):
        """Total context is capped at max_chars."""
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="user", content="A" * 500),
                Message(role="assistant", content="B" * 500),
                Message(role="user", content="Current query"),
            ],
            stream=False,
        )
        result = AnalyticalEngine._build_conversation_context(request, max_chars=100)
        # Should be truncated (100 chars + truncation marker)
        assert "[... earlier conversation truncated]" in result

    def test_long_individual_messages_truncated(self):
        """Individual messages longer than 300 chars get truncated."""
        long_msg = "X" * 500
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="user", content=long_msg),
                Message(role="assistant", content="Short reply"),
                Message(role="user", content="Follow-up"),
            ],
            stream=False,
        )
        result = AnalyticalEngine._build_conversation_context(request)
        # The long message should be truncated to ~300 chars with "..."
        assert "..." in result
        assert long_msg not in result  # Full message should not appear

    def test_braces_escaped(self):
        """Curly braces in conversation are escaped for str.format() safety."""
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="user", content="What does {key: value} mean?"),
                Message(role="assistant", content="It's a dict literal."),
                Message(role="user", content="Thanks"),
            ],
            stream=False,
        )
        result = AnalyticalEngine._build_conversation_context(request)
        assert "{{key: value}}" in result  # Braces escaped
        # Verify the result is safe for str.format() — no KeyError
        result.format(conversation_context="test")

    def test_system_messages_excluded(self):
        """System messages from the original request are not included."""
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="What's up?"),
            ],
            stream=False,
        )
        result = AnalyticalEngine._build_conversation_context(request)
        assert "helpful assistant" not in result
        assert "Hello" in result
        assert "Hi there" in result

    def test_empty_messages_list(self):
        """Empty messages list returns empty string."""
        request = InternalChatRequest(model="test", messages=[], stream=False)
        result = AnalyticalEngine._build_conversation_context(request)
        assert result == ""

    def test_only_system_and_one_user(self):
        """System + one user message = no prior context."""
        request = InternalChatRequest(
            model="test",
            messages=[
                Message(role="system", content="System prompt"),
                Message(role="user", content="First and only message"),
            ],
            stream=False,
        )
        result = AnalyticalEngine._build_conversation_context(request)
        assert result == ""


class TestConversationContextInPrompts:
    """Tests that conversation_context flows through to the shared system
    prefix (moved there from user content for prefix-cache reuse)."""

    def test_assess_prompt_includes_context(self):
        """ASSESS system prompt should include conversation_context."""
        system, user = get_phase_prompt(
            "assess",
            query="How does that compare?",
            conversation_context="\n## Conversation History\nUser: What is X?\nAssistant: X is 42.\n",
        )
        assert "Conversation History" in system
        assert "X is 42" in system

    def test_apply_simple_prompt_includes_context(self):
        """APPLY (simple) system prompt should include conversation_context."""
        system, user = get_phase_prompt(
            "apply",
            query="And the other one?",
            assess_output="TYPE: factual\nCOMPLEXITY: simple",
            is_simple=True,
            conversation_context="\n## Conversation History\nUser: First?\nAssistant: First answer.\n",
        )
        assert "Conversation History" in system
        assert "First answer" in system

    def test_conclude_prompt_includes_context(self):
        """CONCLUDE system prompt should include conversation_context."""
        system, user = get_phase_prompt(
            "conclude",
            query="And the follow-up?",
            apply_output="PRELIMINARY_ANSWER: 42",
            conversation_context="\n## Conversation History\nUser: Setup?\nAssistant: Setup done.\n",
        )
        assert "Conversation History" in system
        assert "Setup done" in system

    def test_verify_prompt_no_context(self):
        """VERIFY does NOT receive conversation context (not relevant for review)."""
        system, user = get_phase_prompt(
            "verify",
            query="Check this",
            apply_output="STEP 1: ...",
            identify_output="KEY_CONCEPTS: ...",
            conversation_context="\n## Conversation History\nUser: Prior?\nAssistant: Prior.\n",
        )
        # VERIFY doesn't pass through conversation_context
        assert "Conversation History" not in user
        assert "Conversation History" not in system

    def test_empty_context_no_pollution(self):
        """Empty conversation_context doesn't add spurious text."""
        system, user = get_phase_prompt("assess", query="Simple question")
        assert "Conversation History" not in user
        assert "Conversation History" not in system


class TestConversationContextIntegration:
    """Integration test: conversation context flows through the full pipeline."""

    @pytest.mark.asyncio
    async def test_multi_turn_context_in_assess(self):
        """Multi-turn request puts conversation context in ASSESS system prompt."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)

        request = InternalChatRequest(
            model="test-model",
            messages=[
                Message(role="user", content="What is the GDP of France?"),
                Message(role="assistant", content="France GDP is about $3 trillion."),
                Message(role="user", content="How does that compare to Germany?"),
            ],
            stream=False,
        )
        await engine.process(request)

        # ASSESS is the first call — conversation context lives in the
        # shared system prefix now, not the per-phase user content.
        assess_call = backend.calls[0]
        assert "Conversation History" in assess_call.system_prompt
        assert "GDP of France" in assess_call.system_prompt
        assert "$3 trillion" in assess_call.system_prompt

    @pytest.mark.asyncio
    async def test_single_turn_no_context(self):
        """Single-turn request has no conversation context in prompts."""
        backend = MockAnalyticalBackend(assess_complexity="simple")
        engine = AnalyticalEngine(backend)

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="What is 2+2?")],
            stream=False,
        )
        await engine.process(request)

        # No conversation history injected anywhere — the shared prefix
        # only populates that section when conversation_context is non-empty.
        assess_call = backend.calls[0]
        assert "Conversation History" not in assess_call.user_content
        assert "Conversation History" not in assess_call.system_prompt


# ===========================================================================
# Tier 1.4 — multi-phase prefix reuse
# ===========================================================================


class TestMultiPhasePrefixReuse:
    """Guard the invariant that the shared prefix is byte-identical across
    phases of one UARF run — otherwise llama-server's cache_prompt can't
    reuse KV across phases and we lose the 3-5x speedup."""

    def test_shared_prefix_identical_across_phases(self):
        """System prompts for assess/identify/apply/conclude all start with
        the same shared prefix when datetime_ctx is pinned."""
        from augmentum.modes.analytical.prompts import build_shared_prefix

        dt = "2026-04-18 12:00:00 UTC"
        query = "Compare X and Y in depth."
        conv = "\n## Conversation History\nUser: hi\nAssistant: hello\n"
        expected = build_shared_prefix(dt, query, conv, include_conversation=True)

        phases = ["assess", "identify", "relevant", "apply", "conclude"]
        prompts = [
            get_phase_prompt(
                p, query=query, conversation_context=conv,
                datetime_ctx=dt,
                # Plausible prior-phase outputs so later phases don't hit
                # the "empty user" fallback path unrelated to this test.
                assess_output="COMPLEXITY: moderate",
                identify_output="KEY_CONCEPTS: x",
                relevant_output="FACTS: ...",
                apply_output="ANSWER: ...",
            )[0]
            for p in phases
        ]
        for p, sys in zip(phases, prompts, strict=True):
            assert sys.startswith(expected), (
                f"phase {p} system prompt did not start with shared prefix"
            )

    def test_verify_prefix_is_proper_prefix_of_full(self):
        """VERIFY suppresses conversation_context historically; keep that —
        its shared prefix should still be a proper prefix of the full-prefix
        variant so token-level KV sharing works up to the divergence point."""
        from augmentum.modes.analytical.prompts import build_shared_prefix

        dt = "2026-04-18 12:00:00 UTC"
        query = "Compare X and Y."
        conv = "\n## Conversation History\nUser: hi\nAssistant: hi\n"
        full = build_shared_prefix(dt, query, conv, include_conversation=True)
        verify = build_shared_prefix(dt, query, conv, include_conversation=False)
        assert full.startswith(verify.rstrip("\n---\n"))

    def test_datetime_drift_breaks_prefix_sharing(self):
        """Two phases called with DIFFERENT datetime_ctx strings no longer
        share a prefix — a regression test showing why the engine must pin."""
        dt_a = "2026-04-18 12:00:00 UTC"
        dt_b = "2026-04-18 12:00:01 UTC"  # one second later
        sys_a, _ = get_phase_prompt("assess", query="q", datetime_ctx=dt_a)
        sys_b, _ = get_phase_prompt("assess", query="q", datetime_ctx=dt_b)
        assert sys_a != sys_b

    @pytest.mark.asyncio
    async def test_engine_pins_datetime_across_phases(self):
        """AnalyticalEngine captures one datetime_ctx in __init__ and passes
        it to every phase, so all phase system prompts share an identical
        prefix in real pipeline runs."""
        backend = MockAnalyticalBackend(assess_complexity="moderate")
        engine = AnalyticalEngine(backend)

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Compare X and Y.")],
            stream=False,
        )
        await engine.process(request)

        # Extract the datetime substring from each phase's system prompt.
        # The prefix layout starts with the pinned datetime_ctx value.
        datetimes = {
            call.system_prompt.split("\n\n")[0]
            for call in backend.calls
        }
        assert len(datetimes) == 1, (
            f"phases saw different datetimes: {datetimes} — prefix cache "
            "cannot reuse across phases"
        )
