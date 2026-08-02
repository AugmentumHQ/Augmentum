"""Tests for UARF prompt improvements — confidence parsing, few-shot,
message structure, heuristic ASSESS, and Tier 3 tool prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.modes.analytical.engine import AnalyticalEngine
from augmentum.modes.analytical.prompts import (
    get_phase_prompt,
    get_tool_prompt_section,
)

# === Confidence Parsing ===


class TestConfidenceParsing:
    """Tests for the robust _parse_confidence() cascade."""

    def test_decimal_standard(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 0.85") == 0.85

    def test_decimal_one(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 1.0") == 1.0

    def test_decimal_zero(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 0") == 0.0

    def test_decimal_one_bare(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 1") == 1.0

    def test_percentage(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 85%") == 0.85

    def test_percentage_100(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 100%") == 1.0

    def test_percentage_0(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 0%") == 0.0

    def test_fraction_decimal(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 0.85/1.0") == 0.85

    def test_fraction_integer(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 85/100") == 0.85

    def test_bare_integer_50(self):
        """Bare integer >1 should be treated as percentage."""
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 50") == 0.5

    def test_bare_integer_85(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 85") == 0.85

    def test_clamp_over_1(self):
        """Values > 1.0 after conversion should be clamped to 1.0."""
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 150%") == 1.0

    def test_missing_confidence_returns_0(self):
        assert AnalyticalEngine._parse_confidence("no confidence here") == 0.0

    def test_case_insensitive(self):
        assert AnalyticalEngine._parse_confidence("confidence: 0.9") == 0.9

    def test_fraction_zero_denominator(self):
        """Zero denominator should not crash — falls back to bare integer."""
        result = AnalyticalEngine._parse_confidence("CONFIDENCE: 5/0")
        assert 0.0 <= result <= 1.0

    def test_embedded_in_text(self):
        """Confidence line embedded in longer verify output."""
        text = (
            "ERRORS_FOUND:\n- None\n\n"
            "VERIFIED: yes\n"
            "CONFIDENCE: 0.92\n"
            "VERIFICATION_NOTES: Looks good."
        )
        assert AnalyticalEngine._parse_confidence(text) == 0.92

    def test_percentage_with_space(self):
        assert AnalyticalEngine._parse_confidence("CONFIDENCE: 75 %") == 0.75


# === Few-Shot Examples ===


class TestFewShotExamples:
    """Tests that prompts contain examples and key structural elements."""

    def test_assess_has_examples(self):
        system, user = get_phase_prompt("assess", query="test")
        # Examples are inline in the prompt
        assert "COMPLEXITY: simple" in system
        assert "COMPLEXITY: complex" in system

    def test_verify_has_output_format(self):
        system, user = get_phase_prompt(
            "verify", query="test", apply_output="test",
        )
        assert "CONFIDENCE:" in system
        assert "VERIFIED:" in system
        assert "ERRORS_FOUND:" in system

    def test_verify_has_review_framing(self):
        system, user = get_phase_prompt(
            "verify", query="test", apply_output="test",
        )
        assert "review" in system.lower()
        assert "errors" in system.lower()

    def test_verify_has_review_identity(self):
        """Verify prompt should identify the review role."""
        system, user = get_phase_prompt(
            "verify", query="test", apply_output="test",
        )
        assert "review this analysis" in system.lower()


# === Message Structure ===


class TestMessageStructure:
    """Tests that get_phase_prompt returns (system, user) with correct separation."""

    def test_returns_tuple(self):
        result = get_phase_prompt("assess", query="test")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_has_instructions(self):
        system, user = get_phase_prompt("assess", query="test")
        # System prompt contains the classification instructions
        assert "classify" in system.lower()
        assert "COMPLEXITY:" in system

    def test_user_has_query(self):
        system, user = get_phase_prompt("assess", query="What is the meaning of life?")
        assert "What is the meaning of life?" in user
        assert "What is the meaning of life?" not in system

    def test_query_not_in_system(self):
        """Query data should only be in user content, not system."""
        system, user = get_phase_prompt(
            "identify", query="unique_query_xyzzy",
            assess_output="COMPLEXITY: moderate",
        )
        assert "unique_query_xyzzy" not in system
        assert "unique_query_xyzzy" in user

    def test_phase_data_in_user(self):
        """Prior phase output should be in user content."""
        system, user = get_phase_prompt(
            "apply", query="test",
            identify_output="KEY_CONCEPTS: alpha",
            relevant_output="RELEVANT_KNOWLEDGE: beta",
        )
        assert "KEY_CONCEPTS: alpha" in user
        assert "RELEVANT_KNOWLEDGE: beta" in user
        assert "KEY_CONCEPTS: alpha" not in system

    def test_conversation_context_in_user(self):
        system, user = get_phase_prompt(
            "assess", query="test",
            conversation_context="## Conversation History\nprior chat",
        )
        assert "Conversation History" in user
        assert "Conversation History" not in system

    def test_search_context_in_user(self):
        # Use APPLY which gets full search context
        system, user = get_phase_prompt(
            "apply", query="test",
            identify_output="concepts",
            relevant_output="info",
            search_context="## Web Search Results\nresults here",
        )
        assert "Web Search Results" in user
        assert "Web Search Results" not in system

    def test_backtrack_context_in_user(self):
        system, user = get_phase_prompt(
            "apply", query="test",
            backtrack_context="## Revision Guidance\nfix step 2",
        )
        assert "Revision Guidance" in user
        assert "Revision Guidance" not in system

    def test_simple_conclude_no_verify(self):
        """Simple conclude should not include verification output."""
        system, user = get_phase_prompt(
            "conclude", query="test",
            apply_output="analysis here",
            verify_output="VERIFIED: yes",
            is_simple=True,
        )
        assert "analysis here" in user
        assert "VERIFIED: yes" not in user

    def test_full_conclude_has_verify(self):
        """Full conclude should include verification output."""
        system, user = get_phase_prompt(
            "conclude", query="test",
            apply_output="analysis here",
            verify_output="VERIFIED: yes",
            is_simple=False,
        )
        assert "analysis here" in user
        assert "VERIFIED: yes" in user

    def test_unknown_phase_returns_empty_tuple(self):
        assert get_phase_prompt("bogus", query="test") == ("", "")

    def test_all_phases_have_role_identity(self):
        """Each phase system prompt should contain a key directive phrase."""
        roles = {
            "assess": "classify this query",
            "identify": "decompose this query",
            "relevant": "gather the specific evidence",
            "apply": "solve this query",
            "verify": "review this analysis",
            "conclude": "write the final answer",
        }
        for phase, phrase in roles.items():
            system, _ = get_phase_prompt(phase, query="test")
            assert phrase in system.lower(), f"{phase} system prompt missing '{phrase}'"


# === Tier 3 Tool Prompt ===


@dataclass
class _MockTool:
    name: str
    description: str
    input_schema: dict | None = None


class TestTier3ToolPrompt:
    """Tests for the concise get_tool_prompt_section()."""

    def test_empty_tools_returns_empty(self):
        assert get_tool_prompt_section([]) == ""

    def test_has_function_signature(self):
        tools = [_MockTool(name="web_search", description="Search the web",
                           input_schema={"required": ["query"], "properties": {"query": {"type": "string"}}})]
        prompt = get_tool_prompt_section(tools)
        assert "web_search(query: str)" in prompt
        assert "call tools as functions" in prompt.lower() or "Call a tool" in prompt

    def test_lists_tool(self):
        tools = [_MockTool(name="calculator", description="Do math",
                           input_schema={"required": ["expression"], "properties": {"expression": {"type": "string"}}})]
        prompt = get_tool_prompt_section(tools)
        assert "calculator(expression: str)" in prompt
        assert "Do math" in prompt

    def test_web_search_example(self):
        tools = [_MockTool(name="web_search", description="Search",
                           input_schema={"required": ["query"], "properties": {"query": {"type": "string"}}})]
        prompt = get_tool_prompt_section(tools)
        assert 'web_search(query="' in prompt
        assert "Seattle WA" in prompt

    def test_fallback_example_for_unknown_tool(self):
        tools = [_MockTool(name="custom_tool", description="Custom",
                           input_schema={"required": ["input"], "properties": {"input": {"type": "string"}}})]
        prompt = get_tool_prompt_section(tools)
        assert 'custom_tool(input="' in prompt

    def test_concise_no_redundant_emphasis(self):
        """The prompt should not have redundant emphasis phrases."""
        tools = [_MockTool(name="web_search", description="Search",
                           input_schema={"required": ["query"], "properties": {"query": {}}})]
        prompt = get_tool_prompt_section(tools)
        # Old prompt had these — they should be gone
        assert "STOP and call" not in prompt
        assert "Rules:" not in prompt
        assert "IMPORTANT:" not in prompt

    def test_section_header_is_tools(self):
        """Section header should be '## Tools' not '## Available Tools'."""
        tools = [_MockTool(name="web_search", description="Search",
                           input_schema={"required": ["query"], "properties": {"query": {}}})]
        prompt = get_tool_prompt_section(tools)
        assert "## Tools" in prompt


# === Heuristic ASSESS ===


class TestHeuristicAssess:
    """Tests for the _heuristic_assess() static method."""

    def test_simple_what_is(self):
        assert AnalyticalEngine._heuristic_assess("What is the capital of France?") == "simple"

    def test_simple_who_is(self):
        assert AnalyticalEngine._heuristic_assess("Who is Albert Einstein?") == "simple"

    def test_simple_where_is(self):
        assert AnalyticalEngine._heuristic_assess("Where is Tokyo?") == "simple"

    def test_simple_when_was(self):
        assert AnalyticalEngine._heuristic_assess("When was WWII?") == "simple"

    def test_simple_define(self):
        assert AnalyticalEngine._heuristic_assess("Define photosynthesis") == "simple"

    def test_simple_too_long(self):
        """Long questions starting with 'what is' should not be simple."""
        q = "What is the relationship between quantum mechanics and general relativity in modern physics?"
        assert AnalyticalEngine._heuristic_assess(q) is None

    def test_complex_multiple_questions(self):
        assert AnalyticalEngine._heuristic_assess(
            "What is GDP? And how does it compare to GNP?"
        ) == "complex"

    def test_complex_compare_long(self):
        q = (
            "Compare the economic policies of the United States and the European Union "
            "regarding artificial intelligence regulation and their impact on startups"
        )
        assert AnalyticalEngine._heuristic_assess(q) == "complex"

    def test_ambiguous_returns_none(self):
        """Ambiguous queries should return None (fall through to LLM)."""
        assert AnalyticalEngine._heuristic_assess("Explain quantum computing") is None

    def test_compare_short_returns_none(self):
        """Short compare query should be ambiguous, not complex."""
        assert AnalyticalEngine._heuristic_assess("Compare X and Y") is None

    def test_case_insensitive(self):
        assert AnalyticalEngine._heuristic_assess("WHAT IS THE SPEED OF LIGHT?") == "simple"


# === Preamble Consolidation ===


class TestPromptStructure:
    """Tests that prompts have concise, directive-first structure."""

    def test_assess_starts_with_directive(self):
        system, _ = get_phase_prompt("assess", query="test")
        # Date/time prefix is injected, directive follows after
        assert "current date:" in system.lower()
        assert "classify" in system.lower()

    def test_identify_starts_with_directive(self):
        system, _ = get_phase_prompt("identify", query="test")
        assert "decompose" in system.lower()

    def test_relevant_starts_with_directive(self):
        system, _ = get_phase_prompt("relevant", query="test")
        assert "gather" in system.lower()

    def test_apply_starts_with_directive(self):
        system, _ = get_phase_prompt("apply", query="test")
        assert "solve" in system.lower() or "answer" in system.lower()

    def test_verify_starts_with_directive(self):
        system, _ = get_phase_prompt("verify", query="test", apply_output="x")
        assert "review" in system.lower()

    def test_conclude_starts_with_directive(self):
        system, _ = get_phase_prompt("conclude", query="test")
        assert "write" in system.lower()

    def test_no_verbose_sections(self):
        """Prompts should not have verbose section headers."""
        for phase in ("assess", "identify", "relevant", "apply", "verify", "conclude"):
            system, _ = get_phase_prompt(phase, query="test", apply_output="x")
            assert "## Your Goal" not in system, f"{phase} still has '## Your Goal'"

    def test_tool_instructions_in_system(self):
        """Tool instructions nudge should be in system prompt when has_tools=True."""
        system, _ = get_phase_prompt("apply", query="test", has_tools=True)
        assert "tool" in system.lower()
        system_no_tools, _ = get_phase_prompt("apply", query="test", has_tools=False)
        assert "call a tool first" not in system_no_tools.lower()


# === Positive Backtrack Framing ===


class TestPositiveBacktrackFraming:
    """Tests that backtrack context uses constructive language."""

    def test_no_failed_verification_language(self):
        """The old 'Failed Verification' language should be gone."""
        _, user = get_phase_prompt(
            "apply", query="test",
            backtrack_context=(
                "## Revision Guidance\n"
                "A reviewer identified these issues in the previous draft:\n"
                "Step 2 was incorrect\n\n"
                "Write a revised analysis that addresses each issue."
            ),
        )
        assert "Revision Guidance" in user
        assert "Failed Verification" not in user
        assert "rejected" not in user.lower()
