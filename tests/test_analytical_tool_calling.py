"""Tests for analytical mode tool calling tiers and auto-verification."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.modes.analytical.tool_calling import (
    ToolCallingTier,
    build_structured_output_schema,
    coerce_tool_params,
    extract_structured_text,
    parse_native_tool_call,
    parse_native_tool_calls_all,
    parse_python_style_tool_call,
    parse_structured_output,
    tools_to_native_format,
)


def _mock_tool(name: str = "web_search", description: str = "Search the web",
               schema: dict | None = None):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.input_schema = schema or {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer"},
        },
    }
    return tool


class TestToolCallingTierEnum:
    """Tier enum values."""

    def test_tier_values(self):
        assert ToolCallingTier.NATIVE.value == "native"
        assert ToolCallingTier.STRUCTURED.value == "structured"
        assert ToolCallingTier.TEXT.value == "text"


class TestToolsToNativeFormat:
    """Convert Tool objects to OpenAI function calling format."""

    def test_single_tool_conversion(self):
        tool = _mock_tool()
        result = tools_to_native_format([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "web_search"
        assert result[0]["function"]["description"] == "Search the web"
        assert "properties" in result[0]["function"]["parameters"]

    def test_empty_tools_list(self):
        result = tools_to_native_format([])
        assert result == []

    def test_tool_without_schema_gets_empty_object(self):
        tool = _mock_tool(schema=None)
        tool.input_schema = None
        result = tools_to_native_format([tool])
        assert result[0]["function"]["parameters"]["type"] == "object"


class TestBuildStructuredOutputSchema:
    """Tier 2 structured output schema generation."""

    def test_schema_has_required_action(self):
        tools = [_mock_tool("calc"), _mock_tool("search")]
        schema = build_structured_output_schema(tools)
        assert "action" in schema["properties"]
        assert schema["properties"]["action"]["enum"] == ["tool_call", "text_response"]
        assert "action" in schema["required"]

    def test_schema_includes_tool_names(self):
        tools = [_mock_tool("calculator"), _mock_tool("web_search")]
        schema = build_structured_output_schema(tools)
        assert "calculator" in schema["properties"]["tool_name"]["enum"]
        assert "web_search" in schema["properties"]["tool_name"]["enum"]


class TestParseNativeToolCall:
    """Parse native (Tier 1) tool calls from LLM responses."""

    def test_parse_openai_style(self):
        response = MagicMock()
        response.message.tool_calls = [{
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"query": "test"}),
            },
        }]
        result = parse_native_tool_call(response)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args["query"] == "test"

    def test_parse_ollama_style_dict_args(self):
        response = MagicMock()
        response.message.tool_calls = [{
            "function": {
                "name": "calculator",
                "arguments": {"expression": "2+2"},
            },
        }]
        result = parse_native_tool_call(response)
        assert result is not None
        name, args = result
        assert name == "calculator"
        assert args["expression"] == "2+2"

    def test_parse_returns_none_when_no_tool_calls(self):
        response = MagicMock()
        response.message.tool_calls = None
        result = parse_native_tool_call(response)
        assert result is None

    def test_parse_returns_none_when_empty_name(self):
        response = MagicMock()
        response.message.tool_calls = [{"function": {"name": "", "arguments": {}}}]
        result = parse_native_tool_call(response)
        assert result is None


class TestParseNativeToolCallsAll:
    """Parse multiple native tool calls."""

    def test_parse_multiple_calls(self):
        response = MagicMock()
        response.message.tool_calls = [
            {"function": {"name": "search", "arguments": {"query": "a"}}},
            {"function": {"name": "calc", "arguments": {"expr": "1+1"}}},
        ]
        results = parse_native_tool_calls_all(response)
        assert len(results) == 2
        assert results[0][0] == "search"
        assert results[1][0] == "calc"

    def test_empty_tool_calls(self):
        response = MagicMock()
        response.message.tool_calls = []
        results = parse_native_tool_calls_all(response)
        assert results == []


class TestParseStructuredOutput:
    """Parse Tier 2 structured JSON output."""

    def test_parse_tool_call_action(self):
        text = json.dumps({
            "action": "tool_call",
            "tool_name": "web_search",
            "tool_input": {"query": "weather"},
        })
        result = parse_structured_output(text)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args["query"] == "weather"

    def test_parse_text_response_returns_none(self):
        text = json.dumps({"action": "text_response", "text": "Here is the answer"})
        result = parse_structured_output(text)
        assert result is None

    def test_parse_malformed_json_returns_none(self):
        result = parse_structured_output("{not valid json")
        assert result is None

    def test_parse_missing_tool_name_returns_none(self):
        text = json.dumps({"action": "tool_call", "tool_name": "", "tool_input": {}})
        result = parse_structured_output(text)
        assert result is None


class TestExtractStructuredText:
    """Extract text from Tier 2 text_response."""

    def test_extract_text_field(self):
        text = json.dumps({"action": "text_response", "text": "The answer is 42"})
        result = extract_structured_text(text)
        assert result == "The answer is 42"

    def test_fallback_to_raw_on_parse_failure(self):
        result = extract_structured_text("Just plain text")
        assert result == "Just plain text"


class TestParsePythonStyleToolCall:
    """Parse Python-style function calls from LLM text."""

    def test_parse_simple_call(self):
        result = parse_python_style_tool_call('web_search(query="test weather")')
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args.get("query") == "test weather"

    def test_parse_with_known_tools_filter(self):
        result = parse_python_style_tool_call(
            'unknown_func("test")',
            known_tools={"web_search", "calculator"},
        )
        assert result is None

    def test_parse_returns_none_on_no_match(self):
        result = parse_python_style_tool_call("Just regular text without any calls")
        assert result is None


class TestCoerceToolParams:
    """Type coercion for tool parameters."""

    def test_coerce_string_to_int(self):
        tool = _mock_tool(schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        })
        params = {"count": "42"}
        coerce_tool_params(tool, params)
        assert params["count"] == 42

    def test_coerce_leaves_correct_types(self):
        tool = _mock_tool(schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        })
        params = {"query": "test"}
        coerce_tool_params(tool, params)
        assert params["query"] == "test"


class TestAutoVerify:
    """Automated verification — math, code, fact extraction."""

    def test_extract_math_expressions(self):
        from augmentum.modes.analytical.auto_verify import extract_math_expressions
        text = "The total is 25 * 4 = 100 and tax is 100 * 0.08 = 8"
        results = extract_math_expressions(text)
        assert len(results) >= 1
        # At least one expression should be found
        assert any("25" in expr for expr, _ in results)

    def test_extract_code_blocks(self):
        from augmentum.modes.analytical.auto_verify import extract_code_blocks
        text = "Here is some code:\n```python\nprint('hello world')\nx = 42\n```\nDone."
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        assert "print" in blocks[0]

    def test_extract_code_blocks_skips_tiny(self):
        from augmentum.modes.analytical.auto_verify import extract_code_blocks
        text = "```python\nx=1\n```"
        blocks = extract_code_blocks(text)
        # Too short (< 10 chars) should be skipped
        assert len(blocks) == 0

    async def test_run_auto_verification_no_registry(self):
        from augmentum.modes.analytical.auto_verify import run_auto_verification
        result = await run_auto_verification("Some text", None)
        assert result.checks == []
        assert result.all_passed is True

    async def test_verification_result_properties(self):
        from augmentum.modes.analytical.auto_verify import AutoVerifyResult, VerificationCheck
        result = AutoVerifyResult()
        result.checks.append(VerificationCheck(
            check_type="math", input_text="2+2=4", passed=True, details="correct",
        ))
        result.checks.append(VerificationCheck(
            check_type="code", input_text="print(1)", passed=False, details="error",
        ))
        assert result.pass_count == 1
        assert result.fail_count == 1
        assert result.skip_count == 0
        assert result.has_checks is True

    def test_check_code_dependencies_stdlib_ok(self):
        from augmentum.modes.analytical.auto_verify import check_code_dependencies
        code = "import json\nimport os\nprint('hello')"
        unavailable = check_code_dependencies(code)
        assert unavailable == []

    def test_check_code_dependencies_flags_unknown(self):
        from augmentum.modes.analytical.auto_verify import check_code_dependencies
        code = "import sklearn\nfrom transformers import pipeline"
        unavailable = check_code_dependencies(code)
        assert "sklearn" in unavailable or "transformers" in unavailable
