"""Tests for the three-tier tool calling fallback chain.

Covers:
- Tier selection based on backend type and model name
- Config override for tier selection
- Native format conversion (Tool → OpenAI function calling)
- Structured output schema generation
- Response parsing for all three tiers
- Type coercion for tool parameters
- Engine integration with tier selection
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import patch

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
from augmentum.models.engine import AugmentumEngineBackend as _RealEngine
from augmentum.models.llama_cpp import LlamaCppBackend as _RealLlamaCpp
from augmentum.models.ollama import OllamaBackend as _RealOllama
from augmentum.models.openai_compat import OpenAIBackend as _RealOpenAI
from augmentum.modes.analytical.engine import AnalyticalEngine
from augmentum.modes.analytical.tool_calling import (
    ToolCallingTier,
    build_structured_output_schema,
    coerce_tool_params,
    extract_structured_text,
    parse_native_tool_call,
    parse_python_style_tool_call,
    parse_structured_output,
    select_tier,
    tools_to_native_format,
)
from augmentum.tools.base import Tool, ToolCategory, ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockTool(Tool):
    """Minimal concrete Tool for testing."""

    def __init__(
        self, name: str = "web_search", description: str = "Search the web",
        category: ToolCategory = ToolCategory.SEARCH,
        input_schema: dict | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._category = category
        self._input_schema = (
            input_schema
            if input_schema is not None
            else {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            }
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> ToolCategory:
        return self._category

    @property
    def input_schema(self) -> dict:
        return self._input_schema

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="mock result")


class _StubBackend(ModelBackend):
    """Minimal backend for isinstance checks — never called."""

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        raise NotImplementedError

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        raise NotImplementedError
        yield  # noqa: RET503 — make it an async generator

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails()


# Real backend subclasses so isinstance() works with lazy imports in select_tier()


class _OllamaStub(_RealOllama):
    def __init__(self) -> None:
        pass  # skip real __init__


class _OpenAIStub(_RealOpenAI):
    def __init__(self, base_url: str = "https://api.openai.com/v1") -> None:
        self._base_url = base_url  # needed for cloud vs local detection


class _LlamaCppStub(_RealLlamaCpp):
    def __init__(self) -> None:
        pass  # skip real __init__


class _EngineStub(_RealEngine):
    def __init__(self) -> None:
        pass  # skip real __init__


# =========================================================================
# Tier selection tests
# =========================================================================


class TestSelectTier:
    """Tests for select_tier()."""

    def test_openai_cloud_backend_native(self):
        backend = _OpenAIStub("https://api.openai.com/v1")
        assert select_tier(backend, "gpt-4o") == ToolCallingTier.NATIVE

    def test_openai_gemini_cloud_native(self):
        backend = _OpenAIStub("https://generativelanguage.googleapis.com/v1beta")
        assert select_tier(backend, "gemini-2.0-flash") == ToolCallingTier.NATIVE

    def test_openai_local_unknown_model_text(self):
        """Local OpenAI-compat servers with unknown models fall back to TEXT."""
        backend = _OpenAIStub("http://localhost:1234/v1")
        assert select_tier(backend, "rocinante-x-12b-v1") == ToolCallingTier.TEXT

    def test_openai_local_known_family_native(self):
        """Local servers with known tool-capable model families still get NATIVE."""
        backend = _OpenAIStub("http://192.168.1.100:1234/v1")
        assert select_tier(backend, "qwen2.5-7b-instruct") == ToolCallingTier.NATIVE

    def test_ollama_capable_model_qwen(self):
        backend = _OllamaStub()
        assert select_tier(backend, "qwen2.5:7b") == ToolCallingTier.NATIVE

    def test_ollama_capable_model_llama31(self):
        backend = _OllamaStub()
        assert select_tier(backend, "llama3.1:8b") == ToolCallingTier.NATIVE

    def test_ollama_capable_model_mistral_nemo(self):
        backend = _OllamaStub()
        assert select_tier(backend, "mistral-nemo:12b") == ToolCallingTier.NATIVE

    def test_ollama_capable_model_command_r(self):
        backend = _OllamaStub()
        assert select_tier(backend, "command-r:35b") == ToolCallingTier.NATIVE

    def test_ollama_unknown_model_native(self):
        """Optimistic native: an unknown Ollama model is assumed tool-capable
        (modern models follow the OpenAI tool-calling standard) rather than
        being demoted by a stale hardcoded allowlist."""
        backend = _OllamaStub()
        assert select_tier(backend, "phi3:3.8b") == ToolCallingTier.NATIVE

    def test_engine_unknown_model_native(self):
        """Augmentum engine (llama-server, --jinja always on) assumes NATIVE
        for any model — no static family allowlist gate."""
        backend = _EngineStub()
        assert select_tier(backend, "brand-new-model-2027") == ToolCallingTier.NATIVE

    def test_llamacpp_native(self):
        """LlamaCppBackend is llama-server with --jinja — it emits/streams
        native tool_calls (identical to AugmentumEngineBackend), so it gets
        NATIVE. (2026-06-30: previously misclassified as TEXT, which forced
        the non-streaming peek-then-blob path for every local-model chat turn
        that didn't fire a tool.)"""
        backend = _LlamaCppStub()
        assert select_tier(backend, "some-model") == ToolCallingTier.NATIVE

    def test_unknown_backend_text(self):
        backend = _StubBackend()
        assert select_tier(backend, "any") == ToolCallingTier.TEXT

    def test_config_override(self):
        with patch(
            "augmentum.modes.analytical.tool_calling.settings",
        ) as mock_settings:
            mock_settings.uarf_tool_tier_override = "text"
            backend = _OllamaStub()
            assert select_tier(backend, "qwen2.5:7b") == ToolCallingTier.TEXT

    def test_config_override_invalid_ignored(self):
        with patch(
            "augmentum.modes.analytical.tool_calling.settings",
        ) as mock_settings:
            mock_settings.uarf_tool_tier_override = "bogus"
            backend = _OllamaStub()
            assert select_tier(backend, "qwen2.5:7b") == ToolCallingTier.NATIVE


# =========================================================================
# Schema generation tests
# =========================================================================


class TestSchemaGeneration:

    def test_native_format_conversion(self):
        tool = _MockTool(
            name="calculator",
            description="Evaluate math expressions",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        )
        result = tools_to_native_format([tool])
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "calculator"
        assert result[0]["function"]["description"] == "Evaluate math expressions"
        assert result[0]["function"]["parameters"]["required"] == ["expression"]

    def test_native_format_multiple_tools(self):
        tools = [
            _MockTool(name="web_search", description="Search"),
            _MockTool(name="calculator", description="Calculate"),
        ]
        result = tools_to_native_format(tools)
        assert len(result) == 2
        names = {r["function"]["name"] for r in result}
        assert names == {"web_search", "calculator"}

    def test_native_format_empty_schema(self):
        tool = _MockTool(name="test", description="Test", input_schema={})
        result = tools_to_native_format([tool])
        params = result[0]["function"]["parameters"]
        assert params == {"type": "object", "properties": {}}

    def test_structured_schema_tool_names(self):
        tools = [
            _MockTool(name="web_search"),
            _MockTool(name="calculator"),
        ]
        schema = build_structured_output_schema(tools)
        assert schema["properties"]["tool_name"]["enum"] == ["web_search", "calculator"]

    def test_structured_schema_action_enum(self):
        tools = [_MockTool()]
        schema = build_structured_output_schema(tools)
        assert schema["properties"]["action"]["enum"] == ["tool_call", "text_response"]
        assert "action" in schema["required"]


# =========================================================================
# Parsing tests
# =========================================================================


class TestParseNativeToolCall:

    def test_ollama_format(self):
        """Ollama returns tool_calls with dict arguments."""
        response = InternalChatResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "test query"},
                    },
                }],
            ),
            model="qwen2.5:7b",
        )
        result = parse_native_tool_call(response)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1] == {"query": "test query"}

    def test_openai_format(self):
        """OpenAI returns tool_calls with string arguments."""
        response = InternalChatResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "2 + 2"}',
                    },
                }],
            ),
            model="gpt-4o",
        )
        result = parse_native_tool_call(response)
        assert result is not None
        assert result[0] == "calculator"
        assert result[1] == {"expression": "2 + 2"}

    def test_no_tool_call(self):
        response = InternalChatResponse(
            message=Message(role="assistant", content="Just some text"),
            model="test",
        )
        assert parse_native_tool_call(response) is None

    def test_empty_tool_calls_list(self):
        response = InternalChatResponse(
            message=Message(role="assistant", content="", tool_calls=[]),
            model="test",
        )
        assert parse_native_tool_call(response) is None

    def test_no_message(self):
        response = InternalChatResponse(
            message=Message(role="assistant", content=""),
            model="test",
        )
        assert parse_native_tool_call(response) is None

    def test_flat_tool_call(self):
        """Some backends return tool_calls without nested function key."""
        response = InternalChatResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "name": "web_search",
                    "arguments": {"query": "test"},
                }],
            ),
            model="test",
        )
        result = parse_native_tool_call(response)
        assert result is not None
        assert result[0] == "web_search"

    def test_invalid_arguments_string(self):
        """Non-JSON string arguments → empty dict."""
        response = InternalChatResponse(
            message=Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "function": {
                        "name": "web_search",
                        "arguments": "not json at all",
                    },
                }],
            ),
            model="test",
        )
        result = parse_native_tool_call(response)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1] == {}


class TestParseStructuredOutput:

    def test_valid_tool_call(self):
        text = json.dumps({
            "action": "tool_call",
            "tool_name": "web_search",
            "tool_input": {"query": "test"},
        })
        result = parse_structured_output(text)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1] == {"query": "test"}

    def test_text_response(self):
        text = json.dumps({
            "action": "text_response",
            "text": "The answer is 42.",
        })
        assert parse_structured_output(text) is None

    def test_malformed_json(self):
        assert parse_structured_output("not json {{{") is None

    def test_missing_tool_name(self):
        text = json.dumps({"action": "tool_call", "tool_input": {"query": "test"}})
        assert parse_structured_output(text) is None

    def test_empty_tool_input(self):
        text = json.dumps({
            "action": "tool_call",
            "tool_name": "calculator",
        })
        result = parse_structured_output(text)
        assert result is not None
        assert result[0] == "calculator"
        assert result[1] == {}

    def test_non_dict_tool_input(self):
        text = json.dumps({
            "action": "tool_call",
            "tool_name": "calculator",
            "tool_input": "not a dict",
        })
        result = parse_structured_output(text)
        assert result is not None
        assert result[1] == {}


class TestExtractStructuredText:

    def test_text_response(self):
        text = json.dumps({
            "action": "text_response",
            "text": "The answer is 42.",
        })
        assert extract_structured_text(text) == "The answer is 42."

    def test_malformed_returns_raw(self):
        raw = "not json at all"
        assert extract_structured_text(raw) == raw

    def test_missing_text_field(self):
        text = json.dumps({"action": "text_response"})
        result = extract_structured_text(text)
        assert result == text  # returns raw when "text" key missing


# =========================================================================
# Type coercion tests
# =========================================================================


class TestCoerceToolParams:

    def _make_tool(self, properties: dict) -> _MockTool:
        return _MockTool(
            name="test", description="Test",
            input_schema={
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
            },
        )

    def test_string_to_int(self):
        tool = self._make_tool({"count": {"type": "integer"}})
        params = {"count": "42"}
        result = coerce_tool_params(tool, params)
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_string_to_float(self):
        tool = self._make_tool({"value": {"type": "number"}})
        params = {"value": "3.14"}
        result = coerce_tool_params(tool, params)
        assert result["value"] == pytest.approx(3.14)
        assert isinstance(result["value"], float)

    def test_string_to_bool(self):
        tool = self._make_tool({"flag": {"type": "boolean"}})
        params = {"flag": "true"}
        result = coerce_tool_params(tool, params)
        assert result["flag"] is True

    def test_string_to_bool_no(self):
        tool = self._make_tool({"flag": {"type": "boolean"}})
        params = {"flag": "false"}
        result = coerce_tool_params(tool, params)
        assert result["flag"] is False

    def test_already_correct_type(self):
        tool = self._make_tool({"count": {"type": "integer"}})
        params = {"count": 42}
        result = coerce_tool_params(tool, params)
        assert result["count"] == 42

    def test_unknown_key_stripped(self):
        """Unknown parameters are stripped to prevent TypeError on execute()."""
        tool = self._make_tool({"count": {"type": "integer"}})
        params = {"count": "5", "extra": "hello"}
        result = coerce_tool_params(tool, params)
        assert result["count"] == 5
        assert "extra" not in result  # stripped to prevent TypeError

    def test_no_schema(self):
        tool = _MockTool(name="test", description="Test", input_schema={})
        params = {"key": "value"}
        result = coerce_tool_params(tool, params)
        assert result == {"key": "value"}

    def test_invalid_int_string(self):
        tool = self._make_tool({"count": {"type": "integer"}})
        params = {"count": "not_a_number"}
        result = coerce_tool_params(tool, params)
        assert result["count"] == "not_a_number"  # unchanged on failure

    def test_unwrap_nested_parameters(self):
        """Models that wrap args in {"action": "...", "parameters": {...}}."""
        tool = self._make_tool({
            "prompt": {"type": "string"},
            "style": {"type": "string"},
        })
        params = {
            "action": "generate_image",
            "parameters": {"prompt": "a cat", "style": "anime"},
        }
        result = coerce_tool_params(tool, params)
        assert result["prompt"] == "a cat"
        assert result["style"] == "anime"
        assert "action" not in result
        assert "parameters" not in result

    def test_unwrap_nested_arguments(self):
        """Models that wrap args in {"arguments": {...}}."""
        tool = self._make_tool({"query": {"type": "string"}})
        params = {"arguments": {"query": "weather today"}}
        result = coerce_tool_params(tool, params)
        assert result["query"] == "weather today"
        assert "arguments" not in result

    def test_no_unwrap_when_outer_matches(self):
        """Don't unwrap if the outer dict already matches the schema."""
        tool = self._make_tool({
            "parameters": {"type": "string"},  # "parameters" is a real param name
            "query": {"type": "string"},
        })
        params = {"parameters": "some value", "query": "test"}
        result = coerce_tool_params(tool, params)
        assert result["parameters"] == "some value"
        assert result["query"] == "test"


# =========================================================================
# Engine integration tests
# =========================================================================


class _ToolCallBackend(ModelBackend):
    """Backend that returns native tool_calls on first call, text on second."""

    def __init__(
        self,
        tool_name: str = "web_search",
        tool_args: dict | None = None,
        tool_result_text: str = "Analysis complete with tool data.",
    ) -> None:
        self._tool_name = tool_name
        self._tool_args = tool_args or {"query": "test"}
        self._tool_result_text = tool_result_text
        self._call_count = 0

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        self._call_count += 1

        # First call: return a tool call
        if self._call_count == 1:
            return InternalChatResponse(
                message=Message(
                    role="assistant",
                    content="",
                    tool_calls=[{
                        "function": {
                            "name": self._tool_name,
                            "arguments": self._tool_args,
                        },
                    }],
                ),
                model=request.model,
                usage=Usage(total_tokens=100),
            )

        # Second call (after tool result injected): return text
        return InternalChatResponse(
            message=Message(role="assistant", content=self._tool_result_text),
            model=request.model,
            usage=Usage(total_tokens=150),
        )

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        raise NotImplementedError
        yield  # noqa: RET503

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails()


class _StructuredBackend(ModelBackend):
    """Backend that returns structured JSON output."""

    def __init__(
        self,
        first_response: dict | None = None,
        second_response: str = "Analysis done.",
    ) -> None:
        self._first = first_response or {
            "action": "tool_call",
            "tool_name": "web_search",
            "tool_input": {"query": "test"},
        }
        self._second = second_response
        self._call_count = 0

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        self._call_count += 1
        if self._call_count == 1:
            content = json.dumps(self._first)
        else:
            content = json.dumps({
                "action": "text_response",
                "text": self._second,
            })
        return InternalChatResponse(
            message=Message(role="assistant", content=content),
            model=request.model,
            usage=Usage(total_tokens=100),
        )

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        raise NotImplementedError
        yield  # noqa: RET503

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails()


class _TextBackend(ModelBackend):
    """Backend that returns text with TOOL_CALL markers."""

    def __init__(
        self,
        first_response: str = "",
        second_response: str = "Final analysis.",
    ) -> None:
        self._first = first_response or (
            "I need to search for information.\n\n"
            "TOOL_CALL: web_search\n"
            'TOOL_INPUT: {"query": "test query"}'
        )
        self._second = second_response
        self._call_count = 0

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        self._call_count += 1
        content = self._first if self._call_count == 1 else self._second
        return InternalChatResponse(
            message=Message(role="assistant", content=content),
            model=request.model,
            usage=Usage(total_tokens=100),
        )

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        raise NotImplementedError
        yield  # noqa: RET503

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails()


def _make_registry():
    """Build a ToolRegistry with a mock web_search tool."""
    from augmentum.tools.registry import ToolRegistry

    registry = ToolRegistry()
    tool = _MockTool(
        name="web_search",
        description="Search the web",
        category=ToolCategory.SEARCH,
    )
    registry.register(tool)
    return registry


class TestEngineTierIntegration:
    """Integration tests verifying the engine uses the correct tier."""

    @pytest.mark.asyncio
    async def test_engine_tier1_native_tool_call(self):
        """Tier 1: Backend returns native tool_calls → tool executes."""
        backend = _ToolCallBackend()
        registry = _make_registry()
        engine = AnalyticalEngine(backend, tool_registry=registry)

        # Force NATIVE tier
        with patch(
            "augmentum.modes.analytical.engine.select_tier",
            return_value=ToolCallingTier.NATIVE,
        ):
            from augmentum.modes.analytical.state import AnalyticalPhase

            result = await engine._run_phase_with_tools(
                AnalyticalPhase.APPLY,
                model="test-model",
                query="What is the weather?",
            )

        assert result.output == "Analysis complete with tool data."
        assert len(engine.state.tool_calls) == 1
        assert engine.state.tool_calls[0].tool_name == "web_search"
        assert engine.state.tool_calls[0].success is True

    @pytest.mark.asyncio
    async def test_engine_tier2_structured_tool_call(self):
        """Tier 2: Backend returns structured JSON → tool executes."""
        backend = _StructuredBackend()
        registry = _make_registry()
        engine = AnalyticalEngine(backend, tool_registry=registry)

        with patch(
            "augmentum.modes.analytical.engine.select_tier",
            return_value=ToolCallingTier.STRUCTURED,
        ):
            from augmentum.modes.analytical.state import AnalyticalPhase

            result = await engine._run_phase_with_tools(
                AnalyticalPhase.APPLY,
                model="test-model",
                query="What is the weather?",
            )

        assert result.output == "Analysis done."
        assert len(engine.state.tool_calls) == 1
        assert engine.state.tool_calls[0].tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_engine_tier3_text_tool_call(self):
        """Tier 3: Backend returns TOOL_CALL text → tool executes."""
        backend = _TextBackend()
        registry = _make_registry()
        engine = AnalyticalEngine(backend, tool_registry=registry)

        with patch(
            "augmentum.modes.analytical.engine.select_tier",
            return_value=ToolCallingTier.TEXT,
        ):
            from augmentum.modes.analytical.state import AnalyticalPhase

            result = await engine._run_phase_with_tools(
                AnalyticalPhase.APPLY,
                model="test-model",
                query="What is the weather?",
            )

        assert result.output == "Final analysis."
        assert len(engine.state.tool_calls) == 1
        assert engine.state.tool_calls[0].tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_tier_fallback_on_parse_failure(self):
        """Tier 1 returns no tool_calls but text has TOOL_CALL → falls back to Tier 3."""
        # Backend returns text with TOOL_CALL markers but no native tool_calls
        tool_text = (
            "Let me search.\n\n"
            "TOOL_CALL: web_search\n"
            'TOOL_INPUT: {"query": "fallback test"}'
        )
        backend = _TextBackend(
            first_response=tool_text,
            second_response="Fallback worked.",
        )
        registry = _make_registry()
        engine = AnalyticalEngine(backend, tool_registry=registry)

        with patch(
            "augmentum.modes.analytical.engine.select_tier",
            return_value=ToolCallingTier.NATIVE,
        ):
            from augmentum.modes.analytical.state import AnalyticalPhase

            result = await engine._run_phase_with_tools(
                AnalyticalPhase.APPLY,
                model="test-model",
                query="test fallback",
            )

        # Should have fallen back to text parsing and executed the tool
        assert result.output == "Fallback worked."
        assert len(engine.state.tool_calls) == 1
        assert engine.state.tool_calls[0].tool_name == "web_search"

    @pytest.mark.asyncio
    async def test_engine_no_tools_skips_tier(self):
        """When no tools are available, tier selection is skipped."""
        backend = _TextBackend(
            first_response="Just analysis, no tools needed.",
            second_response="",
        )
        engine = AnalyticalEngine(backend, tool_registry=None)

        from augmentum.modes.analytical.state import AnalyticalPhase

        result = await engine._run_phase_with_tools(
            AnalyticalPhase.APPLY,
            model="test-model",
            query="What is 2+2?",
        )
        assert result.output == "Just analysis, no tools needed."
        assert len(engine.state.tool_calls) == 0


# =========================================================================
# Prompt function tests
# =========================================================================


class TestPromptFunctions:

    def test_native_prompt_section(self):
        from augmentum.modes.analytical.prompts import get_native_tool_prompt_section

        result = get_native_tool_prompt_section()
        assert "tools available" in result.lower()
        assert "before" in result.lower()

    def test_structured_prompt_section_lists_tools(self):
        from augmentum.modes.analytical.prompts import get_structured_tool_prompt_section

        tools = [
            _MockTool(name="web_search", description="Search the web"),
            _MockTool(name="calculator", description="Do math"),
        ]
        schema = build_structured_output_schema(tools)
        result = get_structured_tool_prompt_section(tools, schema)
        assert "web_search" in result
        assert "calculator" in result
        assert "tool_call" in result
        assert "text_response" in result

    def test_structured_prompt_section_empty(self):
        from augmentum.modes.analytical.prompts import get_structured_tool_prompt_section

        result = get_structured_tool_prompt_section([], {})
        assert result == ""


# =========================================================================
# LlamaCpp tools passthrough test
# =========================================================================


class TestLlamaCppToolsPassthrough:

    def test_tools_in_payload(self):
        from augmentum.models.llama_cpp import LlamaCppBackend

        tools = [{"type": "function", "function": {"name": "test"}}]
        request = InternalChatRequest(
            model="test", messages=[Message(role="user", content="hi")],
            stream=False, tools=tools,
        )
        payload = LlamaCppBackend._to_openai_payload(request)
        assert payload["tools"] == tools

    def test_no_tools_no_key(self):
        from augmentum.models.llama_cpp import LlamaCppBackend

        request = InternalChatRequest(
            model="test", messages=[Message(role="user", content="hi")],
            stream=False,
        )
        payload = LlamaCppBackend._to_openai_payload(request)
        assert "tools" not in payload


# =========================================================================
# Python-style tool call parsing tests
# =========================================================================


class TestParsePythonStyleToolCall:
    """Tests for parse_python_style_tool_call — catches models that emit
    tool calls as Python function calls instead of using native/structured
    or TOOL_CALL: format."""

    KNOWN = {"web_search", "calculator", "web_fetch", "python_exec"}

    def test_simple_positional_string(self):
        text = 'web_search("current weather in NYC")'
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args["query"] == "current weather in NYC"

    def test_keyword_arg(self):
        text = 'web_search(query="latest AI news")'
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args["query"] == "latest AI news"

    def test_multiple_keyword_args(self):
        text = 'calculator(expression="2 + 2", precision=2)'
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        name, args = result
        assert name == "calculator"
        assert args["expression"] == "2 + 2"
        assert args["precision"] == 2

    def test_code_fenced(self):
        text = '```python\nweb_search("current weather")\n```'
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        name, args = result
        assert name == "web_search"
        assert args["query"] == "current weather"

    def test_code_fenced_no_lang(self):
        text = '```\nweb_search("test query")\n```'
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        assert result[0] == "web_search"

    def test_surrounded_by_text(self):
        text = (
            "I'll search for that information.\n\n"
            'web_search("current weather in Seattle")\n\n'
            "Let me check the results."
        )
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1]["query"] == "current weather in Seattle"

    def test_unknown_tool_rejected(self):
        text = 'unknown_tool("some arg")'
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is None

    def test_no_known_tools_matches_any(self):
        text = 'any_function("arg")'
        result = parse_python_style_tool_call(text, known_tools=None)
        assert result is not None
        assert result[0] == "any_function"

    def test_single_quotes(self):
        text = "web_search('latest news')"
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        assert result[1]["query"] == "latest news"

    def test_empty_args(self):
        text = "calculator()"
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        assert result[0] == "calculator"
        assert result[1] == {}

    def test_no_function_call(self):
        text = "The answer to your question is 42."
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is None

    def test_mixed_positional_and_keyword(self):
        text = 'web_fetch(url="https://example.com", max_chars=5000)'
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        name, args = result
        assert name == "web_fetch"
        assert args["url"] == "https://example.com"
        assert args["max_chars"] == 5000

    def test_bool_coercion(self):
        text = 'python_exec(code="print(1)", safe=True)'
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        assert result[1]["safe"] is True

    def test_multiline_code_block(self):
        text = (
            "I need to search for this.\n\n"
            "```python\n"
            "result = web_search(query=\"AI news 2026\")\n"
            "print(result)\n"
            "```"
        )
        result = parse_python_style_tool_call(text, known_tools=self.KNOWN)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1]["query"] == "AI news 2026"


# ---------------------------------------------------------------------------
# ReAct Action/Action Input parsing
# ---------------------------------------------------------------------------


class TestParseActionInput:
    """Test parse_action_input_tool_call for ReAct-style output."""

    KNOWN = {"web_search", "calculator", "web_fetch"}

    def test_basic_action_input(self):
        from augmentum.modes.analytical.tool_calling import parse_action_input_tool_call
        text = "Thought: I need to search.\nAction: web_search\nAction Input: {\"query\": \"weather today\"}"
        result = parse_action_input_tool_call(text, self.KNOWN)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1]["query"] == "weather today"

    def test_bold_markdown_action(self):
        from augmentum.modes.analytical.tool_calling import parse_action_input_tool_call
        text = "**Action**: web_search\n**Input**: {\"query\": \"test\"}"
        result = parse_action_input_tool_call(text, self.KNOWN)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1]["query"] == "test"

    def test_tool_keyword(self):
        from augmentum.modes.analytical.tool_calling import parse_action_input_tool_call
        text = "Tool: calculator\nArguments: {\"expression\": \"2+2\"}"
        result = parse_action_input_tool_call(text, self.KNOWN)
        assert result is not None
        assert result[0] == "calculator"
        assert result[1]["expression"] == "2+2"

    def test_plain_string_input(self):
        from augmentum.modes.analytical.tool_calling import parse_action_input_tool_call
        text = "Action: web_search\nAction Input: weather forecast tomorrow"
        result = parse_action_input_tool_call(text, self.KNOWN)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1]["query"] == "weather forecast tomorrow"

    def test_unknown_tool_rejected(self):
        from augmentum.modes.analytical.tool_calling import parse_action_input_tool_call
        text = "Action: unknown_tool\nAction Input: {\"x\": 1}"
        result = parse_action_input_tool_call(text, self.KNOWN)
        assert result is None

    def test_no_input_line(self):
        from augmentum.modes.analytical.tool_calling import parse_action_input_tool_call
        text = "Action: web_search"
        result = parse_action_input_tool_call(text, self.KNOWN)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1] == {}

    def test_placeholder_name_rejected(self):
        from augmentum.modes.analytical.tool_calling import parse_action_input_tool_call
        text = "Action: tool_name\nAction Input: {\"query\": \"test\"}"
        result = parse_action_input_tool_call(text, known_tools=None)
        assert result is None


# ---------------------------------------------------------------------------
# XML tool call parsing
# ---------------------------------------------------------------------------


class TestParseXmlToolCalls:
    """Test parse_xml_tool_calls for XML-style tool call blocks."""

    KNOWN = {"web_search", "calculator", "web_fetch"}

    def test_tool_use_block(self):
        from augmentum.modes.analytical.tool_calling import parse_xml_tool_calls
        text = '<tool_use><name>web_search</name><input>{"query": "test"}</input></tool_use>'
        result = parse_xml_tool_calls(text, self.KNOWN)
        assert result is not None
        assert len(result) == 1
        assert result[0][0] == "web_search"
        assert result[0][1]["query"] == "test"

    def test_tool_call_inline_json(self):
        from augmentum.modes.analytical.tool_calling import parse_xml_tool_calls
        text = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
        result = parse_xml_tool_calls(text, self.KNOWN)
        assert result is not None
        assert result[0][0] == "calculator"
        assert result[0][1]["expression"] == "2+2"

    def test_multiple_xml_blocks(self):
        from augmentum.modes.analytical.tool_calling import parse_xml_tool_calls
        text = (
            '<tool_use><name>web_search</name><input>{"query": "a"}</input></tool_use>\n'
            '<tool_use><name>calculator</name><input>{"expression": "1+1"}</input></tool_use>'
        )
        result = parse_xml_tool_calls(text, self.KNOWN)
        assert result is not None
        assert len(result) == 2

    def test_unknown_tool_filtered(self):
        from augmentum.modes.analytical.tool_calling import parse_xml_tool_calls
        text = '<tool_use><name>unknown</name><input>{"x": 1}</input></tool_use>'
        result = parse_xml_tool_calls(text, self.KNOWN)
        assert result is None

    def test_hermes_qwen_format(self):
        from augmentum.modes.analytical.tool_calling import parse_xml_tool_calls
        text = '<tool_call>\n{"name": "web_fetch", "arguments": {"url": "https://example.com"}}\n</tool_call>'
        result = parse_xml_tool_calls(text, self.KNOWN)
        assert result is not None
        assert result[0][0] == "web_fetch"
        assert result[0][1]["url"] == "https://example.com"

    def test_parameters_key(self):
        from augmentum.modes.analytical.tool_calling import parse_xml_tool_calls
        text = '<tool_call>{"name": "web_search", "parameters": {"query": "test"}}</tool_call>'
        result = parse_xml_tool_calls(text, self.KNOWN)
        assert result is not None
        assert result[0][1]["query"] == "test"

    def test_no_xml_returns_none(self):
        from augmentum.modes.analytical.tool_calling import parse_xml_tool_calls
        text = "Just a normal response with no XML tags."
        result = parse_xml_tool_calls(text, self.KNOWN)
        assert result is None


# ---------------------------------------------------------------------------
# Fuzzy tool name + JSON parsing
# ---------------------------------------------------------------------------


class TestParseFuzzyToolCall:
    """Test parse_fuzzy_tool_call for last-resort matching."""

    KNOWN = {"web_search", "calculator"}

    def test_tool_name_with_json(self):
        from augmentum.modes.analytical.tool_calling import parse_fuzzy_tool_call
        text = 'I\'ll use web_search with {"query": "weather today"}'
        result = parse_fuzzy_tool_call(text, self.KNOWN)
        assert result is not None
        assert result[0] == "web_search"
        assert result[1]["query"] == "weather today"

    def test_tool_name_colon_json(self):
        from augmentum.modes.analytical.tool_calling import parse_fuzzy_tool_call
        text = 'Let me call the calculator tool: {"expression": "2+2"}'
        result = parse_fuzzy_tool_call(text, self.KNOWN)
        assert result is not None
        assert result[0] == "calculator"
        assert result[1]["expression"] == "2+2"

    def test_no_json_returns_none(self):
        from augmentum.modes.analytical.tool_calling import parse_fuzzy_tool_call
        text = "I should use web_search to find that information."
        result = parse_fuzzy_tool_call(text, self.KNOWN)
        assert result is None

    def test_empty_known_tools_returns_none(self):
        from augmentum.modes.analytical.tool_calling import parse_fuzzy_tool_call
        text = 'web_search {"query": "test"}'
        result = parse_fuzzy_tool_call(text, set())
        assert result is None

    def test_json_too_far_from_name(self):
        from augmentum.modes.analytical.tool_calling import parse_fuzzy_tool_call
        text = "web_search" + " " * 250 + '{"query": "test"}'
        result = parse_fuzzy_tool_call(text, self.KNOWN)
        assert result is None

    def test_negation_rejected(self):
        from augmentum.modes.analytical.tool_calling import parse_fuzzy_tool_call
        text = "I don't need to use web_search for this. {\"query\": \"test\"}"
        result = parse_fuzzy_tool_call(text, self.KNOWN)
        assert result is None

    def test_negation_wont(self):
        from augmentum.modes.analytical.tool_calling import parse_fuzzy_tool_call
        text = "I won't use calculator here {\"expression\": \"2+2\"}"
        result = parse_fuzzy_tool_call(text, self.KNOWN)
        assert result is None

    def test_negation_instead_of(self):
        from augmentum.modes.analytical.tool_calling import parse_fuzzy_tool_call
        text = "Instead of web_search, I'll answer from memory {\"query\": \"x\"}"
        result = parse_fuzzy_tool_call(text, self.KNOWN)
        assert result is None


# ---------------------------------------------------------------------------
# Fuzzy stripping
# ---------------------------------------------------------------------------


class TestStripFuzzyToolCalls:
    """Test _strip_fuzzy_tool_calls removes tool artifacts from assistant text."""

    def test_strip_tool_name_and_json(self):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        text = "I'll use web_search with {\"query\": \"weather today\"} to find that."
        result = PassthroughHandler._strip_fuzzy_tool_calls(
            text, [("web_search", {"query": "weather today"}, "call_1")]
        )
        assert "web_search" not in result
        assert "{" not in result

    def test_strip_python_style(self):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        text = "Let me check: calculator(expression=\"2+2\") for you."
        result = PassthroughHandler._strip_fuzzy_tool_calls(
            text, [("calculator", {"expression": "2+2"}, "call_1")]
        )
        assert "calculator(" not in result

    def test_no_match_is_noop(self):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        text = "Just a normal response."
        result = PassthroughHandler._strip_fuzzy_tool_calls(
            text, [("web_search", {"query": "test"}, "call_1")]
        )
        assert result == "Just a normal response."
