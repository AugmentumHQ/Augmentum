"""Tests for Claude Messages API converter and backend adapter."""

from __future__ import annotations

from augmentum.models.converters.claude import (
    ClaudeConverter,
    apply_prompt_caching,
    calculate_thinking_budget,
    convert_response,
    get_thinking_config,
    is_adaptive_model,
    is_no_prefill_model,
    is_thinking_model,
)
from augmentum.models.converters.utils import ZWS

# ===================================================================
# TestClaudeMessageConversion
# ===================================================================


class TestClaudeMessageConversion:
    """Test ClaudeConverter.convert_messages."""

    def setup_method(self) -> None:
        self.converter = ClaudeConverter()

    def test_system_extraction_single(self) -> None:
        """Leading system message extracted to system blocks."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = self.converter.convert_messages(msgs)
        assert len(result["system"]) == 1
        assert result["system"][0] == {"type": "text", "text": "You are helpful."}
        # Messages should only contain user
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_system_extraction_multiple_prefix(self) -> None:
        """Multiple leading system messages all extracted."""
        msgs = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "Hi"},
        ]
        result = self.converter.convert_messages(msgs)
        assert len(result["system"]) == 2
        assert result["system"][0]["text"] == "System 1"
        assert result["system"][1]["text"] == "System 2"

    def test_mid_conversation_system_becomes_user(self) -> None:
        """System messages after the first non-system become user role."""
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Mid-conversation system"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = self.converter.convert_messages(msgs)
        # System extracted
        assert len(result["system"]) == 1
        # Mid-system converted to user, then merged or kept separate
        roles = [m["role"] for m in result["messages"]]
        assert "system" not in roles

    def test_consecutive_merge(self) -> None:
        """Consecutive same-role messages are merged."""
        msgs = [
            {"role": "user", "content": "Part 1"},
            {"role": "user", "content": "Part 2"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = self.converter.convert_messages(msgs)
        # Should be merged into one user + one assistant
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
        # Merged user should have both content blocks
        user_texts = [
            b["text"]
            for b in result["messages"][0]["content"]
            if b.get("type") == "text"
        ]
        assert "Part 1" in user_texts[0]
        assert "Part 2" in user_texts[1]

    def test_image_conversion_data_uri(self) -> None:
        """Data URI images converted to Claude base64 format."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBOR"
                        },
                    },
                ],
            },
        ]
        result = self.converter.convert_messages(msgs)
        content = result["messages"][0]["content"]
        # Should have text + image blocks
        types = [b["type"] for b in content]
        assert "text" in types
        assert "image" in types
        img_block = next(b for b in content if b["type"] == "image")
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == "iVBOR"

    def test_tool_calls_conversion(self) -> None:
        """OpenAI tool_calls converted to Claude tool_use blocks."""
        msgs = [
            {"role": "user", "content": "Search for cats"},
            {
                "role": "assistant",
                "content": "Let me search.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "cats"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Found 5 results about cats.",
            },
        ]
        result = self.converter.convert_messages(msgs)
        # Find the assistant message with tool_use
        assistant_msg = next(
            m for m in result["messages"] if m["role"] == "assistant"
        )
        tool_blocks = [
            b for b in assistant_msg["content"] if b.get("type") == "tool_use"
        ]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["id"] == "call_123"
        assert tool_blocks[0]["name"] == "search"
        assert tool_blocks[0]["input"] == {"query": "cats"}

        # Tool result should be user role with tool_result content
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        tool_result_blocks = []
        for um in user_msgs:
            for b in um.get("content", []):
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tool_result_blocks.append(b)
        assert len(tool_result_blocks) == 1
        assert tool_result_blocks[0]["tool_use_id"] == "call_123"

    def test_prefill_appended(self) -> None:
        """Prefill text appended as assistant message."""
        msgs = [{"role": "user", "content": "Hello"}]
        result = self.converter.convert_messages(msgs, prefill="Sure, I")
        assert result["messages"][-1]["role"] == "assistant"
        texts = [
            b["text"]
            for b in result["messages"][-1]["content"]
            if b.get("type") == "text"
        ]
        assert any("Sure, I" in t for t in texts)

    def test_empty_content_replaced_with_zws(self) -> None:
        """Empty content replaced with zero-width space."""
        msgs = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Reply"},
        ]
        result = self.converter.convert_messages(msgs)
        user_msg = result["messages"][0]
        first_text = user_msg["content"][0]["text"]
        assert first_text == ZWS

    def test_name_prepended(self) -> None:
        """Message name field prepended to content text."""
        msgs = [
            {"role": "user", "content": "Hello", "name": "Alice"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = self.converter.convert_messages(msgs)
        user_content = result["messages"][0]["content"]
        text = user_content[0]["text"]
        assert text.startswith("Alice:")

    def test_starts_with_user(self) -> None:
        """Conversation always starts with user message."""
        msgs = [
            {"role": "assistant", "content": "I start"},
            {"role": "user", "content": "OK"},
        ]
        result = self.converter.convert_messages(msgs)
        assert result["messages"][0]["role"] == "user"


# ===================================================================
# TestClaudeThinkingConfig
# ===================================================================


class TestClaudeThinkingConfig:
    """Test thinking model detection and budget calculation."""

    def test_thinking_model_detection(self) -> None:
        assert is_thinking_model("claude-3-7-sonnet-latest")
        assert is_thinking_model("claude-opus-4")
        assert is_thinking_model("claude-sonnet-4")
        assert is_thinking_model("claude-opus-4-6")
        assert is_thinking_model("claude-sonnet-4-6")
        assert not is_thinking_model("claude-3-5-sonnet-latest")
        assert not is_thinking_model("gpt-4o")

    def test_adaptive_model_detection(self) -> None:
        assert is_adaptive_model("claude-opus-4-6")
        assert is_adaptive_model("claude-sonnet-4-6")
        assert not is_adaptive_model("claude-opus-4")
        assert not is_adaptive_model("claude-3-7-sonnet-latest")

    def test_no_prefill_model_detection(self) -> None:
        assert is_no_prefill_model("claude-opus-4-6")
        assert not is_no_prefill_model("claude-sonnet-4")

    def test_budget_min_1024(self) -> None:
        """Budget never goes below 1024 even at 0% effort."""
        budget = calculate_thinking_budget("min", 2000)
        assert budget == 1024

    def test_budget_medium(self) -> None:
        budget = calculate_thinking_budget("medium", 8000)
        assert budget == 2000  # 25% of 8000

    def test_budget_high(self) -> None:
        budget = calculate_thinking_budget("high", 10000)
        assert budget == 5000  # 50% of 10000

    def test_budget_max(self) -> None:
        budget = calculate_thinking_budget("max", 10000)
        assert budget == 9500  # 95% of 10000

    def test_budget_low_clamps_to_min(self) -> None:
        """Low effort with small max_tokens clamps to 1024."""
        budget = calculate_thinking_budget("low", 4000)
        assert budget == 1024  # 10% of 4000 = 400, clamped to 1024

    def test_adaptive_returns_effort(self) -> None:
        config = get_thinking_config("claude-opus-4-6", "medium", 8000)
        assert config["thinking"]["type"] == "adaptive"
        assert config["output_config"]["effort"] == "medium"
        assert "budget_tokens" not in config.get("thinking", {})

    def test_traditional_returns_budget(self) -> None:
        config = get_thinking_config("claude-3-7-sonnet-latest", "high", 10000)
        assert config["thinking"]["type"] == "enabled"
        assert config["thinking"]["budget_tokens"] == 5000


# ===================================================================
# TestClaudeCaching
# ===================================================================


class TestClaudeCaching:
    """Test prompt caching marker application."""

    def test_system_prompt_caching(self) -> None:
        system = [
            {"type": "text", "text": "Block 1"},
            {"type": "text", "text": "Block 2"},
        ]
        apply_prompt_caching(system, None)
        # Only last block gets cache_control
        assert "cache_control" not in system[0]
        assert system[1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    def test_tool_caching(self) -> None:
        tools = [
            {"name": "tool_a", "input_schema": {}},
            {"name": "tool_b", "input_schema": {}},
        ]
        apply_prompt_caching([], tools)
        assert "cache_control" not in tools[0]
        assert tools[1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    def test_custom_ttl(self) -> None:
        system = [{"type": "text", "text": "sys"}]
        apply_prompt_caching(system, None, cache_ttl="10m")
        assert system[0]["cache_control"]["ttl"] == "10m"

    def test_empty_lists_no_error(self) -> None:
        """Caching with empty lists should not raise."""
        apply_prompt_caching([], None)
        apply_prompt_caching([], [])


# ===================================================================
# TestClaudeResponseConversion
# ===================================================================


class TestClaudeResponseConversion:
    """Test convert_response normalisation."""

    def test_text_response(self) -> None:
        data = {
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = convert_response(data)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_thinking_response(self) -> None:
        data = {
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        result = convert_response(data)
        assert result["content"] == "Answer"
        assert result["thinking"] == "Let me think..."

    def test_tool_use_response(self) -> None:
        data = {
            "content": [
                {"type": "text", "text": "I'll search for that."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "search",
                    "input": {"query": "cats"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 30},
        }
        result = convert_response(data)
        assert result["finish_reason"] == "tool_calls"
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "toolu_01"
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == {"query": "cats"}

    def test_stop_reason_mapping(self) -> None:
        for claude_reason, expected in [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("stop_sequence", "stop"),
        ]:
            data = {
                "content": [{"type": "text", "text": "x"}],
                "stop_reason": claude_reason,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
            assert convert_response(data)["finish_reason"] == expected


# ===================================================================
# TestClaudeRequestBuilding
# ===================================================================


class TestClaudeRequestBuilding:
    """Test ClaudeBackend request body and headers."""

    def test_headers_include_api_key(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.claude import ClaudeBackend

        backend = ClaudeBackend(
            client=MagicMock(),
            api_key="sk-ant-test-key",
        )
        headers = backend._headers()
        assert headers["x-api-key"] == "sk-ant-test-key"
        assert headers["anthropic-version"] == "2023-06-01"

    def test_headers_beta_flags(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.claude import ClaudeBackend

        backend = ClaudeBackend(
            client=MagicMock(),
            api_key="sk-test",
            cache_enabled=True,
        )
        headers = backend._headers(tools=True, thinking=True)
        beta = headers.get("anthropic-beta", "")
        assert "tools-2024-04-04" in beta
        assert "interleaved-thinking-2025-05-14" in beta
        assert "prompt-caching-2024-07-31" in beta

    def test_full_request_body(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.claude import ClaudeBackend
        from augmentum.models.base import InternalChatRequest, Message

        backend = ClaudeBackend(
            client=MagicMock(),
            api_key="sk-test",
            cache_enabled=False,
        )
        request = InternalChatRequest(
            model="claude-sonnet-4",
            messages=[
                Message(role="system", content="Be brief."),
                Message(role="user", content="Hi"),
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        body = backend._build_request_body(request)
        assert body["model"] == "claude-sonnet-4"
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.7
        assert "system" in body
        assert body["system"][0]["text"] == "Be brief."
        assert len(body["messages"]) >= 1

    def test_tool_conversion(self) -> None:
        from augmentum.models.adapters.claude import ClaudeBackend

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        result = ClaudeBackend._convert_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather"
        assert "input_schema" in result[0]
        assert result[0]["input_schema"]["type"] == "object"

    def test_continue_last_assistant_passes_trailing_assistant_through(self) -> None:
        """Continue button on Claude: Anthropic's Messages API natively
        continues a trailing assistant — the adapter must NOT inject a
        synthetic user, and must rstrip whitespace from the partial so
        Anthropic's "final assistant content cannot end with trailing
        whitespace" 400 doesn't fire."""
        from unittest.mock import MagicMock

        from augmentum.models.adapters.claude import ClaudeBackend
        from augmentum.models.base import InternalChatRequest, Message

        backend = ClaudeBackend(
            client=MagicMock(),
            api_key="sk-test",
            cache_enabled=False,
        )
        # claude-opus-4-5 supports assistant prefill (only claude-*-4-6
        # variants are in _NO_PREFILL_MODEL_RE per converters/claude.py).
        request = InternalChatRequest(
            model="claude-opus-4-5",
            messages=[
                Message(role="user", content="Continue the story."),
                Message(role="assistant", content="Once upon a time   \n  "),
            ],
            max_tokens=1024,
            continue_last_assistant=True,
        )
        body = backend._build_request_body(request)

        # Trailing assistant survives, no synthetic user injected. The
        # Claude converter renders content as a list of typed blocks,
        # not a bare string — find the text block.
        last = body["messages"][-1]
        assert last["role"] == "assistant"
        text_blocks = [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "text"]
        assert text_blocks, "expected at least one text block on trailing assistant"
        final_text = text_blocks[-1]["text"]
        # Trailing whitespace stripped per Anthropic's "final assistant
        # content cannot end with trailing whitespace" constraint.
        assert final_text.rstrip() == final_text, (
            f"trailing whitespace not stripped: {final_text!r}"
        )

    def test_continue_last_assistant_falls_back_for_no_prefill_model(self) -> None:
        """Claude models flagged by is_no_prefill_model (some Haiku
        variants) don't support assistant prefill — fall back to a
        synthetic user "continue from where you left off" message."""
        from unittest.mock import MagicMock, patch

        from augmentum.models.adapters.claude import ClaudeBackend
        from augmentum.models.base import InternalChatRequest, Message

        backend = ClaudeBackend(
            client=MagicMock(),
            api_key="sk-test",
            cache_enabled=False,
        )
        request = InternalChatRequest(
            model="claude-3-5-haiku-latest",
            messages=[
                Message(role="user", content="Story?"),
                Message(role="assistant", content="Once upon a time"),
            ],
            max_tokens=1024,
            continue_last_assistant=True,
        )

        # Force is_no_prefill_model=True regardless of current regex —
        # we want to validate the fallback branch, not the model list.
        with patch(
            "augmentum.models.adapters.claude.is_no_prefill_model",
            return_value=True,
        ):
            body = backend._build_request_body(request)

        # Last message should be the synthetic user prompt, not the
        # original assistant partial.
        last = body["messages"][-1]
        assert last["role"] == "user"
        assert "continue" in (
            last["content"] if isinstance(last["content"], str)
            else " ".join(p.get("text", "") for p in last["content"])
        ).lower()

    def test_thinking_removes_temperature(self) -> None:
        """When thinking is enabled, temperature and top_p are removed."""
        from unittest.mock import MagicMock

        from augmentum.models.adapters.claude import ClaudeBackend
        from augmentum.models.base import InternalChatRequest, Message

        backend = ClaudeBackend(
            client=MagicMock(),
            api_key="sk-test",
            cache_enabled=False,
        )
        request = InternalChatRequest(
            model="claude-opus-4",
            messages=[Message(role="user", content="Think hard")],
            temperature=0.9,
            top_p=0.95,
            max_tokens=8000,
            think=True,
        )
        body = backend._build_request_body(request)
        assert "temperature" not in body
        assert "top_p" not in body
        assert "thinking" in body

    def test_context_length(self) -> None:
        """get_context_length returns correct values."""
        import asyncio
        from unittest.mock import MagicMock

        from augmentum.models.adapters.claude import ClaudeBackend

        backend = ClaudeBackend(
            client=MagicMock(),
            api_key="sk-test",
        )
        assert asyncio.get_event_loop().run_until_complete(
            backend.get_context_length("claude-opus-4-6")
        ) == 1_000_000
        assert asyncio.get_event_loop().run_until_complete(
            backend.get_context_length("claude-sonnet-4")
        ) == 200_000
