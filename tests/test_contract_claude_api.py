"""Contract tests for ClaudeBackend — verify headers, beta flags, body schema, parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.models.adapters.claude import ClaudeBackend
from augmentum.models.base import InternalChatRequest, Message


def _make_backend(**overrides) -> tuple[ClaudeBackend, MagicMock]:
    client = MagicMock()
    defaults = {
        "client": client,
        "api_key": "sk-ant-test-key",
    }
    defaults.update(overrides)
    backend = ClaudeBackend(**defaults)
    return backend, client


def _make_request(**overrides) -> InternalChatRequest:
    defaults = {
        "model": "claude-sonnet-4-20250514",
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(overrides)
    return InternalChatRequest(**defaults)


class TestHeaders:
    """Verify x-api-key and anthropic-version headers."""

    def test_api_key_header(self):
        backend, _ = _make_backend()
        headers = backend._headers()
        assert headers["x-api-key"] == "sk-ant-test-key"

    def test_anthropic_version_header(self):
        backend, _ = _make_backend()
        headers = backend._headers()
        assert headers["anthropic-version"] == "2023-06-01"

    def test_content_type_json(self):
        backend, _ = _make_backend()
        headers = backend._headers()
        assert headers["Content-Type"] == "application/json"


class TestBetaFlags:
    """Verify beta flags are set correctly."""

    def test_tools_beta_flag(self):
        backend, _ = _make_backend()
        headers = backend._headers(tools=True)
        assert "tools-2024-04-04" in headers["anthropic-beta"]

    def test_thinking_beta_flag(self):
        backend, _ = _make_backend()
        headers = backend._headers(thinking=True)
        assert "interleaved-thinking-2025-05-14" in headers["anthropic-beta"]

    def test_caching_beta_flag(self):
        backend, _ = _make_backend(cache_enabled=True)
        headers = backend._headers()
        assert "prompt-caching-2024-07-31" in headers["anthropic-beta"]

    def test_no_beta_when_disabled(self):
        backend, _ = _make_backend(cache_enabled=False)
        headers = backend._headers()
        assert "anthropic-beta" not in headers

    def test_multiple_beta_flags(self):
        backend, _ = _make_backend(cache_enabled=True)
        headers = backend._headers(tools=True, thinking=True)
        beta = headers["anthropic-beta"]
        assert "tools-2024-04-04" in beta
        assert "interleaved-thinking-2025-05-14" in beta
        assert "prompt-caching-2024-07-31" in beta


class TestRequestBodySchema:
    """Verify Messages API request body shape."""

    @pytest.mark.asyncio
    async def test_chat_posts_to_messages_endpoint(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("claude_messages.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        await backend.chat(_make_request())
        url = client.post.call_args[0][0]
        assert url.endswith("/messages")

    @pytest.mark.asyncio
    async def test_body_has_required_fields(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("claude_messages.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        await backend.chat(_make_request(max_tokens=1024))
        body = client.post.call_args[1]["json"]
        assert "model" in body
        assert "messages" in body
        assert "max_tokens" in body
        assert body["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_system_messages_extracted(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("claude_messages.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        req = _make_request(
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="hi"),
            ]
        )
        await backend.chat(req)
        body = client.post.call_args[1]["json"]
        # System should be extracted to the top-level 'system' key
        assert "system" in body

    @pytest.mark.asyncio
    async def test_tools_converted_to_claude_format(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("claude_messages.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        tools = [{"type": "function", "function": {"name": "search", "description": "Search", "parameters": {}}}]
        req = _make_request(tools=tools)
        await backend.chat(req)
        body = client.post.call_args[1]["json"]
        assert "tools" in body
        # Claude format: name, description, input_schema (not function wrapper)
        assert body["tools"][0]["name"] == "search"
        assert "input_schema" in body["tools"][0]


class TestThinkingConfig:
    """Verify thinking config is added for thinking models."""

    @pytest.mark.asyncio
    async def test_thinking_added_for_thinking_model(self, load_fixture):
        backend, client = _make_backend(thinking_effort="medium")
        fixture = load_fixture("claude_messages.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        req = _make_request(model="claude-sonnet-4-20250514", think=True)
        await backend.chat(req)
        body = client.post.call_args[1]["json"]
        assert "thinking" in body

    @pytest.mark.asyncio
    async def test_temperature_removed_with_thinking(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("claude_messages.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        req = _make_request(model="claude-sonnet-4-20250514", think=True, temperature=0.7)
        await backend.chat(req)
        body = client.post.call_args[1]["json"]
        assert "temperature" not in body


class TestResponseParsing:
    """Verify Claude response parsing."""

    @pytest.mark.asyncio
    async def test_parse_response(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("claude_messages.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        result = await backend.chat(_make_request())
        assert result.message.role == "assistant"
        assert result.message.content is not None
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_error_raises_runtime_error(self):
        backend, client = _make_backend()

        resp_mock = MagicMock()
        resp_mock.status_code = 429
        resp_mock.text = "Rate limit exceeded"
        client.post = AsyncMock(return_value=resp_mock)

        with pytest.raises(RuntimeError, match="429"):
            await backend.chat(_make_request())


class TestModelListing:
    """Verify hardcoded model list."""

    @pytest.mark.asyncio
    async def test_list_returns_known_models(self):
        backend, _ = _make_backend()
        models = await backend.list_models()
        names = [m.name for m in models]
        assert "claude-sonnet-4-6" in names
        assert "claude-opus-4-6" in names
        assert all(m.vision is True for m in models)

    @pytest.mark.asyncio
    async def test_context_length_4_6(self):
        backend, _ = _make_backend()
        ctx = await backend.get_context_length("claude-opus-4-6")
        assert ctx == 1_000_000

    @pytest.mark.asyncio
    async def test_context_length_older(self):
        backend, _ = _make_backend()
        ctx = await backend.get_context_length("claude-3-opus-latest")
        assert ctx == 200_000


class TestConverterHelpers:
    """Verify Claude-specific converter utility functions."""

    def test_is_thinking_model_positive(self):
        from augmentum.models.converters.claude import is_thinking_model

        assert is_thinking_model("claude-sonnet-4-20250514") is True
        assert is_thinking_model("claude-opus-4-6") is True

    def test_is_thinking_model_negative(self):
        from augmentum.models.converters.claude import is_thinking_model

        assert is_thinking_model("claude-3-opus-20240229") is False

    def test_is_no_prefill_model(self):
        from augmentum.models.converters.claude import is_no_prefill_model

        assert is_no_prefill_model("claude-opus-4-6") is True
        assert is_no_prefill_model("claude-sonnet-4-20250514") is False

    def test_convert_response(self):
        from augmentum.models.converters.claude import convert_response

        data = {
            "content": [{"type": "text", "text": "hello world"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = convert_response(data)
        assert result["content"] == "hello world"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
