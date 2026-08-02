"""Contract tests for OpenAIBackend — verify URL construction, headers, and parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import (
    OpenAIBackend,
    to_openai_chat_response,
    to_openai_models_response,
)


def _make_backend(base_url="https://api.openai.com/v1", api_key="sk-test") -> tuple[OpenAIBackend, MagicMock]:
    client = MagicMock()
    backend = OpenAIBackend(client, base_url, api_key)
    return backend, client


def _make_request(**overrides) -> InternalChatRequest:
    defaults = {
        "model": "gpt-4",
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(overrides)
    return InternalChatRequest(**defaults)


class TestUrlConstruction:
    """Verify correct URL paths for all endpoints."""

    @pytest.mark.asyncio
    async def test_chat_url(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("openai_chat_completion.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        await backend.chat(_make_request())
        url = client.post.call_args[0][0]
        assert url == "https://api.openai.com/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_models_url(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("openai_models.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.get = AsyncMock(return_value=resp_mock)

        # Patch _probe_lmstudio_types to avoid side effects
        backend._probe_lmstudio_types = AsyncMock(return_value={})

        await backend.list_models()
        url = client.get.call_args[0][0]
        assert url == "https://api.openai.com/v1/models"

    @pytest.mark.asyncio
    async def test_show_model_url(self):
        backend, client = _make_backend()

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {"id": "gpt-4"}
        resp_mock.raise_for_status = MagicMock()
        client.get = AsyncMock(return_value=resp_mock)

        await backend.show_model("gpt-4")
        url = client.get.call_args[0][0]
        assert url == "https://api.openai.com/v1/models/gpt-4"


class TestAuthHeaders:
    """Verify Bearer auth header is set correctly."""

    def test_bearer_auth_present(self):
        backend, _ = _make_backend(api_key="sk-test-key")
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_no_auth_when_no_key(self):
        backend, _ = _make_backend(api_key=None)
        headers = backend._headers()
        assert "Authorization" not in headers

    def test_profile_auth_type_x_api_key(self):
        from augmentum.models.provider_profiles import ProviderProfile

        profile = ProviderProfile(
            id="test",
            name="Test",
            base_url="https://test.com/v1",
            auth_type="x-api-key",
            auth_header="x-api-key",
        )
        client = MagicMock()
        backend = OpenAIBackend(client, "https://test.com/v1", "my-key", profile=profile)
        headers = backend._headers()
        assert headers["x-api-key"] == "my-key"
        assert "Authorization" not in headers

    def test_profile_extra_headers(self):
        from augmentum.models.provider_profiles import ProviderProfile

        profile = ProviderProfile(
            id="openrouter",
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "https://augmentum.dev", "X-Title": "Augmentum"},
        )
        client = MagicMock()
        backend = OpenAIBackend(client, "https://openrouter.ai/api/v1", "sk-or-test", profile=profile)
        headers = backend._headers()
        assert headers["HTTP-Referer"] == "https://augmentum.dev"
        assert headers["X-Title"] == "Augmentum"


class TestPayloadShape:
    """Verify request body has correct OpenAI shape."""

    def test_payload_required_fields(self):
        backend, _ = _make_backend()
        req = _make_request()
        payload = backend._build_openai_payload(req)
        assert "model" in payload
        assert "messages" in payload
        assert "stream" in payload

    def test_payload_with_options(self):
        backend, _ = _make_backend()
        req = _make_request(temperature=0.5, top_p=0.9, max_tokens=100, seed=42)
        payload = backend._build_openai_payload(req)
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9
        assert payload["max_tokens"] == 100
        assert payload["seed"] == 42

    def test_payload_json_format(self):
        backend, _ = _make_backend()
        req = _make_request(format="json")
        payload = backend._build_openai_payload(req)
        assert payload["response_format"] == {"type": "json_object"}

    def test_payload_vision_content_array(self):
        backend, _ = _make_backend()
        req = _make_request(
            messages=[Message(role="user", content="describe", images=["data:image/png;base64,abc"])]
        )
        payload = backend._build_openai_payload(req)
        content = payload["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    def test_payload_strip_images(self):
        backend, _ = _make_backend()
        req = _make_request(
            messages=[Message(role="user", content="describe", images=["data:image/png;base64,abc"])]
        )
        payload = backend._build_openai_payload(req, strip_images=True)
        content = payload["messages"][0]["content"]
        assert isinstance(content, str)
        assert content == "describe"


class TestResponseParsing:
    """Verify OpenAI response JSON is parsed correctly."""

    @pytest.mark.asyncio
    async def test_parse_chat_response(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("openai_chat_completion.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        result = await backend.chat(_make_request())
        assert result.message.role == "assistant"
        assert "Hello" in result.message.content
        assert result.model == "gpt-4"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 9
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_model_list_parsing(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("openai_models.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        resp_mock.raise_for_status = MagicMock()
        client.get = AsyncMock(return_value=resp_mock)

        backend._probe_lmstudio_types = AsyncMock(return_value={})

        models = await backend.list_models()
        assert len(models) == 2
        assert models[0].name == "gpt-4"


class TestVisionRetryFallback:
    """Verify 400 + image_url triggers text-only retry."""

    @pytest.mark.asyncio
    async def test_vision_rejected_retries_without_images(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("openai_chat_completion.json")

        # First call returns 400 with image_url in body
        error_resp = MagicMock()
        error_resp.status_code = 400
        error_resp.text = "image_url is not supported"

        # Second call succeeds
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = fixture

        client.post = AsyncMock(side_effect=[error_resp, ok_resp])

        req = _make_request(
            messages=[Message(role="user", content="describe", images=["data:image/png;base64,abc"])]
        )
        result = await backend.chat(req)
        assert result.message.content is not None
        assert client.post.call_count == 2

        # Second call should have string content (images stripped)
        retry_payload = client.post.call_args_list[1][1]["json"]
        assert isinstance(retry_payload["messages"][0]["content"], str)

    @pytest.mark.asyncio
    async def test_non_vision_error_raises(self):
        backend, client = _make_backend()

        resp_mock = MagicMock()
        resp_mock.status_code = 500
        resp_mock.text = "Internal server error"
        client.post = AsyncMock(return_value=resp_mock)

        with pytest.raises(RuntimeError, match="500"):
            await backend.chat(_make_request())


class TestConverterFunctions:
    """Verify response format converter functions."""

    def test_to_openai_chat_response_shape(self):
        from augmentum.models.base import InternalChatResponse, Message, Usage

        internal = InternalChatResponse(
            message=Message(role="assistant", content="test response"),
            model="gpt-4",
            finish_reason="stop",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        result = to_openai_chat_response(internal)
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-4"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "test response"
        assert result["usage"]["total_tokens"] == 8

    def test_to_openai_models_response_shape(self):
        from augmentum.models.base import ModelInfo

        models = [ModelInfo(name="m1", model="m1"), ModelInfo(name="m2", model="m2")]
        result = to_openai_models_response(models)
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == "m1"


class TestCacheAndReasoningTelemetry:
    """Verify DeepSeek + OpenAI cache and reasoning fields land in Usage."""

    @pytest.mark.asyncio
    async def test_parse_deepseek_cache_telemetry(self):
        """DeepSeek puts cache hit/miss at the top of usage."""
        backend, client = _make_backend(base_url="https://api.deepseek.com/beta")
        deepseek_response = {
            "id": "cc-1",
            "model": "deepseek-chat",
            "choices": [{
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 1500,
                "completion_tokens": 50,
                "total_tokens": 1550,
                "prompt_cache_hit_tokens": 1200,
                "prompt_cache_miss_tokens": 300,
            },
        }
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = deepseek_response
        client.post = AsyncMock(return_value=resp_mock)

        result = await backend.chat(_make_request(model="deepseek-chat"))
        assert result.usage.prompt_tokens == 1500
        assert result.usage.cache_hit_tokens == 1200
        assert result.usage.cache_miss_tokens == 300

    @pytest.mark.asyncio
    async def test_parse_openai_nested_cache_telemetry(self):
        """OpenAI nests cached tokens under prompt_tokens_details.cached_tokens."""
        backend, client = _make_backend()
        openai_response = {
            "id": "cc-2",
            "model": "gpt-5.5",
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 2000,
                "completion_tokens": 80,
                "total_tokens": 2080,
                "prompt_tokens_details": {"cached_tokens": 1800},
                "completion_tokens_details": {"reasoning_tokens": 45},
            },
        }
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = openai_response
        client.post = AsyncMock(return_value=resp_mock)

        result = await backend.chat(_make_request(model="gpt-5.5"))
        assert result.usage.cache_hit_tokens == 1800
        assert result.usage.reasoning_tokens == 45

    @pytest.mark.asyncio
    async def test_missing_telemetry_defaults_to_zero(self):
        """Providers that don't report cache/reasoning leave fields at 0."""
        backend, client = _make_backend()
        plain_response = {
            "id": "cc-3",
            "model": "gpt-4",
            "choices": [{
                "message": {"role": "assistant", "content": "fine"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
            },
        }
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = plain_response
        client.post = AsyncMock(return_value=resp_mock)

        result = await backend.chat(_make_request())
        assert result.usage.cache_hit_tokens == 0
        assert result.usage.cache_miss_tokens == 0
        assert result.usage.reasoning_tokens == 0

    def test_usage_serializer_omits_zero_cache_and_reasoning(self):
        """to_openai_chat_response should not emit zero-valued cache/reasoning."""
        from augmentum.models.base import InternalChatResponse, Message, Usage

        resp = InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="gpt-4",
            finish_reason="stop",
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        out = to_openai_chat_response(resp)
        assert "prompt_cache_hit_tokens" not in out["usage"]
        assert "prompt_cache_miss_tokens" not in out["usage"]
        assert "completion_tokens_details" not in out["usage"]

    def test_usage_serializer_emits_nonzero_telemetry(self):
        from augmentum.models.base import InternalChatResponse, Message, Usage

        resp = InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="deepseek-chat",
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=1500, completion_tokens=50, total_tokens=1550,
                cache_hit_tokens=1200, cache_miss_tokens=300, reasoning_tokens=15,
            ),
        )
        out = to_openai_chat_response(resp)
        assert out["usage"]["prompt_cache_hit_tokens"] == 1200
        assert out["usage"]["prompt_cache_miss_tokens"] == 300
        assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 15

    @pytest.mark.asyncio
    async def test_streaming_final_chunk_carries_cache_telemetry(self):
        """The include_usage final SSE chunk's usage block must populate
        Usage.cache_hit_tokens / cache_miss_tokens / reasoning_tokens."""
        from augmentum.utils.thinking import ThinkingStreamBuffer

        backend, _ = _make_backend(base_url="https://api.deepseek.com/beta")

        # Hand-rolled SSE stream: content delta, then the final
        # include_usage chunk with DeepSeek-shaped telemetry.
        sse_lines = [
            'data: {"choices":[{"delta":{"role":"assistant","content":"hi"}}],"model":"deepseek-chat"}',
            'data: {"choices":[],"usage":{"prompt_tokens":1500,"completion_tokens":50,'
            '"total_tokens":1550,"prompt_cache_hit_tokens":1200,'
            '"prompt_cache_miss_tokens":300,'
            '"completion_tokens_details":{"reasoning_tokens":42}}}',
            'data: [DONE]',
        ]

        class _FakeResp:
            async def aiter_lines(self):
                for line in sse_lines:
                    yield line

        thinking_buf = ThinkingStreamBuffer(
            model="deepseek-chat", thinking_enabled=False, preserve_thinking=False,
        )

        chunks = []
        async for chunk in backend._iter_stream(_FakeResp(), thinking_buf):
            chunks.append(chunk)

        usage_chunks = [c for c in chunks if c.usage is not None]
        assert usage_chunks, "expected at least one chunk carrying usage"
        usage = usage_chunks[-1].usage
        assert usage.prompt_tokens == 1500
        assert usage.cache_hit_tokens == 1200
        assert usage.cache_miss_tokens == 300
        assert usage.reasoning_tokens == 42
