"""Contract tests for GeminiBackend — verify URL format, body shape, and parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.models.adapters.gemini import GeminiBackend
from augmentum.models.base import InternalChatRequest, Message


def _make_backend(**overrides) -> tuple[GeminiBackend, MagicMock]:
    client = MagicMock()
    defaults = {
        "client": client,
        "api_key": "test-gemini-key",
    }
    defaults.update(overrides)
    backend = GeminiBackend(**defaults)
    return backend, client


def _make_request(**overrides) -> InternalChatRequest:
    defaults = {
        "model": "gemini-2.0-flash",
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(overrides)
    return InternalChatRequest(**defaults)


class TestUrlConstruction:
    """Verify AI Studio URL has key in query param."""

    def test_ai_studio_url_format(self):
        backend, _ = _make_backend()
        url = backend._endpoint("gemini-2.0-flash", "generateContent")
        assert "key=test-gemini-key" in url
        assert "models/gemini-2.0-flash:generateContent" in url

    def test_ai_studio_stream_url_includes_alt_sse(self):
        backend, _ = _make_backend()
        url = backend._endpoint("gemini-2.0-flash", "streamGenerateContent", stream=True)
        assert "alt=sse" in url
        assert "key=test-gemini-key" in url

    def test_vertex_url_format(self):
        backend, _ = _make_backend(
            vertex=True, vertex_project="my-project", vertex_region="us-east1",
        )
        url = backend._endpoint("gemini-2.0-flash", "generateContent")
        assert "us-east1-aiplatform.googleapis.com" in url
        assert "projects/my-project" in url
        assert "models/gemini-2.0-flash:generateContent" in url
        assert "key=" not in url


class TestHeaders:
    """Verify headers for AI Studio vs Vertex."""

    def test_ai_studio_no_bearer(self):
        backend, _ = _make_backend()
        headers = backend._headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_vertex_has_bearer(self):
        backend, _ = _make_backend(vertex=True)
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer test-gemini-key"


class TestBodyFormat:
    """Verify generateContent body format."""

    @pytest.mark.asyncio
    async def test_chat_body_has_contents(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("gemini_generate.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        await backend.chat(_make_request())
        body = client.post.call_args[1]["json"]
        assert "contents" in body
        assert "safetySettings" in body

    @pytest.mark.asyncio
    async def test_body_generation_config(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("gemini_generate.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        req = _make_request(temperature=0.7, max_tokens=256, top_p=0.9)
        await backend.chat(req)
        body = client.post.call_args[1]["json"]
        gen_config = body.get("generationConfig", {})
        assert gen_config["temperature"] == 0.7
        assert gen_config["maxOutputTokens"] == 256
        assert gen_config["topP"] == 0.9

    @pytest.mark.asyncio
    async def test_system_instruction_extracted(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("gemini_generate.json")

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
        assert "systemInstruction" in body

    @pytest.mark.asyncio
    async def test_tools_converted_to_function_declarations(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("gemini_generate.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        tools = [{"type": "function", "function": {"name": "search", "description": "Search web", "parameters": {}}}]
        req = _make_request(tools=tools)
        await backend.chat(req)
        body = client.post.call_args[1]["json"]
        assert "tools" in body
        assert "function_declarations" in body["tools"][0]
        assert body["tools"][0]["function_declarations"][0]["name"] == "search"


class TestSafetySettings:
    """Verify safety settings are included."""

    def test_ai_studio_safety_settings(self):
        from augmentum.models.converters.gemini import get_safety_settings

        settings = get_safety_settings(vertex=False)
        assert len(settings) == 5
        assert all(s["threshold"] == "OFF" for s in settings)

    def test_vertex_includes_extra_categories(self):
        from augmentum.models.converters.gemini import get_safety_settings

        settings = get_safety_settings(vertex=True)
        assert len(settings) > 5
        categories = [s["category"] for s in settings]
        assert "HARM_CATEGORY_JAILBREAK" in categories


class TestResponseParsing:
    """Verify Gemini response parsing."""

    @pytest.mark.asyncio
    async def test_parse_response(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("gemini_generate.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        client.post = AsyncMock(return_value=resp_mock)

        result = await backend.chat(_make_request())
        assert result.message.role == "assistant"
        assert "Hello" in result.message.content
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10

    @pytest.mark.asyncio
    async def test_error_raises_runtime_error(self):
        backend, client = _make_backend()

        resp_mock = MagicMock()
        resp_mock.status_code = 403
        resp_mock.text = "API key invalid"
        client.post = AsyncMock(return_value=resp_mock)

        with pytest.raises(RuntimeError, match="403"):
            await backend.chat(_make_request())


class TestConverterFunctions:
    """Verify Gemini converter utility functions."""

    def test_convert_response(self):
        from augmentum.models.converters.gemini import convert_response

        data = {
            "candidates": [{
                "content": {"parts": [{"text": "hello world"}], "role": "model"},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        result = convert_response(data)
        assert result["content"] == "hello world"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["total_tokens"] == 15

    def test_convert_response_empty_candidates(self):
        from augmentum.models.converters.gemini import convert_response

        data = {"candidates": []}
        result = convert_response(data)
        assert result["content"] == ""

    def test_finish_reason_map(self):
        from augmentum.models.converters.gemini import _FINISH_REASON_MAP

        assert _FINISH_REASON_MAP["STOP"] == "stop"
        assert _FINISH_REASON_MAP["MAX_TOKENS"] == "length"
        assert _FINISH_REASON_MAP["SAFETY"] == "content_filter"

    def test_thinking_config_flash(self):
        from augmentum.models.converters.gemini import get_thinking_config

        config = get_thinking_config("gemini-2.5-flash-latest", "medium")
        assert config is not None
        assert "thinkingConfig" in config
        assert "thinkingBudget" in config["thinkingConfig"]

    def test_thinking_config_unsupported_model(self):
        from augmentum.models.converters.gemini import get_thinking_config

        config = get_thinking_config("gemini-1.0-pro", "medium")
        assert config is None


class TestModelListing:
    """Verify model list endpoint and parsing."""

    @pytest.mark.asyncio
    async def test_list_models_url(self):
        backend, client = _make_backend()

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "models": [
                {
                    "name": "models/gemini-2.0-flash",
                    "supportedGenerationMethods": ["generateContent"],
                    "inputTokenLimit": 1000000,
                },
            ]
        }
        client.get = AsyncMock(return_value=resp_mock)

        models = await backend.list_models()
        url = client.get.call_args[0][0]
        assert "key=test-gemini-key" in url
        assert "/models" in url
        assert len(models) == 1
        # "models/" prefix should be stripped
        assert models[0].name == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_list_models_filters_non_generate(self):
        backend, client = _make_backend()

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "models": [
                {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
                {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
            ]
        }
        client.get = AsyncMock(return_value=resp_mock)

        models = await backend.list_models()
        assert len(models) == 1
        assert models[0].name == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_context_length_flash(self):
        backend, _ = _make_backend()
        ctx = await backend.get_context_length("gemini-2.0-flash")
        assert ctx == 1_000_000

    @pytest.mark.asyncio
    async def test_context_length_pro(self):
        backend, _ = _make_backend()
        ctx = await backend.get_context_length("gemini-2.5-pro")
        assert ctx == 2_000_000
