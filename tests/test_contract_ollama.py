"""Contract tests for OllamaBackend — verify URL paths, payloads, and parsing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.ollama import OllamaBackend


def _make_backend(mock_client=None) -> tuple[OllamaBackend, MagicMock]:
    client = mock_client or MagicMock()
    backend = OllamaBackend(client, "http://ollama:11434")
    return backend, client


def _make_request(**overrides) -> InternalChatRequest:
    defaults = {
        "model": "llama3.1:8b",
        "messages": [Message(role="user", content="hello")],
    }
    defaults.update(overrides)
    return InternalChatRequest(**defaults)


class TestChatEndpoint:
    """Verify /api/chat is called with correct payload shape."""

    @pytest.mark.asyncio
    async def test_chat_posts_to_correct_url(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("ollama_chat.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        resp_mock.raise_for_status = MagicMock()
        client.post = AsyncMock(return_value=resp_mock)

        req = _make_request()
        await backend.chat(req)

        client.post.assert_called_once()
        call_args = client.post.call_args
        assert call_args[0][0] == "http://ollama:11434/api/chat"

    @pytest.mark.asyncio
    async def test_chat_payload_contains_required_fields(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("ollama_chat.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        resp_mock.raise_for_status = MagicMock()
        client.post = AsyncMock(return_value=resp_mock)

        req = _make_request(temperature=0.5)
        await backend.chat(req)

        payload = client.post.call_args[1]["json"]
        assert payload["model"] == "llama3.1:8b"
        assert isinstance(payload["messages"], list)
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_chat_payload_includes_tools_when_present(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("ollama_chat.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        resp_mock.raise_for_status = MagicMock()
        client.post = AsyncMock(return_value=resp_mock)

        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        req = _make_request(tools=tools)
        await backend.chat(req)

        payload = client.post.call_args[1]["json"]
        assert payload["tools"] == tools

    @pytest.mark.asyncio
    async def test_chat_payload_includes_think_flag(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("ollama_chat.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        resp_mock.raise_for_status = MagicMock()
        client.post = AsyncMock(return_value=resp_mock)

        req = _make_request(think=True)
        await backend.chat(req)

        payload = client.post.call_args[1]["json"]
        assert payload["think"] is True

    @pytest.mark.asyncio
    async def test_chat_response_parsing(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("ollama_chat.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        resp_mock.raise_for_status = MagicMock()
        client.post = AsyncMock(return_value=resp_mock)

        result = await backend.chat(_make_request())
        assert result.message.role == "assistant"
        assert result.message.content is not None
        assert result.model == "llama3.1:8b"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 8
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_chat_raises_on_error(self):
        backend, client = _make_backend()

        resp_mock = MagicMock()
        resp_mock.status_code = 500
        resp_mock.raise_for_status.side_effect = Exception("Server error")
        client.post = AsyncMock(return_value=resp_mock)

        with pytest.raises(Exception):
            await backend.chat(_make_request())


class TestListModels:
    """Verify /api/tags is called and response is parsed."""

    @pytest.mark.asyncio
    async def test_list_models_url(self, load_fixture):
        backend, client = _make_backend()
        fixture = load_fixture("ollama_tags.json")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        resp_mock.raise_for_status = MagicMock()
        client.get = AsyncMock(return_value=resp_mock)

        models = await backend.list_models()
        client.get.assert_called_once_with("http://ollama:11434/api/tags")
        assert len(models) == 1
        assert models[0].name == "llama3.1:8b"

    @pytest.mark.asyncio
    async def test_list_models_detects_vision(self):
        backend, client = _make_backend()
        fixture = {
            "models": [
                {"name": "llava:7b", "details": {"families": ["llama", "clip"]}, "size": 1000},
            ]
        }
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = fixture
        resp_mock.raise_for_status = MagicMock()
        client.get = AsyncMock(return_value=resp_mock)

        models = await backend.list_models()
        assert models[0].vision is True


class TestShowModel:
    """Verify /api/show is called correctly."""

    @pytest.mark.asyncio
    async def test_show_model_url_and_payload(self):
        backend, client = _make_backend()

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "modelfile": "FROM llama3",
            "parameters": "temp 0.7",
            "template": "",
            "details": {"family": "llama"},
            "model_info": {},
        }
        resp_mock.raise_for_status = MagicMock()
        client.post = AsyncMock(return_value=resp_mock)

        details = await backend.show_model("llama3.1:8b")
        client.post.assert_called_once_with(
            "http://ollama:11434/api/show",
            json={"name": "llama3.1:8b"},
        )
        assert details.modelfile == "FROM llama3"


class TestPayloadMapping:
    """Verify InternalChatRequest fields map to correct Ollama payload keys."""

    def test_max_tokens_maps_to_num_predict(self):
        backend, _ = _make_backend()
        req = _make_request(max_tokens=512)
        payload = backend._build_ollama_payload(req)
        assert payload["options"]["num_predict"] == 512

    def test_frequency_penalty_maps_to_repeat_penalty(self):
        backend, _ = _make_backend()
        req = _make_request(frequency_penalty=1.2)
        payload = backend._build_ollama_payload(req)
        assert payload["options"]["repeat_penalty"] == 1.2

    def test_format_json_passthrough(self):
        backend, _ = _make_backend()
        req = _make_request(format="json")
        payload = backend._build_ollama_payload(req)
        assert payload["format"] == "json"

    def test_images_included_in_messages(self):
        backend, _ = _make_backend()
        req = _make_request(
            messages=[Message(role="user", content="look", images=["data:image/png;base64,abc"])]
        )
        payload = backend._build_ollama_payload(req)
        assert payload["messages"][0]["images"] == ["data:image/png;base64,abc"]
