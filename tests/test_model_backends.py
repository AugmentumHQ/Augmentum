"""Tests for model backend implementations and provider registry."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelDetails,
    ModelInfo,
    Usage,
)
from augmentum.models.kv_session_manifest import KVSessionManifest
from augmentum.models.llama_cpp import LlamaCppBackend
from augmentum.models.model_manager import ModelManager, ModelStatus, RunningModel
from augmentum.models.ollama import OllamaBackend
from augmentum.models.openai_compat import (
    OpenAIBackend,
    to_openai_chat_response,
    to_openai_models_response,
    to_openai_stream_chunk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**overrides) -> InternalChatRequest:
    """Build a simple InternalChatRequest."""
    defaults = {
        "model": "test-model",
        "messages": [Message(role="user", content="Hello")],
        "stream": False,
    }
    defaults.update(overrides)
    return InternalChatRequest(**defaults)


def _mock_response(status_code: int = 200, json_data: dict | None = None, text: str = "") -> httpx.Response:
    """Build a fake httpx.Response."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        text=text,
        request=httpx.Request("POST", "http://test"),
    )


def _make_inflight_manager() -> MagicMock:
    """MagicMock manager with a working ``request_in_flight()``.

    The real LlamaServerManager exposes ``request_in_flight`` as an
    async context manager that increments/decrements an in-flight
    counter. Tests that mock the manager need that interface to be
    awaitable; otherwise ``async with manager.request_in_flight():``
    raises a TypeError.

    The returned MagicMock also tracks the live counter on
    ``in_flight_during`` (a list of values observed inside the ctx) so
    tests can assert request_in_flight was actually entered.
    """
    manager = MagicMock()
    counter = {"count": 0}
    manager.in_flight_during = []  # captured by callers via side_effect

    @contextlib.asynccontextmanager
    async def _in_flight():
        counter["count"] += 1
        manager.in_flight_during.append(counter["count"])
        try:
            yield
        finally:
            counter["count"] -= 1

    manager.request_in_flight = _in_flight
    manager._in_flight_count_view = lambda: counter["count"]
    return manager


class MockTransport(httpx.AsyncBaseTransport):
    """Programmable async transport for httpx.AsyncClient tests."""

    def __init__(self) -> None:
        self.responses: dict[str, httpx.Response] = {}
        self.requests: list[httpx.Request] = []

    def add_response(self, method: str, url: str, *, status: int = 200, json_data: dict | None = None) -> None:
        key = f"{method.upper()} {url}"
        body = json.dumps(json_data or {}).encode()
        self.responses[key] = httpx.Response(
            status_code=status,
            content=body,
            headers={"content-type": "application/json"},
            request=httpx.Request(method, url),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {str(request.url)}"
        if key in self.responses:
            return self.responses[key]
        # Fallback: return 404
        return httpx.Response(404, request=request, text="Not found")


# ===========================================================================
# Ollama Backend
# ===========================================================================

class TestOllamaBackend:
    """Tests for OllamaBackend."""

    def _make_backend(self, transport: MockTransport | None = None) -> tuple[OllamaBackend, MockTransport]:
        t = transport or MockTransport()
        client = httpx.AsyncClient(transport=t)
        return OllamaBackend(client, "http://ollama:11434"), t

    @pytest.mark.asyncio
    async def test_chat_basic(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://ollama:11434/api/chat", json_data={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "Hi there!"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 15,
            "eval_count": 8,
        })

        req = _make_request(model="llama3.1:8b")
        result = await backend.chat(req)

        assert isinstance(result, InternalChatResponse)
        assert result.message.role == "assistant"
        assert result.message.content == "Hi there!"
        assert result.model == "llama3.1:8b"
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 15
        assert result.usage.completion_tokens == 8
        assert result.usage.total_tokens == 23

    @pytest.mark.asyncio
    async def test_chat_with_options(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://ollama:11434/api/chat", json_data={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "Ok"},
            "done": True,
        })

        req = _make_request(
            model="llama3.1:8b",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["END"],
            frequency_penalty=1.2,
            seed=42,
            format="json",
            keep_alive="10m",
            raw_options={"mirostat": 2},
        )
        await backend.chat(req)

        # Verify the payload sent
        sent = transport.requests[-1]
        body = json.loads(sent.content)
        assert body["model"] == "llama3.1:8b"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.7
        assert body["options"]["top_p"] == 0.9
        assert body["options"]["num_predict"] == 100
        assert body["options"]["stop"] == ["END"]
        assert body["options"]["repeat_penalty"] == 1.2
        assert body["options"]["seed"] == 42
        assert body["options"]["mirostat"] == 2
        assert body["format"] == "json"
        assert body["keep_alive"] == "10m"

    @pytest.mark.asyncio
    async def test_chat_with_images(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://ollama:11434/api/chat", json_data={
            "model": "llava",
            "message": {"role": "assistant", "content": "A cat"},
            "done": True,
        })

        req = _make_request(
            model="llava",
            messages=[Message(role="user", content="What's this?", images=["base64data"])],
        )
        await backend.chat(req)

        sent = transport.requests[-1]
        body = json.loads(sent.content)
        assert body["messages"][0]["images"] == ["base64data"]

    @pytest.mark.asyncio
    async def test_chat_with_tools(self):
        backend, transport = self._make_backend()
        tool_calls = [{"function": {"name": "search", "arguments": '{"q": "test"}'}}]
        transport.add_response("POST", "http://ollama:11434/api/chat", json_data={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
            "done": True,
        })

        req = _make_request(tools=[{"type": "function", "function": {"name": "search"}}])
        result = await backend.chat(req)
        assert result.message.tool_calls == tool_calls

    @pytest.mark.asyncio
    async def test_chat_timing_stats(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://ollama:11434/api/chat", json_data={
            "model": "llama3.1:8b",
            "message": {"role": "assistant", "content": "Hi"},
            "done": True,
            "total_duration": 5000000000,
            "load_duration": 1000000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 2000000000,
            "eval_count": 5,
            "eval_duration": 3000000000,
        })

        req = _make_request()
        result = await backend.chat(req)
        assert result.timing is not None
        assert result.timing["total_duration"] == 5000000000
        assert result.timing["load_duration"] == 1000000000

    @pytest.mark.asyncio
    async def test_list_models(self):
        backend, transport = self._make_backend()
        transport.add_response("GET", "http://ollama:11434/api/tags", json_data={
            "models": [
                {
                    "name": "llama3.1:8b",
                    "model": "llama3.1:8b",
                    "size": 4_000_000_000,
                    "digest": "abc123",
                    "modified_at": "2024-01-01T00:00:00Z",
                    "details": {"family": "llama"},
                },
                {
                    "name": "mistral:7b",
                    "model": "mistral:7b",
                    "size": 3_500_000_000,
                    "digest": "def456",
                    "modified_at": "2024-02-01T00:00:00Z",
                },
            ],
        })

        models = await backend.list_models()
        assert len(models) == 2
        assert models[0].name == "llama3.1:8b"
        assert models[0].size == 4_000_000_000
        assert models[0].details == {"family": "llama"}
        assert models[1].name == "mistral:7b"

    @pytest.mark.asyncio
    async def test_show_model(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://ollama:11434/api/show", json_data={
            "modelfile": "FROM llama3.1:8b",
            "parameters": "temperature 0.7",
            "template": "{{ .System }}",
            "details": {"family": "llama", "parameter_size": "8B"},
            "model_info": {"general.architecture": "llama"},
        })

        details = await backend.show_model("llama3.1:8b")
        assert isinstance(details, ModelDetails)
        assert details.modelfile == "FROM llama3.1:8b"
        assert details.parameters == "temperature 0.7"
        assert details.details == {"family": "llama", "parameter_size": "8B"}

    @pytest.mark.asyncio
    async def test_build_payload_minimal(self):
        backend, _ = self._make_backend()
        req = _make_request()
        payload = backend._build_ollama_payload(req)

        assert payload["model"] == "test-model"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert "options" not in payload  # No options when none set

    @pytest.mark.asyncio
    async def test_parse_empty_response(self):
        backend, _ = self._make_backend()
        result = backend._parse_ollama_response({})

        assert result.message.role == "assistant"
        assert result.message.content == ""
        assert result.model == ""
        assert result.usage.total_tokens == 0


# ===========================================================================
# OpenAI Backend
# ===========================================================================

class TestOpenAIBackend:
    """Tests for OpenAIBackend."""

    def _make_backend(self, api_key: str | None = "sk-test") -> tuple[OpenAIBackend, MockTransport]:
        t = MockTransport()
        client = httpx.AsyncClient(transport=t)
        return OpenAIBackend(client, "https://api.openai.com/v1", api_key), t

    def _backend_with_max_output(self, max_output: int) -> OpenAIBackend:
        from augmentum.models.provider_profiles import ProviderProfile
        b, _ = self._make_backend()
        b._profile = ProviderProfile(
            id="t", name="t", base_url="https://api.openai.com/v1",
            max_output=max_output,
        )
        return b

    def test_coder_budget_raises_capable_model_to_floor(self):
        """DeepSeek-like: huge ceiling → a coder request (8192) is raised to the
        cloud floor so a large file_write fits in one response."""
        from unittest.mock import patch
        b = self._backend_with_max_output(384_000)
        with patch("augmentum.config.settings") as s:
            s.coder_cloud_max_tokens_floor = 32768
            assert b._coder_output_budget(8192) == 32768

    def test_coder_budget_clamps_small_cap_model(self):
        """Cohere-R+-like: 4096 ceiling → the 8192 coder ask is clamped DOWN to
        4096 so we never send a value the API rejects."""
        from unittest.mock import patch
        b = self._backend_with_max_output(4096)
        with patch("augmentum.config.settings") as s:
            s.coder_cloud_max_tokens_floor = 32768
            assert b._coder_output_budget(8192) == 4096

    def test_coder_budget_leaves_tight_modes_alone(self):
        """A small request (analytical's 512-tok UARF) stays put — only
        large-output (>=4096) requests get the coder floor."""
        from unittest.mock import patch
        b = self._backend_with_max_output(128_000)
        with patch("augmentum.config.settings") as s:
            s.coder_cloud_max_tokens_floor = 32768
            assert b._coder_output_budget(512) == 512

    def test_coder_budget_unknown_ceiling_is_status_quo(self):
        """max_output=0 (aggregators / undocumented) → never raised or clamped;
        we don't invent a ceiling."""
        from unittest.mock import patch
        b = self._backend_with_max_output(0)
        with patch("augmentum.config.settings") as s:
            s.coder_cloud_max_tokens_floor = 32768
            assert b._coder_output_budget(8192) == 8192

    @pytest.mark.asyncio
    async def test_chat_basic(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "https://api.openai.com/v1/chat/completions", json_data={
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

        req = _make_request(model="gpt-4")
        result = await backend.chat(req)

        assert result.message.content == "Hello!"
        assert result.model == "gpt-4"
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_headers_with_api_key(self):
        backend, transport = self._make_backend(api_key="sk-secret")
        transport.add_response("POST", "https://api.openai.com/v1/chat/completions", json_data={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {},
        })

        await backend.chat(_make_request())
        sent = transport.requests[-1]
        assert sent.headers["authorization"] == "Bearer sk-secret"
        assert sent.headers["content-type"] == "application/json"

    @pytest.mark.asyncio
    async def test_headers_without_api_key(self):
        backend, transport = self._make_backend(api_key=None)
        transport.add_response("POST", "https://api.openai.com/v1/chat/completions", json_data={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {},
        })

        await backend.chat(_make_request())
        sent = transport.requests[-1]
        assert "authorization" not in sent.headers

    @pytest.mark.asyncio
    async def test_chat_with_options(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "https://api.openai.com/v1/chat/completions", json_data={
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {},
        })

        req = _make_request(
            temperature=0.5,
            top_p=0.8,
            max_tokens=200,
            stop=["<END>"],
            frequency_penalty=0.5,
            presence_penalty=0.3,
            seed=123,
            format="json",
        )
        await backend.chat(req)

        body = json.loads(transport.requests[-1].content)
        assert body["temperature"] == 0.5
        assert body["top_p"] == 0.8
        assert body["max_tokens"] == 200
        assert body["stop"] == ["<END>"]
        assert body["frequency_penalty"] == 0.5
        assert body["presence_penalty"] == 0.3
        assert body["seed"] == 123
        assert body["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self):
        backend, transport = self._make_backend()
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
        transport.add_response("POST", "https://api.openai.com/v1/chat/completions", json_data={
            "choices": [{
                "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        })

        req = _make_request(tools=[{"type": "function", "function": {"name": "search"}}])
        result = await backend.chat(req)
        assert result.message.tool_calls == tool_calls
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_list_models(self):
        backend, transport = self._make_backend()
        transport.add_response("GET", "https://api.openai.com/v1/models", json_data={
            "data": [
                {"id": "gpt-4", "object": "model", "created": 1700000000},
                {"id": "gpt-3.5-turbo", "object": "model", "created": 1690000000},
            ],
        })

        models = await backend.list_models()
        assert len(models) == 2
        assert models[0].name == "gpt-4"
        assert models[1].name == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_show_model(self):
        backend, transport = self._make_backend()
        transport.add_response("GET", "https://api.openai.com/v1/models/gpt-4", json_data={
            "id": "gpt-4",
            "object": "model",
            "created": 1700000000,
            "owned_by": "openai",
        })

        details = await backend.show_model("gpt-4")
        assert isinstance(details, ModelDetails)
        assert details.details is not None
        assert details.details["id"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_parse_empty_choices(self):
        backend, _ = self._make_backend()
        result = backend._parse_openai_response({"choices": [{}], "usage": {}})
        assert result.message.role == "assistant"
        assert result.message.content == ""

    @pytest.mark.asyncio
    async def test_build_payload_strips_none_options(self):
        backend, _ = self._make_backend()
        req = _make_request()
        payload = backend._build_openai_payload(req)

        # Should not have optional fields when not set
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "max_tokens" not in payload
        assert "stop" not in payload
        assert "response_format" not in payload

    @pytest.mark.asyncio
    async def test_build_payload_emits_reasoning_content_on_replay(self):
        """Regression: DeepSeek 400s if a previous assistant turn had
        reasoning_content and we replay the message without it (typical
        mid-tool-loop scenario). Message.thinking captures it on the parse
        path; this verifies it round-trips back into the request payload.
        """
        backend, _ = self._make_backend()
        req = _make_request(messages=[
            Message(role="user", content="weather?"),
            Message(
                role="assistant",
                content="",
                thinking="user wants weather → call web tool",
                tool_calls=[{"id": "c1", "type": "function",
                             "function": {"name": "web", "arguments": "{}"}}],
            ),
            Message(role="tool", tool_call_id="c1", content="sunny"),
        ])
        payload = backend._build_openai_payload(req)

        assistant_msg = payload["messages"][1]
        assert assistant_msg["reasoning_content"] == "user wants weather → call web tool"
        assert assistant_msg["tool_calls"][0]["id"] == "c1"

    @pytest.mark.asyncio
    async def test_build_payload_omits_reasoning_content_when_absent(self):
        """Don't emit reasoning_content when the assistant message had no
        thinking — avoids polluting payloads sent to non-reasoning models."""
        backend, _ = self._make_backend()
        req = _make_request(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ])
        payload = backend._build_openai_payload(req)

        assistant_msg = payload["messages"][1]
        assert "reasoning_content" not in assistant_msg

    @pytest.mark.asyncio
    async def test_build_payload_omits_reasoning_content_for_user_messages(self):
        """Defensive: only assistant messages get reasoning_content emitted,
        even if a user message somehow had thinking populated."""
        backend, _ = self._make_backend()
        req = _make_request(messages=[
            Message(role="user", content="hi", thinking="leaked"),
        ])
        payload = backend._build_openai_payload(req)

        assert "reasoning_content" not in payload["messages"][0]

    # -- Continue button: assistant-prefix continuation -----------------

    def _make_backend_with_profile(self, profile):
        """OpenAIBackend constructed with a specific ProviderProfile."""
        t = MockTransport()
        client = httpx.AsyncClient(transport=t)
        return OpenAIBackend(
            client, profile.base_url, "sk-test", profile=profile,
        ), t

    @pytest.mark.asyncio
    async def test_continue_deepseek_chat_sets_prefix_on_trailing_assistant(self):
        """DeepSeek's /beta endpoint extends a trailing assistant message
        when ``prefix: true`` is set on it. The backend should merge the
        profile's prefix marker onto the last assistant dict.

        Only ``deepseek-chat`` (the non-reasoning variant) routes
        prefix completion to the visible content channel. The
        reasoning lineup (v4-pro / v4-flash / v3.2 / reasoner) falls
        back to the synthetic-user path — see
        test_continue_deepseek_reasoning_model_falls_back_to_synthetic_user.
        """
        from augmentum.models.provider_profiles import PROFILES
        backend, _ = self._make_backend_with_profile(PROFILES["deepseek"])

        req = _make_request(
            model="deepseek-chat",
            messages=[
                Message(role="user", content="Tell me a story."),
                Message(role="assistant", content="Once upon a time, there was a"),
            ],
            continue_last_assistant=True,
        )
        payload = backend._build_openai_payload(req)

        last = payload["messages"][-1]
        assert last["role"] == "assistant"
        assert last.get("prefix") is True
        # No synthetic user message — DeepSeek does this natively.
        assert payload["messages"][-2]["role"] == "user"
        assert len(payload["messages"]) == 2

    @pytest.mark.asyncio
    async def test_continue_deepseek_reasoning_model_strips_prefix_marker(self):
        """DeepSeek reasoning models (v4-pro / v4-flash / etc.) accept
        prefix completion requests but route 100% of generated tokens
        into ``reasoning_content`` — confirmed live 2026-05-17 with
        v4-flash emitting 24KB reasoning + 0 content bytes when
        prefix:true is set. The backend must detect these models and
        send the trailing-assistant continuation WITHOUT the prefix
        marker so the reasoning model emits visible content instead
        of looping in its reasoning channel.
        """
        from augmentum.models.provider_profiles import PROFILES
        backend, _ = self._make_backend_with_profile(PROFILES["deepseek"])

        for reasoning_model in ("deepseek-v4-pro", "deepseek-v4-flash",
                                "deepseek-reasoner", "deepseek-v3.2"):
            req = _make_request(
                model=reasoning_model,
                messages=[
                    Message(role="user", content="Story?"),
                    Message(role="assistant", content="Once upon a time"),
                ],
                continue_last_assistant=True,
            )
            payload = backend._build_openai_payload(req)

            # Trailing assistant kept, no prefix marker, no synthetic
            # user. Model completes the assistant turn naturally.
            assert payload["messages"][-1]["role"] == "assistant", (
                f"{reasoning_model}: expected trailing assistant, "
                f"got {payload['messages'][-1]['role']}"
            )
            assert "prefix" not in payload["messages"][-1], (
                f"{reasoning_model}: prefix:true marker should be stripped "
                f"for reasoning models"
            )
            assert payload["messages"][-1]["content"] == "Once upon a time"
            # Same message count as input — no synthetic-user appended.
            assert len(payload["messages"]) == 2

    @pytest.mark.asyncio
    async def test_continue_deepseek_reasoning_model_does_not_route_to_beta(self):
        """Reasoning-model continue requests use synthetic-user (not
        prefix), so they should target the user's stored base URL
        rather than the /beta override. /beta accepts the call either
        way, but using the standard endpoint matches what the user
        configured."""
        from augmentum.models.provider_profiles import PROFILES
        t = MockTransport()
        client = httpx.AsyncClient(transport=t)
        backend = OpenAIBackend(
            client, "https://api.deepseek.com/v1", "sk-test",
            profile=PROFILES["deepseek"],
        )

        # Reasoning model: continue should NOT route to /beta
        reasoning_req = _make_request(
            model="deepseek-v4-pro",
            messages=[
                Message(role="user", content="Story?"),
                Message(role="assistant", content="Once"),
            ],
            continue_last_assistant=True,
        )
        assert backend._chat_url(reasoning_req) == (
            "https://api.deepseek.com/v1/chat/completions"
        )

        # Chat model: continue DOES route to /beta (where prefix lives)
        chat_req = _make_request(
            model="deepseek-chat",
            messages=[
                Message(role="user", content="Story?"),
                Message(role="assistant", content="Once"),
            ],
            continue_last_assistant=True,
        )
        assert backend._chat_url(chat_req) == (
            "https://api.deepseek.com/beta/chat/completions"
        )

    @pytest.mark.asyncio
    async def test_continue_openai_preserves_trailing_assistant(self):
        """OpenAI doesn't support a native prefix marker, but the
        Open-WebUI-style continuation pattern still works: send messages
        with the partial assistant as the trailing turn, with no
        synthetic-user instruction. OpenAI's chat completions endpoint
        completes the trailing assistant turn naturally without
        re-introducing. Mirrors the approach in Open WebUI's
        backend/open_webui/utils/middleware.py.
        """
        backend, _ = self._make_backend()  # No profile → OpenAI defaults

        req = _make_request(
            model="gpt-4o",
            messages=[
                Message(role="user", content="Tell me a story."),
                Message(role="assistant", content="Once upon a time, there was a"),
            ],
            continue_last_assistant=True,
        )
        payload = backend._build_openai_payload(req)

        # Trailing assistant intact, no synthetic-user instruction.
        assert payload["messages"][-1]["role"] == "assistant"
        assert payload["messages"][-1]["content"] == "Once upon a time, there was a"
        # No prefix marker because OpenAI's profile doesn't support it.
        assert "prefix" not in payload["messages"][-1]
        # Same message count as the input — nothing appended.
        assert len(payload["messages"]) == 2

    @pytest.mark.asyncio
    async def test_continue_noop_when_last_msg_not_assistant(self):
        """Defensive: continue_last_assistant=True with a trailing user
        message should be a no-op, not a malformed prefix or synthetic
        user. The Continue button only appears on assistant nodes, so
        this is defense-in-depth against a programmatic caller getting
        it wrong."""
        from augmentum.models.provider_profiles import PROFILES
        backend, _ = self._make_backend_with_profile(PROFILES["deepseek"])

        req = _make_request(
            model="deepseek-chat",
            messages=[
                Message(role="assistant", content="prior"),
                Message(role="user", content="next question"),
            ],
            continue_last_assistant=True,
        )
        payload = backend._build_openai_payload(req)

        # Trailing user — no marker added anywhere, no synthetic message.
        assert payload["messages"][-1]["role"] == "user"
        assert "prefix" not in payload["messages"][-1]
        # Don't touch the earlier assistant either.
        assert "prefix" not in payload["messages"][0]

    @pytest.mark.asyncio
    async def test_continue_strips_tools_for_prefix_providers(self):
        """DeepSeek's prefix endpoint 400s on "Function call should not
        be used with prefix" when tools are attached. The backend must
        drop ``tools`` and ``tool_choice`` from the payload for any
        prefix-completion request to a supporting provider — the model
        is continuing a prior response, not making fresh tool calls.

        Regression for the 400 observed after the prefix-endpoint
        routing fix landed."""
        from augmentum.models.provider_profiles import PROFILES
        backend, _ = self._make_backend_with_profile(PROFILES["deepseek"])

        req = _make_request(
            model="deepseek-chat",
            messages=[
                Message(role="user", content="Story?"),
                Message(role="assistant", content="Once upon a time"),
            ],
            tools=[{"type": "function", "function": {"name": "web_search"}}],
            tool_choice="auto",
            continue_last_assistant=True,
        )
        payload = backend._build_openai_payload(req)

        # Tools stripped; prefix marker intact; trailing assistant kept.
        assert "tools" not in payload
        assert "tool_choice" not in payload
        assert payload["messages"][-1]["role"] == "assistant"
        assert payload["messages"][-1].get("prefix") is True

    @pytest.mark.asyncio
    async def test_continue_strips_reasoning_content_from_trailing_assistant(self):
        """The trailing assistant message must have ``reasoning_content``
        stripped on continue requests. Without this strip, a reasoning
        model (DeepSeek V4 Pro / Flash, etc.) sees its prior reasoning
        chain in the prefix message and keeps reasoning instead of
        emitting visible content — observed live 2026-05-17 with V4
        Flash emitting 24KB of reasoning and 0 content bytes."""
        from augmentum.models.provider_profiles import PROFILES
        backend, _ = self._make_backend_with_profile(PROFILES["deepseek"])

        req = _make_request(
            model="deepseek-v4-pro",
            messages=[
                Message(role="user", content="Story?"),
                Message(
                    role="assistant",
                    content="Once upon a time",
                    thinking="I should write a fairy tale opening...",
                ),
            ],
            continue_last_assistant=True,
        )
        payload = backend._build_openai_payload(req)

        assert payload["messages"][-1]["role"] == "assistant"
        assert "reasoning_content" not in payload["messages"][-1], (
            "reasoning_content must be stripped from trailing assistant on continue"
        )

    @pytest.mark.asyncio
    async def test_continue_strips_tools_for_any_provider_with_trailing_assistant(self):
        """Continue requests always strip tools, regardless of provider.
        The model is continuing a prior assistant turn — no fresh tool
        decisions are appropriate. Confirmed at the openai_compat
        level for OpenAI (no profile / no native prefix); same applies
        to DeepSeek with or without prefix:true."""
        backend, _ = self._make_backend()  # No profile → OpenAI defaults

        req = _make_request(
            model="gpt-4o",
            messages=[
                Message(role="user", content="Story?"),
                Message(role="assistant", content="Once upon a time"),
            ],
            tools=[{"type": "function", "function": {"name": "web_search"}}],
            tool_choice="auto",
            continue_last_assistant=True,
        )
        payload = backend._build_openai_payload(req)

        # Tools stripped, trailing assistant kept.
        assert "tools" not in payload
        assert "tool_choice" not in payload
        assert payload["messages"][-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_continue_deepseek_routes_to_beta_endpoint(self):
        """DeepSeek's prefix completion is gated to the /beta path. If
        the user stored their provider with /v1 base URL (the standard
        OpenAI-compat endpoint), the request still needs to route to
        /beta — the profile's prefix_endpoint_override carries the
        canonical URL.

        Regression for "prefix is only available when using beta api"
        400 observed in production after the initial Continue feature
        landed.
        """
        from augmentum.models.provider_profiles import PROFILES
        deepseek = PROFILES["deepseek"]

        # Simulate the user-stored /v1 base URL — same hostname as the
        # profile, so get_profile_for_url() attaches the DeepSeek
        # profile even though the path is wrong for prefix.
        t = MockTransport()
        client = httpx.AsyncClient(transport=t)
        backend = OpenAIBackend(
            client, "https://api.deepseek.com/v1", "sk-test",
            profile=deepseek,
        )

        # Normal request → goes to the stored /v1 base
        normal_req = _make_request(model="deepseek-chat")
        assert backend._chat_url(normal_req) == (
            "https://api.deepseek.com/v1/chat/completions"
        )

        # Continue request → routes to /beta override regardless of stored base
        continue_req = _make_request(
            model="deepseek-chat",
            messages=[
                Message(role="user", content="Continue."),
                Message(role="assistant", content="Partial"),
            ],
            continue_last_assistant=True,
        )
        assert backend._chat_url(continue_req) == (
            "https://api.deepseek.com/beta/chat/completions"
        )

    @pytest.mark.asyncio
    async def test_continue_prefix_skips_post_process_alternation(self):
        """DeepSeek's post_process="semi" alternation would normally
        rewrite a trailing assistant. Prefix-completion requests must
        bypass it so the prefix marker stays on the trailing assistant."""
        from augmentum.models.provider_profiles import PROFILES
        backend, _ = self._make_backend_with_profile(PROFILES["deepseek"])

        req = _make_request(
            model="deepseek-chat",
            messages=[
                Message(role="user", content="Tell me a story."),
                Message(role="assistant", content="Once upon a time, there was a"),
            ],
            continue_last_assistant=True,
        )
        payload = backend._build_openai_payload(req)

        # Last message kept as assistant + prefix:true even though
        # DeepSeek's profile would otherwise force alternation.
        assert payload["messages"][-1]["role"] == "assistant"
        assert payload["messages"][-1].get("prefix") is True


# ===========================================================================
# OpenAI Conversion Functions
# ===========================================================================

class TestOpenAIConversions:
    """Tests for OpenAI format conversion helpers."""

    def test_to_openai_chat_response(self):
        response = InternalChatResponse(
            message=Message(role="assistant", content="Hello world"),
            model="gpt-4",
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        result = to_openai_chat_response(response)

        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-4"
        assert result["choices"][0]["message"]["content"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["total_tokens"] == 15
        assert result["id"].startswith("chatcmpl-")

    def test_to_openai_chat_response_no_finish_reason(self):
        response = InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="gpt-4",
        )
        result = to_openai_chat_response(response)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_to_openai_stream_chunk_with_content(self):
        chunk = InternalStreamChunk(
            content_delta="Hello",
            role="assistant",
            model="gpt-4",
        )
        result = to_openai_stream_chunk(chunk)

        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["delta"]["content"] == "Hello"
        assert result["choices"][0]["delta"]["role"] == "assistant"

    def test_to_openai_stream_chunk_empty_delta(self):
        chunk = InternalStreamChunk(content_delta="", model="gpt-4")
        result = to_openai_stream_chunk(chunk)
        assert "content" not in result["choices"][0]["delta"]
        assert "role" not in result["choices"][0]["delta"]

    def test_to_openai_stream_chunk_with_id(self):
        chunk = InternalStreamChunk(content_delta="Hi", model="gpt-4")
        result = to_openai_stream_chunk(chunk, chunk_id="chatcmpl-custom")
        assert result["id"] == "chatcmpl-custom"

    def test_to_openai_stream_chunk_finish_reason(self):
        chunk = InternalStreamChunk(
            content_delta="",
            model="gpt-4",
            finish_reason="stop",
            done=True,
        )
        result = to_openai_stream_chunk(chunk)
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_to_openai_models_response(self):
        models = [
            ModelInfo(name="model-a", model="model-a"),
            ModelInfo(name="model-b", model="model-b"),
        ]
        result = to_openai_models_response(models)

        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["data"][0]["id"] == "model-a"
        assert result["data"][0]["owned_by"] == "augmentum"
        assert result["data"][1]["id"] == "model-b"

    def test_to_openai_models_response_empty(self):
        result = to_openai_models_response([])
        assert result["data"] == []


# ===========================================================================
# LlamaCpp Backend
# ===========================================================================

class TestLlamaCppBackend:
    """Tests for LlamaCppBackend."""

    def _make_backend(self) -> tuple[LlamaCppBackend, MockTransport]:
        t = MockTransport()
        client = httpx.AsyncClient(transport=t)
        return LlamaCppBackend(client, "http://llamacpp:8080"), t

    @pytest.mark.asyncio
    async def test_chat_basic(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://llamacpp:8080/v1/chat/completions", json_data={
            "model": "my-model.gguf",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from llama.cpp!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
        })

        req = _make_request(model="my-model.gguf")
        result = await backend.chat(req)

        assert result.message.content == "Hello from llama.cpp!"
        assert result.model == "my-model.gguf"
        assert result.finish_reason == "stop"
        assert result.usage.total_tokens == 18

    @pytest.mark.asyncio
    async def test_chat_with_params(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://llamacpp:8080/v1/chat/completions", json_data={
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {},
        })

        req = _make_request(temperature=0.8, top_p=0.95, max_tokens=512)
        await backend.chat(req)

        body = json.loads(transport.requests[-1].content)
        assert body["temperature"] == 0.8
        assert body["top_p"] == 0.95
        assert body["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_list_models(self):
        backend, transport = self._make_backend()
        transport.add_response("GET", "http://llamacpp:8080/v1/models", json_data={
            "data": [
                {"id": "my-model.gguf", "object": "model"},
            ],
        })

        models = await backend.list_models()
        assert len(models) == 1
        assert models[0].name == "my-model.gguf"
        assert models[0].model == "my-model.gguf"

    @pytest.mark.asyncio
    async def test_list_models_error(self):
        backend, transport = self._make_backend()
        # No response set → 404
        models = await backend.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_show_model(self):
        backend, transport = self._make_backend()
        transport.add_response("GET", "http://llamacpp:8080/props", json_data={
            "default_generation_settings": {"model": "llama-3.1"},
            "system_prompt": "You are helpful.",
        })

        details = await backend.show_model("my-model")
        assert details.format == "gguf"
        assert details.family == "llama-3.1"
        assert details.system_prompt == "You are helpful."

    @pytest.mark.asyncio
    async def test_show_model_error_fallback(self):
        backend, transport = self._make_backend()
        # No response → 404 → fallback
        details = await backend.show_model("my-model")
        assert details.format == "gguf"
        assert details.family == "my-model"

    @pytest.mark.asyncio
    async def test_get_slots(self):
        backend, transport = self._make_backend()
        transport.add_response("GET", "http://llamacpp:8080/slots", json_data=[
            {"id": 0, "state": 1, "prompt": "test"},
        ])

        slots = await backend.get_slots()
        assert len(slots) == 1
        assert slots[0]["id"] == 0

    @pytest.mark.asyncio
    async def test_get_slots_error(self):
        backend, _ = self._make_backend()
        slots = await backend.get_slots()
        assert slots == []

    @pytest.mark.asyncio
    async def test_tokenize(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://llamacpp:8080/tokenize", json_data={
            "tokens": [1, 234, 567, 8],
        })

        tokens = await backend.tokenize("Hello world")
        assert tokens == [1, 234, 567, 8]

    @pytest.mark.asyncio
    async def test_tokenize_error(self):
        backend, _ = self._make_backend()
        tokens = await backend.tokenize("Hello")
        assert tokens == []

    def test_to_openai_payload_static(self):
        backend, _ = self._make_backend()
        req = _make_request(model="test", temperature=0.5, max_tokens=100)
        payload = backend._to_openai_payload(req)
        assert payload["model"] == "test"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 100
        assert len(payload["messages"]) == 1

    def test_to_openai_payload_forwards_native_tool_choice(self):
        backend, _ = self._make_backend()
        req = _make_request(
            model="Qwen3.6-35B-A3B",
            tools=[{"type": "function", "function": {"name": "file_read"}}],
            tool_choice="required",
        )
        payload = backend._to_openai_payload(req)

        assert payload["tools"] == req.tools
        assert payload["tool_choice"] == "required"

    def test_reasoning_options_honor_explicit_chat_template_kwargs(self):
        backend, _ = self._make_backend()
        req = _make_request(
            model="Qwen3.6-35B-A3B",
            think=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        payload: dict = {}

        backend._apply_reasoning_request_options(payload, req)

        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["reasoning_format"] == "none"

    def test_deepseek_reasoning_effort_forwarded_to_template(self):
        """DeepSeek V3.2/V4 GGUF templates consume ``reasoning_effort``
        ("high"/"max") next to ``enable_thinking`` — the UI's Off/High/Max
        picker must reach the local template, mirroring the cloud adapter's
        nested ``thinking:{reasoning_effort}``."""
        backend, _ = self._make_backend()
        req = _make_request(
            model="DeepSeek-V4-Flash-UD-IQ3_XXS",
            think=True,
            reasoning_effort="max",
        )
        payload: dict = {}
        backend._apply_reasoning_request_options(payload, req)
        kwargs = payload["chat_template_kwargs"]
        assert kwargs["enable_thinking"] is True
        assert kwargs["reasoning_effort"] == "max"
        assert payload["reasoning_format"] == "deepseek"

    def test_deepseek_reasoning_effort_invalid_levels_dropped(self):
        """Mode hints set 'low'/'medium' globally; DS4 only documents
        high/max — anything else must not reach the template. 'xhigh'
        (OpenAI-enum top tier) maps to 'max'."""
        backend, _ = self._make_backend()
        for effort, expected in (("low", None), ("medium", None), ("xhigh", "max")):
            req = _make_request(
                model="DeepSeek-V4-Flash-UD-IQ3_XXS",
                think=True,
                reasoning_effort=effort,
            )
            payload: dict = {}
            backend._apply_reasoning_request_options(payload, req)
            assert payload["chat_template_kwargs"].get("reasoning_effort") == expected

    def test_deepseek_reasoning_effort_off_forces_thinking_off(self):
        """Belt-and-braces: an explicit 'off' effort disables thinking even
        if a stale think=True rides along."""
        backend, _ = self._make_backend()
        req = _make_request(
            model="DeepSeek-V4-Flash-UD-IQ3_XXS",
            think=True,
            reasoning_effort="off",
        )
        payload: dict = {}
        backend._apply_reasoning_request_options(payload, req)
        kwargs = payload["chat_template_kwargs"]
        assert kwargs["enable_thinking"] is False
        assert "reasoning_effort" not in kwargs
        assert payload["reasoning_format"] == "none"

    def test_qwen_thinking_ignores_reasoning_effort(self):
        """The effort kwarg is DeepSeek-gated — other hybrid families keep
        their plain enable_thinking payload."""
        backend, _ = self._make_backend()
        req = _make_request(
            model="Qwen3.6-35B-A3B",
            think=True,
            reasoning_effort="max",
        )
        payload: dict = {}
        backend._apply_reasoning_request_options(payload, req)
        assert "reasoning_effort" not in payload["chat_template_kwargs"]

    def test_continue_last_assistant_forces_no_generation_prompt(self):
        """Continue button: llama-server needs add_generation_prompt:false
        so the chat template formats the trailing assistant without a
        fresh turn marker, AND enable_thinking:false because llama-server
        400s on "Assistant response prefill is incompatible with
        enable_thinking" (per the prewarm path constraint)."""
        backend, _ = self._make_backend()
        req = _make_request(
            model="Qwen3.6-35B-A3B",
            think=True,
            continue_last_assistant=True,
        )
        payload: dict = {}

        backend._apply_reasoning_request_options(payload, req)

        kwargs = payload["chat_template_kwargs"]
        assert kwargs["add_generation_prompt"] is False
        assert kwargs["enable_thinking"] is False
        # reasoning_format follows the effective enable_thinking — for
        # Qwen this becomes "none" when thinking is forced off.
        assert payload["reasoning_format"] == "none"

    # -- OOM backoff -----------------------------------------------------

    @staticmethod
    def _make_oom_manager(
        outcomes: list[str | None],
        autofit_layers: int = 40,
        n_layers: int = 50,
    ) -> "_OomFakeManager":
        """Build an OOM-test manager keyed off ``outcomes``.

        Each entry in ``outcomes`` is the result of one ``start()``:
        ``None`` = succeed, ``"oom"`` = raise OOM-class RuntimeError,
        ``"crash"`` = raise non-OOM RuntimeError. ``start()`` records
        the override layer count it was called with on each invocation.
        """
        from augmentum.models.llama_server_manager import ProcessState
        from augmentum.models.model_profile_cache import ModelProfile

        class _OomFakeManager:
            def __init__(self) -> None:
                self.state = ProcessState.IDLE
                self.model_id = ""
                self.model_path = ""
                self.process = None
                self._last_crashed_model = ""
                self._last_profile = ModelProfile(
                    model_path="/models/x.gguf",
                    model_name="x",
                    n_layers=n_layers,
                )
                self.start_calls: list[int | None] = []  # override per call
                self._outcomes = list(outcomes)
                self._autofit_layers = autofit_layers

            def check_alive(self) -> bool:
                return True

            def _resolve_model_path(self, model: str) -> str:
                return f"/models/{model}.gguf"

            def _autofit_gpu_layers(self, profile) -> int:  # noqa: ARG002
                return self._autofit_layers

            async def start(self, path: str, gpu_layers_override=None) -> None:
                self.start_calls.append(gpu_layers_override)
                if not self._outcomes:
                    raise AssertionError(
                        f"start() called more than {len(outcomes)} times "
                        "(test outcomes exhausted)"
                    )
                outcome = self._outcomes.pop(0)
                if outcome == "oom":
                    raise RuntimeError(
                        f"llama-server exited during startup with code 137 "
                        f"(out of memory)"
                    )
                if outcome == "crash":
                    raise RuntimeError("invalid model file: bad magic")
                self.state = ProcessState.READY
                self.model_path = path
                self.model_id = path.rsplit("/", 1)[-1].removesuffix(".gguf")

        return _OomFakeManager()

    @pytest.mark.asyncio
    async def test_oom_backoff_first_attempt_succeeds(self):
        """No OOM = no retry: start() called exactly once with no override."""
        backend, _ = self._make_backend()
        manager = self._make_oom_manager([None])
        backend._manager = manager

        await backend._start_with_oom_backoff("/models/x.gguf")

        assert manager.start_calls == [None]

    @pytest.mark.asyncio
    async def test_oom_backoff_retries_with_reduced_layers(self):
        """First OOM, retry succeeds: second call uses 0.85 × autofit."""
        backend, _ = self._make_backend()
        manager = self._make_oom_manager(["oom", None], autofit_layers=40)
        backend._manager = manager

        await backend._start_with_oom_backoff("/models/x.gguf")

        # Two calls: first with no override (attempt 0), second with
        # int(40 * 0.85) = 34 layers.
        assert len(manager.start_calls) == 2
        assert manager.start_calls[0] is None
        assert manager.start_calls[1] == 34

    @pytest.mark.asyncio
    async def test_oom_backoff_iterates_through_factors(self):
        """Multiple OOMs walk the factor schedule 1.0, 0.85, 0.70, 0.55…"""
        backend, _ = self._make_backend()
        manager = self._make_oom_manager(
            ["oom", "oom", "oom", "oom", None], autofit_layers=100,
        )
        backend._manager = manager

        await backend._start_with_oom_backoff("/models/x.gguf")

        # Five calls total. attempt-0 unconstrained, then 0.85/0.70/
        # 0.55/0.40 of autofit=100 → 85, 70, 55, 40.
        assert manager.start_calls == [None, 85, 70, 55, 40]

    @pytest.mark.asyncio
    async def test_oom_backoff_exhausts_after_max_attempts(self):
        """All attempts OOM → raise after _OOM_RETRY_MAX_ATTEMPTS retries."""
        backend, _ = self._make_backend()
        # 6 OOMs (1 initial + 5 retries) — exhausts the cap.
        manager = self._make_oom_manager(
            ["oom"] * (backend._OOM_RETRY_MAX_ATTEMPTS + 1), autofit_layers=50,
        )
        backend._manager = manager

        with pytest.raises(RuntimeError, match="out of memory"):
            await backend._start_with_oom_backoff("/models/x.gguf")

        # All retry slots consumed: 1 initial + _OOM_RETRY_MAX_ATTEMPTS retries.
        assert len(manager.start_calls) == backend._OOM_RETRY_MAX_ATTEMPTS + 1

    @pytest.mark.asyncio
    async def test_oom_backoff_non_oom_propagates_immediately(self):
        """Non-OOM RuntimeError on attempt 0 raises without any retry."""
        backend, _ = self._make_backend()
        manager = self._make_oom_manager(["crash"])
        backend._manager = manager

        with pytest.raises(RuntimeError, match="invalid model file"):
            await backend._start_with_oom_backoff("/models/x.gguf")

        assert len(manager.start_calls) == 1, (
            "non-OOM error should not trigger backoff"
        )

    def test_is_oom_class_error_matches_known_strings(self):
        """Lock the OOM-error matcher against the strings llama-server emits."""
        match = LlamaCppBackend._is_oom_class_error
        # Real production strings observed from llama-server crashes.
        assert match(RuntimeError("llama-server exited during startup with code 137"))
        assert match(RuntimeError("OOM: failed to allocate KV cache"))
        assert match(RuntimeError("CUDA out of memory while loading"))
        assert match(RuntimeError("ggml_cuda_init: insufficient memory"))
        # False positives we don't want to match.
        assert not match(RuntimeError("invalid model file: bad magic"))
        assert not match(RuntimeError("connection refused"))
        # Wrong exception type stays False.
        assert not match(TimeoutError("health check timed out"))
        assert not match(ValueError("oom"))  # type guard catches non-RuntimeError

    @pytest.mark.asyncio
    async def test_oom_backoff_under_concurrent_ensure_server(self):
        """Two ``_ensure_server`` calls racing into the OOM-retry path
        must not double-spawn.

        Without ``_ensure_lock`` serializing them, both callers would
        independently observe ``IDLE``, both would call
        ``_start_with_oom_backoff``, and the OOM retry sequence would
        run twice — burning CPU on a duplicate startup attempt and
        potentially triggering a second (real) OOM crash on the
        already-recovering box.

        With the lock in place, the second caller waits, sees
        ``READY`` after the first finishes, and short-circuits.

        We assert: exactly 2 ``start()`` invocations total (1 OOM +
        1 successful retry), NOT 4 (which would be 2 OOMs + 2
        retries running concurrently).
        """
        from augmentum.models.llama_server_manager import ProcessState
        from augmentum.models.model_profile_cache import ModelProfile

        class _ConcurrentOomManager:
            def __init__(self) -> None:
                self.state = ProcessState.IDLE
                self.model_id = ""
                self.model_path = ""
                self.process = None
                self._last_crashed_model = ""
                self._last_profile = ModelProfile(
                    model_path="/models/x.gguf",
                    model_name="x",
                    n_layers=50,
                )
                self.start_calls: list[int | None] = []
                # OOM once, succeed on retry.
                self._outcomes = ["oom", None]

            def check_alive(self) -> bool:
                return True

            def _resolve_model_path(self, model: str) -> str:
                return f"/models/{model}.gguf"

            def _autofit_gpu_layers(self, profile) -> int:  # noqa: ARG002
                return 40

            async def start(self, path: str, gpu_layers_override=None) -> None:
                self.start_calls.append(gpu_layers_override)
                # Simulate real startup latency so the second concurrent
                # caller has a real chance to overlap if the lock leaks.
                await asyncio.sleep(0.02)
                if not self._outcomes:
                    raise AssertionError(
                        "start() called more times than the OOM scenario "
                        "permits — double-spawn detected"
                    )
                outcome = self._outcomes.pop(0)
                if outcome == "oom":
                    raise RuntimeError(
                        "llama-server exited during startup with code 137 "
                        "(out of memory)"
                    )
                self.state = ProcessState.READY
                self.model_path = path
                self.model_id = path.rsplit("/", 1)[-1].removesuffix(".gguf")

        backend, _ = self._make_backend()
        manager = _ConcurrentOomManager()
        backend._manager = manager

        # Two concurrent ensure_server callers. Both should resolve
        # without raising; the second is a no-op once state == READY.
        await asyncio.gather(
            backend._ensure_server("x"),
            backend._ensure_server("x"),
        )

        # CRITICAL invariant: exactly 2 start calls — the OOM and its
        # successful retry. Anything higher means the lock leaked and
        # the retry sequence ran twice.
        assert len(manager.start_calls) == 2, (
            f"OOM retry sequence ran more than once under concurrency: "
            f"{manager.start_calls!r}"
        )
        # First call: no override (initial autofit). Second call:
        # int(40 * 0.85) = 34 (post-OOM reduction). If the second
        # caller raced past the lock it would have made a third call
        # with no override.
        assert manager.start_calls[0] is None
        assert manager.start_calls[1] == 34
        assert manager.state == ProcessState.READY

    @pytest.mark.asyncio
    async def test_ensure_server_serializes_concurrent_lazy_loads(self):
        from augmentum.models.llama_server_manager import ProcessState

        class FakeManager:
            def __init__(self) -> None:
                self.state = ProcessState.IDLE
                self.model_id = ""
                self.model_path = ""
                self.process = None
                self._last_crashed_model = ""
                self.start_calls = 0

            def check_alive(self) -> bool:
                return True

            def _resolve_model_path(self, model: str) -> str:
                return f"/models/{model}.gguf"

            async def start(self, path: str, gpu_layers_override=None) -> None:
                self.start_calls += 1
                self.state = ProcessState.STARTING
                await asyncio.sleep(0.01)
                self.model_path = path
                self.model_id = path.rsplit("/", 1)[-1].removesuffix(".gguf")
                self.state = ProcessState.READY

            async def swap(self, path: str) -> None:
                self.model_path = path
                self.model_id = path.rsplit("/", 1)[-1].removesuffix(".gguf")
                self.state = ProcessState.READY

        transport = MockTransport()
        client = httpx.AsyncClient(transport=transport)
        manager = FakeManager()
        backend = LlamaCppBackend(client, "http://llamacpp:8080", server_manager=manager)

        await asyncio.gather(
            backend._ensure_server("gemma-4"),
            backend._ensure_server("gemma-4"),
        )

        assert manager.start_calls == 1
        assert manager.model_id == "gemma-4"

    def test_session_fingerprint_prefers_kv_session_key(self):
        backend, _ = self._make_backend()
        req = _make_request(
            messages=[Message(role="system", content="system prompt")],
            kv_session_key="sess-live-123",
        )
        assert backend._session_fingerprint(req) == "sess-live-123"

    def test_session_fingerprint_uses_stable_checkpoint_key_when_available(self):
        backend, _ = self._make_backend()
        stable_messages = [
            Message(role="system", content="card prompt"),
            Message(role="assistant", content="Previous reply"),
            Message(role="user", content="New turn"),
        ]
        req = _make_request(
            kv_session_key="sess-live-123",
            kv_stable_messages=stable_messages,
        )

        expected_digest = backend._messages_tail_fingerprint(stable_messages[:-1])
        assert backend._session_fingerprint(req) == f"sess-live-123::stable::{expected_digest}"

    @pytest.mark.asyncio
    async def test_prepare_stable_checkpoint_prewarms_and_saves_new_checkpoint(self):
        backend, _ = self._make_backend()
        backend._manager = _make_inflight_manager()
        backend.prewarm_context = AsyncMock(return_value=True)
        backend.save_session_state = AsyncMock(return_value=True)

        stable_messages = [
            Message(role="system", content="card prompt"),
            Message(role="user", content="Hello there"),
        ]
        req = _make_request(
            model="test-model",
            kv_session_key="sess-live-123",
            kv_mode="narrative",
            kv_stable_messages=stable_messages,
        )

        prepared = await backend.prepare_stable_checkpoint(req, "Hi back")

        assert prepared is True
        backend.prewarm_context.assert_awaited_once()
        checkpoint_payload = backend.prewarm_context.await_args.args[0]
        assert checkpoint_payload == [
            {"role": "system", "content": "card prompt"},
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi back"},
        ]

        checkpoint_messages = stable_messages + [Message(role="assistant", content="Hi back")]
        expected_key = f"sess-live-123::stable::{backend._messages_tail_fingerprint(checkpoint_messages)}"
        save_call = backend.save_session_state.await_args
        assert save_call.args[0] == expected_key
        assert save_call.kwargs["request"].kv_session_key == expected_key
        assert backend._get_session_for_slot(0) == expected_key

    @pytest.mark.asyncio
    async def test_stream_completion_emits_done_exactly_once(self):
        """Regression: /completion path must emit done=True exactly once.

        Pre-fix the data chunk with ``stop: true`` AND the ``[DONE]``
        SSE marker each emitted an InternalStreamChunk with done=True.
        Outer SSE wrappers that exit on first done=True silently dropped
        the [DONE] flush — losing any partial-tag content held in the
        thinking buffer at end-of-stream and discarding the Usage that
        was attached to the [DONE] chunk.

        Match the documented rule from _stream_chat_completions: the
        intermediate stop:true chunk carries its delta, but [DONE] (or
        the EOF fallback) is the sole source of done=True. Usage
        captured from stop:true's timings propagates to the [DONE]
        chunk so the consumer sees both signals together.
        """
        backend, _ = self._make_backend()

        def handler(request: httpx.Request) -> httpx.Response:
            sse_lines = [
                'data: {"content": "Hello", "stop": false}',
                'data: {"content": " world", "stop": false}',
                ('data: {"content": "!", "stop": true, '
                 '"timings": {"prompt_n": 12, "predicted_n": 6}}'),
                "data: [DONE]",
                "",
            ]
            body = ("\n".join(sse_lines) + "\n").encode("utf-8")
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        req = _make_request(model="m")
        chunks = []
        async for c in backend._stream_completion(req, [1, 2, 3]):
            chunks.append(c)

        done_chunks = [c for c in chunks if c.done]
        assert len(done_chunks) == 1, (
            f"expected exactly one done=True chunk, got {len(done_chunks)}: "
            f"{[c.content_delta for c in done_chunks]}"
        )
        # Usage from the stop:true chunk's timings propagates to the
        # terminal chunk.
        terminal = done_chunks[0]
        assert terminal.usage is not None
        assert terminal.usage.prompt_tokens == 12
        assert terminal.usage.completion_tokens == 6
        assert terminal.usage.total_tokens == 18
        # All visible content emitted across the stream.
        joined = "".join(c.content_delta or "" for c in chunks)
        assert joined == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_completion_eof_without_done_still_terminates(self):
        """If [DONE] never arrives, the EOF fallback still emits one
        done=True chunk with whatever Usage was captured."""
        backend, _ = self._make_backend()

        def handler(request: httpx.Request) -> httpx.Response:
            sse_lines = [
                'data: {"content": "x", "stop": false}',
                ('data: {"content": "y", "stop": true, '
                 '"timings": {"prompt_n": 3, "predicted_n": 2}}'),
                "",  # EOF without [DONE]
            ]
            body = ("\n".join(sse_lines) + "\n").encode("utf-8")
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        req = _make_request(model="m")
        chunks = []
        async for c in backend._stream_completion(req, [1]):
            chunks.append(c)

        done_chunks = [c for c in chunks if c.done]
        assert len(done_chunks) == 1
        assert done_chunks[0].usage is not None
        assert done_chunks[0].usage.total_tokens == 5

    @pytest.mark.asyncio
    async def test_stream_chat_completions_forwards_tool_call_deltas(self):
        """llama.cpp's OpenAI stream carries tool calls in delta.tool_calls.

        The coder hybrid/native parser consumes them from
        chunk.augmentum["tool_calls"], so dropping this field makes native
        tool turns look like blank text stops.
        """
        backend, _ = self._make_backend()
        tc_delta = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "file_list",
                "arguments": '{"path":"/workspace"}',
            },
        }

        def handler(request: httpx.Request) -> httpx.Response:
            sse_events = [
                {
                    "model": "m",
                    "choices": [{
                        "delta": {
                            "tool_calls": [tc_delta],
                        },
                        "finish_reason": None,
                    }],
                },
                {
                    "model": "m",
                    "choices": [{
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }],
                },
            ]
            sse_lines = [
                f"data: {json.dumps(event)}" for event in sse_events
            ] + ["data: [DONE]", ""]
            body = ("\n".join(sse_lines) + "\n").encode("utf-8")
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        req = _make_request(
            model="m",
            tools=[{"type": "function", "function": {"name": "file_list"}}],
        )

        chunks = []
        async for c in backend._stream_chat_completions(req):
            chunks.append(c)

        tool_chunks = [
            c for c in chunks
            if c.augmentum and c.augmentum.get("tool_calls")
        ]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].augmentum["tool_calls"] == [tc_delta]

    @pytest.mark.asyncio
    async def test_stream_completion_eof_after_stop_true_logs_debug_not_warning(
        self, monkeypatch,
    ):
        """Clean /completion termination (saw stop:true → EOF) must NOT
        log a warning. llama-server's raw /completion endpoint never
        emits ``[DONE]``, so EOF is the normal terminator — warning
        level here drowns dashboards in false alarms.
        """
        from augmentum.models import llama_cpp as llm_mod
        warnings_seen: list[tuple[str, dict]] = []
        debugs_seen: list[tuple[str, dict]] = []

        def fake_warning(event, **kw):
            warnings_seen.append((event, kw))
        def fake_debug(event, **kw):
            debugs_seen.append((event, kw))

        # Wrap, don't fully replace — other call sites in the same
        # function use the same logger and we only care about the EOF
        # branch's choice. Capture all calls; assert on the EOF event.
        monkeypatch.setattr(llm_mod.log, "warning", fake_warning)
        monkeypatch.setattr(llm_mod.log, "debug", fake_debug)

        backend, _ = self._make_backend()

        def handler(request: httpx.Request) -> httpx.Response:
            sse_lines = [
                'data: {"content": "ok", "stop": true, "timings": {"prompt_n": 1, "predicted_n": 1}}',
                "",  # EOF without [DONE]
            ]
            body = ("\n".join(sse_lines) + "\n").encode("utf-8")
            return httpx.Response(
                200, content=body,
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        req = _make_request(model="m")
        async for _ in backend._stream_completion(req, [1]):
            pass

        truncation_warnings = [w for w in warnings_seen if "truncated" in w[0]]
        assert not truncation_warnings, (
            f"clean EOF after stop:true must not log a truncation warning; "
            f"got: {truncation_warnings}"
        )
        assert any("clean_eof" in d[0] for d in debugs_seen), (
            f"expected a clean_eof debug log; got debugs={debugs_seen}"
        )

    @pytest.mark.asyncio
    async def test_stream_completion_eof_without_stop_true_logs_warning(
        self, monkeypatch,
    ):
        """Real truncation (EOF before any stop:true) must surface as a
        warning. This is the case the warning was originally meant to
        catch — model crash, upstream socket reset, or context-window
        overflow that produced no output.
        """
        from augmentum.models import llama_cpp as llm_mod
        warnings_seen: list[tuple[str, dict]] = []

        def fake_warning(event, **kw):
            warnings_seen.append((event, kw))
        monkeypatch.setattr(llm_mod.log, "warning", fake_warning)

        backend, _ = self._make_backend()

        def handler(request: httpx.Request) -> httpx.Response:
            sse_lines = [
                'data: {"content": "partial", "stop": false}',
                "",  # EOF mid-generation, no stop:true ever
            ]
            body = ("\n".join(sse_lines) + "\n").encode("utf-8")
            return httpx.Response(
                200, content=body,
                headers={"content-type": "text/event-stream"},
                request=request,
            )

        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        req = _make_request(model="m")
        async for _ in backend._stream_completion(req, [1]):
            pass

        truncation_warnings = [w for w in warnings_seen if "truncated" in w[0]]
        assert truncation_warnings, (
            f"EOF without stop:true must warn — that's a real model "
            f"truncation, not the normal /completion terminator. "
            f"Warnings seen: {warnings_seen}"
        )

    @pytest.mark.asyncio
    async def test_prepare_stable_checkpoint_blocks_idle_unload_via_in_flight(self):
        """Regression: checkpoint prewarm must hold the in-flight counter.

        prepare_stable_checkpoint runs as a background asyncio.task from
        the narrative handler, AFTER the original chat-stream's
        request_in_flight() scope has closed. Without explicit
        re-wrapping, the idle monitor's countdown could fire mid-prewarm
        on long contexts (90k tokens × cold prefill = 5-10 s of silence),
        unload the subprocess, and 502 the save.

        Asserts the counter is observed at >= 1 from inside prewarm —
        proving the wrap is in place and the idle monitor would refuse
        to fire while it runs.
        """
        backend, _ = self._make_backend()
        manager = _make_inflight_manager()
        backend._manager = manager
        # Capture the live count at the moment prewarm runs.
        captured_count: list[int] = []

        async def _fake_prewarm(*args, **kwargs):
            captured_count.append(manager._in_flight_count_view())
            return True

        backend.prewarm_context = _fake_prewarm
        backend.save_session_state = AsyncMock(return_value=True)

        req = _make_request(
            kv_session_key="sess-x",
            kv_stable_messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
            ],
        )
        prepared = await backend.prepare_stable_checkpoint(req, "reply")

        assert prepared is True
        # request_in_flight was entered exactly once and prewarm saw the
        # counter at >= 1.
        assert manager.in_flight_during == [1]
        assert captured_count == [1]
        # Counter dropped back to 0 after the wrap exited.
        assert manager._in_flight_count_view() == 0

    @pytest.mark.asyncio
    async def test_prepare_stable_checkpoint_releases_in_flight_on_cancel(self):
        """Cancelling the checkpoint task mid-prewarm must still
        release the in-flight counter.

        prepare_stable_checkpoint runs as a fire-and-forget background
        task. If the chat session is closed, the parent route may
        cancel the task — the wrap's finally block must still
        decrement the counter, otherwise the idle monitor would never
        consider the engine idle again.

        Pre-T1-2 the counter wasn't reliably released on the cancel
        path because prewarm_context didn't propagate CancelledError
        cleanly through the slot lock. This test forces a cancel
        during prewarm and asserts the wrap exited correctly.
        """
        backend, _ = self._make_backend()
        manager = _make_inflight_manager()
        backend._manager = manager

        # Prewarm hangs forever — exactly the shape of a real long
        # narrative prewarm that gets cancelled before completion.
        async def hanging_prewarm(*args, **kwargs):
            await asyncio.sleep(10.0)
            return True

        backend.prewarm_context = hanging_prewarm
        backend.save_session_state = AsyncMock(return_value=True)

        req = _make_request(
            kv_session_key="sess-x",
            kv_stable_messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
            ],
        )

        task = asyncio.create_task(
            backend.prepare_stable_checkpoint(req, "reply")
        )
        # Let the task enter request_in_flight + reach prewarm.
        await asyncio.sleep(0.02)
        assert manager._in_flight_count_view() == 1, (
            "task hadn't entered request_in_flight yet"
        )

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The in-flight counter MUST be back to zero. If this fails
        # in production, the idle monitor stops working until restart.
        assert manager._in_flight_count_view() == 0, (
            f"in-flight counter leaked after cancel: "
            f"{manager._in_flight_count_view()}"
        )

    @pytest.mark.asyncio
    async def test_save_slot_hashes_session_filename(self):
        backend, transport = self._make_backend()
        transport.add_response("POST", "http://llamacpp:8080/slots/0?action=save", json_data={"status": "ok"})

        saved = await backend.save_slot(0, "sess:abc/123")

        assert saved is True
        body = json.loads(transport.requests[-1].content.decode("utf-8"))
        expected = "session_" + hashlib.sha256("sess:abc/123".encode("utf-8")).hexdigest()[:32]
        assert body["filename"] == expected

    @pytest.mark.asyncio
    async def test_slot_save_501_latches_and_stops_retrying(self):
        """A 501 (server started without --slot-save-path) means slot I/O is
        unavailable for this server's life. We must latch it so we stop hitting
        the endpoint (and stop warning) every turn — and restore_slot must
        honour the same latch without firing a request."""
        backend, transport = self._make_backend()
        transport.add_response(
            "POST", "http://llamacpp:8080/slots/0?action=save", status=501,
        )

        first = await backend.save_slot(0, "sess-1")
        assert first is False
        assert backend._slot_io_unsupported is True
        n_after_first = len(transport.requests)

        # Subsequent save/restore short-circuit — no further HTTP traffic.
        assert await backend.save_slot(0, "sess-2") is False
        assert await backend.restore_slot(0, "sess-2") is False
        assert len(transport.requests) == n_after_first, (
            "latched slot I/O must not issue more requests"
        )

    @pytest.mark.asyncio
    async def test_slot_save_skipped_when_manager_reports_unsupported(self):
        """When the loaded model ran multi-slot/--kv-unified, the manager sets
        _slot_save_supported=False — the backend must skip slot I/O entirely
        (no HTTP, no 501) since those models persist via --ctx-checkpoints."""
        backend, transport = self._make_backend()

        class _Mgr:
            _slot_save_supported = False
            current_mmproj_path = ""

        backend._manager = _Mgr()
        assert await backend.save_slot(0, "s") is False
        assert await backend.restore_slot(0, "s") is False
        assert len(transport.requests) == 0, "must not hit the server at all"

    @pytest.mark.asyncio
    async def test_restore_session_state_skips_incompatible_manifest(self, tmp_path):
        transport = MockTransport()
        client = httpx.AsyncClient(transport=transport)
        manifest = KVSessionManifest(str(tmp_path / "kv_manifest.db"))

        class FakeManager:
            def __init__(self) -> None:
                self._session_manifest = manifest
                self._slot_dir = "/slots/test-model"
                self.model_id = "test-model"
                self.model_path = "/models/test-model.gguf"
                self.current_ctx_size = 8192
                self.kv_cache_type = "q8_0"

            def current_runtime_signature(self) -> dict:
                return {
                    "model_key": "test-model",
                    "model_id": "test-model",
                    "model_path": "/models/test-model.gguf",
                    "model_mtime": 111.0,
                    "ctx_size": 8192,
                    "kv_cache_type": "q8_0",
                }

            def kv_ttl_days_for_mode(self, mode: str = "") -> int:
                return 2

            def session_is_pinned(self, session_key: str, mode: str = "") -> bool:
                return False

        manager = FakeManager()
        backend = LlamaCppBackend(client, "http://llamacpp:8080", server_manager=manager)
        manifest.record_save(
            model_key="test-model",
            session_key="sess-1",
            mode="passthrough",
            slot_dir="/slots/test-model",
            slot_filename=backend._slot_storage_name("sess-1"),
            model_id="test-model",
            model_path="/models/test-model.gguf",
            model_mtime=111.0,
            ctx_size=4096,
            kv_cache_type="q8_0",
            template_fingerprint="tpl",
            system_prompt_hash="sys",
            prompt_fingerprint="prompt",
            prompt_message_count=4,
            ttl_days=2,
            pinned=False,
        )

        restored = await backend.restore_session_state(
            "sess-1",
            request=_make_request(model="test-model", kv_session_key="sess-1"),
        )

        assert restored is False
        assert transport.requests == []
        record = manifest.get_session("test-model", "sess-1")
        assert record is not None
        assert record["last_restore_result"] == "skipped"
        assert record["last_skip_reason"] == "context size changed"

    @pytest.mark.parametrize(
        "mutate,expected_reason",
        [
            ({"flash_attn": True}, "flash_attn changed"),
            ({"gpu_layers_mode": "fixed"}, "gpu_layers_mode changed"),
            ({"gpu_layers": 99}, "gpu_layers changed"),
            ({"batch_size": 999}, "batch_size changed"),
            ({"draft_model": "draft-v2"}, "draft_model changed"),
            ({"draft_max": 99}, "draft_max changed"),
            # Architectural fingerprints (T2-2). Each independently
            # rejects a saved slot whose underlying model's
            # architecture differs from the live runtime — catches the
            # rare case where ``model_id`` + ``model_mtime`` collide
            # across two different models.
            ({"n_embed": 5120}, "n_embed changed"),
            ({"n_layers_total": 80}, "n_layers_total changed"),
            ({"n_heads_kv": 16}, "n_heads_kv changed"),
        ],
    )
    @pytest.mark.asyncio
    async def test_restore_skip_reason_per_load_shape_dimension(
        self, tmp_path, mutate, expected_reason,
    ):
        """Each KV-shaping dimension must independently invalidate a stored slot."""
        transport = MockTransport()
        client = httpx.AsyncClient(transport=transport)
        manifest = KVSessionManifest(str(tmp_path / "kv_manifest.db"))

        baseline = {
            "model_key": "test-model",
            "model_id": "test-model",
            "model_path": "/models/test-model.gguf",
            "model_mtime": 111.0,
            "ctx_size": 8192,
            "kv_cache_type": "q8_0",
            "flash_attn": False,
            "gpu_layers": 32,
            "gpu_layers_mode": "manual",
            "batch_size": 256,
            "draft_model": "",
            "draft_max": 0,
            "n_embed": 4096,
            "n_layers_total": 32,
            "n_heads_kv": 8,
        }

        class FakeManager:
            def __init__(self, sig: dict) -> None:
                self._session_manifest = manifest
                self._slot_dir = "/slots/test-model"
                self.model_id = "test-model"
                self.model_path = "/models/test-model.gguf"
                self.current_ctx_size = 8192
                self.kv_cache_type = "q8_0"
                self._sig = sig

            def current_runtime_signature(self) -> dict:
                return dict(self._sig)

            def kv_ttl_days_for_mode(self, mode: str = "") -> int:
                return 2

            def session_is_pinned(self, session_key: str, mode: str = "") -> bool:
                return False

        # Save the slot under baseline runtime.
        baseline_manager = FakeManager(baseline)
        backend = LlamaCppBackend(client, "http://llamacpp:8080", server_manager=baseline_manager)
        manifest.record_save(
            model_key="test-model",
            session_key="sess-1",
            mode="passthrough",
            slot_dir="/slots/test-model",
            slot_filename=backend._slot_storage_name("sess-1"),
            model_id=baseline["model_id"],
            model_path=baseline["model_path"],
            model_mtime=baseline["model_mtime"],
            ctx_size=baseline["ctx_size"],
            kv_cache_type=baseline["kv_cache_type"],
            template_fingerprint="tpl",
            system_prompt_hash="sys",
            prompt_fingerprint="prompt",
            prompt_message_count=4,
            ttl_days=2,
            pinned=False,
            flash_attn=baseline["flash_attn"],
            gpu_layers=baseline["gpu_layers"],
            gpu_layers_mode=baseline["gpu_layers_mode"],
            batch_size=baseline["batch_size"],
            draft_model=baseline["draft_model"],
            draft_max=baseline["draft_max"],
            n_embed=baseline["n_embed"],
            n_layers_total=baseline["n_layers_total"],
            n_heads_kv=baseline["n_heads_kv"],
        )

        # Now flip one dimension in the runtime signature.
        mutated = dict(baseline)
        mutated.update(mutate)
        backend._manager = FakeManager(mutated)

        restored = await backend.restore_session_state(
            "sess-1",
            request=_make_request(model="test-model", kv_session_key="sess-1"),
        )

        assert restored is False, (
            f"expected restore to be skipped after changing {mutate}"
        )
        assert transport.requests == []  # no HTTP call to llama-server
        record = manifest.get_session("test-model", "sess-1")
        assert record is not None
        assert record["last_skip_reason"] == expected_reason

    @pytest.mark.asyncio
    async def test_restore_session_state_no_checkpoint_skips_without_erasing(
        self, tmp_path,
    ):
        """Regression for the regenerate-fast fix:

        When no on-disk slot file exists for the requested session_id,
        ``restore_session_state`` must return False WITHOUT calling
        ``restore_slot`` — because ``restore_slot`` always erases slot 0
        before attempting the load (mitigation for an upstream "failed
        to find available cells" error). On a guaranteed-miss restore
        that erase destroys the slot's existing KV that
        ``cache_prompt: true`` could prefix-match against the new
        prompt — turning a fast regenerate into a full re-prefill.

        The fix is a one-liner: check ``_slot_state_exists`` BEFORE
        calling ``restore_slot``. This test asserts:

        1. Returns False (no checkpoint = no restoration)
        2. NO HTTP request fires (no erase, no restore POST)
        3. Manifest is NOT touched (we didn't attempt anything)
        """
        transport = MockTransport()
        client = httpx.AsyncClient(transport=transport)
        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()
        # Empty directory ⇒ _slot_state_exists returns False for any
        # session_id.

        class FakeManager:
            def __init__(self) -> None:
                self._session_manifest = None  # no manifest = no skip-reason path
                self._slot_dir = str(slot_dir)
                self.model_id = "test-model"
                self.model_path = "/models/test-model.gguf"
                self.current_ctx_size = 8192
                self.kv_cache_type = "q8_0"

            def current_runtime_signature(self) -> dict:
                return {
                    "model_key": "test-model",
                    "model_id": "test-model",
                    "model_path": "/models/test-model.gguf",
                    "model_mtime": 0.0,
                    "ctx_size": 8192,
                    "kv_cache_type": "q8_0",
                }

            def kv_ttl_days_for_mode(self, mode: str = "") -> int:
                return 2

            def session_is_pinned(self, session_key: str, mode: str = "") -> bool:
                return False

        backend = LlamaCppBackend(client, "http://llamacpp:8080", server_manager=FakeManager())

        restored = await backend.restore_session_state(
            "no-such-session",
            request=_make_request(model="test-model", kv_session_key="no-such-session"),
        )

        assert restored is False, (
            "no checkpoint should return False, not attempt restore"
        )
        assert transport.requests == [], (
            "no HTTP request should fire — calling restore_slot would "
            "erase the slot, which is the bug we're fixing. requests: "
            f"{[r.url for r in transport.requests]}"
        )

    @pytest.mark.asyncio
    async def test_restore_session_state_calls_restore_slot_when_checkpoint_exists(
        self, tmp_path,
    ):
        """Sibling case: when a slot file IS present on disk, the
        existing restore path runs unchanged. Without this test the
        fix could regress to "always skip restore" and we'd silently
        lose KV cache restoration entirely.
        """
        transport = MockTransport()
        client = httpx.AsyncClient(transport=transport)
        slot_dir = tmp_path / "slots"
        slot_dir.mkdir()

        class FakeManager:
            def __init__(self) -> None:
                self._session_manifest = None
                self._slot_dir = str(slot_dir)
                self.model_id = "test-model"
                self.model_path = "/models/test-model.gguf"
                self.current_ctx_size = 8192
                self.kv_cache_type = "q8_0"

            def current_runtime_signature(self) -> dict:
                return {
                    "model_key": "test-model",
                    "model_id": "test-model",
                    "model_path": "/models/test-model.gguf",
                    "model_mtime": 0.0,
                    "ctx_size": 8192,
                    "kv_cache_type": "q8_0",
                }

            def kv_ttl_days_for_mode(self, mode: str = "") -> int:
                return 2

            def session_is_pinned(self, session_key: str, mode: str = "") -> bool:
                return False

        backend = LlamaCppBackend(client, "http://llamacpp:8080", server_manager=FakeManager())
        # Drop a placeholder file at the expected slot path so
        # _slot_state_exists picks it up (matches by filename prefix).
        slot_filename = backend._slot_storage_name("sess-real")
        (slot_dir / f"{slot_filename}.bin").write_bytes(b"placeholder")

        # Stub out the upstream restore endpoints. Erase POST returns 200,
        # restore POST returns 200 with success — that's the normal happy path.
        transport.add_response(
            "POST", "http://llamacpp:8080/slots/0?action=erase", json_data={"status": "ok"},
        )
        transport.add_response(
            "POST", "http://llamacpp:8080/slots/0?action=restore", json_data={"status": "ok"},
        )

        restored = await backend.restore_session_state(
            "sess-real",
            request=_make_request(model="test-model", kv_session_key="sess-real"),
        )

        assert restored is True, "restore should succeed when slot file exists"
        # erase + restore were both called — that's the existing behavior we kept.
        urls = [str(r.url) for r in transport.requests]
        assert any("action=erase" in u for u in urls)
        assert any("action=restore" in u for u in urls)

    def test_arch_fingerprint_tolerates_zero_on_either_side(self):
        """Architectural fingerprints accept a 0/missing on either side.

        Backward-compat: pre-T2-2 manifest rows have n_embed/n_layers_
        total/n_heads_kv defaulting to 0; rejecting on a 0-vs-real
        mismatch would force a cold prefill on every legacy row's
        first restore attempt. Rows that DO have non-zero stored
        values still get rejected when they disagree with non-zero
        live values — that's the symmetric guard we want.
        """
        from augmentum.models.llama_cpp import kv_restore_skip_reason

        # Compatible baseline fully populated on both sides.
        baseline_record = {
            "model_id": "m", "model_path": "/m.gguf", "model_mtime": 1.0,
            "ctx_size": 8192, "kv_cache_type": "q8_0",
            "n_embed": 4096, "n_layers_total": 32, "n_heads_kv": 8,
        }
        baseline_runtime = {
            "model_id": "m", "model_path": "/m.gguf", "model_mtime": 1.0,
            "ctx_size": 8192, "kv_cache_type": "q8_0",
            "n_embed": 4096, "n_layers_total": 32, "n_heads_kv": 8,
        }
        assert kv_restore_skip_reason(baseline_record, baseline_runtime) is None

        # Stored 0 + live populated → tolerated (legacy row, fresh runtime).
        legacy_record = {**baseline_record, "n_embed": 0, "n_layers_total": 0, "n_heads_kv": 0}
        assert kv_restore_skip_reason(legacy_record, baseline_runtime) is None

        # Stored populated + live 0 → tolerated (manifest written before
        # _last_profile was set, e.g. seeded by another process).
        unloaded_runtime = {**baseline_runtime, "n_embed": 0, "n_layers_total": 0, "n_heads_kv": 0}
        assert kv_restore_skip_reason(baseline_record, unloaded_runtime) is None

        # Both populated AND mismatched → reject (the protective case).
        wrong_arch_runtime = {**baseline_runtime, "n_embed": 5120}
        assert kv_restore_skip_reason(baseline_record, wrong_arch_runtime) == "n_embed changed"

    def test_to_openai_payload_moves_late_system_messages_to_front(self):
        backend, _ = self._make_backend()
        req = _make_request(
            messages=[
                Message(role="system", content="stable prompt"),
                Message(role="user", content="hello"),
                Message(role="system", content="dynamic memory"),
                Message(role="assistant", content="hi"),
            ],
        )

        payload = backend._to_openai_payload(req)

        assert payload["messages"][0] == {
            "role": "system",
            "content": "stable prompt\n\ndynamic memory",
        }
        assert [msg["role"] for msg in payload["messages"][1:]] == ["user", "assistant"]

    def test_to_openai_payload_rewrites_narrative_late_system_messages(self):
        backend, _ = self._make_backend()
        req = _make_request(
            messages=[
                Message(role="system", content="stable prompt"),
                Message(role="assistant", content="previous reply"),
                Message(role="system", content="dynamic memory"),
                Message(role="user", content="next turn"),
            ],
            kv_session_key="sess-live-123",
            kv_stable_messages=[
                Message(role="system", content="stable prompt"),
                Message(role="assistant", content="previous reply"),
                Message(role="user", content="next turn"),
            ],
        )

        payload = backend._to_openai_payload(req)

        assert payload["messages"][0] == {
            "role": "system",
            "content": "stable prompt",
        }
        # Leading-assistant guard prepends a synthetic user "scene opens"
        # carrier so strict alternation templates don't 500. See
        # `_ensure_user_first_after_system`.
        assert payload["messages"][1]["role"] == "user"
        assert "scene opens" in payload["messages"][1]["content"]
        assert payload["messages"][2] == {
            "role": "assistant",
            "content": "previous reply",
        }
        assert payload["messages"][3]["role"] == "user"
        assert "Augmentum narrative context" in payload["messages"][3]["content"]
        assert "dynamic memory" in payload["messages"][3]["content"]
        assert payload["messages"][4] == {
            "role": "user",
            "content": "next turn",
        }

    def test_to_openai_payload_inserts_user_carrier_when_first_role_is_assistant(self):
        # Narrative mode: session opens with a character/narrator turn before
        # the user has typed. Strict Llama 3.x / Mistral templates 500 with
        # "Conversation roles must alternate user/assistant/user/..." unless
        # the first non-system message is `user`.
        backend, _ = self._make_backend()
        req = _make_request(
            messages=[
                Message(role="system", content="narrator system prompt"),
                Message(role="assistant", content="The tavern door swings open."),
                Message(role="user", content="I step inside."),
            ],
        )

        payload = backend._to_openai_payload(req)

        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["system", "user", "assistant", "user"]
        assert "scene opens" in payload["messages"][1]["content"]
        assert payload["messages"][2]["content"] == "The tavern door swings open."
        assert payload["messages"][3]["content"] == "I step inside."

    def test_to_openai_payload_user_first_carrier_skipped_when_already_user(self):
        backend, _ = self._make_backend()
        req = _make_request(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ],
        )

        payload = backend._to_openai_payload(req)

        assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant"]

    def test_checkpoint_requests_defer_autosave(self):
        req = _make_request(
            kv_stable_messages=[Message(role="system", content="stable prompt")],
        )

        assert LlamaCppBackend._should_defer_session_save(req) is True
        assert LlamaCppBackend._should_defer_session_save(
            _make_request(kv_stable_messages=None)
        ) is False

    @staticmethod
    def _ready_manager():
        """Minimal FakeManager whose state passes _manage_slot's gate."""
        from augmentum.models.llama_server_manager import ProcessState

        m = MagicMock()
        m.state = ProcessState.READY
        m.model_id = "test-model"
        m.touch = MagicMock()
        # Token cache attribute the backend may inspect; None disables caching.
        m.token_cache = None
        # Restart-warm hint must be explicitly empty — a MagicMock auto-attr
        # is truthy and would trip the warm-pickup branch in _manage_slot.
        m._warm_session_key = ""
        return m

    @staticmethod
    def _ok_chat_response() -> httpx.Response:
        body = json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {},
        }).encode()
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "http://test"),
        )

    @pytest.mark.asyncio
    async def test_slot_lock_serializes_cross_session_chat(self, monkeypatch):
        """Two concurrent chats from different sessions must not interleave
        slot save/restore around generation. Tests single-slot
        semantics explicitly — under multi-slot, concurrent chats
        should NOT serialize this way; the multi-slot path is covered
        by tests/test_multislot_phase2.py.
        """
        from augmentum.config import settings
        monkeypatch.setattr(settings, "engine_multislot_enabled", False)
        backend, _ = self._make_backend()
        backend._manager = self._ready_manager()

        gen_started = asyncio.Event()
        gen_can_finish = asyncio.Event()
        call_log: list[str] = []

        async def fake_completions(*_args, **_kwargs):
            call_log.append("gen_start")
            gen_started.set()
            await gen_can_finish.wait()
            call_log.append("gen_end")
            return self._ok_chat_response()

        backend._client.post = AsyncMock(side_effect=fake_completions)

        async def fake_save(session_id, *_a, **_k):
            call_log.append(f"save:{session_id}")
            return True

        async def fake_restore(session_id, *_a, **_k):
            call_log.append(f"restore:{session_id}")
            return True

        backend.save_session_state = AsyncMock(side_effect=fake_save)
        backend.restore_session_state = AsyncMock(side_effect=fake_restore)

        # Stub _ensure_server / _build_token_prompt so we exercise the
        # /v1/chat/completions fallback path which uses fake_completions.
        backend._ensure_server = AsyncMock()
        backend._build_token_prompt = AsyncMock(return_value=None)

        req_a = _make_request(kv_session_key="sess-A")
        req_b = _make_request(kv_session_key="sess-B")

        task_a = asyncio.create_task(backend.chat(req_a))
        await gen_started.wait()  # A holds the slot lock and is mid-generation
        gen_started.clear()

        task_b = asyncio.create_task(backend.chat(req_b))
        # Give B a chance to attempt the lock. It must block.
        await asyncio.sleep(0.05)

        # B must NOT have called restore yet — A still holds the lock.
        restores_so_far = [e for e in call_log if e.startswith("restore:")]
        assert restores_so_far == ["restore:sess-A"], (
            f"B's restore ran before A finished: {call_log}"
        )

        # Release A; let B proceed.
        gen_can_finish.set()
        await task_a
        gen_can_finish.clear()
        await gen_started.wait()  # B reaches generation
        gen_can_finish.set()
        await task_b

        # A's full lifecycle must complete entirely before any of B's runs.
        a_end = call_log.index("save:sess-A")
        b_start = call_log.index("restore:sess-B")
        assert a_end < b_start, f"B started before A's save: {call_log}"

    @pytest.mark.asyncio
    async def test_slot_lock_skips_redundant_restore_for_same_session(self, monkeypatch):
        """Two consecutive chats with the same session don't re-save/restore.
        Tests single-slot semantics; multi-slot's hot-path no-op is
        covered by tests/test_multislot_phase2.py.
        """
        from augmentum.config import settings
        monkeypatch.setattr(settings, "engine_multislot_enabled", False)
        backend, _ = self._make_backend()
        backend._manager = self._ready_manager()
        backend._ensure_server = AsyncMock()
        backend._build_token_prompt = AsyncMock(return_value=None)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)
        backend._client.post = AsyncMock(side_effect=lambda *a, **k: self._ok_chat_response())

        req = _make_request(kv_session_key="sess-same")

        await backend.chat(req)
        await backend.chat(req)

        # restore should fire once: the first chat's _manage_slot does the
        # restore, the second chat sees _active_session==session_fp and skips.
        assert backend.restore_session_state.await_count == 1
        # save fires per generation (twice), guarding the slot for the next
        # restorer.
        assert backend.save_session_state.await_count == 2

    @pytest.mark.asyncio
    async def test_manage_slot_consumes_restart_warm_session(self, monkeypatch):
        """When the manager pre-loaded slot 0 with the MRU session, a
        matching request must skip the redundant restore round-trip.
        Tests single-slot semantics — multi-slot uses observed
        id_slot rather than the warm-session marker.
        """
        from augmentum.config import settings
        monkeypatch.setattr(settings, "engine_multislot_enabled", False)
        backend, _ = self._make_backend()
        backend._manager = self._ready_manager()
        backend._manager._warm_session_key = "warm-sess"

        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        # Request whose fingerprint matches the warm key.
        await backend._manage_slot(_make_request(kv_session_key="warm-sess"))

        # No restore fired — slot 0 is already warm.
        backend.restore_session_state.assert_not_awaited()
        backend.save_session_state.assert_not_awaited()
        # Active session is now the warm one and the warm hint is consumed.
        assert backend._get_session_for_slot(0) == "warm-sess"
        assert backend._manager._warm_session_key == ""

    @pytest.mark.asyncio
    async def test_manage_slot_warm_session_swaps_when_request_differs(self, monkeypatch):
        """If the warm session doesn't match the request, save it and
        restore the requested one. No KV is lost across the swap.
        Tests single-slot semantics; multi-slot doesn't pre-claim
        warm sessions (they're observed via id_slot post-response).
        """
        from augmentum.config import settings
        monkeypatch.setattr(settings, "engine_multislot_enabled", False)
        backend, _ = self._make_backend()
        backend._manager = self._ready_manager()
        backend._manager._warm_session_key = "warm-sess"

        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        await backend._manage_slot(_make_request(kv_session_key="other-sess"))

        # Warm session got persisted before being evicted from slot 0.
        # Phase 1 added explicit slot_id kwarg threading; the call now
        # passes slot_id=0.
        backend.save_session_state.assert_awaited_once_with("warm-sess", slot_id=0)
        backend.restore_session_state.assert_awaited_once()
        assert backend._get_session_for_slot(0) == "other-sess"

    @pytest.mark.asyncio
    async def test_apply_template_moves_late_system_messages_to_front(self):
        backend, transport = self._make_backend()
        transport.add_response(
            "POST",
            "http://llamacpp:8080/apply-template",
            json_data={"prompt": "ok"},
        )

        await backend.apply_template([
            {"role": "system", "content": "stable prompt"},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "dynamic memory"},
        ])

        sent = json.loads(transport.requests[0].content.decode("utf-8"))
        assert sent["messages"][0] == {
            "role": "system",
            "content": "stable prompt\n\ndynamic memory",
        }
        assert sent["messages"][1:] == [{"role": "user", "content": "hello"}]


class TestLlamaCppListModelsCache:
    """``list_models()`` is polled by Open WebUI, Settings, and Model
    Manager — pre-cache the disk scan ran 600-820ms per call. These tests
    pin the cache contract:

    - First call drives the scan; second call within TTL is served from
      memory.
    - Currently-loaded model is spliced AFTER the cache lookup so swaps
      are visible immediately regardless of TTL.
    - Concurrent callers on a cold cache collapse to one scan
      (singleflight via the lazy lock).
    """

    def _make_backend_with_fake_manager(
        self, files: list[dict], *, loaded_model: str = "",
    ):
        from unittest.mock import MagicMock
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={}))
        client = httpx.AsyncClient(transport=transport)
        scan_count = [0]

        def _fake_discover():
            scan_count[0] += 1
            return list(files)

        manager = MagicMock()
        manager.discover_gguf_files = _fake_discover
        manager.profile_cache = {}
        manager.model_id = loaded_model
        manager.current_mmproj_path = ""
        manager._find_paired_mmproj = MagicMock(return_value="")
        backend = LlamaCppBackend(
            client, "http://llamacpp:8080", server_manager=manager,
        )
        return backend, scan_count

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(self):
        backend, scan_count = self._make_backend_with_fake_manager([
            {"filename": "modelA.gguf", "path": "/m/modelA.gguf", "size": 1},
        ])
        await backend.list_models()
        await backend.list_models()
        assert scan_count[0] == 1, (
            "second call must not re-scan; got %d" % scan_count[0]
        )

    @pytest.mark.asyncio
    async def test_expired_cache_refreshes(self):
        backend, scan_count = self._make_backend_with_fake_manager([
            {"filename": "modelA.gguf", "path": "/m/modelA.gguf", "size": 1},
        ])
        await backend.list_models()
        # Backdate the cache stamp beyond TTL to force a refresh.
        ts, payload = backend._models_cache
        backend._models_cache = (ts - 9999.0, payload)
        await backend.list_models()
        assert scan_count[0] == 2

    @pytest.mark.asyncio
    async def test_loaded_model_spliced_outside_cache(self):
        """Loaded model can change between calls without TTL waiting."""
        backend, scan_count = self._make_backend_with_fake_manager(
            [{"filename": "modelA.gguf", "path": "/m/modelA.gguf", "size": 1}],
            loaded_model="",
        )
        first = await backend.list_models()
        assert all(m.name != "modelB" for m in first)
        # Simulate a model load — should appear on next list_models
        # WITHOUT a re-scan, because it splices outside the cache.
        backend._manager.model_id = "modelB"
        second = await backend.list_models()
        assert any(m.name == "modelB" for m in second)
        assert scan_count[0] == 1, "loaded-model splice must not invalidate cache"

    @pytest.mark.asyncio
    async def test_concurrent_callers_share_one_scan(self):
        """Singleflight: 5 concurrent cold callers do ONE scan."""
        backend, scan_count = self._make_backend_with_fake_manager([
            {"filename": "modelA.gguf", "path": "/m/modelA.gguf", "size": 1},
        ])
        results = await asyncio.gather(*[backend.list_models() for _ in range(5)])
        assert scan_count[0] == 1
        assert all(len(r) >= 1 for r in results)


# ===========================================================================
# Provider Registry
# ===========================================================================

class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    @patch("augmentum.models.provider_registry.settings")
    def test_init_ollama_only(self, mock_settings):
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = None
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "ollama"

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock()
        registry = ProviderRegistry(client)

        assert "ollama" in registry.available_backends
        assert "openai" not in registry.available_backends
        assert "llamacpp" not in registry.available_backends

    @patch("augmentum.models.provider_registry.settings")
    def test_init_all_backends(self, mock_settings):
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.openai_base_url = "https://api.openai.com/v1"
        mock_settings.llamacpp_base_url = "http://llamacpp:8080"
        mock_settings.default_backend = "ollama"

        from augmentum.models.provider_registry import ProviderRegistry

        client = MagicMock()
        registry = ProviderRegistry(client)

        assert set(registry.available_backends) == {"ollama", "openai", "llamacpp"}

    @patch("augmentum.models.provider_registry.settings")
    def test_get_backend_default(self, mock_settings):
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = None
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "ollama"

        from augmentum.models.provider_registry import ProviderRegistry

        registry = ProviderRegistry(MagicMock())
        backend = registry.get_backend()
        assert isinstance(backend, OllamaBackend)

    @patch("augmentum.models.provider_registry.settings")
    def test_get_backend_by_name(self, mock_settings):
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = "sk-test"
        mock_settings.openai_base_url = "https://api.openai.com/v1"
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "ollama"

        from augmentum.models.provider_registry import ProviderRegistry

        registry = ProviderRegistry(MagicMock())
        backend = registry.get_backend("openai")
        assert isinstance(backend, OpenAIBackend)

    @patch("augmentum.models.provider_registry.settings")
    def test_get_backend_missing_returns_none(self, mock_settings):
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = None
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "ollama"

        from augmentum.models.provider_registry import ProviderRegistry

        registry = ProviderRegistry(MagicMock())
        result = registry.get_backend("llamacpp")
        assert result is None

    @patch("augmentum.models.provider_registry.settings")
    def test_get_backend_missing_default_raises(self, mock_settings):
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = None
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "nonexistent"

        from augmentum.models.provider_registry import ProviderRegistry

        registry = ProviderRegistry(MagicMock())
        with pytest.raises(ValueError, match="not available"):
            registry.get_backend()

    @patch("augmentum.models.provider_registry.settings")
    def test_default_backend_property(self, mock_settings):
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = None
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "ollama"

        from augmentum.models.provider_registry import ProviderRegistry

        registry = ProviderRegistry(MagicMock())
        assert isinstance(registry.default_backend, OllamaBackend)

    @patch("augmentum.models.provider_registry.settings")
    def test_backends_property(self, mock_settings):
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = None
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "ollama"

        from augmentum.models.provider_registry import ProviderRegistry

        registry = ProviderRegistry(MagicMock())
        backends = registry.backends
        assert isinstance(backends, dict)
        assert "ollama" in backends


# ===========================================================================
# Model Manager
# ===========================================================================

class TestModelManager:
    """Tests for ModelManager."""

    def _make_manager(self) -> tuple[ModelManager, MagicMock]:
        registry = MagicMock()
        return ModelManager(registry), registry

    @pytest.mark.asyncio
    async def test_list_all_models_single_backend(self):
        manager, registry = self._make_manager()
        mock_backend = AsyncMock()
        mock_backend.list_models.return_value = [
            ModelInfo(name="model-a", model="model-a", size=1000),
        ]
        registry.backends = {"ollama": mock_backend}

        models = await manager.list_all_models()
        assert len(models) == 1
        assert models[0].name == "model-a"

    @pytest.mark.asyncio
    async def test_list_all_models_multi_backend(self):
        manager, registry = self._make_manager()
        ollama_backend = AsyncMock()
        ollama_backend.list_models.return_value = [
            ModelInfo(name="llama3:8b", model="llama3:8b"),
        ]
        openai_backend = AsyncMock()
        openai_backend.list_models.return_value = [
            ModelInfo(name="gpt-4", model="gpt-4"),
        ]
        registry.backends = {"ollama": ollama_backend, "openai": openai_backend}

        models = await manager.list_all_models()
        assert len(models) == 2
        # Multi-backend mode appends backend name
        names = [m.name for m in models]
        assert "llama3:8b (ollama)" in names
        assert "gpt-4 (openai)" in names

    @pytest.mark.asyncio
    async def test_list_all_models_does_not_mutate_cached_modelinfo(self):
        """Regression: backends cache and return SHARED ModelInfo objects
        (OpenAICompatibleBackend._list_models_cache). The " (provider)" display
        suffix added in multi-backend mode must be applied to COPIES — never by
        mutating the cached objects in place. Otherwise the provider_registry
        probe later reads the polluted "name (provider)" name and maps the model
        under an unrequestable key, so the cloud model silently becomes
        unroutable (and the suffix compounds on every call)."""
        manager, registry = self._make_manager()
        # Each backend returns the SAME cached list/objects on every call.
        cloud_cache = [ModelInfo(name="deepseek-v4-flash", model="deepseek-v4-flash")]
        other_cache = [ModelInfo(name="gpt-4", model="gpt-4")]
        cloud = AsyncMock()
        cloud.list_models.return_value = cloud_cache
        other = AsyncMock()
        other.list_models.return_value = other_cache
        registry.backends = {"deepseek": cloud, "openai": other}

        # The aggregated result carries the display suffix...
        models = await manager.list_all_models()
        assert "deepseek-v4-flash (deepseek)" in [m.name for m in models]

        # ...but the backend's CACHED object must stay pristine, so the registry
        # still maps the clean, requestable name.
        assert cloud_cache[0].name == "deepseek-v4-flash", (
            "list_all_models must not mutate the backend's cached ModelInfo"
        )

        # Calling again must NOT compound the suffix ("... (deepseek) (deepseek)").
        models2 = await manager.list_all_models()
        assert "deepseek-v4-flash (deepseek)" in [m.name for m in models2]
        assert cloud_cache[0].name == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_list_all_models_backend_failure(self):
        manager, registry = self._make_manager()
        working = AsyncMock()
        working.list_models.return_value = [ModelInfo(name="ok", model="ok")]
        failing = AsyncMock()
        failing.list_models.side_effect = Exception("Connection refused")
        registry.backends = {"working": working, "failing": failing}

        models = await manager.list_all_models()
        assert len(models) == 1  # Only working backend's models

    @pytest.mark.asyncio
    async def test_get_model_status_found(self):
        manager, registry = self._make_manager()
        mock_backend = AsyncMock()
        mock_backend.show_model.return_value = ModelDetails(
            quantization_level="Q4_K_M",
            parameter_size="8B",
        )
        registry.backends = {"ollama": mock_backend}

        status = await manager.get_model_status("llama3:8b")
        assert isinstance(status, ModelStatus)
        assert status.available is True
        assert status.backend == "ollama"
        assert status.quantization == "Q4_K_M"
        assert status.parameter_count == "8B"

    @pytest.mark.asyncio
    async def test_get_model_status_not_found(self):
        manager, registry = self._make_manager()
        mock_backend = AsyncMock()
        mock_backend.show_model.side_effect = Exception("Not found")
        registry.backends = {"ollama": mock_backend}

        status = await manager.get_model_status("nonexistent")
        assert status.available is False
        assert status.backend == "unknown"

    @pytest.mark.asyncio
    async def test_get_running_models(self):
        manager, registry = self._make_manager()
        mock_backend = MagicMock()
        mock_backend._base_url = "http://ollama:11434"
        mock_client = AsyncMock()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "models": [
                    {
                        "name": "llama3:8b",
                        "size": 5_000_000_000,
                        "size_vram": 4_000_000_000,
                        "expires_at": "2024-01-01T01:00:00Z",
                        "details": {"family": "llama"},
                    }
                ]
            },
        )
        mock_client.get.return_value.raise_for_status = MagicMock()
        mock_backend._client = mock_client
        registry.get_backend.return_value = mock_backend

        running = await manager.get_running_models()
        assert len(running) == 1
        assert running[0].name == "llama3:8b"
        assert running[0].backend == "ollama"
        assert running[0].size_vram == 4_000_000_000

    @pytest.mark.asyncio
    async def test_get_running_models_no_ollama(self):
        manager, registry = self._make_manager()
        registry.get_backend.return_value = None

        running = await manager.get_running_models()
        assert running == []

    @pytest.mark.asyncio
    async def test_load_model(self):
        manager, registry = self._make_manager()
        mock_backend = MagicMock()
        mock_backend._base_url = "http://ollama:11434"
        mock_client = AsyncMock()
        mock_client.post.return_value = MagicMock(status_code=200)
        mock_backend._client = mock_client
        registry.get_backend.return_value = mock_backend

        success = await manager.load_model("llama3:8b")
        assert success is True
        mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_no_ollama(self):
        manager, registry = self._make_manager()
        registry.get_backend.return_value = None

        success = await manager.load_model("llama3:8b")
        assert success is False

    @pytest.mark.asyncio
    async def test_load_model_failure(self):
        manager, registry = self._make_manager()
        mock_backend = MagicMock()
        mock_backend._base_url = "http://ollama:11434"
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("Connection refused")
        mock_backend._client = mock_client
        registry.get_backend.return_value = mock_backend

        success = await manager.load_model("llama3:8b")
        assert success is False

    @pytest.mark.asyncio
    async def test_unload_model(self):
        manager, registry = self._make_manager()
        mock_backend = MagicMock()
        mock_backend._base_url = "http://ollama:11434"
        mock_client = AsyncMock()
        mock_client.post.return_value = MagicMock(status_code=200)
        mock_backend._client = mock_client
        registry.get_backend.return_value = mock_backend

        success = await manager.unload_model("llama3:8b")
        assert success is True

    @pytest.mark.asyncio
    async def test_unload_model_no_ollama(self):
        manager, registry = self._make_manager()
        registry.get_backend.return_value = None

        success = await manager.unload_model("llama3:8b")
        assert success is False


# ===========================================================================
# Data Model Tests
# ===========================================================================

class TestDataModels:
    """Tests for base data models."""

    def test_message_defaults(self):
        msg = Message(role="user", content="hello")
        assert msg.images is None
        assert msg.tool_calls is None

    def test_message_with_images(self):
        msg = Message(role="user", content="describe", images=["base64abc"])
        assert msg.images == ["base64abc"]

    def test_internal_chat_request_defaults(self):
        req = InternalChatRequest(
            model="test",
            messages=[Message(role="user", content="hi")],
        )
        assert req.stream is False
        assert req.temperature is None
        assert req.tools is None
        assert req.raw_options is None

    def test_usage_defaults(self):
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_internal_chat_response_defaults(self):
        resp = InternalChatResponse(
            message=Message(role="assistant", content="hi"),
            model="test",
        )
        assert resp.finish_reason is None
        assert resp.timing is None
        assert resp.usage.total_tokens == 0

    def test_stream_chunk_defaults(self):
        chunk = InternalStreamChunk()
        assert chunk.content_delta == ""
        assert chunk.role is None
        assert chunk.done is False
        assert chunk.usage is None

    def test_model_info_defaults(self):
        info = ModelInfo(name="test", model="test")
        assert info.size == 0
        assert info.digest == ""
        assert info.details is None

    def test_model_details_extended_fields(self):
        details = ModelDetails(
            format="gguf",
            family="llama",
            parameter_size="8B",
            quantization_level="Q4_K_M",
            system_prompt="Be helpful",
        )
        assert details.format == "gguf"
        assert details.family == "llama"
        assert details.parameter_size == "8B"
        assert details.quantization_level == "Q4_K_M"
        assert details.system_prompt == "Be helpful"

    def test_model_details_backwards_compat(self):
        """Existing code that only uses the original fields still works."""
        details = ModelDetails(
            modelfile="FROM llama3.1:8b",
            parameters="temperature 0.7",
            template="{{ .System }}",
            details={"family": "llama"},
        )
        assert details.modelfile == "FROM llama3.1:8b"
        assert details.format == ""  # default
        assert details.family == ""  # default

    def test_model_status(self):
        status = ModelStatus(name="test", available=True, backend="ollama")
        assert status.loaded is False
        assert status.vram_usage == 0

    def test_running_model(self):
        rm = RunningModel(name="llama3:8b", backend="ollama", size_vram=4_000_000_000)
        assert rm.size_ram == 0
        assert rm.expires_at == ""


# ===========================================================================
# Config Routes
# ===========================================================================

class TestConfigRoutes:
    """Tests for config API routes."""

    def test_get_config(self, client):
        resp = client.get("/api/config/")
        assert resp.status_code == 200
        data = resp.json()
        # API key should be redacted
        assert data.get("openai_api_key") in (None, "***")
        # Non-sensitive fields should be present
        assert "host" in data
        assert "port" in data

    def test_get_config_section(self, client):
        resp = client.get("/api/config/section/ollama")
        assert resp.status_code == 200
        data = resp.json()
        assert "ollama_base_url" in data

    def test_get_config_section_empty(self, client):
        resp = client.get("/api/config/section/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {}


# ===========================================================================
# Model Routes
# ===========================================================================

class TestModelRoutes:
    """Tests for model management API routes."""

    def test_models_status(self, client):
        resp = client.get("/api/models/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data

    def test_running_models(self, client):
        resp = client.get("/api/models/running")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data

    def test_model_info(self, client):
        resp = client.get("/api/models/llama3.1:8b/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "name" in data
        assert "available" in data

    def test_load_model(self, client):
        resp = client.post("/api/models/llama3.1:8b/load")
        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data

    def test_unload_model(self, client):
        resp = client.post("/api/models/llama3.1:8b/unload")
        assert resp.status_code == 200
        data = resp.json()
        assert "success" in data


class TestModelLimitExtraction:
    """_pick_limit_int captures a real per-model context/output window from a
    provider /v1/models entry, so /v1/models reports the source-of-truth limit
    instead of a hand-guessed number (external harnesses size their compactor
    off this)."""

    CTX = ("max_input_tokens", "context_length", "context_window",
           "context_size", "max_model_len")
    OUT = ("max_completion_tokens", "max_output_tokens", "max_tokens")

    def test_toplevel_field(self):
        # OpenRouter list entries carry context_length at the top level
        assert OpenAIBackend._pick_limit_int(
            {"context_length": 200000}, self.CTX) == 200000

    def test_nested_top_provider(self):
        # OpenRouter also nests under top_provider
        assert OpenAIBackend._pick_limit_int(
            {"top_provider": {"context_length": 400000}}, self.CTX) == 400000
        assert OpenAIBackend._pick_limit_int(
            {"top_provider": {"max_completion_tokens": 128000}}, self.OUT) == 128000

    def test_vllm_and_alt_fields(self):
        assert OpenAIBackend._pick_limit_int(
            {"max_model_len": 32768}, self.CTX) == 32768

    def test_absent_or_invalid_is_zero(self):
        assert OpenAIBackend._pick_limit_int({}, self.CTX) == 0
        assert OpenAIBackend._pick_limit_int({"context_length": 0}, self.CTX) == 0
        assert OpenAIBackend._pick_limit_int({"context_length": "big"}, self.CTX) == 0
        assert OpenAIBackend._pick_limit_int("not-a-dict", self.CTX) == 0
