"""Integration tests for provider backends and converters.

Verifies full request building for Claude, Gemini, and profiled OpenAI
backends, plus converter round-trip preservation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from augmentum.models.base import InternalChatRequest, Message

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    *,
    model: str = "test-model",
    system: str | None = None,
    user: str = "Hello",
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    tools: list[dict] | None = None,
    think: bool = False,
) -> InternalChatRequest:
    """Build a minimal InternalChatRequest for testing."""
    msgs: list[Message] = []
    if system:
        msgs.append(Message(role="system", content=system))
    msgs.append(Message(role="user", content=user))
    return InternalChatRequest(
        model=model,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        tools=tools,
        think=think,
    )


def _mock_httpx_client() -> AsyncMock:
    return AsyncMock()


# ===================================================================
# TestClaudeRequestBuilding
# ===================================================================


class TestClaudeRequestBuilding:
    """Verify ClaudeBackend builds correct request bodies and headers."""

    def _make_backend(self, **kwargs):
        from augmentum.models.adapters.claude import ClaudeBackend

        return ClaudeBackend(
            client=_mock_httpx_client(),
            api_key="sk-test-key",
            cache_enabled=False,
            **kwargs,
        )

    def test_full_request_body(self):
        """System + user message with temperature and max_tokens."""
        backend = self._make_backend()
        req = _make_request(
            model="claude-3-5-sonnet-latest",
            system="You are helpful.",
            user="What is 2+2?",
            temperature=0.7,
            max_tokens=1024,
        )
        body = backend._build_request_body(req)

        assert body["model"] == "claude-3-5-sonnet-latest"
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.7
        # System is a list of text blocks
        assert isinstance(body["system"], list)
        assert any(b.get("text") for b in body["system"])
        # Messages should have 1 user message (system extracted)
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_headers_include_api_key(self):
        """x-api-key and anthropic-version headers must be present."""
        backend = self._make_backend()
        headers = backend._headers()

        assert headers["x-api-key"] == "sk-test-key"
        assert "anthropic-version" in headers

    def test_thinking_enabled_removes_sampling(self):
        """When think=True on a thinking model, temperature/top_p are removed."""
        backend = self._make_backend()
        req = _make_request(
            model="claude-3-7-sonnet-latest",
            user="Think about this",
            temperature=0.9,
            top_p=0.95,
            think=True,
        )
        body = backend._build_request_body(req)

        assert "temperature" not in body
        assert "top_p" not in body
        assert "thinking" in body


# ===================================================================
# TestGeminiRequestBuilding
# ===================================================================


class TestGeminiRequestBuilding:
    """Verify GeminiBackend builds correct bodies, endpoints, and headers."""

    def _make_backend(self, **kwargs):
        from augmentum.models.adapters.gemini import GeminiBackend

        return GeminiBackend(
            client=_mock_httpx_client(),
            api_key="gemini-test-key",
            **kwargs,
        )

    def test_full_body(self):
        """System + user produces contents, systemInstruction, generationConfig, safetySettings."""
        backend = self._make_backend()
        req = _make_request(
            model="gemini-2.0-flash",
            system="You are a tutor.",
            user="Explain gravity.",
            temperature=0.5,
            max_tokens=2048,
        )
        body = backend._build_body(req)

        assert "contents" in body
        assert isinstance(body["contents"], list)
        assert body["systemInstruction"] is not None
        gen = body.get("generationConfig", {})
        assert gen.get("temperature") == 0.5
        assert gen.get("maxOutputTokens") == 2048
        assert "safetySettings" in body

    def test_endpoint_with_api_key(self):
        """AI Studio endpoint must include key= query param."""
        backend = self._make_backend()
        url = backend._endpoint("gemini-2.0-flash", "generateContent")

        assert "key=gemini-test-key" in url

    def test_vertex_endpoint_format(self):
        """Vertex endpoint must contain region and project."""
        backend = self._make_backend(
            vertex=True,
            vertex_project="my-project",
            vertex_region="us-east1",
        )
        url = backend._endpoint("gemini-2.0-flash", "generateContent")

        assert "us-east1" in url
        assert "my-project" in url
        assert "publishers/google/models/gemini-2.0-flash" in url

    def test_tools_as_function_declarations(self):
        """Tools must be wrapped in function_declarations."""
        backend = self._make_backend()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        req = _make_request(
            model="gemini-2.0-flash",
            user="What is the weather?",
            tools=tools,
        )
        body = backend._build_body(req)

        assert "tools" in body
        assert isinstance(body["tools"], list)
        assert "function_declarations" in body["tools"][0]


# ===================================================================
# TestProfiledOpenAIRequestBuilding
# ===================================================================


class TestProfiledOpenAIRequestBuilding:
    """Verify OpenAIBackend headers differ based on provider profile."""

    def _make_backend(self, profile_id: str, api_key: str = "test-key"):
        from augmentum.models.openai_compat import OpenAIBackend
        from augmentum.models.provider_profiles import get_profile

        profile = get_profile(profile_id)
        assert profile is not None, f"Profile {profile_id!r} not found"
        return OpenAIBackend(
            client=_mock_httpx_client(),
            base_url=profile.base_url or "https://example.com/v1",
            api_key=api_key,
            profile=profile,
        )

    def test_openrouter_headers(self):
        """OpenRouter profile must set HTTP-Referer, X-Title, and Authorization."""
        backend = self._make_backend("openrouter", api_key="or-key")
        headers = backend._headers()

        assert headers.get("HTTP-Referer") == "https://augmentum.dev"
        assert headers.get("X-Title") == "Augmentum"
        assert headers.get("Authorization") == "Bearer or-key"

    def test_azure_api_key_header(self):
        """Azure profile must use api-key header instead of Authorization."""
        backend = self._make_backend("azure", api_key="az-key")
        headers = backend._headers()

        assert headers.get("api-key") == "az-key"
        assert "Authorization" not in headers

    def test_nanogpt_bearer_auth(self):
        """NanoGPT authenticates with Authorization: Bearer (not x-api-key).

        Regression guard: a prior x-api-key override produced silent 401s;
        NanoGPT's docs use bearer auth across all examples (2026-06-15).
        """
        backend = self._make_backend("nanogpt", api_key="nano-key")
        headers = backend._headers()

        assert headers.get("Authorization") == "Bearer nano-key"
        assert "x-api-key" not in headers


# ===================================================================
# TestConverterRoundTrips
# ===================================================================


class TestConverterRoundTrips:
    """Verify message converters preserve content across transformations."""

    def test_claude_tool_round_trip(self):
        """User → assistant+tool_calls → tool_result → assistant preserves all 4 messages."""
        from augmentum.models.converters.claude import ClaudeConverter

        messages = [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "22°C, sunny",
            },
            {"role": "assistant", "content": "It is 22°C and sunny in Paris."},
        ]

        converter = ClaudeConverter()
        result = converter.convert_messages(messages)

        claude_msgs = result["messages"]
        # All 4 logical messages should be preserved (tool result becomes user role)
        assert len(claude_msgs) == 4

        # First message: user
        assert claude_msgs[0]["role"] == "user"

        # Second message: assistant with tool_use block
        assert claude_msgs[1]["role"] == "assistant"
        tool_use_blocks = [
            b for b in claude_msgs[1]["content"] if b.get("type") == "tool_use"
        ]
        assert len(tool_use_blocks) == 1

        # Third message: user with tool_result block
        assert claude_msgs[2]["role"] == "user"
        tool_result_blocks = [
            b for b in claude_msgs[2]["content"] if b.get("type") == "tool_result"
        ]
        assert len(tool_result_blocks) == 1

        # Fourth message: assistant
        assert claude_msgs[3]["role"] == "assistant"

    def test_gemini_preserves_all_messages(self):
        """System + user + assistant + user → systemInstruction + 3 contents."""
        from augmentum.models.converters.gemini import GeminiConverter

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]

        converter = GeminiConverter()
        result = converter.convert_messages(messages)

        assert result["systemInstruction"] is not None
        assert len(result["contents"]) == 3

    def test_mistral_tool_ids_deterministic(self):
        """Same tool ID hashed twice must give the same result."""
        from augmentum.models.converters.mistral import _hash_tool_id

        id1 = _hash_tool_id("call_abc123")
        id2 = _hash_tool_id("call_abc123")

        assert id1 == id2
        assert len(id1) == 9

    def test_cohere_params_round_trip(self):
        """map_params with top_k and top_p returns k and p."""
        from augmentum.models.converters.cohere import CohereConverter

        converter = CohereConverter()
        mapped = converter.map_params({"top_k": 40, "top_p": 0.9, "temperature": 0.7})

        assert mapped["k"] == 40
        assert mapped["p"] == 0.9
        assert mapped["temperature"] == 0.7
        assert "top_k" not in mapped
        assert "top_p" not in mapped
