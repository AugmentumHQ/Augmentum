"""Tests for Google Gemini converter and backend adapter."""

from __future__ import annotations

from augmentum.models.converters.gemini import (
    GeminiConverter,
    convert_response,
    get_safety_settings,
    get_thinking_config,
)

# ===================================================================
# TestGeminiMessageConversion
# ===================================================================


class TestGeminiMessageConversion:
    """Test GeminiConverter.convert_messages."""

    def setup_method(self) -> None:
        self.converter = GeminiConverter()

    def test_system_to_instruction(self) -> None:
        """Leading system messages become systemInstruction."""
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = self.converter.convert_messages(msgs)
        si = result["systemInstruction"]
        assert si is not None
        assert len(si["parts"]) == 1
        assert si["parts"][0]["text"] == "You are helpful."
        # Contents should only have the user message.
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"

    def test_multiple_system_to_instruction(self) -> None:
        """Multiple leading system messages all go to systemInstruction."""
        msgs = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Go"},
        ]
        result = self.converter.convert_messages(msgs)
        assert len(result["systemInstruction"]["parts"]) == 2
        assert result["systemInstruction"]["parts"][0]["text"] == "Rule 1"
        assert result["systemInstruction"]["parts"][1]["text"] == "Rule 2"

    def test_role_mapping(self) -> None:
        """assistant -> model, user -> user."""
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = self.converter.convert_messages(msgs)
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][1]["role"] == "model"

    def test_image_to_inline_data(self) -> None:
        """Data URI images converted to inlineData parts."""
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
        parts = result["contents"][0]["parts"]
        assert any(p.get("text") == "What is this?" for p in parts)
        inline = next(p for p in parts if "inlineData" in p)
        assert inline["inlineData"]["mimeType"] == "image/png"
        assert inline["inlineData"]["data"] == "iVBOR"

    def test_tool_calls_to_function_call(self) -> None:
        """Assistant tool_calls converted to functionCall parts."""
        msgs = [
            {"role": "user", "content": "Search cats"},
            {
                "role": "assistant",
                "content": "Searching...",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "cats"}',
                        },
                    }
                ],
            },
        ]
        result = self.converter.convert_messages(msgs)
        model_parts = result["contents"][1]["parts"]
        fc_parts = [p for p in model_parts if "functionCall" in p]
        assert len(fc_parts) == 1
        assert fc_parts[0]["functionCall"]["name"] == "search"
        assert fc_parts[0]["functionCall"]["args"] == {"query": "cats"}

    def test_tool_result_to_function_response(self) -> None:
        """Tool result messages become functionResponse parts."""
        msgs = [
            {"role": "user", "content": "Go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search",
                "content": "Found results",
            },
        ]
        result = self.converter.convert_messages(msgs)
        # Tool result should be in a user-role content (since tool -> user).
        user_contents = [c for c in result["contents"] if c["role"] == "user"]
        fr_parts = []
        for uc in user_contents:
            for p in uc["parts"]:
                if "functionResponse" in p:
                    fr_parts.append(p)
        assert len(fr_parts) == 1
        assert fr_parts[0]["functionResponse"]["name"] == "search"
        assert fr_parts[0]["functionResponse"]["response"]["content"] == "Found results"

    def test_consecutive_merged(self) -> None:
        """Consecutive same-role messages are merged (parts extended)."""
        msgs = [
            {"role": "user", "content": "Part 1"},
            {"role": "user", "content": "Part 2"},
            {"role": "assistant", "content": "Reply"},
        ]
        result = self.converter.convert_messages(msgs)
        # Should merge into 1 user + 1 model.
        assert len(result["contents"]) == 2
        user_parts = result["contents"][0]["parts"]
        assert len(user_parts) == 2
        assert user_parts[0]["text"] == "Part 1"
        assert user_parts[1]["text"] == "Part 2"

    def test_name_prepended(self) -> None:
        """Message name field prepended to text."""
        msgs = [
            {"role": "user", "content": "Hello", "name": "Alice"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = self.converter.convert_messages(msgs)
        text = result["contents"][0]["parts"][0]["text"]
        assert text.startswith("Alice: ")

    def test_no_system_returns_none(self) -> None:
        """No system messages -> systemInstruction is None."""
        msgs = [{"role": "user", "content": "Hi"}]
        result = self.converter.convert_messages(msgs)
        assert result["systemInstruction"] is None


# ===================================================================
# TestGeminiSafetySettings
# ===================================================================


class TestGeminiSafetySettings:
    """Test safety settings generation."""

    def test_all_categories_off(self) -> None:
        """All base categories set to OFF."""
        settings = get_safety_settings()
        assert len(settings) == 5
        for s in settings:
            assert s["threshold"] == "OFF"
        cats = {s["category"] for s in settings}
        assert "HARM_CATEGORY_HARASSMENT" in cats
        assert "HARM_CATEGORY_HATE_SPEECH" in cats
        assert "HARM_CATEGORY_SEXUALLY_EXPLICIT" in cats
        assert "HARM_CATEGORY_DANGEROUS_CONTENT" in cats
        assert "HARM_CATEGORY_CIVIC_INTEGRITY" in cats

    def test_vertex_adds_extra_categories(self) -> None:
        """Vertex mode adds extra categories."""
        settings = get_safety_settings(vertex=True)
        assert len(settings) == 10  # 5 base + 5 vertex
        cats = {s["category"] for s in settings}
        assert "HARM_CATEGORY_IMAGE_HATE" in cats
        assert "HARM_CATEGORY_JAILBREAK" in cats
        for s in settings:
            assert s["threshold"] == "OFF"


# ===================================================================
# TestGeminiThinking
# ===================================================================


class TestGeminiThinking:
    """Test thinking configuration for different Gemini models."""

    def test_flash_25_budget(self) -> None:
        """Gemini 2.5 Flash uses thinkingBudget, capped [0, 24576]."""
        cfg = get_thinking_config("gemini-2.5-flash", "medium", max_tokens=8192)
        assert cfg is not None
        assert "thinkingBudget" in cfg["thinkingConfig"]
        assert cfg["thinkingConfig"]["includeThoughts"] is True
        budget = cfg["thinkingConfig"]["thinkingBudget"]
        assert 0 <= budget <= 24576

    def test_flash_25_budget_min_zero(self) -> None:
        """Gemini 2.5 Flash at min effort -> budget clamped to 0."""
        cfg = get_thinking_config("gemini-2.5-flash", "min", max_tokens=8192)
        assert cfg["thinkingConfig"]["thinkingBudget"] == 0

    def test_pro_25_budget(self) -> None:
        """Gemini 2.5 Pro uses thinkingBudget, capped [128, 32768]."""
        cfg = get_thinking_config("gemini-2.5-pro", "medium", max_tokens=8192)
        assert cfg is not None
        assert "thinkingBudget" in cfg["thinkingConfig"]
        budget = cfg["thinkingConfig"]["thinkingBudget"]
        assert 128 <= budget <= 32768

    def test_pro_25_budget_min_clamps_to_128(self) -> None:
        """Gemini 2.5 Pro at min effort -> budget clamped to 128."""
        cfg = get_thinking_config("gemini-2.5-pro", "min", max_tokens=8192)
        assert cfg["thinkingConfig"]["thinkingBudget"] == 128

    def test_gemini3_flash_level(self) -> None:
        """Gemini 3 Flash uses thinkingLevel string."""
        cfg = get_thinking_config("gemini-3-flash", "medium", max_tokens=8192)
        assert cfg is not None
        assert cfg["thinkingConfig"]["thinkingLevel"] == "medium"
        assert cfg["thinkingConfig"]["includeThoughts"] is True

    def test_gemini3_flash_min_is_minimal(self) -> None:
        cfg = get_thinking_config("gemini-3-flash", "min", max_tokens=8192)
        assert cfg["thinkingConfig"]["thinkingLevel"] == "minimal"

    def test_gemini3_flash_max_is_high(self) -> None:
        cfg = get_thinking_config("gemini-3-flash", "max", max_tokens=8192)
        assert cfg["thinkingConfig"]["thinkingLevel"] == "high"

    def test_gemini3_pro_level(self) -> None:
        """Gemini 3 Pro uses thinkingLevel with different mapping."""
        cfg = get_thinking_config("gemini-3-pro", "medium", max_tokens=8192)
        assert cfg is not None
        assert cfg["thinkingConfig"]["thinkingLevel"] == "low"

    def test_gemini3_pro_high_is_high(self) -> None:
        cfg = get_thinking_config("gemini-3-pro", "high", max_tokens=8192)
        assert cfg["thinkingConfig"]["thinkingLevel"] == "high"

    def test_unknown_model_returns_none(self) -> None:
        """Unknown models get no thinking config."""
        cfg = get_thinking_config("gemini-1.5-pro", "medium", max_tokens=8192)
        assert cfg is None


# ===================================================================
# TestGeminiResponseConversion
# ===================================================================


class TestGeminiResponseConversion:
    """Test convert_response normalisation."""

    def test_text_response(self) -> None:
        data = {
            "candidates": [{
                "content": {"parts": [{"text": "Hello!"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        result = convert_response(data)
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_tool_call_response(self) -> None:
        data = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me search."},
                        {
                            "functionCall": {
                                "name": "search",
                                "args": {"query": "cats"},
                            }
                        },
                    ]
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 30,
                "totalTokenCount": 50,
            },
        }
        result = convert_response(data)
        assert result["content"] == "Let me search."
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == {"query": "cats"}
        assert result["finish_reason"] == "tool_calls"

    def test_thought_parts_skipped(self) -> None:
        """Parts with thought=True are skipped."""
        data = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "thinking...", "thought": True},
                        {"text": "Answer"},
                    ]
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 10,
                "totalTokenCount": 15,
            },
        }
        result = convert_response(data)
        assert result["content"] == "Answer"

    def test_finish_reason_mapping(self) -> None:
        for gemini_reason, expected in [
            ("STOP", "stop"),
            ("MAX_TOKENS", "length"),
            ("SAFETY", "content_filter"),
            ("RECITATION", "content_filter"),
        ]:
            data = {
                "candidates": [{
                    "content": {"parts": [{"text": "x"}]},
                    "finishReason": gemini_reason,
                }],
                "usageMetadata": {
                    "promptTokenCount": 0,
                    "candidatesTokenCount": 0,
                    "totalTokenCount": 0,
                },
            }
            assert convert_response(data)["finish_reason"] == expected

    def test_empty_candidates(self) -> None:
        """Empty candidates returns defaults."""
        result = convert_response({"candidates": []})
        assert result["content"] == ""
        assert result["tool_calls"] is None


# ===================================================================
# TestGeminiRequestBuilding
# ===================================================================


class TestGeminiRequestBuilding:
    """Test GeminiBackend request body and endpoint construction."""

    def test_full_body(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.gemini import GeminiBackend
        from augmentum.models.base import InternalChatRequest, Message

        backend = GeminiBackend(
            client=MagicMock(),
            api_key="test-key",
        )
        request = InternalChatRequest(
            model="gemini-2.5-flash",
            messages=[
                Message(role="system", content="Be brief."),
                Message(role="user", content="Hi"),
            ],
            temperature=0.7,
            max_tokens=1024,
            stop=["END"],
        )
        body = backend._build_body(request)
        assert "contents" in body
        assert "systemInstruction" in body
        assert body["systemInstruction"]["parts"][0]["text"] == "Be brief."
        gen = body["generationConfig"]
        assert gen["maxOutputTokens"] == 1024
        assert gen["temperature"] == 0.7
        assert gen["stopSequences"] == ["END"]
        assert "safetySettings" in body

    def test_endpoint_with_api_key(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.gemini import GeminiBackend

        backend = GeminiBackend(
            client=MagicMock(),
            api_key="test-key",
        )
        url = backend._endpoint("gemini-2.5-flash", "generateContent")
        assert "key=test-key" in url
        assert "models/gemini-2.5-flash:generateContent" in url
        assert "&alt=sse" not in url

    def test_endpoint_stream_adds_alt_sse(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.gemini import GeminiBackend

        backend = GeminiBackend(
            client=MagicMock(),
            api_key="test-key",
        )
        url = backend._endpoint(
            "gemini-2.5-flash", "streamGenerateContent", stream=True
        )
        assert "&alt=sse" in url

    def test_vertex_endpoint(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.gemini import GeminiBackend

        backend = GeminiBackend(
            client=MagicMock(),
            api_key="vertex-token",
            vertex=True,
            vertex_project="my-project",
            vertex_region="us-east1",
        )
        url = backend._endpoint("gemini-2.5-pro", "generateContent")
        assert "us-east1-aiplatform.googleapis.com" in url
        assert "projects/my-project" in url
        assert "locations/us-east1" in url
        assert "models/gemini-2.5-pro:generateContent" in url
        # Vertex doesn't use key in URL.
        assert "key=" not in url

    def test_vertex_headers_bearer(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.gemini import GeminiBackend

        backend = GeminiBackend(
            client=MagicMock(),
            api_key="vertex-token",
            vertex=True,
        )
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer vertex-token"

    def test_ai_studio_headers_no_auth(self) -> None:
        from unittest.mock import MagicMock

        from augmentum.models.adapters.gemini import GeminiBackend

        backend = GeminiBackend(
            client=MagicMock(),
            api_key="test-key",
        )
        headers = backend._headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_tool_conversion(self) -> None:
        from augmentum.models.adapters.gemini import GeminiBackend

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
        result = GeminiBackend._convert_tools(tools)
        assert result is not None
        assert len(result) == 1
        decls = result[0]["function_declarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"

    def test_context_length(self) -> None:
        """get_context_length returns correct values."""
        import asyncio
        from unittest.mock import MagicMock

        from augmentum.models.adapters.gemini import GeminiBackend

        backend = GeminiBackend(client=MagicMock(), api_key="test-key")
        loop = asyncio.new_event_loop()
        try:
            assert loop.run_until_complete(
                backend.get_context_length("gemini-2.5-flash")
            ) == 1_000_000
            assert loop.run_until_complete(
                backend.get_context_length("gemini-2.5-pro")
            ) == 2_000_000
            assert loop.run_until_complete(
                backend.get_context_length("gemini-nano")
            ) == 128_000
        finally:
            loop.close()
