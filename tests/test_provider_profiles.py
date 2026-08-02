"""Tests for provider profiles, Mistral converter, and Cohere converter."""

from __future__ import annotations

import pytest

from augmentum.models.converters.cohere import CohereConverter
from augmentum.models.converters.mistral import MistralConverter
from augmentum.models.provider_profiles import get_profile, list_profiles

# ---------------------------------------------------------------------------
# ProviderProfile catalog tests
# ---------------------------------------------------------------------------


class TestProviderProfile:
    """Verify built-in profile entries exist and have correct values."""

    def test_openai_profile(self) -> None:
        p = get_profile("openai")
        assert p is not None
        assert "openai.com" in p.base_url
        assert p.auth_type == "bearer"
        assert p.supports_thinking is True
        assert p.thinking_param == "reasoning_effort"

    def test_groq_profile(self) -> None:
        p = get_profile("groq")
        assert p is not None
        assert "groq.com" in p.base_url
        assert p.auth_type == "bearer"

    def test_openrouter_profile(self) -> None:
        p = get_profile("openrouter")
        assert p is not None
        assert "openrouter.ai" in p.base_url
        assert "HTTP-Referer" in p.extra_headers
        assert "X-Title" in p.extra_headers
        assert p.supports_thinking is True

    def test_deepseek_profile(self) -> None:
        p = get_profile("deepseek")
        assert p is not None
        assert "deepseek.com" in p.base_url
        assert p.post_process == "semi"
        assert p.model_list_url != ""

    def test_azure_profile(self) -> None:
        p = get_profile("azure")
        assert p is not None
        assert p.auth_type == "api-key"
        assert p.auth_header == "api-key"

    def test_get_profile_unknown_returns_none(self) -> None:
        assert get_profile("nonexistent_provider_xyz") is None

    def test_list_profiles_returns_all(self) -> None:
        profiles = list_profiles()
        assert len(profiles) >= 18
        ids = [p.id for p in profiles]
        assert "openai" in ids
        assert "groq" in ids
        assert "mistral" in ids

    def test_profile_is_frozen(self) -> None:
        p = get_profile("openai")
        assert p is not None
        with pytest.raises(AttributeError):
            p.name = "hacked"  # type: ignore[misc]

    def test_mistral_profile_has_converter_id(self) -> None:
        p = get_profile("mistral")
        assert p is not None
        assert p.converter_id == "mistral"

    def test_nanogpt_auth(self) -> None:
        p = get_profile("nanogpt")
        assert p is not None
        # Bearer auth since 2026-06-15 (x-api-key caused silent 401s).
        assert p.auth_type == "bearer"
        assert p.auth_header == "Authorization"


# ---------------------------------------------------------------------------
# MistralConverter tests
# ---------------------------------------------------------------------------


class TestMistralConverter:
    """Verify Mistral message conversion."""

    def test_tool_id_hashing(self) -> None:
        """Tool IDs are hashed to 9-char hex via sha512."""
        converter = MistralConverter()
        messages = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "sunny",
            },
        ]
        result = converter.convert_messages(messages)
        out_msgs = result["messages"]

        # Find the assistant message with tool_calls
        assistant_msg = next(m for m in out_msgs if m.get("tool_calls"))
        hashed_id = assistant_msg["tool_calls"][0]["id"]
        assert len(hashed_id) == 9
        assert all(c in "0123456789abcdef" for c in hashed_id)

        # Tool result should have the same hashed ID
        tool_msg = next(m for m in out_msgs if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == hashed_id

    def test_tool_id_is_sha512_based(self) -> None:
        """Verify hashing uses sha512 truncated to 9 chars."""
        import hashlib

        converter = MistralConverter()
        tool_id = "call_abc123"
        expected = hashlib.sha512(tool_id.encode()).hexdigest()[:9]

        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tool_id, "content": "ok"},
        ]
        result = converter.convert_messages(messages)
        assistant_msg = next(m for m in result["messages"] if m.get("tool_calls"))
        assert assistant_msg["tool_calls"][0]["id"] == expected

    def test_prefix_on_last_assistant(self) -> None:
        """When enable_prefix=True, last assistant message gets prefix=True."""
        converter = MistralConverter(enable_prefix=True)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "goodbye"},
        ]
        result = converter.convert_messages(messages)
        out_msgs = result["messages"]
        # Last message should be assistant with prefix
        last_assistant = [m for m in out_msgs if m["role"] == "assistant"][-1]
        assert last_assistant.get("prefix") is True

    def test_prefix_off_by_default(self) -> None:
        """By default, no prefix is added."""
        converter = MistralConverter()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = converter.convert_messages(messages)
        last = result["messages"][-1]
        assert "prefix" not in last

    def test_map_params_seed(self) -> None:
        """seed → random_seed, skip if -1."""
        converter = MistralConverter()
        params = {"seed": 42, "temperature": 0.7}
        mapped = converter.map_params(params)
        assert "random_seed" in mapped
        assert mapped["random_seed"] == 42
        assert "seed" not in mapped

    def test_map_params_seed_minus_one(self) -> None:
        """seed=-1 is skipped entirely."""
        converter = MistralConverter()
        params = {"seed": -1, "temperature": 0.7}
        mapped = converter.map_params(params)
        assert "random_seed" not in mapped
        assert "seed" not in mapped

    def test_prepend_names(self) -> None:
        """Names are prepended to content."""
        converter = MistralConverter()
        messages = [
            {"role": "user", "name": "Alice", "content": "hello"},
        ]
        result = converter.convert_messages(messages)
        assert "Alice: hello" in result["messages"][0]["content"]

    def test_system_after_assistant_becomes_user(self) -> None:
        """System messages after assistant are converted to user role."""
        converter = MistralConverter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "ok"},
        ]
        result = converter.convert_messages(messages)
        roles = [m["role"] for m in result["messages"]]
        # No system messages in the middle
        for i, role in enumerate(roles):
            if i > 0:  # skip leading system
                assert role != "system" or i == 0

    def test_convert_response_passthrough(self) -> None:
        """Response conversion is a passthrough."""
        converter = MistralConverter()
        data = {"choices": [{"message": {"content": "hello"}}]}
        assert converter.convert_response(data) == data


# ---------------------------------------------------------------------------
# CohereConverter tests
# ---------------------------------------------------------------------------


class TestCohereConverter:
    """Verify Cohere message conversion."""

    def test_top_k_to_k(self) -> None:
        """top_k → k mapping."""
        converter = CohereConverter()
        params = {"top_k": 50, "temperature": 0.8}
        mapped = converter.map_params(params)
        assert "k" in mapped
        assert mapped["k"] == 50
        assert "top_k" not in mapped

    def test_top_p_to_p(self) -> None:
        """top_p → p mapping."""
        converter = CohereConverter()
        params = {"top_p": 0.9, "temperature": 0.8}
        mapped = converter.map_params(params)
        assert "p" in mapped
        assert mapped["p"] == 0.9
        assert "top_p" not in mapped

    def test_both_params_mapped(self) -> None:
        """Both top_k and top_p mapped simultaneously."""
        converter = CohereConverter()
        params = {"top_k": 40, "top_p": 0.95}
        mapped = converter.map_params(params)
        assert mapped["k"] == 40
        assert mapped["p"] == 0.95

    def test_prepend_names(self) -> None:
        """Names are prepended to content."""
        converter = CohereConverter()
        messages = [
            {"role": "user", "name": "Bob", "content": "hey"},
        ]
        result = converter.convert_messages(messages)
        assert "Bob: hey" in result["messages"][0]["content"]

    def test_convert_response_passthrough(self) -> None:
        """Response conversion is a passthrough."""
        converter = CohereConverter()
        data = {"choices": [{"message": {"content": "hi"}}]}
        assert converter.convert_response(data) == data

    def test_other_params_preserved(self) -> None:
        """Params not mapped are preserved unchanged."""
        converter = CohereConverter()
        params = {"temperature": 0.5, "max_tokens": 100}
        mapped = converter.map_params(params)
        assert mapped["temperature"] == 0.5
        assert mapped["max_tokens"] == 100
