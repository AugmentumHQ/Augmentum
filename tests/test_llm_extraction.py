"""Tests for LLM-based memory extraction pipeline."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.memory.llm_extractor import (
    _SYSTEM_PROMPT,
    _clamp,
    _parse_extraction_response,
    llm_extract,
)
from augmentum.memory.models import MemoryType

# ===========================================================================
# _clamp
# ===========================================================================


class TestClamp:
    def test_clamp_normal(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_string(self):
        assert _clamp("0.7") == 0.7

    def test_clamp_non_numeric(self):
        assert _clamp("hello") == 0.5

    def test_clamp_none(self):
        assert _clamp(None) == 0.5


# ===========================================================================
# _parse_extraction_response
# ===========================================================================


class TestParseExtractionResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "facts": [
                {"content": "User is a Python developer", "type": "fact", "importance": 0.8, "confidence": 0.9},
            ],
        })
        facts = _parse_extraction_response(raw)
        assert len(facts) == 1
        assert facts[0].content == "User is a Python developer"
        assert facts[0].type == MemoryType.FACT
        assert facts[0].importance == 0.8
        assert facts[0].confidence == 0.9
        assert facts[0].is_explicit is False

    def test_markdown_wrapped_json(self):
        raw = '```json\n{"facts": [{"content": "User likes Flask", "type": "preference"}]}\n```'
        facts = _parse_extraction_response(raw)
        assert len(facts) == 1
        assert facts[0].content == "User likes Flask"
        assert facts[0].type == MemoryType.PREFERENCE

    def test_empty_facts(self):
        raw = '{"facts": []}'
        assert _parse_extraction_response(raw) == []

    def test_invalid_json(self):
        assert _parse_extraction_response("not json") == []

    def test_not_a_dict(self):
        assert _parse_extraction_response("[1, 2, 3]") == []

    def test_no_facts_key(self):
        assert _parse_extraction_response('{"other": "data"}') == []

    def test_facts_not_a_list(self):
        assert _parse_extraction_response('{"facts": "string"}') == []

    def test_short_content_skipped(self):
        raw = json.dumps({"facts": [{"content": "hi"}]})
        assert _parse_extraction_response(raw) == []

    def test_invalid_type_defaults_to_fact(self):
        raw = json.dumps({"facts": [{"content": "User works at Google", "type": "invalid"}]})
        facts = _parse_extraction_response(raw)
        assert len(facts) == 1
        assert facts[0].type == MemoryType.FACT

    def test_missing_fields_defaults(self):
        raw = json.dumps({"facts": [{"content": "User is a developer"}]})
        facts = _parse_extraction_response(raw)
        assert len(facts) == 1
        assert facts[0].importance == 0.5
        assert facts[0].confidence == 0.8

    def test_importance_clamped(self):
        raw = json.dumps({"facts": [{"content": "Clamped value", "importance": 5.0, "confidence": -1.0}]})
        facts = _parse_extraction_response(raw)
        assert facts[0].importance == 1.0
        assert facts[0].confidence == 0.0

    def test_multiple_facts(self):
        raw = json.dumps({
            "facts": [
                {"content": "User uses Python", "type": "fact"},
                {"content": "User prefers dark mode", "type": "preference"},
                {"content": "User works at ACME Corp", "type": "entity"},
            ],
        })
        facts = _parse_extraction_response(raw)
        assert len(facts) == 3
        assert facts[0].type == MemoryType.FACT
        assert facts[1].type == MemoryType.PREFERENCE
        assert facts[2].type == MemoryType.ENTITY

    def test_non_dict_items_skipped(self):
        raw = json.dumps({"facts": ["string", 42, {"content": "Valid fact here"}]})
        facts = _parse_extraction_response(raw)
        assert len(facts) == 1
        assert facts[0].content == "Valid fact here"

    def test_source_context_set(self):
        raw = json.dumps({"facts": [{"content": "User likes Python"}]})
        facts = _parse_extraction_response(raw)
        assert facts[0].source_context["extraction"] == "llm"


# ===========================================================================
# llm_extract
# ===========================================================================


class TestLlmExtract:
    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        backend = MagicMock()
        response = MagicMock()
        response.message.content = json.dumps({
            "facts": [{"content": "User is a data scientist", "type": "fact", "importance": 0.8}],
        })
        backend.chat = AsyncMock(return_value=response)

        facts = await llm_extract(
            "I've been doing data science for 5 years",
            "That's great experience!",
            backend, "test-model",
        )
        assert len(facts) == 1
        assert facts[0].content == "User is a data scientist"
        backend.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_backend_error_returns_empty(self):
        backend = MagicMock()
        backend.chat = AsyncMock(side_effect=RuntimeError("connection failed"))

        facts = await llm_extract("test", "test", backend, "model")
        assert facts == []

    @pytest.mark.asyncio
    async def test_mode_passthrough(self):
        backend = MagicMock()
        response = MagicMock()
        response.message.content = '{"facts": []}'
        backend.chat = AsyncMock(return_value=response)

        await llm_extract("hi", "hello", backend, "model", mode="passthrough")
        call = backend.chat.call_args[0][0]
        assert "extract" in call.messages[1].content.lower()
        assert "user" in call.messages[1].content.lower()

    @pytest.mark.asyncio
    async def test_mode_narrative_uses_passthrough_fallback(self):
        """Narrative mode is skipped at the integration layer, but if llm_extract
        is called directly it falls back to the passthrough instruction."""
        backend = MagicMock()
        response = MagicMock()
        response.message.content = '{"facts": []}'
        backend.chat = AsyncMock(return_value=response)

        await llm_extract("hi", "hello", backend, "model", mode="narrative")
        call = backend.chat.call_args[0][0]
        # Falls back to passthrough since narrative has no dedicated instruction
        assert "extract" in call.messages[1].content.lower()

    @pytest.mark.asyncio
    async def test_mode_analytical(self):
        backend = MagicMock()
        response = MagicMock()
        response.message.content = '{"facts": []}'
        backend.chat = AsyncMock(return_value=response)

        await llm_extract("hi", "hello", backend, "model", mode="analytical")
        call = backend.chat.call_args[0][0]
        assert "technical" in call.messages[1].content.lower()

    @pytest.mark.asyncio
    async def test_input_truncation(self):
        backend = MagicMock()
        response = MagicMock()
        response.message.content = '{"facts": []}'
        backend.chat = AsyncMock(return_value=response)

        long_msg = "x" * 5000
        await llm_extract(long_msg, long_msg, backend, "model")
        call = backend.chat.call_args[0][0]
        # The user prompt should contain truncated content
        assert len(call.messages[1].content) < 8500

    @pytest.mark.asyncio
    async def test_request_format(self):
        backend = MagicMock()
        response = MagicMock()
        response.message.content = '{"facts": []}'
        backend.chat = AsyncMock(return_value=response)

        await llm_extract("test", "response", backend, "my-model")
        call = backend.chat.call_args[0][0]
        assert call.model == "my-model"
        assert call.format is None  # Don't force JSON mode — not all backends support it
        assert call.temperature == 0.1
        assert call.max_tokens == 800
        assert call.stream is False

    @pytest.mark.asyncio
    async def test_brace_escaping(self):
        """User content with braces should not cause .format() errors."""
        backend = MagicMock()
        response = MagicMock()
        response.message.content = '{"facts": []}'
        backend.chat = AsyncMock(return_value=response)

        await llm_extract("function() { return {}; }", "ok", backend, "model")
        # Should not raise — braces are escaped

    def test_system_prompt_has_rules(self):
        assert "user" in _SYSTEM_PROMPT.lower()
        assert "facts" in _SYSTEM_PROMPT.lower()
        assert "JSON" in _SYSTEM_PROMPT


# ===========================================================================
# Fallback chain (smart_extract_and_store)
# ===========================================================================


class TestSmartExtractAndStore:
    @pytest.mark.asyncio
    async def test_explicit_only_no_backend(self):
        """With no backend, explicit 'remember' should still work."""
        from augmentum.memory.extractor import smart_extract_and_store

        store = MagicMock()
        store.store_fact = AsyncMock(return_value="mem_1")
        store.recall = AsyncMock(return_value=[])
        store.update_content = AsyncMock(return_value=True)

        import augmentum.memory.extractor as ext_mod
        orig = ext_mod.settings
        ext_mod.settings = MagicMock()
        ext_mod.settings.memory_llm_extraction_enabled = False
        try:
            count = await smart_extract_and_store(
                session_id="s1", user_id="default",
                user_message="Remember that I like dark mode",
                assistant_response="Got it!",
                store=store, backend=None,
            )
        finally:
            ext_mod.settings = orig

        assert count >= 1

    @pytest.mark.asyncio
    async def test_heuristic_fallback_when_no_llm(self):
        """Without LLM, should fallback to heuristic extraction."""
        from augmentum.memory.extractor import smart_extract_and_store

        store = MagicMock()
        store.store_fact = AsyncMock(return_value="mem_1")
        store.recall = AsyncMock(return_value=[])
        store.update_content = AsyncMock(return_value=True)

        import augmentum.memory.extractor as ext_mod
        orig = ext_mod.settings
        ext_mod.settings = MagicMock()
        ext_mod.settings.memory_llm_extraction_enabled = False
        try:
            count = await smart_extract_and_store(
                session_id="s1", user_id="default",
                user_message="I prefer Python over JavaScript",
                assistant_response="Good choice!",
                store=store, backend=None,
            )
        finally:
            ext_mod.settings = orig

        assert count >= 1

    @pytest.mark.asyncio
    async def test_returns_zero_on_no_facts(self):
        from augmentum.memory.extractor import smart_extract_and_store

        store = MagicMock()
        store.store_fact = AsyncMock(return_value="mem_1")
        store.recall = AsyncMock(return_value=[])
        store.update_content = AsyncMock(return_value=True)

        import augmentum.memory.extractor as ext_mod
        orig = ext_mod.settings
        ext_mod.settings = MagicMock()
        ext_mod.settings.memory_llm_extraction_enabled = False
        try:
            count = await smart_extract_and_store(
                session_id="s1", user_id="default",
                user_message="Hello, how are you?",
                assistant_response="I'm good!",
                store=store, backend=None,
            )
        finally:
            ext_mod.settings = orig

        assert count == 0
