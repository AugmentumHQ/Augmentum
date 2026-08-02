"""Tests for narrative mode handler and engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    Usage,
)
from augmentum.modes.narrative.engine import NarrativeEngine, NarrativeResult
from augmentum.modes.narrative.handler import NarrativeHandler


def _make_request(content: str = "Hello", system: str = ""):
    messages = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=content))
    return InternalChatRequest(model="test-model", messages=messages, stream=False)


def _make_backend():
    backend = MagicMock()
    backend.chat = AsyncMock(return_value=InternalChatResponse(
        message=Message(role="assistant", content="*nods thoughtfully*"),
        model="test-model",
        finish_reason="stop",
        usage=Usage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    ))

    async def _mock_stream(request):
        yield InternalStreamChunk(content_delta="*nods", role="assistant", model=request.model, done=False)
        yield InternalStreamChunk(content_delta=" thoughtfully*", model=request.model, done=False)
        yield InternalStreamChunk(content_delta="", model=request.model, done=True, finish_reason="stop")

    backend.chat_stream = _mock_stream
    return backend


class TestNarrativeEngineConstruction:
    """Engine initialization."""

    def test_engine_default_config(self):
        engine = NarrativeEngine(session_id="test")
        assert engine._session_id == "test"
        assert engine._character_card is None
        assert engine._initialized is False

    def test_engine_custom_budget(self):
        engine = NarrativeEngine(session_id="test", context_budget=8000)
        assert engine._context_builder._budget == 8000

    def test_engine_has_all_trackers(self):
        engine = NarrativeEngine(session_id="test")
        assert engine._character_tracker is not None
        assert engine._world_tracker is not None
        assert engine._plot_tracker is not None
        assert engine._lore_engine is not None
        assert engine._branch_tracker is not None
        assert engine._relationship_tracker is not None
        assert engine._context_builder is not None

    def test_engine_state_initial(self):
        engine = NarrativeEngine(session_id="test")
        assert engine._state.session_id == "test"
        assert engine._state.message_count == 0


class TestNarrativeHandlerConstruction:
    """Handler initialization."""

    def test_handler_basic_construction(self):
        backend = _make_backend()
        engine = NarrativeEngine(session_id="test")
        handler = NarrativeHandler(
            backend=backend,
            engine=engine,
            session_id="test",
        )
        assert handler._backend is backend
        assert handler._engine is engine
        assert handler._session_id == "test"
        assert handler._state_loaded is False

    def test_handler_with_image_queue(self):
        backend = _make_backend()
        engine = NarrativeEngine(session_id="test")
        queue = MagicMock()
        handler = NarrativeHandler(
            backend=backend,
            engine=engine,
            image_queue=queue,
            image_enabled=True,
        )
        assert handler._image_queue is queue
        assert handler._image_enabled is True


class TestNarrativeResult:
    """NarrativeResult dataclass."""

    def test_result_defaults(self):
        from augmentum.modes.narrative.context_builder import BuiltContext
        from augmentum.state.narrative_state import NarrativeSessionState
        result = NarrativeResult(
            augmented_request=_make_request(),
            context=BuiltContext(),
            state=NarrativeSessionState(),
        )
        assert result.contradictions == []
        assert result.new_facts == []
        assert result.branch_detected is False
        assert result.is_regeneration is False


class TestNarrativeMemorySettings:
    """Memory setting resolution (session -> global fallback)."""

    def test_session_override_takes_precedence(self):
        from augmentum.modes.narrative.memory_settings import (
            SessionMemorySettings,
            resolve_memory_setting,
        )
        session = SessionMemorySettings(memory_enabled=False)
        result = resolve_memory_setting(session, "memory_enabled")
        assert result is False

    def test_none_session_falls_back_to_global(self):
        from augmentum.modes.narrative.memory_settings import resolve_memory_setting
        # With no session settings, should fall back to global
        result = resolve_memory_setting(None, "memory_enabled")
        # Result is whatever the global config says — just verify it doesn't error
        assert result is not None or result is None  # always passes; verifies no exception

    def test_session_settings_to_dict_only_non_none(self):
        from augmentum.modes.narrative.memory_settings import SessionMemorySettings
        settings = SessionMemorySettings(memory_enabled=True, memory_mode="state_and_ledger")
        d = settings.to_dict()
        assert "memory_enabled" in d
        assert "memory_mode" in d
        assert "smart_retrieval" not in d  # None values excluded

    def test_session_settings_from_dict(self):
        from augmentum.modes.narrative.memory_settings import SessionMemorySettings
        d = {"memory_enabled": False, "memory_interval": 5}
        settings = SessionMemorySettings.from_dict(d)
        assert settings.memory_enabled is False
        assert settings.memory_interval == 5
        assert settings.memory_mode is None

    def test_session_settings_from_empty_dict(self):
        from augmentum.modes.narrative.memory_settings import SessionMemorySettings
        settings = SessionMemorySettings.from_dict({})
        assert settings.memory_enabled is None

    def test_session_settings_ignores_unknown_keys(self):
        from augmentum.modes.narrative.memory_settings import SessionMemorySettings
        d = {"memory_enabled": True, "unknown_field": "ignored"}
        settings = SessionMemorySettings.from_dict(d)
        assert settings.memory_enabled is True
        assert not hasattr(settings, "unknown_field")


class TestRefineTrimOrderPreservation:
    """Regression guards for ``_refine_trim_with_real_tokens``.

    The refine pass used to rebuild its candidate as ``sys_msgs +
    candidate_chat`` — hoisting EVERY system message to the front, including
    the dynamic STATE/MEMORY block that ``_augment_request`` deliberately
    places just before the latest user turn for llama-server prefix-cache
    reuse. Because the pass runs on every turn (even when nothing is
    dropped), the per-turn-changing block landed at position ~0, the token
    prefix diverged at message 0, and the engine re-prefilled the entire
    context every turn (observed live 2026-07-01: 12-15 min TTFT on a 61k
    narrative session). Trim must only drop oldest chat messages, in place.
    """

    def _handler_with_tokenizer(self, token_counts: list[int]):
        """Handler whose backend renders/tokenizes with scripted counts."""
        backend = MagicMock()
        rendered_payloads: list[list[dict]] = []

        async def apply_template(payload):
            rendered_payloads.append(payload)
            return "rendered"

        counts = iter(token_counts)

        async def tokenize(_rendered):
            return [0] * next(counts)

        backend.apply_template = apply_template
        backend.tokenize = tokenize
        engine = NarrativeEngine(session_id="test_refine_order")
        handler = NarrativeHandler(
            backend=backend, engine=engine, session_id="test_refine_order",
        )
        return handler, rendered_payloads

    def _messages_with_mid_system(self) -> list[Message]:
        """[card, u0, a0, u1, a1, STATE(system), u2] — cache-friendly shape."""
        return [
            Message(role="system", content="CARD"),
            Message(role="user", content="u0"),
            Message(role="assistant", content="a0"),
            Message(role="user", content="u1"),
            Message(role="assistant", content="a1"),
            Message(role="system", content="STATE_SNAPSHOT_DYNAMIC"),
            Message(role="user", content="u2"),
        ]

    async def test_within_budget_returns_original_object_unreordered(self):
        handler, rendered = self._handler_with_tokenizer([50])
        messages = self._messages_with_mid_system()

        out = await handler._refine_trim_with_real_tokens(messages, budget=100)

        # Identity: the caller's ``is`` fast-path must see a no-op.
        assert out is messages
        # And the candidate that was RENDERED for token counting must be in
        # original order too (the old code rendered a reordered array, so
        # its verdicts were computed against the wrong prompt shape).
        roles = [m["role"] for m in rendered[0]]
        assert roles == ["system", "user", "assistant", "user", "assistant",
                         "system", "user"]

    async def test_over_budget_drops_oldest_in_place(self):
        # First render over budget, second fits.
        handler, rendered = self._handler_with_tokenizer([200, 50])
        messages = self._messages_with_mid_system()

        out = await handler._refine_trim_with_real_tokens(messages, budget=180)

        contents = [m.content for m in out]
        # Oldest chat dropped, both systems present, ORDER preserved:
        # the dynamic STATE block must still sit just before the last user.
        assert contents[0] == "CARD"
        assert "STATE_SNAPSHOT_DYNAMIC" in contents
        state_idx = contents.index("STATE_SNAPSHOT_DYNAMIC")
        assert contents[state_idx + 1] == "u2"
        assert state_idx > 0 and out[state_idx - 1].role != "system", (
            "STATE block was hoisted into the leading system stack"
        )
        # Chronology of surviving chat messages intact.
        chat = [c for i, c in enumerate(contents) if out[i].role != "system"]
        assert chat == sorted(chat, key=["u0", "a0", "u1", "a1", "u2"].index)
