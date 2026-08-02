"""Tests for narrative mode routing through proxy endpoints.

Verifies that:
- Character card requests get routed through narrative mode
- Plain requests stay in passthrough mode
- Narrative mode augments the request with injected context
- Both Ollama and OpenAI endpoints support narrative routing
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from augmentum.classifier.router import RequestClassifier
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)
from augmentum.models.provider_registry import ProviderRegistry
from augmentum.proxy.handler_factory import get_handler_for_mode, get_session_id_from_request
from augmentum.proxy.server import create_app
from augmentum.state.backends.memory import MemoryBackend
from augmentum.state.manager import StateManager

# --- Helpers ---


SILLYTAVERN_CARD = (
    "{{char}} is Lyra, a 200-year-old elven sorceress.\n"
    "Personality: Wise, patient, slightly sardonic\n"
    "Appearance: Tall with silver hair that flows like moonlight\n"
    "{{char}} lives in a tower at the edge of the Whispering Forest.\n"
    "{{user}} is a young adventurer seeking {{char}}'s wisdom.\n"
    "{{scenario}}: {{user}} arrives at the tower during a thunderstorm.\n"
    "Stay in character at all times. Use *asterisks* for actions."
)


class SpyBackend(ModelBackend):
    """Mock backend that records the requests it receives."""

    def __init__(self) -> None:
        self.last_chat_request: InternalChatRequest | None = None
        self.last_stream_request: InternalChatRequest | None = None

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        self.last_chat_request = request
        return InternalChatResponse(
            message=Message(role="assistant", content="Hello from mock!"),
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        self.last_stream_request = request
        yield InternalStreamChunk(
            content_delta="Hello",
            role="assistant",
            model=request.model,
            done=False,
        )
        yield InternalStreamChunk(
            content_delta=" from mock!",
            model=request.model,
            done=False,
        )
        yield InternalStreamChunk(
            content_delta="",
            model=request.model,
            done=True,
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                name="llama3.1:8b",
                model="llama3.1:8b",
                size=4_000_000_000,
                digest="abc123",
                modified_at="2024-01-01T00:00:00Z",
                details={"family": "llama", "parameter_size": "8B"},
            ),
        ]

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails(
            modelfile="FROM llama3.1:8b",
            parameters="temperature 0.7",
        )


@pytest.fixture
def spy_backend():
    return SpyBackend()


@pytest.fixture
def narrative_app(spy_backend):
    """Create a test app with a spy backend to inspect request augmentation."""
    from augmentum.config import settings

    # Disable LLM extraction to prevent background tasks overwriting spy_backend state
    original_llm_extraction = settings.narrative_llm_extraction
    object.__setattr__(settings, "narrative_llm_extraction", False)

    # Enable backend state tracking so tests that check context injection pass
    originals = {}
    for attr in ("narrative_state_tracking_enabled", "narrative_consistency_enabled",
                 "narrative_backend_lorebook", "narrative_backend_card_summary"):
        originals[attr] = getattr(settings, attr)
        object.__setattr__(settings, attr, True)

    application = create_app()

    application.state.http_client = MagicMock()
    application.state.provider_registry = MagicMock(spec=ProviderRegistry)
    application.state.provider_registry.get_backend.return_value = spy_backend
    application.state.provider_registry.default_backend = spy_backend
    application.state.provider_registry.resolve_backend_for_model = AsyncMock(return_value=(spy_backend, "llama3.1:8b"))
    application.state.provider_registry.refresh_model_map = AsyncMock(return_value={})
    application.state.provider_registry.backends = {"ollama": spy_backend}
    application.state.state_manager = StateManager(MemoryBackend())
    application.state.classifier = RequestClassifier()
    application.state.narrative_engines = {}

    yield application

    object.__setattr__(settings, "narrative_llm_extraction", original_llm_extraction)
    for attr, val in originals.items():
        object.__setattr__(settings, attr, val)


@pytest.fixture
def narrative_client(narrative_app):
    return TestClient(narrative_app)


# === Unit tests for handler factory ===


class TestHandlerFactory:
    def test_passthrough_for_plain_request(self, narrative_app):
        """Plain request should resolve to PassthroughHandler."""
        from augmentum.classifier.router import Mode
        from augmentum.modes.passthrough.handler import PassthroughHandler

        backend = narrative_app.state.provider_registry.default_backend
        handler = get_handler_for_mode(
            Mode.PASSTHROUGH, backend, "ses_test", narrative_app.state,
        )
        assert isinstance(handler, PassthroughHandler)

    def test_narrative_for_narrative_mode(self, narrative_app):
        """Narrative mode should resolve to NarrativeHandler."""
        from augmentum.classifier.router import Mode
        from augmentum.modes.narrative.handler import NarrativeHandler

        backend = narrative_app.state.provider_registry.default_backend
        handler = get_handler_for_mode(
            Mode.NARRATIVE, backend, "ses_test", narrative_app.state,
        )
        assert isinstance(handler, NarrativeHandler)

    def test_narrative_engine_reuse(self, narrative_app):
        """Same session ID should reuse the same NarrativeEngine."""
        from augmentum.classifier.router import Mode

        backend = narrative_app.state.provider_registry.default_backend
        h1 = get_handler_for_mode(
            Mode.NARRATIVE, backend, "ses_abc", narrative_app.state,
        )
        h2 = get_handler_for_mode(
            Mode.NARRATIVE, backend, "ses_abc", narrative_app.state,
        )
        assert h1._engine is h2._engine  # noqa: SLF001
        assert len(narrative_app.state.narrative_engines) == 1

    def test_different_sessions_get_different_engines(self, narrative_app):
        """Different session IDs should get different NarrativeEngines."""
        from augmentum.classifier.router import Mode

        backend = narrative_app.state.provider_registry.default_backend
        h1 = get_handler_for_mode(
            Mode.NARRATIVE, backend, "ses_111", narrative_app.state,
        )
        h2 = get_handler_for_mode(
            Mode.NARRATIVE, backend, "ses_222", narrative_app.state,
        )
        assert h1._engine is not h2._engine  # noqa: SLF001
        assert len(narrative_app.state.narrative_engines) == 2

    def test_analytical_returns_analytical_handler(self, narrative_app):
        """Analytical mode should return an AnalyticalHandler."""
        from augmentum.classifier.router import Mode
        from augmentum.modes.analytical.handler import AnalyticalHandler

        backend = narrative_app.state.provider_registry.default_backend
        handler = get_handler_for_mode(
            Mode.ANALYTICAL, backend, "ses_test", narrative_app.state,
        )
        assert isinstance(handler, AnalyticalHandler)

    def test_session_id_deterministic(self):
        """Same system prompt should produce the same session ID."""
        req1 = InternalChatRequest(
            model="llama3.1:8b",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ],
        )
        req2 = InternalChatRequest(
            model="llama3.1:8b",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Goodbye"),
            ],
        )
        # Same system prompt → same session ID
        assert get_session_id_from_request(req1) == get_session_id_from_request(req2)

    def test_session_id_differs_for_different_system(self):
        """Different system prompts should produce different session IDs."""
        req1 = InternalChatRequest(
            model="llama3.1:8b",
            messages=[
                Message(role="system", content="You are a pirate."),
                Message(role="user", content="Hello"),
            ],
        )
        req2 = InternalChatRequest(
            model="llama3.1:8b",
            messages=[
                Message(role="system", content="You are a wizard."),
                Message(role="user", content="Hello"),
            ],
        )
        assert get_session_id_from_request(req1) != get_session_id_from_request(req2)


# === Integration tests: Ollama endpoint ===


class TestOllamaNarrativeRouting:
    def test_character_card_routes_through_narrative(self, narrative_client, spy_backend):
        """A request with a character card system prompt should be processed by narrative handler."""
        resp = narrative_client.post(
            "/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": SILLYTAVERN_CARD},
                    {"role": "user", "content": "*waves hello*"},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200

        # The spy backend should have received the request through narrative handler
        augmented = spy_backend.last_chat_request
        assert augmented is not None
        system_msg = next(
            (m.content for m in augmented.messages if m.role == "system"), ""
        )
        # Original character card content should be preserved
        assert "Lyra" in system_msg
        assert "elven sorceress" in system_msg

    def test_plain_request_stays_passthrough(self, narrative_client, spy_backend):
        """A plain request without a character card stays in passthrough."""
        resp = narrative_client.post(
            "/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200

        # Passthrough should send the request unchanged
        sent = spy_backend.last_chat_request
        assert sent is not None
        assert len(sent.messages) == 1
        assert sent.messages[0].content == "What is 2+2?"

    def test_narrative_augments_context(self, narrative_client, spy_backend):
        """Narrative mode should inject character state into system prompt."""
        resp = narrative_client.post(
            "/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": SILLYTAVERN_CARD},
                    {"role": "user", "content": "*enters the tower*"},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200

        augmented = spy_backend.last_chat_request
        system_msg = next(
            (m.content for m in augmented.messages if m.role == "system"), ""
        )
        # The context builder should inject character info
        assert "Lyra" in system_msg
        # Original prompt content should be preserved
        assert "elven sorceress" in system_msg

    def test_mode_prefix_forces_narrative(self, narrative_client, spy_backend):
        """n/ model prefix should force narrative mode even without a character card."""
        resp = narrative_client.post(
            "/api/chat",
            json={
                "model": "n/llama3.1:8b",
                "messages": [
                    {"role": "system", "content": SILLYTAVERN_CARD},
                    {"role": "user", "content": "Hello"},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        augmented = spy_backend.last_chat_request
        system_msg = next(
            (m.content for m in augmented.messages if m.role == "system"), ""
        )
        # Mode prefix stripped and request routed through narrative handler
        assert "Lyra" in system_msg

    def test_header_override_forces_narrative(self, narrative_client, spy_backend):
        """X-Augmentum-Mode: narrative header should force narrative mode."""
        resp = narrative_client.post(
            "/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": SILLYTAVERN_CARD},
                    {"role": "user", "content": "Hello"},
                ],
                "stream": False,
            },
            headers={"X-Augmentum-Mode": "narrative"},
        )
        assert resp.status_code == 200
        augmented = spy_backend.last_chat_request
        system_msg = next(
            (m.content for m in augmented.messages if m.role == "system"), ""
        )
        # Header override routes through narrative handler, content preserved
        assert "Lyra" in system_msg

    def test_narrative_engine_persists_across_requests(self, narrative_client, narrative_app, spy_backend):
        """Same session (same system prompt) should reuse the NarrativeEngine."""
        for _ in range(2):
            narrative_client.post(
                "/api/chat",
                json={
                    "model": "llama3.1:8b",
                    "messages": [
                        {"role": "system", "content": SILLYTAVERN_CARD},
                        {"role": "user", "content": "*waves*"},
                    ],
                    "stream": False,
                },
            )
        # Only one engine should have been created for this session
        assert len(narrative_app.state.narrative_engines) == 1


# === Integration tests: OpenAI endpoint ===


class TestOpenAINarrativeRouting:
    def test_character_card_routes_through_narrative(self, narrative_client, spy_backend):
        """OpenAI endpoint should also route narrative requests correctly."""
        resp = narrative_client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": SILLYTAVERN_CARD},
                    {"role": "user", "content": "*waves hello*"},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200

        augmented = spy_backend.last_chat_request
        system_msg = next(
            (m.content for m in augmented.messages if m.role == "system"), ""
        )
        # Routed through narrative handler, original content preserved
        assert "Lyra" in system_msg
        assert "elven sorceress" in system_msg

    def test_plain_request_stays_passthrough(self, narrative_client, spy_backend):
        """Plain OpenAI requests should remain in passthrough."""
        resp = narrative_client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "user", "content": "Summarize this article."},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200

        sent = spy_backend.last_chat_request
        assert len(sent.messages) == 1
        assert sent.messages[0].content == "Summarize this article."

    def test_narrative_response_format(self, narrative_client):
        """Narrative responses should still conform to OpenAI format."""
        resp = narrative_client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": SILLYTAVERN_CARD},
                    {"role": "user", "content": "*enters*"},
                ],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data


# === Streaming tests ===


class TestNarrativeStreaming:
    def test_ollama_streaming_with_narrative(self, narrative_client, spy_backend):
        """Streaming Ollama chat should work with narrative mode."""
        resp = narrative_client.post(
            "/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": SILLYTAVERN_CARD},
                    {"role": "user", "content": "*waves*"},
                ],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        # Parse NDJSON lines
        lines = [
            json.loads(line)
            for line in resp.text.strip().split("\n")
            if line.strip()
        ]
        assert len(lines) >= 2

        # Last line should be done=True
        assert lines[-1]["done"] is True

        # The backend should have received the request through narrative handler
        augmented = spy_backend.last_stream_request
        system_msg = next(
            (m.content for m in augmented.messages if m.role == "system"), ""
        )
        assert "Lyra" in system_msg

    def test_openai_streaming_with_narrative(self, narrative_client, spy_backend):
        """Streaming OpenAI chat should work with narrative mode."""
        resp = narrative_client.post(
            "/v1/chat/completions",
            json={
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": SILLYTAVERN_CARD},
                    {"role": "user", "content": "*waves*"},
                ],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        # Parse SSE events
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))

        assert len(events) >= 2

        # The backend should have received the request through narrative handler
        augmented = spy_backend.last_stream_request
        system_msg = next(
            (m.content for m in augmented.messages if m.role == "system"), ""
        )
        assert "Lyra" in system_msg
