"""Tests for the Direct mode tier — raw pass-through to backend.

Direct mode is the external-API "no Augmentum injection" tier. The
explicit contract is: what the caller sends is what the model sees.
These tests pin every layer that contract depends on — the classifier
recognising the prefix/header, the handler not injecting anything, and
the factory routing the mode correctly.

Spec: see augmentum/modes/direct/handler.py module docstring.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from augmentum.classifier.router import (
    MODE_MAP,
    MODE_PREFIXES,
    Mode,
    RequestClassifier,
)
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
from augmentum.modes.direct.handler import DirectHandler
from augmentum.proxy.handler_factory import get_handler_for_mode

# ----------------------------------------------------------------------
# Recording backend — captures the messages it receives so tests can
# verify nothing was injected before dispatch.
# ----------------------------------------------------------------------


class _RecordingBackend(ModelBackend):
    """Backend that records every request it handles and echoes content."""

    def __init__(self) -> None:
        self.received_requests: list[InternalChatRequest] = []

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        self.received_requests.append(request)
        return InternalChatResponse(
            message=Message(role="assistant", content="echo"),
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        self.received_requests.append(request)
        yield InternalStreamChunk(
            content_delta="echo", role="assistant",
            model=request.model, done=False,
        )
        yield InternalStreamChunk(
            content_delta="", model=request.model, done=True,
            finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails(modelfile="", parameters="", template="", details={})


def _req(model: str = "test", *, messages: list[Message] | None = None) -> InternalChatRequest:
    if messages is None:
        messages = [Message(role="user", content="hi")]
    return InternalChatRequest(model=model, messages=messages)


# ----------------------------------------------------------------------
# Classifier registration
# ----------------------------------------------------------------------


def test_mode_direct_string_value():
    assert Mode.DIRECT.value == "direct"


def test_mode_map_includes_direct():
    assert MODE_MAP["direct"] == Mode.DIRECT


def test_mode_prefixes_includes_direct():
    assert MODE_PREFIXES["d/"] == Mode.DIRECT


def test_direct_does_not_collide_with_becca_direct():
    # Distinct values keep the two from being aliased — guards against
    # someone changing one without thinking about the other.
    assert Mode.DIRECT.value != Mode.BECCA_DIRECT.value
    assert MODE_MAP["direct"] is not MODE_MAP["becca_direct"]


def test_classifier_recognises_d_prefix():
    classifier = RequestClassifier()
    result = classifier.classify(_req(model="d/qwen3-coder"))
    assert result.mode == Mode.DIRECT
    assert result.confidence == 1.0


def test_classifier_recognises_direct_header_override():
    classifier = RequestClassifier()
    result = classifier.classify(_req(), mode_override="direct")
    assert result.mode == Mode.DIRECT


def test_classifier_never_returns_direct_from_heuristics():
    # Build a request rich in narrative + complexity signals — none of
    # them should produce DIRECT, because DIRECT is explicit-only.
    classifier = RequestClassifier()
    messages = [
        Message(role="system",
                content="You are Aria, a witty alchemist in a fantasy tavern."),
        Message(role="user",
                content="Walk me through synthesising the elixir step by step."),
    ]
    result = classifier.classify(_req(messages=messages))
    assert result.mode != Mode.DIRECT


# ----------------------------------------------------------------------
# Handler contract — nothing gets injected
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_handler_does_not_inject_datetime_non_stream():
    backend = _RecordingBackend()
    handler = DirectHandler(backend=backend)
    req = _req(messages=[Message(role="user", content="hi")])

    await handler.handle(req)

    received = backend.received_requests[0]
    assert len(received.messages) == 1
    assert received.messages[0].role == "user"
    assert received.messages[0].content == "hi"
    # No system message added.
    assert not any(m.role == "system" for m in received.messages)


@pytest.mark.asyncio
async def test_direct_handler_does_not_inject_datetime_stream():
    backend = _RecordingBackend()
    handler = DirectHandler(backend=backend)
    req = _req(messages=[Message(role="user", content="hi")])

    chunks = []
    async for chunk in handler.handle_stream(req):
        chunks.append(chunk)

    received = backend.received_requests[0]
    assert not any(m.role == "system" for m in received.messages)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_direct_handler_preserves_user_supplied_system_message():
    # If the caller DOES send a system message, it must pass through
    # untouched — direct callers are pinning their own prompt.
    backend = _RecordingBackend()
    handler = DirectHandler(backend=backend)
    custom = "You are a strict JSON-only assistant."
    req = _req(messages=[
        Message(role="system", content=custom),
        Message(role="user", content="hi"),
    ])

    await handler.handle(req)

    received = backend.received_requests[0]
    assert len(received.messages) == 2
    assert received.messages[0].role == "system"
    assert received.messages[0].content == custom


# ----------------------------------------------------------------------
# Handler factory dispatches DIRECT correctly
# ----------------------------------------------------------------------


def test_handler_factory_returns_direct_handler():
    backend = _RecordingBackend()
    # Minimal app_state — DirectHandler doesn't read from it, so a stub
    # with no attributes suffices. The factory branch must catch the
    # mode and return DirectHandler before touching app_state.
    class _StubAppState:
        pass
    app_state = _StubAppState()

    handler = get_handler_for_mode(
        Mode.DIRECT, backend, session_id="s1", app_state=app_state,
    )
    assert isinstance(handler, DirectHandler)
