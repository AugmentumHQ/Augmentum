"""LLM bridge tests against a stubbed ProviderRegistry.

Verifies that:
* the bridge resolves a backend via the registry
* prompts are forwarded verbatim as a single user-role Message
* a frame is attached as a base64 image when the resolved model is
  vision-capable, and dropped otherwise
* the assistant content is returned unchanged for the agent to parse

The real ProviderRegistry constructs real backends; the tests instead
build a duck-typed stub with just the surface the bridge touches.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import pytest

from augmentum.game_agent.llm_bridge import make_game_agent_llm


@dataclass
class _StubMessage:
    role: str
    content: str
    images: list[str] | None = None


@dataclass
class _StubResponse:
    message: _StubMessage


class _StubBackend:
    def __init__(self) -> None:
        self.last_request: Any = None

    async def chat(self, request: Any) -> _StubResponse:
        self.last_request = request
        return _StubResponse(_StubMessage(role="assistant", content="{}"))


class _StubRegistry:
    def __init__(self, *, model_name: str) -> None:
        self._model_name = model_name
        self.backend = _StubBackend()

    async def resolve_backend_for_model(
        self, _model_name: str
    ) -> tuple[_StubBackend, str]:
        return (self.backend, self._model_name)

    async def resolve_backend_with_fabric(
        self, model_name: str
    ) -> tuple[_StubBackend, str]:
        # Production (llm_bridge._call) calls the fabric-aware resolver;
        # delegate to the plain one so instrumented subclasses that
        # override resolve_backend_for_model still capture the request.
        return await self.resolve_backend_for_model(model_name)

    async def resolve_model_for_role(
        self, role: str, override: str = "", settings: Any = None,
    ) -> tuple[_StubBackend, str]:
        # Both bridge lanes now resolve through the ROLE chain rather than
        # through an empty model name — an empty name bottomed out at "first
        # model on the default backend", which is a silent auto-pick. The
        # pin arrives as ``override`` and still wins, which is what the
        # pinned-model tests below assert.
        self.last_role = role
        return await self.resolve_backend_with_fabric(override)


@pytest.mark.asyncio
async def test_bridge_forwards_prompt_as_user_message() -> None:
    """@example: prompt text arrives unchanged in a single user message."""

    registry = _StubRegistry(model_name="text-only-1b")
    llm = make_game_agent_llm(registry)  # type: ignore[arg-type]

    reply = await llm("hello, model", [])
    assert reply == "{}"

    req = registry.backend.last_request
    assert req is not None
    assert req.model == "text-only-1b"
    assert len(req.messages) == 1
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "hello, model"
    assert req.messages[0].images is None


@pytest.mark.asyncio
async def test_bridge_attaches_frame_only_for_vision_capable_model() -> None:
    """@example: a frame is base64-encoded for vision models, dropped otherwise.

    ROOT CAUSE:
      Sending images to a non-vision local model raises at the
      backend layer with an unhelpful error. Dropping at the bridge
      keeps the slow path running on log entries alone, which is the
      designed-for fallback.
    """

    frame = b"PNG-bytes-pretend"
    # Bridge wraps each frame as a full data-URL so llama-server's
    # image_url validator accepts them (bare base64 → 500 "Invalid url
    # value"). Multi-frame: each frame becomes its own data URL in the
    # images list, oldest first.
    expected_b64 = base64.b64encode(frame).decode("ascii")
    expected_data_url = f"data:image/png;base64,{expected_b64}"

    vision_reg = _StubRegistry(model_name="gemini-2.5-pro-vision")
    vision_llm = make_game_agent_llm(vision_reg)  # type: ignore[arg-type]
    await vision_llm("look at this", [frame])
    assert vision_reg.backend.last_request.messages[0].images == [expected_data_url]

    text_reg = _StubRegistry(model_name="qwen-2.5-7b")
    text_llm = make_game_agent_llm(text_reg)  # type: ignore[arg-type]
    await text_llm("look at this", [frame])
    assert text_reg.backend.last_request.messages[0].images is None


@pytest.mark.asyncio
async def test_bridge_respects_pinned_model() -> None:
    """@example: pinned_model overrides the registry default."""

    registry = _StubRegistry(model_name="default-1b")
    llm = make_game_agent_llm(registry, pinned_model="my-pinned-model")  # type: ignore[arg-type]

    await llm("hi", [])
    # The stub registry ignores the requested name and returns its
    # configured one, so we instead assert the bridge passed the
    # pinned name through:
    #
    # We need to make _StubRegistry capture the requested name.
    # Re-run with an instrumented stub.

    class _CapturingStub(_StubRegistry):
        def __init__(self) -> None:
            super().__init__(model_name="default-1b")
            self.requested: str | None = None

        async def resolve_backend_for_model(self, name: str) -> tuple[_StubBackend, str]:
            self.requested = name
            return (self.backend, self._model_name)

    cap = _CapturingStub()
    cap_llm = make_game_agent_llm(cap, pinned_model="my-pinned-model")  # type: ignore[arg-type]
    await cap_llm("hi", [])
    assert cap.requested == "my-pinned-model"
