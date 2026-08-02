"""Agentic eval harness — scripted backend + mock tools for flow-level regression.

Each case YAML declares scripted LLM responses (ordered) and property
assertions against the final delivered text + artifact metadata. This lets
prompt/flow changes be regression-tested without hitting a live model.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from augmentum.models.base import (
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
    ModelInfo,
    Usage,
)


@dataclass
class ScriptedResponse:
    """A single pre-recorded LLM response.

    ``content`` is the prose body. ``tool_calls`` may carry structured calls
    (OpenAI format: ``[{"function": {"name": ..., "arguments": "..."}}]``).
    """

    content: str = ""
    tool_calls: list[dict] | None = None


@dataclass
class ScriptedBackend(ModelBackend):
    """Returns a deterministic, ordered sequence of scripted responses.

    Tests push responses via ``queue(...)`` or construct with a list. The
    backend raises if the script is exhausted — that's a signal the flow
    asked for more LLM turns than the case anticipated.
    """

    responses: list[ScriptedResponse] = field(default_factory=list)
    _calls_made: int = 0

    async def chat(self, request) -> InternalChatResponse:
        if self._calls_made >= len(self.responses):
            raise AssertionError(
                f"ScriptedBackend exhausted at call #{self._calls_made + 1}; "
                "add more scripted responses to the case."
            )
        resp = self.responses[self._calls_made]
        self._calls_made += 1
        return InternalChatResponse(
            message=Message(
                role="assistant",
                content=resp.content,
                tool_calls=resp.tool_calls,
            ),
            model=request.model,
            finish_reason="stop",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def chat_stream(self, request) -> AsyncIterator[InternalStreamChunk]:
        # Streaming is emulated by yielding the full content as one chunk.
        resp_msg = (await self.chat(request)).message
        yield InternalStreamChunk(
            content_delta=resp_msg.content,
            role="assistant",
            model=request.model,
            done=True,
            finish_reason="stop",
        )

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails(modelfile="", parameters="", template="", details={})


@dataclass
class MockToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockTool:
    """Minimal tool surrogate for eval cases.

    Canned ``output`` and ``metadata`` are returned from ``execute()``. Tools
    that should fail set ``success=False``.
    """

    name: str
    description: str = "mock tool"
    input_schema: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    timeout: float = 5.0
    output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    model_hint: str = ""

    def health_check(self) -> bool:
        return True

    def enrich_error(self, err: str, args: dict) -> str:
        return err

    async def execute(self, **kwargs) -> MockToolResult:
        return MockToolResult(
            success=self.success,
            output=self.output,
            error="" if self.success else "mock failure",
            metadata=dict(self.metadata),
        )


class MockToolRegistry:
    def __init__(self, tools: list[MockTool]) -> None:
        self._tools = {t.name: t for t in tools}

    def resolve(self, name: str) -> MockTool | None:
        return self._tools.get(name)

    def all_tools(self) -> list[MockTool]:
        return list(self._tools.values())

    def list_tools(self, category=None) -> list[MockTool]:
        # Tests don't model categories; return everything.
        return list(self._tools.values())


def _eval_minimal_deliver_flow():
    """Minimal two-step flow used by eval harness tests.

    Plan (generative) -> Deliver (generative, streamed). No chain/tool
    steps — exercises just the generative path + delivery composition.
    """
    from augmentum.reasoning.models import FlowStep, ReasoningFlow
    from augmentum.reasoning.templates import _DELIVER_SYSTEM_BASE, _DELIVER_USER_TEMPLATE
    import uuid

    def _sid() -> str:
        return uuid.uuid4().hex[:16]

    steps = [
        FlowStep(
            id=_sid(),
            sort_order=1,
            name="Plan",
            system_prompt="Write a 2-line plan.",
            user_template="{query}",
            role="plan",
            tool_names=[],
            stream_to_user=False,
            output_cap=400,
        ),
        FlowStep(
            id=_sid(),
            sort_order=2,
            name="Deliver",
            system_prompt=_DELIVER_SYSTEM_BASE,
            user_template=_DELIVER_USER_TEMPLATE,
            role="deliver",
            tool_names=[],
            stream_to_user=True,
            output_cap=600,
        ),
    ]
    return ReasoningFlow(
        id=_sid(),
        name="EvalMinimalDeliver",
        description="Eval harness test flow",
        steps=steps,
    )


@pytest.fixture
def scripted_backend() -> ScriptedBackend:
    return ScriptedBackend()


@pytest.fixture
def mock_tool_registry() -> MockToolRegistry:
    return MockToolRegistry([])
