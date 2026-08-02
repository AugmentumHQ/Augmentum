"""Reflexion-style self-critique on streak-break (PR3).

When _act_hybrid hits a *_break detector, the loop emits the
deterministic ``[Stopped: ...]`` message AND then calls
``_reflect_on_streak_break`` to stream a model-generated critique.
The reflection chunks land with ``status="reflection"`` so the UI can
style them differently. Disabled via ``coder_reflexion_on_break=False``
to skip the extra LLM call.

Reflexion (arxiv:2303.11366) — generalises every handcrafted
*_streak nudge into a model-generated diagnosis of what went wrong.
"""
from __future__ import annotations

import pytest

from augmentum.config import settings as _global_settings
from augmentum.models.base import (
    InternalChatRequest,
    InternalStreamChunk,
)
from augmentum.modes.coder.handler import CoderHandler
from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeChunk,
    _force_native_tier,
    _make_request,
    _tc_delta,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ReflectingBackend:
    """Backend that emits malformed tool calls every iteration so the
    validation_error_streak detector trips after 5 iters; on the
    reflection request (no tools), emits a 3-line critique."""

    def __init__(self) -> None:
        self.iter_calls = 0
        self.reflect_calls = 0

    async def chat_stream(self, request: InternalChatRequest):
        # Reflection request has no `tools` field set.
        if not request.tools:
            self.reflect_calls += 1
            yield _FakeChunk(content_delta="1. I assumed the file existed.")
            yield _FakeChunk(content_delta=" 2. The ENOENT error said it didn't.")
            yield _FakeChunk(
                content_delta=" 3. I'd run file_list first.",
            )
            yield _FakeChunk(done=True, finish_reason="stop")
            return

        self.iter_calls += 1
        # Always a malformed tool call (missing required args) so the
        # validation_error tracker fires every iteration.
        yield _FakeChunk(augmentum={"tool_calls": [
            _tc_delta(0, f"tc-{self.iter_calls}", "code_edit", {}),
        ]})
        yield _FakeChunk(done=True, finish_reason="tool_calls")

    async def chat(self, request):
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflection_streams_on_validation_error_break(monkeypatch):
    """When validation_error_streak fires, the reflection helper runs
    and streams chunks with status=reflection."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(_global_settings, "coder_reflexion_on_break", True)
    # Real CodeEditTool needed so validation_error counter actually
    # ticks (FakeTool doesn't validate args).
    from augmentum.coder.tools import CodeEditTool

    def _make_tools(cm, ws, state, **_):
        return [CodeEditTool(
            container_manager=cm, workspace_id=ws, state=state,
        )]

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools", _make_tools,
    )

    backend = _ReflectingBackend()
    handler = CoderHandler(
        backend, session_id="sess-rfx",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-rfx",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("edit a file"), workspace_context="",
    ):
        chunks.append(c)

    # The validation_error_break must have fired.
    break_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "validation_error_break"
    ]
    assert break_chunks, "expected validation_error_break to fire"

    # Reflection chunks must have followed it.
    reflection_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "reflection"
    ]
    assert reflection_chunks, (
        "expected reflection chunks after the break"
    )
    # Concatenated reflection content includes the model's 3 numbered points.
    rfx_text = "".join(c.content_delta for c in reflection_chunks)
    assert "1." in rfx_text
    assert "2." in rfx_text
    assert "3." in rfx_text
    assert backend.reflect_calls == 1


@pytest.mark.asyncio
async def test_reflection_skipped_when_disabled(monkeypatch):
    """coder_reflexion_on_break=False → no extra backend call, no
    reflection chunks. The terse [Stopped: ...] message remains."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(_global_settings, "coder_reflexion_on_break", False)
    from augmentum.coder.tools import CodeEditTool

    def _make_tools(cm, ws, state, **_):
        return [CodeEditTool(
            container_manager=cm, workspace_id=ws, state=state,
        )]

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools", _make_tools,
    )

    backend = _ReflectingBackend()
    handler = CoderHandler(
        backend, session_id="sess-noref",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-noref",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("edit a file"), workspace_context="",
    ):
        chunks.append(c)

    reflection_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "reflection"
    ]
    assert not reflection_chunks
    assert backend.reflect_calls == 0


@pytest.mark.asyncio
async def test_reflection_swallows_backend_errors(monkeypatch):
    """A reflection backend error must not propagate — the [Stopped]
    line is the only contractual output."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(_global_settings, "coder_reflexion_on_break", True)
    from augmentum.coder.tools import CodeEditTool

    def _make_tools(cm, ws, state, **_):
        return [CodeEditTool(
            container_manager=cm, workspace_id=ws, state=state,
        )]

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools", _make_tools,
    )

    class _BrokenReflection(_ReflectingBackend):
        async def chat_stream(self, request):
            if not request.tools:
                raise RuntimeError("reflection backend down")
            async for c in super().chat_stream(request):
                yield c

    backend = _BrokenReflection()
    handler = CoderHandler(
        backend, session_id="sess-broke",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-broke",
    )

    chunks: list[InternalStreamChunk] = []
    # Must complete without raising.
    async for c in handler._act_hybrid(
        _make_request("edit a file"), workspace_context="",
    ):
        chunks.append(c)

    break_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "validation_error_break"
    ]
    assert break_chunks
    # No reflection chunks because the backend errored out.
    reflection_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "reflection"
    ]
    assert not reflection_chunks
