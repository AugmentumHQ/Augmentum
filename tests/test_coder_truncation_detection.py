"""Tests for D1: truncation-aware streaming assembly in CoderHandler.

When a model emits a tool_call header (name set) but the arguments
JSON never arrives because the response was cut off by max_tokens,
``_stream_and_parse`` must:

1. Flag the assembled entry with ``_truncation_reason``.
2. Carry the provider's ``finish_reason`` for diagnostics.
3. Trigger the dispatcher's short-circuit so the model gets a
   structured "your output was truncated" error instead of the
   misleading bare "missing required arg" downstream error.

Regression target: the empty-args loop documented in
``feedback_no_bandaid_fixes.md``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.modes.analytical.tool_calling import ToolCallingTier
from augmentum.modes.coder.handler import (
    CoderHandler,
    _TRUNCATION_REASON_EMPTY_ARGS,
    _build_truncation_error,
)


@dataclass
class _FakeChunk:
    content_delta: str = ""
    thinking_delta: str = ""
    role: str | None = None
    finish_reason: str | None = None
    usage: Any = None
    model: str = ""
    done: bool = False
    augmentum: dict | None = None


class _FakeBackend:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[_FakeChunk]:
        for c in self._chunks:
            yield c

    async def chat(self, request: InternalChatRequest):
        return None


def _handler(backend: _FakeBackend) -> CoderHandler:
    return CoderHandler(
        backend,
        session_id="ws-test",
        workspace_id="ws-test",
        container_manager=None,
        user_id="alice",
    )


def _req() -> InternalChatRequest:
    return InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="hi")],
        stream=True,
    )


def _empty_args_chunks() -> list[_FakeChunk]:
    """Simulate the failure: model emits file_write name + 2-char
    whitespace content, then a terminal chunk with finish_reason=length
    — but no arguments delta ever arrives."""
    return [
        _FakeChunk(content_delta="\n\n"),
        _FakeChunk(augmentum={
            "tool_calls": [{
                "index": 0,
                "id": "call_abc",
                "function": {"name": "file_write"},
            }],
        }),
        _FakeChunk(done=True, finish_reason="length"),
    ]


def _valid_args_chunks() -> list[_FakeChunk]:
    """Same model behaviour but the arguments JSON streams through
    fully — no truncation."""
    return [
        _FakeChunk(augmentum={
            "tool_calls": [{
                "index": 0,
                "id": "call_ok",
                "function": {"name": "file_write"},
            }],
        }),
        _FakeChunk(augmentum={
            "tool_calls": [{
                "index": 0,
                "function": {
                    "arguments": '{"path":"/workspace/a.txt","content":"hi"}',
                },
            }],
        }),
        _FakeChunk(done=True, finish_reason="stop"),
    ]


def _malformed_args_chunks() -> list[_FakeChunk]:
    """Args arrive but JSON is broken (truncated mid-content)."""
    return [
        _FakeChunk(augmentum={
            "tool_calls": [{
                "index": 0,
                "id": "call_bad",
                "function": {"name": "file_write"},
            }],
        }),
        _FakeChunk(augmentum={
            "tool_calls": [{
                "index": 0,
                "function": {
                    "arguments": '{"path":"/workspace/a.txt","content":"function foo() { ',
                },
            }],
        }),
        _FakeChunk(done=True, finish_reason="length"),
    ]


@pytest.mark.asyncio
async def test_empty_args_with_length_finish_flags_truncation():
    h = _handler(_FakeBackend(_empty_args_chunks()))
    _, tool_calls, error_kind, _, _, _ = await h._stream_and_parse(
        _req(), messages=[], tool_schemas=[], tool_map={"file_write": None},
        tier=ToolCallingTier.NATIVE, iteration=1,
    )
    assert error_kind == ""
    assert len(tool_calls) == 1
    entry = tool_calls[0]
    assert entry["name"] == "file_write"
    assert entry["input"] == {}
    assert entry["_truncation_reason"] == _TRUNCATION_REASON_EMPTY_ARGS
    assert entry["_finish_reason"] == "length"
    # The malformed-JSON marker must NOT be set — these are different
    # failure modes and the dispatcher chooses different error messages.
    assert "_parse_error_raw" not in entry


@pytest.mark.asyncio
async def test_valid_args_does_not_flag_truncation():
    h = _handler(_FakeBackend(_valid_args_chunks()))
    _, tool_calls, error_kind, _, _, _ = await h._stream_and_parse(
        _req(), messages=[], tool_schemas=[], tool_map={"file_write": None},
        tier=ToolCallingTier.NATIVE, iteration=1,
    )
    assert error_kind == ""
    assert len(tool_calls) == 1
    entry = tool_calls[0]
    assert entry["input"] == {"path": "/workspace/a.txt", "content": "hi"}
    assert "_truncation_reason" not in entry
    assert "_parse_error_raw" not in entry


@pytest.mark.asyncio
async def test_malformed_args_flags_parse_error_not_truncation():
    """Malformed-but-present args go through the existing parse_error
    path, not the new truncation path. Important: the model needs
    different recovery guidance for these two cases."""
    h = _handler(_FakeBackend(_malformed_args_chunks()))
    _, tool_calls, _, _, _, _ = await h._stream_and_parse(
        _req(), messages=[], tool_schemas=[], tool_map={"file_write": None},
        tier=ToolCallingTier.NATIVE, iteration=1,
    )
    entry = tool_calls[0]
    assert entry["input"] == {}
    assert "_parse_error_raw" in entry
    assert "_truncation_reason" not in entry


@pytest.mark.asyncio
async def test_dispatcher_short_circuits_truncated_call():
    """The serial dispatcher (``_run_tool_tracked``) must NOT call
    ``_execute_tool_with_verification`` when ``_truncation_reason``
    is set — that would run the tool with empty args and return the
    misleading bare error. Instead it should synthesize a
    truncation-specific ToolResult directly."""
    h = _handler(_FakeBackend([]))
    # Spy on the executor so we can assert it wasn't called.
    h._execute_tool_with_verification = AsyncMock(
        return_value=(MagicMock(success=False, error="X"), None, None),
    )
    # Fabricate a truncated tool_call entry as if assembly produced it.
    tc = {
        "id": "call_z",
        "name": "file_write",
        "input": {},
        "_truncation_reason": _TRUNCATION_REASON_EMPTY_ARGS,
        "_finish_reason": "length",
    }
    # _run_tool_tracked is an async generator that yields meta chunks.
    # We only need to drive it to completion and inspect the recorded
    # tool_result via the meta chunk it emits.
    messages: list = []
    counters: dict = {}
    chunks: list = []
    async for ev in h._run_tool_tracked(
        tc=tc,
        tool_map={"file_write": MagicMock()},
        tier=ToolCallingTier.NATIVE,
        messages=messages,
        model="test-model",
        counters=counters,
    ):
        chunks.append(ev)

    # Critical: the executor must NOT have been called.
    assert h._execute_tool_with_verification.await_count == 0
    # The synthetic tool_result error message must name the truncation
    # mode and point at the recovery path.
    tool_result_extras = [
        ev for ev in chunks
        if isinstance(ev.augmentum, dict)
        and ev.augmentum.get("status") == "tool_result"
    ]
    assert tool_result_extras, "expected a tool_result meta chunk"
    result_payload = tool_result_extras[0].augmentum["tool_result"]
    assert result_payload["success"] is False
    assert "truncated" in (result_payload.get("output_preview") or "").lower()


def test_build_truncation_error_names_tool_and_reason():
    msg = _build_truncation_error(
        tool_name="file_write", finish_reason="length",
    )
    assert "file_write" in msg
    assert "length" in msg
    assert "code_edit" in msg  # points the model at the recovery path


def test_build_truncation_error_handles_blank_finish_reason():
    msg = _build_truncation_error(tool_name="code_edit", finish_reason="")
    assert "code_edit" in msg
    assert "unknown" in msg
