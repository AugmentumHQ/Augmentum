"""Regression tests for external-harness (pi / OpenCode / Cursor / Aider) agentic
tool loops over /v1/chat/completions.

Two bugs, same class — a standard OpenAI client that drives a multi-turn tool
loop was silently broken while plain-text chat worked:

1. Streaming tool calls were emitted only on the proprietary ``augmentum``
   sidecar, never in the spec ``choices[].delta.tool_calls`` — so the harness
   saw ``finish_reason=tool_calls`` with an empty delta and did nothing.
2. The follow-up request 400'd: the assistant tool-call turn sends
   ``content: null`` (OpenAI spec) and Augmentum's inbound schema rejected None;
   and ``tool_calls`` / ``tool_call_id`` were dropped on the way in, severing the
   loop even when content parsed.
"""

from __future__ import annotations

from augmentum.models.base import InternalStreamChunk
from augmentum.proxy.openai_routes import (
    OpenAIChatRequest,
    _parse_openai_content,
    to_internal_chat_request,
)
from augmentum.proxy.streaming import _chunk_to_openai_sse


def test_sse_promotes_sidecar_tool_calls_into_delta():
    """Bug 1: tool_calls on the augmentum sidecar must also appear in the
    standard delta so OpenAI clients can parse them."""
    tc = [{"index": 0, "id": "call_1", "type": "function",
           "function": {"name": "read", "arguments": ""}}]
    chunk = InternalStreamChunk(content_delta="", augmentum={"tool_calls": tc})
    sse = _chunk_to_openai_sse(chunk, "chatcmpl-x")
    assert sse["choices"][0]["delta"].get("tool_calls") == tc
    # sidecar copy preserved for the web UI
    assert sse["augmentum"]["tool_calls"] == tc


def test_sse_text_chunk_has_no_tool_calls_key():
    """A plain text delta must not grow an empty tool_calls key."""
    chunk = InternalStreamChunk(content_delta="hello")
    sse = _chunk_to_openai_sse(chunk, "chatcmpl-x")
    assert "tool_calls" not in sse["choices"][0]["delta"]
    assert sse["choices"][0]["delta"]["content"] == "hello"


def test_inbound_assistant_tool_call_turn_with_null_content():
    """Bug 2: assistant turn with content=null + tool_calls, then a tool result
    with tool_call_id, must parse and forward the tool fields."""
    req = OpenAIChatRequest(
        model="d/deepseek-v4-flash",
        messages=[
            {"role": "user", "content": "read marker.txt"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "read", "arguments": '{"path":"marker.txt"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "SECRET_TOKEN=BANANA42"},
        ],
    )
    internal = to_internal_chat_request(req)
    assistant = internal.messages[1]
    tool = internal.messages[2]
    assert assistant.content == ""  # null coerced, not a crash
    assert assistant.tool_calls and assistant.tool_calls[0]["id"] == "call_1"
    assert tool.tool_call_id == "call_1"
    assert tool.content == "SECRET_TOKEN=BANANA42"


def test_parse_openai_content_handles_none():
    assert _parse_openai_content(None) == ("", None)
    assert _parse_openai_content("hi") == ("hi", None)
