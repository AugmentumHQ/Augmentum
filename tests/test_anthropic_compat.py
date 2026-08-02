"""Anthropic Messages API compat layer — translation tests.

Drives `augmentum.models.anthropic_compat` via TDD. The translator
must produce SSE event sequences that satisfy the same invariants the
Anthropic Python SDK's `MessageStream` enforces internally:

- streams MUST start with `message_start`
- `content_block_start` precedes any delta for its index
- `text_delta` only on `text` blocks; `input_json_delta` only on `tool_use`
- block indices monotonic from 0
- `content_block_stop` closes the block at its index
- `message_delta` after all content blocks closed
- `message_stop` is final

Five scenarios from the proxy-landscape research (each one breaks 90% of
competing proxies): text-only, tool-only, text→tool, tool→text→tool,
reasoning→tool. Plus request-side translation and count_tokens.
"""

from __future__ import annotations

import json

import pytest

from augmentum.models.anthropic_compat import (
    AnthropicMessagesRequest,
    anthropic_request_to_openai,
    compute_prefix_cache_key,
    count_tokens_estimate,
    internal_response_to_anthropic_message,
    stream_internal_response_as_anthropic_sse,
)
from augmentum.models.base import (
    InternalChatResponse,
    Message,
    Usage,
)

# ─── SSE event validator (mirrors Anthropic SDK invariants) ──────────


def parse_sse_events(raw: bytes) -> list[dict]:
    """Parse SSE event/data pairs into `{event, data}` records.

    Tolerates ping events (data-only or event-only). Each `data:` line
    is JSON-parsed; missing event lines default to the type field on
    the data payload (Anthropic style).
    """
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    blocks = [b for b in text.split("\n\n") if b.strip()]
    out: list[dict] = []
    for block in blocks:
        event = None
        data = None
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
        if data is not None:
            out.append({"event": event or data.get("type"), "data": data})
    return out


def assert_valid_anthropic_event_sequence(events: list[dict]) -> None:
    """Validate that an event sequence satisfies the SDK invariants.

    Encodes the rules the anthropic-sdk-python MessageStream checks.
    Raises AssertionError on any violation, naming the rule that broke.
    """
    assert events, "empty event stream"
    # Allow leading ping events but the first non-ping MUST be message_start.
    non_ping = [e for e in events if e["event"] != "ping"]
    assert non_ping, "no non-ping events"
    assert non_ping[0]["event"] == "message_start", (
        f"first non-ping event must be message_start, got {non_ping[0]['event']!r}"
    )
    assert non_ping[-1]["event"] == "message_stop", (
        f"last event must be message_stop, got {non_ping[-1]['event']!r}"
    )

    open_blocks: dict[int, str] = {}  # index → type
    next_expected_index = 0
    saw_message_delta = False

    for ev in non_ping:
        et = ev["event"]
        d = ev["data"]
        if et == "message_start":
            assert d.get("type") == "message_start"
            assert "message" in d
            msg = d["message"]
            assert msg.get("role") == "assistant"
            assert isinstance(msg.get("content"), list)
        elif et == "content_block_start":
            idx = d["index"]
            assert idx == next_expected_index, (
                f"content_block_start index {idx} not next-monotonic "
                f"(expected {next_expected_index})"
            )
            assert idx not in open_blocks, f"block {idx} already open"
            block = d["content_block"]
            btype = block["type"]
            assert btype in ("text", "tool_use", "thinking"), (
                f"unknown block type {btype!r}"
            )
            open_blocks[idx] = btype
            next_expected_index += 1
        elif et == "content_block_delta":
            idx = d["index"]
            assert idx in open_blocks, (
                f"content_block_delta on closed/never-opened index {idx}"
            )
            btype = open_blocks[idx]
            delta_type = d["delta"]["type"]
            if btype == "text":
                assert delta_type == "text_delta", (
                    f"text_delta required on text block (got {delta_type})"
                )
            elif btype == "tool_use":
                assert delta_type == "input_json_delta", (
                    f"input_json_delta required on tool_use block (got {delta_type})"
                )
            elif btype == "thinking":
                assert delta_type == "thinking_delta", (
                    f"thinking_delta required on thinking block (got {delta_type})"
                )
        elif et == "content_block_stop":
            idx = d["index"]
            assert idx in open_blocks, (
                f"content_block_stop on closed/never-opened index {idx}"
            )
            del open_blocks[idx]
        elif et == "message_delta":
            assert not open_blocks, (
                f"message_delta with open blocks: {open_blocks}"
            )
            assert "delta" in d
            assert "stop_reason" in d["delta"]
            saw_message_delta = True
        elif et == "message_stop":
            assert saw_message_delta, "message_stop without preceding message_delta"
            assert not open_blocks, (
                f"message_stop with open blocks: {open_blocks}"
            )
        else:
            raise AssertionError(f"unknown event type: {et!r}")

    assert not open_blocks, f"stream ended with open blocks: {open_blocks}"


# ─── Helpers ─────────────────────────────────────────────────────────


async def _collect_stream(agen) -> bytes:
    buf = bytearray()
    async for chunk in agen:
        buf.extend(chunk)
    return bytes(buf)


def _resp(text: str = "", *, tool_calls=None, thinking=None,
          finish_reason: str = "stop") -> InternalChatResponse:
    return InternalChatResponse(
        message=Message(
            role="assistant",
            content=text,
            tool_calls=tool_calls,
            thinking=thinking,
        ),
        model="test-model",
        finish_reason=finish_reason,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# ─── Request translation ─────────────────────────────────────────────


class TestRequestTranslation:

    def test_string_system_becomes_leading_system_message(self):
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "system": "You are a helper.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        oa = anthropic_request_to_openai(req)
        assert oa.messages[0].role == "system"
        assert oa.messages[0].content == "You are a helper."

    def test_array_system_blocks_joined(self):
        """Anthropic accepts `system` as an array of typed text blocks
        (each potentially with cache_control). They must be joined into
        one system message; cache_control is stripped."""
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "system": [
                {"type": "text", "text": "Part A."},
                {"type": "text", "text": "Part B.", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        oa = anthropic_request_to_openai(req)
        assert oa.messages[0].role == "system"
        assert "Part A." in oa.messages[0].content
        assert "Part B." in oa.messages[0].content

    def test_user_text_message_passthrough(self):
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "what is 2+2?"}],
            "max_tokens": 100,
        })
        oa = anthropic_request_to_openai(req)
        assert oa.messages[-1].role == "user"
        assert "2+2" in str(oa.messages[-1].content)

    def test_content_block_array_text_extracted(self):
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "block-form text"}],
            }],
            "max_tokens": 100,
        })
        oa = anthropic_request_to_openai(req)
        assert "block-form text" in str(oa.messages[-1].content)

    def test_tool_use_block_in_assistant_becomes_tool_calls(self):
        """An assistant turn containing a `tool_use` block in conversation
        history must translate to an OpenAI assistant message with
        `tool_calls`. CC sends these back on subsequent turns."""
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [
                {"role": "user", "content": "list files"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01",
                     "name": "ls", "input": {"path": "/tmp"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01",
                     "content": "a.txt\nb.txt"},
                ]},
            ],
            "max_tokens": 100,
        })
        oa = anthropic_request_to_openai(req)
        # Find the assistant turn with tool_calls
        assistant = [m for m in oa.messages if m.role == "assistant"]
        assert assistant, "no assistant message produced"
        assert assistant[0].tool_calls is not None
        assert assistant[0].tool_calls[0]["id"] == "toolu_01"
        assert assistant[0].tool_calls[0]["function"]["name"] == "ls"
        args = json.loads(assistant[0].tool_calls[0]["function"]["arguments"])
        assert args == {"path": "/tmp"}

    def test_tool_result_block_in_user_becomes_tool_role_message(self):
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [
                {"role": "user", "content": "list files"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01",
                     "name": "ls", "input": {"path": "/tmp"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01",
                     "content": "a.txt\nb.txt"},
                ]},
            ],
            "max_tokens": 100,
        })
        oa = anthropic_request_to_openai(req)
        tool_msgs = [m for m in oa.messages if m.role == "tool"]
        assert tool_msgs, "tool_result did not produce a tool-role message"
        assert tool_msgs[0].tool_call_id == "toolu_01"
        assert "a.txt" in str(tool_msgs[0].content)

    def test_image_block_extracted_to_images_list(self):
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "what's this?"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": "iVBORw0KGgo=",
                }},
            ]}],
            "max_tokens": 100,
        })
        oa = anthropic_request_to_openai(req)
        last = oa.messages[-1]
        # Last user message should contain the text AND surface the image
        # via OpenAI's content-parts shape (so the existing
        # `_parse_openai_content` in openai_routes pulls images out
        # cleanly).
        assert isinstance(last.content, list), (
            "image-bearing message must use content-parts shape so the "
            "existing OpenAI parser extracts the image_url"
        )
        types = [p.get("type") for p in last.content]
        assert "text" in types
        assert "image_url" in types

    def test_tools_translated_with_parameters_rename(self):
        """Anthropic uses `input_schema`; OpenAI uses `parameters`. The
        rest of the tool record passes through."""
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{
                "name": "lookup",
                "description": "Look something up",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }],
        })
        oa = anthropic_request_to_openai(req)
        assert oa.tools is not None
        tool = oa.tools[0]
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] == "lookup"
        assert fn["description"] == "Look something up"
        assert "parameters" in fn
        assert "input_schema" not in fn
        assert fn["parameters"]["required"] == ["q"]

    def test_max_tokens_capped_at_16384(self):
        """Many local backends 400 on max_tokens > 16k. Cap silently."""
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 50000,
        })
        oa = anthropic_request_to_openai(req)
        assert oa.max_tokens == 16384

    def test_stop_sequences_translated(self):
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stop_sequences": ["</done>", "STOP"],
        })
        oa = anthropic_request_to_openai(req)
        assert oa.stop == ["</done>", "STOP"]

    def test_cache_control_with_scope_field_doesnt_break(self):
        """CC 2.1.24+ adds `scope` to cache_control. Earlier proxies
        400'd on the unknown field. We must strip-and-accept."""
        req = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "system": [{
                "type": "text", "text": "system",
                "cache_control": {"type": "ephemeral", "scope": "ephemeral"},
            }],
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text", "text": "hi",
                    "cache_control": {"type": "ephemeral", "scope": "ephemeral"},
                }],
            }],
            "max_tokens": 100,
        })
        # Just constructing + translating must not raise:
        oa = anthropic_request_to_openai(req)
        assert oa.messages  # sanity


# ─── Non-streaming response translation ──────────────────────────────


class TestNonStreamResponseTranslation:

    def test_text_response(self):
        resp = _resp("Hello there.")
        out = internal_response_to_anthropic_message(resp, model_name="claude-test")
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["model"] == "claude-test"
        assert out["stop_reason"] == "end_turn"
        assert out["content"] == [{"type": "text", "text": "Hello there."}]
        assert out["usage"]["input_tokens"] == 10
        assert out["usage"]["output_tokens"] == 5

    def test_tool_use_response(self):
        resp = _resp(
            "",
            tool_calls=[{
                "id": "toolu_42", "type": "function",
                "function": {"name": "ls", "arguments": '{"path":"/tmp"}'},
            }],
            finish_reason="tool_calls",
        )
        out = internal_response_to_anthropic_message(resp, model_name="claude-test")
        assert out["stop_reason"] == "tool_use"
        # Must have a tool_use block; text block omitted when text is empty
        blocks = out["content"]
        tu = [b for b in blocks if b["type"] == "tool_use"]
        assert tu, f"no tool_use block in {blocks}"
        assert tu[0]["id"] == "toolu_42"
        assert tu[0]["name"] == "ls"
        assert tu[0]["input"] == {"path": "/tmp"}

    def test_text_plus_tool(self):
        resp = _resp(
            "Let me check.",
            tool_calls=[{
                "id": "toolu_42", "type": "function",
                "function": {"name": "ls", "arguments": '{"path":"/"}'},
            }],
            finish_reason="tool_calls",
        )
        out = internal_response_to_anthropic_message(resp, model_name="claude-test")
        blocks = out["content"]
        assert blocks[0] == {"type": "text", "text": "Let me check."}
        assert blocks[1]["type"] == "tool_use"
        assert out["stop_reason"] == "tool_use"

    def test_length_stop_reason_mapped(self):
        resp = _resp("partial", finish_reason="length")
        out = internal_response_to_anthropic_message(resp, model_name="claude-test")
        assert out["stop_reason"] == "max_tokens"


# ─── SSE event sequence — the five scenarios ─────────────────────────


@pytest.mark.asyncio
class TestStreamEventSequences:

    async def test_text_only_stream(self):
        resp = _resp("Hello world.")
        raw = await _collect_stream(
            stream_internal_response_as_anthropic_sse(resp, model_name="claude-test"),
        )
        events = parse_sse_events(raw)
        assert_valid_anthropic_event_sequence(events)
        # Concrete shape checks
        types = [e["event"] for e in events]
        assert "message_start" in types
        assert "content_block_start" in types
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        assert "message_stop" in types

    async def test_tool_only_stream(self):
        resp = _resp(
            "",
            tool_calls=[{
                "id": "toolu_X", "type": "function",
                "function": {"name": "search", "arguments": '{"q":"hi"}'},
            }],
            finish_reason="tool_calls",
        )
        raw = await _collect_stream(
            stream_internal_response_as_anthropic_sse(resp, model_name="claude-test"),
        )
        events = parse_sse_events(raw)
        assert_valid_anthropic_event_sequence(events)
        # First content_block is the tool_use (no text block when content empty)
        starts = [e["data"] for e in events if e["event"] == "content_block_start"]
        assert starts[0]["content_block"]["type"] == "tool_use"
        # stop_reason should be tool_use
        deltas = [e["data"] for e in events if e["event"] == "message_delta"]
        assert deltas[-1]["delta"]["stop_reason"] == "tool_use"

    async def test_text_then_tool_stream(self):
        resp = _resp(
            "Let me check.",
            tool_calls=[{
                "id": "toolu_X", "type": "function",
                "function": {"name": "search", "arguments": '{"q":"hi"}'},
            }],
            finish_reason="tool_calls",
        )
        raw = await _collect_stream(
            stream_internal_response_as_anthropic_sse(resp, model_name="claude-test"),
        )
        events = parse_sse_events(raw)
        assert_valid_anthropic_event_sequence(events)
        # Indices: text=0, tool_use=1
        starts = [e["data"] for e in events if e["event"] == "content_block_start"]
        assert starts[0]["index"] == 0
        assert starts[0]["content_block"]["type"] == "text"
        assert starts[1]["index"] == 1
        assert starts[1]["content_block"]["type"] == "tool_use"

    async def test_tool_then_text_then_tool_stream(self):
        """Multiple tool calls in one assistant turn — common when CC asks
        the model to plan + dispatch multiple actions. Indices 0, 1, 2
        with mixed types must round-trip cleanly."""
        resp = _resp(
            "And then this.",   # text in middle index handled by ordering: we put text first
            tool_calls=[
                {"id": "toolu_A", "type": "function",
                 "function": {"name": "ls", "arguments": "{}"}},
                {"id": "toolu_B", "type": "function",
                 "function": {"name": "cat", "arguments": '{"f":"x"}'}},
            ],
            finish_reason="tool_calls",
        )
        raw = await _collect_stream(
            stream_internal_response_as_anthropic_sse(resp, model_name="claude-test"),
        )
        events = parse_sse_events(raw)
        assert_valid_anthropic_event_sequence(events)
        # Three content blocks: text (0), tool_A (1), tool_B (2)
        starts = [e["data"] for e in events if e["event"] == "content_block_start"]
        assert len(starts) == 3
        assert [s["index"] for s in starts] == [0, 1, 2]
        assert starts[0]["content_block"]["type"] == "text"
        assert starts[1]["content_block"]["type"] == "tool_use"
        assert starts[2]["content_block"]["type"] == "tool_use"
        assert starts[1]["content_block"]["id"] == "toolu_A"
        assert starts[2]["content_block"]["id"] == "toolu_B"

    async def test_reasoning_then_tool_stream(self):
        """Reasoning models emit `thinking` content that we expose as a
        native Anthropic `thinking` block. The block ordering is
        thinking → text → tool_use. This is the scenario where CCR/
        LiteLLM/sglang have all shipped bugs (e.g. emitting text_delta
        on an open tool_use block when reasoning ends with a separator)."""
        resp = _resp(
            "Here you go.",
            thinking="I should look this up.",
            tool_calls=[{
                "id": "toolu_X", "type": "function",
                "function": {"name": "search", "arguments": '{"q":"x"}'},
            }],
            finish_reason="tool_calls",
        )
        raw = await _collect_stream(
            stream_internal_response_as_anthropic_sse(resp, model_name="claude-test"),
        )
        events = parse_sse_events(raw)
        assert_valid_anthropic_event_sequence(events)
        starts = [e["data"] for e in events if e["event"] == "content_block_start"]
        # thinking → text → tool_use
        assert starts[0]["content_block"]["type"] == "thinking"
        assert starts[1]["content_block"]["type"] == "text"
        assert starts[2]["content_block"]["type"] == "tool_use"


# ─── count_tokens estimator ──────────────────────────────────────────


class TestCountTokensEstimator:

    def test_empty_returns_zero_or_low(self):
        n = count_tokens_estimate([], system=None, tools=None)
        assert n >= 0
        assert n < 10  # near-zero for empty

    def test_short_user_text_estimate_reasonable(self):
        msgs = [{"role": "user", "content": "hello world how are you"}]
        n = count_tokens_estimate(msgs, system=None, tools=None)
        # 5 short words → ~5-10 tokens
        assert 3 <= n <= 30

    def test_system_prompt_adds_tokens(self):
        msgs = [{"role": "user", "content": "hi"}]
        baseline = count_tokens_estimate(msgs, system=None, tools=None)
        with_sys = count_tokens_estimate(
            msgs, system="You are a helpful assistant who knows many things.", tools=None,
        )
        assert with_sys > baseline

    def test_tools_add_schema_overhead(self):
        msgs = [{"role": "user", "content": "hi"}]
        baseline = count_tokens_estimate(msgs, system=None, tools=None)
        with_tools = count_tokens_estimate(msgs, system=None, tools=[{
            "name": "search",
            "description": "search the web for information",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }])
        assert with_tools > baseline

    def test_multimodal_block_content_handled(self):
        """count_tokens must not crash on content-block arrays (CC sends
        these for any tool_result / image-bearing turn)."""
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "iVBOR",
                }},
            ],
        }]
        n = count_tokens_estimate(msgs, system=None, tools=None)
        assert n > 0


# ─── Route-level integration ─────────────────────────────────────────


def _stub_openai_response(*, content: str = "", tool_calls=None,
                          finish_reason: str = "stop",
                          reasoning_content: str | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    return {
        "id": "chatcmpl-x", "object": "chat.completion", "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


def _fake_request():
    """Minimal stand-in for the FastAPI Request that anthropic_messages
    passes to openai_chat. The stub doesn't dereference it."""
    return type("R", (), {"scope": {"user": type("U", (), {"id": "u1"})()}})()


@pytest.mark.asyncio
class TestAnthropicMessagesRoute:

    async def test_non_stream_text_response(self, monkeypatch):
        from fastapi.responses import JSONResponse

        from augmentum.proxy.anthropic_routes import anthropic_messages

        async def stub(body, request):
            return JSONResponse(_stub_openai_response(content="hello back"))
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", stub,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        resp = await anthropic_messages(body, _fake_request())
        out = json.loads(resp.body)
        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["model"] == "claude-test"
        assert out["stop_reason"] == "end_turn"
        assert out["content"] == [{"type": "text", "text": "hello back"}]
        assert out["usage"] == {"input_tokens": 5, "output_tokens": 3}

    async def test_non_stream_tool_use_response(self, monkeypatch):
        from fastapi.responses import JSONResponse

        from augmentum.proxy.anthropic_routes import anthropic_messages

        async def stub(body, request):
            return JSONResponse(_stub_openai_response(
                content="checking",
                tool_calls=[{"id": "t1", "type": "function",
                             "function": {"name": "ls",
                                          "arguments": '{"path":"/"}'}}],
                finish_reason="tool_calls",
            ))
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", stub,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "ls /"}],
            "max_tokens": 100,
        })
        resp = await anthropic_messages(body, _fake_request())
        out = json.loads(resp.body)
        assert out["stop_reason"] == "tool_use"
        blocks = out["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "ls"
        assert blocks[1]["input"] == {"path": "/"}

    async def test_stream_full_event_sequence(self, monkeypatch):
        from fastapi.responses import JSONResponse, StreamingResponse

        from augmentum.proxy.anthropic_routes import anthropic_messages

        async def stub(body, request):
            return JSONResponse(_stub_openai_response(
                content="streamed text",
                tool_calls=[{"id": "t1", "type": "function",
                             "function": {"name": "go",
                                          "arguments": '{"x":1}'}}],
                finish_reason="tool_calls",
            ))
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", stub,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
        })
        resp = await anthropic_messages(body, _fake_request())
        assert isinstance(resp, StreamingResponse)
        assert resp.headers.get("x-accel-buffering") == "no"

        raw = await _collect_stream(resp.body_iterator)
        events = parse_sse_events(raw)
        assert_valid_anthropic_event_sequence(events)
        # Concrete shape: text block then tool_use block
        starts = [e["data"] for e in events if e["event"] == "content_block_start"]
        assert starts[0]["content_block"]["type"] == "text"
        assert starts[1]["content_block"]["type"] == "tool_use"
        assert starts[1]["content_block"]["name"] == "go"

    async def test_stream_emits_message_start_before_handler_completes(self, monkeypatch):
        """The TTFT-critical invariant — message_start must hit the wire
        before the inner LLM call completes. Without this, CC's SDK sits
        waiting on bytes for the full model duration and the connection
        looks dead to intermediaries."""
        from fastapi.responses import JSONResponse

        from augmentum.proxy.anthropic_routes import anthropic_messages

        handler_started_event = asyncio.Event()
        handler_release = asyncio.Event()

        async def slow_stub(body, request):
            handler_started_event.set()
            await handler_release.wait()
            return JSONResponse(_stub_openai_response(content="ok"))
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", slow_stub,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
        })
        resp = await anthropic_messages(body, _fake_request())
        agen = resp.body_iterator.__aiter__()

        # First chunk should arrive immediately — must be message_start
        # — without waiting on the handler
        first = await agen.__anext__()
        assert b"message_start" in first

        # Now let the handler run and drain
        handler_release.set()
        try:
            async for _ in agen:
                pass
        except StopAsyncIteration:
            pass

    async def test_stream_error_becomes_text_block(self, monkeypatch):
        """Backend errors inside a started stream are surfaced as a text
        block carrying the message — CC's TUI displays the error text
        instead of the user seeing a silent close."""
        from fastapi.responses import JSONResponse

        from augmentum.proxy.anthropic_routes import anthropic_messages

        async def err_stub(body, request):
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "Model unavailable", "type": "x"}},
            )
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", err_stub,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stream": True,
        })
        resp = await anthropic_messages(body, _fake_request())
        raw = await _collect_stream(resp.body_iterator)
        events = parse_sse_events(raw)
        assert_valid_anthropic_event_sequence(events)
        # The error text appears in a text_delta event
        deltas = [e["data"] for e in events if e["event"] == "content_block_delta"]
        assert deltas, "no text content emitted carrying the error"
        text = deltas[0]["delta"].get("text", "")
        assert "Model unavailable" in text

    async def test_non_stream_error_returns_anthropic_error_shape(self, monkeypatch):
        from fastapi.responses import JSONResponse

        from augmentum.proxy.anthropic_routes import anthropic_messages

        async def err_stub(body, request):
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Bad model name", "type": "x"}},
            )
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", err_stub,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "claude-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        resp = await anthropic_messages(body, _fake_request())
        assert resp.status_code == 400
        out = json.loads(resp.body)
        assert out["type"] == "error"
        assert out["error"]["type"] == "invalid_request_error"
        assert "Bad model name" in out["error"]["message"]

    async def test_count_tokens_endpoint(self):
        from augmentum.proxy.anthropic_routes import anthropic_count_tokens

        resp = await anthropic_count_tokens({
            "messages": [{"role": "user", "content": "hello world how are you"}],
            "system": "You help.",
        }, _fake_request())
        out = json.loads(resp.body)
        assert "input_tokens" in out
        assert isinstance(out["input_tokens"], int)
        assert out["input_tokens"] > 0

    async def test_count_tokens_handles_malformed_input(self):
        """CC sometimes sends count_tokens with unusual shapes — must
        never 500; always return some integer."""
        from augmentum.proxy.anthropic_routes import anthropic_count_tokens

        resp = await anthropic_count_tokens({}, _fake_request())
        out = json.loads(resp.body)
        assert "input_tokens" in out
        assert isinstance(out["input_tokens"], int)


@pytest.mark.asyncio
class TestToolCallDispatch:
    """Isolation tests: requests WITH tools must route around
    PassthroughHandler (which filters/rewrites tools for Augmentum's
    chat UI). Requests WITHOUT tools must still flow through
    openai_chat to get memory/knowledge/dream enrichment."""

    async def test_request_with_tools_bypasses_openai_chat(self, monkeypatch):
        """The bug we fixed: PassthroughHandler was filtering CC's tool
        calls. Now /v1/messages with tools must NOT call openai_chat
        at all — it should go straight to the backend."""
        from augmentum.models.base import InternalChatResponse, Message, Usage
        from augmentum.proxy.anthropic_routes import anthropic_messages

        openai_chat_calls = []
        backend_calls = []

        async def stub_openai(body, request):
            openai_chat_calls.append(body)
            from fastapi.responses import JSONResponse
            return JSONResponse(_stub_openai_response(content="should not be called"))

        class StubBackend:
            async def chat(self, req):
                backend_calls.append(req)
                return InternalChatResponse(
                    message=Message(
                        role="assistant",
                        content="checking",
                        tool_calls=[{
                            "id": "tu_1", "type": "function",
                            "function": {"name": "ls", "arguments": '{"path":"/"}'},
                        }],
                    ),
                    model="m", finish_reason="tool_calls",
                    usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
                )

        class StubRegistry:
            async def resolve_backend_with_fabric(self, model, *, user_id, session_id):
                return StubBackend(), model

        class StubState:
            provider_registry = StubRegistry()

        class StubRequest:
            scope = {"user": type("U", (), {"id": "u1"})()}
            app = type("A", (), {"state": StubState()})()

        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", stub_openai,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "test-model",
            "messages": [{"role": "user", "content": "list files"}],
            "max_tokens": 100,
            "tools": [{"name": "ls", "description": "list", "input_schema": {"type": "object"}}],
        })
        resp = await anthropic_messages(body, StubRequest())
        out = json.loads(resp.body)

        # The decisive assertion: openai_chat must NOT have been called.
        # Backend must have been called directly.
        assert len(openai_chat_calls) == 0, (
            "tool-bearing request must bypass openai_chat (which goes through "
            "PassthroughHandler and filters/rewrites tools)"
        )
        assert len(backend_calls) == 1, (
            "tool-bearing request must call backend.chat() directly"
        )
        # And tool_use must round-trip cleanly
        tool_blocks = [b for b in out["content"] if b["type"] == "tool_use"]
        assert tool_blocks, f"tool_use block lost in translation; content={out['content']}"
        assert tool_blocks[0]["name"] == "ls"
        assert tool_blocks[0]["input"] == {"path": "/"}
        assert out["stop_reason"] == "tool_use"

    async def _capture_chat_template_kwargs(self, monkeypatch, model_name: str):
        """Helper: send a tool-bearing request with ``model_name`` and
        capture the chat_template_kwargs that landed on backend.chat."""
        from augmentum.models.base import InternalChatResponse, Message, Usage
        from augmentum.proxy.anthropic_routes import anthropic_messages

        captured = []

        class StubBackend:
            async def chat(self, req):
                captured.append(req.chat_template_kwargs)
                return InternalChatResponse(
                    message=Message(role="assistant", content="ok"),
                    model="m", finish_reason="stop",
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        class StubRegistry:
            async def resolve_backend_with_fabric(self, model, *, user_id, session_id):
                return StubBackend(), model

        class StubState:
            provider_registry = StubRegistry()

        class StubRequest:
            scope = {"user": type("U", (), {"id": "u1"})()}
            app = type("A", (), {"state": StubState()})()

        body = AnthropicMessagesRequest.model_validate({
            "model": model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"name": "ls", "description": "x", "input_schema": {"type": "object"}}],
        })
        await anthropic_messages(body, StubRequest())
        return captured[0] if captured else None

    async def test_haiku_tier_disables_thinking(self, monkeypatch):
        """CC subagents (haiku) call us with hardcoded claude-haiku-*.
        These are utility calls (Explore, title, compaction) that don't
        need reasoning. Disabling enable_thinking saves 500-2000 tokens
        per call on Qwen3.6 (Qwen team's explicit recommendation for
        non-thinking utility work)."""
        from augmentum.config import settings as cfg
        monkeypatch.setattr(cfg, "primary_chat_model", "Local-35B", raising=False)
        kwargs = await self._capture_chat_template_kwargs(
            monkeypatch, "claude-haiku-4-5-20251001",
        )
        assert kwargs == {"enable_thinking": False}

    async def test_sonnet_tier_enables_thinking_with_preserve(self, monkeypatch):
        """CC main loop (sonnet/opus) needs quality reasoning. Qwen3.6's
        preserve_thinking carries prior reasoning across turns — reduces
        redundant re-derivation AND improves KV cache hit rate (per
        Qwen docs, this is the agentic-mode recommended setting)."""
        from augmentum.config import settings as cfg
        monkeypatch.setattr(cfg, "primary_chat_model", "Local-35B", raising=False)
        for model in ("claude-sonnet-4-5-20250101", "claude-opus-4-7-20251022"):
            kwargs = await self._capture_chat_template_kwargs(monkeypatch, model)
            assert kwargs == {"enable_thinking": True, "preserve_thinking": True}, (
                f"sonnet/opus must enable thinking + preserve_thinking; got {kwargs} for {model}"
            )

    async def test_non_claude_model_leaves_thinking_default(self, monkeypatch):
        """When the caller picks an explicit local model name (not
        claude-*), don't force thinking either way. Let llama-server's
        CLI defaults / GGUF chat template apply unchanged. This keeps
        non-CC integrations (Cursor pointing at /v1/messages, custom
        clients, etc.) opinion-free."""
        from augmentum.config import settings as cfg
        monkeypatch.setattr(cfg, "primary_chat_model", "Local-35B", raising=False)
        kwargs = await self._capture_chat_template_kwargs(
            monkeypatch, "Qwen3.6-35B-A3B-IQ4_XS",
        )
        assert kwargs is None, (
            f"non-claude model must not inject chat_template_kwargs; got {kwargs}"
        )

    async def test_direct_path_sets_kv_session_key_for_slot_affinity(self, monkeypatch):
        """Performance: tool-bearing CC requests must set kv_session_key
        so llama-server's prefix cache hits on turn 2+. Without this,
        every turn re-prefills the full CC system prompt (3-5s cold
        cost). The key is per-user + "claude-code" suffix so different
        clients of the same backend don't collide and CC stays on its
        own slot vs Augmentum chat traffic."""
        from augmentum.models.base import InternalChatResponse, Message, Usage
        from augmentum.proxy.anthropic_routes import anthropic_messages

        seen_keys = []

        class StubBackend:
            async def chat(self, req):
                seen_keys.append(req.kv_session_key)
                return InternalChatResponse(
                    message=Message(role="assistant", content="ok"),
                    model="m", finish_reason="stop",
                    usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                )

        class StubRegistry:
            async def resolve_backend_with_fabric(self, model, *, user_id, session_id):
                return StubBackend(), model

        class StubState:
            provider_registry = StubRegistry()

        class StubRequest:
            scope = {"user": type("U", (), {"id": "user_abc"})()}
            app = type("A", (), {"state": StubState()})()

        body = AnthropicMessagesRequest.model_validate({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "tools": [{"name": "ls", "description": "x", "input_schema": {"type": "object"}}],
        })
        await anthropic_messages(body, StubRequest())

        assert len(seen_keys) == 1, f"expected one backend.chat call; saw {seen_keys}"
        key = seen_keys[0]
        assert key is not None, "kv_session_key must be set on tool-bearing CC requests"
        assert key.startswith("user_abc:cc:"), (
            f"kv_session_key must be per-user + 'cc' tag + prefix hash; saw {key!r}"
        )
        # Hash suffix is the first 12 hex chars of SHA-256
        suffix = key.split(":", 2)[2]
        assert len(suffix) == 12 and all(c in "0123456789abcdef" for c in suffix), (
            f"prefix hash must be 12 hex chars; saw {suffix!r}"
        )

    async def test_claude_haiku_model_aliased_to_primary_chat_model(self, monkeypatch):
        """The CC subagent fix: CC fires Agent/Explore subagents with
        hardcoded ``claude-haiku-4-5-*``. Without aliasing, every
        subagent fails with model-unavailable and CC's Agent tool is
        structurally broken. With aliasing, the subagent reaches the
        user's chosen local model."""
        from fastapi.responses import JSONResponse

        from augmentum.config import settings as cfg
        from augmentum.proxy.anthropic_routes import anthropic_messages

        monkeypatch.setattr(cfg, "primary_chat_model", "Local-35B")

        seen_model = []
        async def stub_openai(body, request):
            seen_model.append(body.model)
            return JSONResponse(_stub_openai_response(content="from local"))
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", stub_openai,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "claude-haiku-4-5-20251001",
            "messages": [{"role": "user", "content": "what files exist?"}],
            "max_tokens": 100,
        })
        await anthropic_messages(body, _fake_request())
        assert seen_model == ["Local-35B"], (
            f"claude-haiku-* must alias to primary_chat_model; saw {seen_model}"
        )

    async def test_tier_specific_alias_beats_default(self, monkeypatch):
        """anthropic_alias_haiku → small fast model, anthropic_alias_opus
        → big model. Per-tier aliases beat the global default + primary."""
        from fastapi.responses import JSONResponse

        from augmentum.config import settings as cfg
        from augmentum.proxy.anthropic_routes import anthropic_messages

        monkeypatch.setattr(cfg, "primary_chat_model", "Big-35B", raising=False)
        monkeypatch.setattr(cfg, "anthropic_alias_haiku", "Tiny-4B", raising=False)

        seen = []
        async def stub_openai(body, request):
            seen.append(body.model)
            return JSONResponse(_stub_openai_response(content="ok"))
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", stub_openai,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "claude-haiku-4-5-20251001",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        await anthropic_messages(body, _fake_request())
        assert seen == ["Tiny-4B"], (
            f"per-tier haiku alias must win over primary_chat_model; saw {seen}"
        )

    async def test_non_claude_model_not_aliased(self, monkeypatch):
        """Local model names (Qwen3.6-*, etc.) must pass through
        untouched. Aliasing is claude-specific."""
        from fastapi.responses import JSONResponse

        from augmentum.config import settings as cfg
        from augmentum.proxy.anthropic_routes import anthropic_messages

        monkeypatch.setattr(cfg, "primary_chat_model", "Local-35B", raising=False)

        seen = []
        async def stub_openai(body, request):
            seen.append(body.model)
            return JSONResponse(_stub_openai_response(content="ok"))
        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", stub_openai,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "Qwen3.6-35B-A3B-IQ4_XS",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        await anthropic_messages(body, _fake_request())
        assert seen == ["Qwen3.6-35B-A3B-IQ4_XS"], (
            f"non-claude model must pass through unchanged; saw {seen}"
        )

    async def test_claude_alias_falls_back_through_chain(self, monkeypatch):
        """No tier-specific alias, no anthropic_alias_default — must
        fall back to primary_chat_model."""
        from augmentum.config import settings as cfg
        from augmentum.proxy.anthropic_routes import _resolve_claude_alias

        # Clear any tier/default aliases
        monkeypatch.setattr(cfg, "anthropic_alias_sonnet", "", raising=False)
        monkeypatch.setattr(cfg, "anthropic_alias_default", "", raising=False)
        monkeypatch.setattr(cfg, "primary_chat_model", "Fallback-Model", raising=False)

        resolved, tier = _resolve_claude_alias("claude-sonnet-4-5-20251022")
        assert resolved == "Fallback-Model"
        assert tier == "sonnet"

    async def test_claude_alias_unchanged_when_no_target(self, monkeypatch):
        """When NOTHING is configured (no aliases, no primary), the
        claude-* name passes through unchanged so the downstream
        ModelUnavailableError gives a clear diagnostic instead of
        silently routing to nothing."""
        from augmentum.config import settings as cfg
        from augmentum.proxy.anthropic_routes import _resolve_claude_alias

        monkeypatch.setattr(cfg, "anthropic_alias_haiku", "", raising=False)
        monkeypatch.setattr(cfg, "anthropic_alias_default", "", raising=False)
        monkeypatch.setattr(cfg, "primary_chat_model", "", raising=False)

        resolved, tier = _resolve_claude_alias("claude-haiku-4-5-20251001")
        assert resolved == "claude-haiku-4-5-20251001"
        assert tier is None

    async def test_request_without_tools_still_uses_openai_chat(self, monkeypatch):
        """Augmentum's normal flow MUST be preserved: requests without
        external tools go through openai_chat for full orchestration
        (memory/knowledge/dream/mode/etc)."""
        from fastapi.responses import JSONResponse

        from augmentum.proxy.anthropic_routes import anthropic_messages

        openai_chat_calls = []

        async def stub_openai(body, request):
            openai_chat_calls.append(body)
            return JSONResponse(_stub_openai_response(content="hello"))

        monkeypatch.setattr(
            "augmentum.proxy.anthropic_routes.openai_chat_handler", stub_openai,
        )

        body = AnthropicMessagesRequest.model_validate({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            # NO tools — Augmentum's normal flow
        })
        resp = await anthropic_messages(body, _fake_request())
        out = json.loads(resp.body)

        assert len(openai_chat_calls) == 1, (
            "non-tool request must still flow through openai_chat for "
            "memory/knowledge/dream enrichment"
        )
        assert out["content"] == [{"type": "text", "text": "hello"}]


# ─── Prefix cache key ────────────────────────────────────────────────


class TestPrefixCacheKey:
    """Hash function backing kv_session_key for CC slot routing.

    Behaviour contract:
      1. Hash is stable across turns that share the cache_control prefix
         but differ in the trailing per-turn content (so two CC turns
         land on the same llama-server slot and the prefix cache hits).
      2. Hash changes when the prefix itself changes (system, tools, or
         content before the marker) so a new slot warms instead of
         clobbering an unrelated context.
      3. Hash respects the LAST cache_control marker — content after it
         doesn't affect the key (that's the whole point).
      4. With no markers, falls back to (system + tools) so non-CC
         callers still get slot affinity for matching system+tools.
    """

    def test_stable_across_turns_with_same_prefix(self):
        """Two turns that share system+tools+marked-prefix but differ
        only in the trailing user message MUST hash to the same key."""
        base = {
            "model": "test-model",
            "system": [
                {"type": "text", "text": "You are a helper.",
                 "cache_control": {"type": "ephemeral"}},
            ],
            "tools": [{
                "name": "shell",
                "input_schema": {"type": "object",
                                  "properties": {"cmd": {"type": "string"}}},
            }],
            "max_tokens": 100,
        }
        turn_1 = AnthropicMessagesRequest.model_validate({
            **base,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "env setup info",
                     "cache_control": {"type": "ephemeral"}},
                ]},
                {"role": "user", "content": "first question"},
            ],
        })
        turn_2 = AnthropicMessagesRequest.model_validate({
            **base,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "env setup info",
                     "cache_control": {"type": "ephemeral"}},
                ]},
                {"role": "user", "content": "completely different second question"},
            ],
        })
        assert compute_prefix_cache_key(turn_1) == compute_prefix_cache_key(turn_2)

    def test_changes_when_system_changes(self):
        """A different system prompt MUST produce a different hash so
        the new context warms its own slot instead of clobbering."""
        req_a = AnthropicMessagesRequest.model_validate({
            "model": "test-model",
            "system": "You are a Python expert.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        req_b = AnthropicMessagesRequest.model_validate({
            "model": "test-model",
            "system": "You are a Rust expert.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        })
        assert compute_prefix_cache_key(req_a) != compute_prefix_cache_key(req_b)

    def test_respects_last_cache_control_marker(self):
        """Content AFTER the last cache_control marker MUST NOT affect
        the hash. CC's contract: 'everything up to and including this
        block is stable across turns'. Trailing per-turn content varies
        by definition."""
        base = {
            "model": "test-model",
            "system": "shared system",
            "max_tokens": 100,
        }
        marker = {"type": "text", "text": "stable env block",
                  "cache_control": {"type": "ephemeral"}}
        req_a = AnthropicMessagesRequest.model_validate({
            **base,
            "messages": [
                {"role": "user", "content": [marker]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "trailing A"},
            ],
        })
        req_b = AnthropicMessagesRequest.model_validate({
            **base,
            "messages": [
                {"role": "user", "content": [marker]},
                {"role": "assistant", "content": "completely different reply"},
                {"role": "user", "content": "trailing B and a lot more text after"},
            ],
        })
        assert compute_prefix_cache_key(req_a) == compute_prefix_cache_key(req_b)

    def test_no_marker_falls_back_to_system_plus_tools(self):
        """Without any cache_control markers (non-CC caller), hash MUST
        depend only on system + tools, not message content. That way
        same-system + same-tools traffic still gets slot affinity."""
        base = {
            "model": "test-model",
            "system": "shared system",
            "tools": [{"name": "shell",
                       "input_schema": {"type": "object", "properties": {}}}],
            "max_tokens": 100,
        }
        req_a = AnthropicMessagesRequest.model_validate({
            **base,
            "messages": [{"role": "user", "content": "first question"}],
        })
        req_b = AnthropicMessagesRequest.model_validate({
            **base,
            "messages": [{"role": "user", "content": "totally different second question"}],
        })
        assert compute_prefix_cache_key(req_a) == compute_prefix_cache_key(req_b)

        # And changing tools breaks the equality
        req_c = AnthropicMessagesRequest.model_validate({
            **base,
            "tools": [{"name": "browse",
                       "input_schema": {"type": "object", "properties": {}}}],
            "messages": [{"role": "user", "content": "first question"}],
        })
        assert compute_prefix_cache_key(req_a) != compute_prefix_cache_key(req_c)


# Import asyncio at module level for the slow-stub test above
import asyncio  # noqa: E402
