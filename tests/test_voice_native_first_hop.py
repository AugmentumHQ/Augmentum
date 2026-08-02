"""Voice first-hop native tool calling (2026-06-15).

The fix for the becca_act_gap failure mode: a native-trained model on
the voice first hop refused tool-worthy asks ("I don't have live access
to news" with web_search right there) because the streaming primary hop
exposed NO native tool schemas — it relied on the model voluntarily
leaking a text-format call into the content stream. Native models emit
STRUCTURED tool calls out-of-band instead, which the text sieve never
saw, so nothing fired until the user explicitly insisted.

Coverage:
- ``_assemble_streamed_tool_calls`` reassembles fragmented OpenAI
  streaming tool-call deltas (the out-of-band channel)
- ``_call_primary`` captures structured deltas into the sink while
  still yielding content for the sieve
- ``_attach_native_tools`` attaches schemas ONLY for NATIVE-tier
  backends (TEXT/STRUCTURED keep the prompt + sieve path)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.companion_runtime import voice as voice_mod
from augmentum.companion_runtime.voice import (
    BeccaVoice,
    _assemble_streamed_tool_calls,
)

# ── Delta reassembly ──────────────────────────────────────────────────


def test_assemble_fragmented_single_call():
    """A call split across chunks (name first, args in fragments)
    reassembles into one ToolCall with decoded args."""
    deltas = [
        {"index": 0, "id": "c1", "type": "function",
         "function": {"name": "web_search", "arguments": ""}},
        {"index": 0, "function": {"arguments": '{"query":"top US '}},
        {"index": 0, "function": {"arguments": 'news today"}'}},
    ]
    calls = _assemble_streamed_tool_calls(deltas)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].args == {"query": "top US news today"}


def test_assemble_parallel_calls_with_nested_and_null():
    """Two parallel calls keyed by index; nested/null arg values are
    stringified to the tag protocol's dict[str, str] shape."""
    deltas = [
        {"index": 0, "function": {"name": "a",
         "arguments": '{"x":1,"y":null,"z":[1,2]}'}},
        {"index": 1, "function": {"name": "b", "arguments": "{}"}},
    ]
    calls = _assemble_streamed_tool_calls(deltas)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].args == {"x": "1", "y": "", "z": "[1, 2]"}
    assert calls[1].args == {}


def test_assemble_malformed_args_keeps_name_drops_args():
    """Truncated/garbled argument JSON must not lose the call — name is
    preserved, args fall back to empty (the executor can still ask)."""
    calls = _assemble_streamed_tool_calls(
        [{"index": 0, "function": {"name": "c", "arguments": "{bad"}}]
    )
    assert len(calls) == 1
    assert calls[0].name == "c"
    assert calls[0].args == {}


def test_assemble_skips_nameless_deltas():
    """A delta carrying only arguments and no name is not a call."""
    assert _assemble_streamed_tool_calls(
        [{"index": 0, "function": {"arguments": "{}"}}]
    ) == []


# ── _call_primary delta capture ───────────────────────────────────────


class _Chunk:
    def __init__(self, content="", tool_calls=None):
        self.content_delta = content
        self.augmentum = {"tool_calls": tool_calls} if tool_calls else None


class _StreamBackend:
    """Backend whose chat_stream yields canned chunks; records req."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.seen_req = None

    async def chat_stream(self, req):
        self.seen_req = req
        for c in self._chunks:
            yield c


def _voice():
    return BeccaVoice(SimpleNamespace(
        bus=None, companion_id="becca", _app_state=None,
    ))


@pytest.mark.asyncio
async def test_call_primary_captures_structured_deltas(monkeypatch):
    """The structured tool-call deltas land in the sink; content still
    streams through for the sieve. Schema attach is isolated out."""
    backend = _StreamBackend([
        _Chunk(content="let me look"),
        _Chunk(tool_calls=[{"index": 0, "function": {
            "name": "web_search", "arguments": '{"query":"news"}'}}]),
        _Chunk(content=""),
    ])

    async def _primary(_runtime):
        return backend, "Qwen3.6-35B"
    monkeypatch.setattr(voice_mod.tiers, "primary", _primary)
    # Isolate delta capture from schema selection (its own test below).
    monkeypatch.setattr(BeccaVoice, "_attach_native_tools",
                        lambda self, req, b, m, intent: None)

    import asyncio
    sink: list[dict] = []
    intent = SimpleNamespace(text="any news?", user_id="u1", metadata={})
    out = []
    async for piece in _voice()._call_primary(
        "sys", intent, cancel=asyncio.Event(),
        invocation_id="iv", tool_call_sink=sink,
    ):
        out.append(piece)

    assert "".join(out) == "let me look"
    calls = _assemble_streamed_tool_calls(sink)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].args == {"query": "news"}


# ── _attach_native_tools tier gating ──────────────────────────────────


def test_attach_native_tools_only_for_native_tier(monkeypatch):
    """NATIVE tier → req.tools set; TEXT tier → req.tools untouched."""
    from augmentum.companion_runtime import native_loop
    from augmentum.modes.analytical import tool_calling as tc

    app_state = SimpleNamespace(tool_registry=object())
    v = BeccaVoice(SimpleNamespace(
        bus=None, companion_id="becca", _app_state=app_state,
    ))
    intent = SimpleNamespace(text="hi", user_id="u1", metadata={})
    fake_tool = SimpleNamespace(
        name="web_search", description="search the web", input_schema=None,
    )
    # _attach_native_tools imports this from native_loop at call time.
    monkeypatch.setattr(
        native_loop, "select_companion_tools", lambda *a, **k: [fake_tool],
    )

    # TEXT tier — no schemas attached.
    monkeypatch.setattr(tc, "select_tier", lambda b, m: tc.ToolCallingTier.TEXT)
    req_text = SimpleNamespace(tools=None)
    v._attach_native_tools(req_text, object(), "small-local", intent)
    assert req_text.tools is None

    # NATIVE tier — schemas attached in OpenAI function format.
    monkeypatch.setattr(tc, "select_tier", lambda b, m: tc.ToolCallingTier.NATIVE)
    req_native = SimpleNamespace(tools=None)
    v._attach_native_tools(req_native, object(), "Qwen3.6-35B", intent)
    assert req_native.tools is not None
    assert req_native.tools[0]["function"]["name"] == "web_search"
