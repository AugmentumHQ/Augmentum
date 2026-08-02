"""Tests for passthrough mode tool integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)
from augmentum.modes.passthrough.handler import PassthroughHandler
from augmentum.proxy.handler_factory import _resolve_passthrough_tools
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeSearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"Results for: {kwargs.get('query', '')}")


class FakeCalcTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Calculate math expressions"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"= {kwargs.get('expression', '')}")


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(FakeSearchTool())
    r.register(FakeCalcTool())
    return r


@pytest.fixture
def backend():
    """Create a mock backend that select_tier will recognize as OpenAI (NATIVE tier)."""
    from augmentum.models.openai_compat import OpenAIBackend
    b = AsyncMock(spec=OpenAIBackend)
    b._base_url = "https://api.openai.com/v1"  # cloud URL → NATIVE tier
    return b


def _make_request(content: str = "Hello") -> InternalChatRequest:
    return InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content=content)],
    )


def _make_response(content: str = "Hello back", tool_calls=None) -> InternalChatResponse:
    return InternalChatResponse(
        message=Message(role="assistant", content=content, tool_calls=tool_calls),
        model="test-model",
    )


# ---------------------------------------------------------------------------
# _resolve_passthrough_tools
# ---------------------------------------------------------------------------


class TestResolvePassthroughTools:
    def test_empty_when_no_registry(self):
        state = MagicMock()
        state.tool_registry = None
        assert _resolve_passthrough_tools(state) == []

    # NOTE: the scheduling substrate injects whenever a dispatcher object
    # exists on app.state (companion_runtime OR scheduler_service) — the pure
    # merge tests below null BOTH attributes so the substrate (tested
    # separately in TestScheduleSubstrateAlwaysOn) doesn't pollute the
    # assertions. ``calculator``/``unit_converter`` are the zero-cost
    # PASSTHROUGH_AUTO_TOOLS auto-included whenever any tool is on.

    def test_empty_when_no_config_no_header(self, registry):
        state = MagicMock()
        state.tool_registry = registry
        state.companion_runtime = None
        state.scheduler_service = None
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = ""
            result = _resolve_passthrough_tools(state)
        assert result == []

    def test_config_defaults(self, registry):
        state = MagicMock()
        state.tool_registry = registry
        state.companion_runtime = None
        state.scheduler_service = None
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search,calculator"
            result = _resolve_passthrough_tools(state)
        assert set(result) == {"web_search", "calculator", "unit_converter"}

    def test_header_adds_to_config(self, registry):
        state = MagicMock()
        state.tool_registry = registry
        state.companion_runtime = None
        state.scheduler_service = None
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search"
            result = _resolve_passthrough_tools(state, header_tools="calculator")
        assert set(result) == {"web_search", "calculator", "unit_converter"}

    def test_header_all_returns_everything(self, registry):
        state = MagicMock()
        state.tool_registry = registry
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = ""
            result = _resolve_passthrough_tools(state, header_tools="all")
        assert set(result) == {"web_search", "calculator"}

    def test_header_only(self, registry):
        state = MagicMock()
        state.tool_registry = registry
        state.companion_runtime = None
        state.scheduler_service = None
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = ""
            result = _resolve_passthrough_tools(state, header_tools="web_search")
        assert set(result) == {"web_search", "calculator", "unit_converter"}

    def test_deduplication(self, registry):
        state = MagicMock()
        state.tool_registry = registry
        state.companion_runtime = None
        state.scheduler_service = None
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search"
            result = _resolve_passthrough_tools(state, header_tools="web_search")
        assert result.count("web_search") == 1


_SUBSTRATE = {
    "schedule_briefing", "schedule_request", "watch_for",
    "schedule_deadline", "schedule_action",
    "list_briefings", "cancel_briefing",
}


class TestScheduleSubstrateAutoInclude:
    """The scheduling substrate rides the SAME auto-include mechanism as
    the zero-cost utilities (no bespoke injection path, no keyword
    gating): present whenever any tool is enabled and a dispatcher
    exists, absent on toolless configs, removed by the "none" header."""

    def test_any_enabled_tool_brings_substrate(self, registry):
        state = MagicMock()  # dispatcher attrs auto-truthy = substrate up
        state.tool_registry = registry
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search"
            result = _resolve_passthrough_tools(
                state, query="what is the capital of France?",
            )
        assert set(result) >= _SUBSTRATE
        assert "web_search" in result

    def test_query_never_gates(self, registry):
        # Identical toolset regardless of phrasing — the model decides.
        state = MagicMock()
        state.tool_registry = registry
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search"
            plain = _resolve_passthrough_tools(state, query="tell me a joke")
            sched = _resolve_passthrough_tools(state, query="wake me at 9")
        assert plain == sched

    def test_toolless_config_stays_toolless(self, registry):
        # No enabled tools -> nothing rides along (utilities included) —
        # the pure-streaming fast path is preserved, consistently.
        state = MagicMock()
        state.tool_registry = registry
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = ""
            result = _resolve_passthrough_tools(
                state, query="wake me at 9 every morning",
            )
        assert result == []

    def test_create_verbs_lead_the_injection_order(self, registry):
        # Tiny models bias toward earlier schema entries — create verbs
        # must precede list/cancel.
        state = MagicMock()
        state.tool_registry = registry
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search"
            result = _resolve_passthrough_tools(state, query="hello")
        assert result[0] == "schedule_briefing"
        assert result.index("watch_for") < result.index("cancel_briefing")

    def test_header_none_opts_out_of_substrate(self, registry):
        # The user's explicit "no tools" wins, like for every tool.
        state = MagicMock()
        state.tool_registry = registry
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search"
            result = _resolve_passthrough_tools(
                state, header_tools="none", query="wake me at 9",
            )
        assert result == []

    def test_no_dispatcher_never_injects(self, registry):
        state = MagicMock()
        state.tool_registry = registry
        state.companion_runtime = None
        state.scheduler_service = None
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search"
            result = _resolve_passthrough_tools(
                state, query="wake me at 9 every morning",
            )
        assert not (set(result) & _SUBSTRATE)
        assert "web_search" in result


# ---------------------------------------------------------------------------
# PassthroughHandler — no tools (unchanged behavior)
# ---------------------------------------------------------------------------


class TestPassthroughNoTools:
    @pytest.mark.asyncio
    async def test_pure_passthrough(self, backend):
        """Without tools, requests pass straight through."""
        backend.chat = AsyncMock(return_value=_make_response("Hi"))
        handler = PassthroughHandler(backend=backend)
        req = _make_request("Hello")
        resp = await handler.handle(req)
        assert resp.message.content == "Hi"
        backend.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pure_passthrough_stream(self, backend):
        """Without tools, streaming passes straight through (plus a
        leading "thinking" status chunk the UI uses to mark the
        streaming start)."""
        async def fake_stream(req):
            yield InternalStreamChunk(content_delta="Hi")
            yield InternalStreamChunk(content_delta=" there", done=True)

        backend.chat_stream = fake_stream
        handler = PassthroughHandler(backend=backend)
        req = _make_request("Hello")
        chunks = [c async for c in handler.handle_stream(req)]
        # Status chunk + 2 content chunks.
        assert len(chunks) == 3
        assert chunks[0].augmentum and chunks[0].augmentum.get("status") == "thinking"
        assert chunks[1].content_delta == "Hi"
        assert chunks[2].content_delta == " there"


# ---------------------------------------------------------------------------
# PassthroughHandler — with tools
# ---------------------------------------------------------------------------


class TestPassthroughWithTools:
    @pytest.mark.asyncio
    async def test_no_tool_call_passes_through(self, backend, registry):
        """When LLM doesn't call any tools, response passes through directly."""
        backend.chat = AsyncMock(return_value=_make_response("Just a text answer"))
        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        resp = await handler.handle(_make_request("What is 2+2?"))
        assert resp.message.content == "Just a text answer"

    @pytest.mark.asyncio
    async def test_single_tool_call(self, backend, registry):
        """LLM calls a tool, handler executes it, LLM gives final answer."""
        tool_call_resp = _make_response("", tool_calls=[{
            "id": "call_abc",
            "function": {"name": "web_search", "arguments": {"query": "weather today"}},
        }])
        final_resp = _make_response("The weather is sunny.")
        backend.chat = AsyncMock(side_effect=[tool_call_resp, final_resp])

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        resp = await handler.handle(_make_request("What's the weather?"))
        assert resp.message.content == "The weather is sunny."
        assert backend.chat.await_count == 2

    @pytest.mark.asyncio
    async def test_tool_result_has_tool_call_id(self, backend, registry):
        """Tool results include tool_call_id for backend compatibility."""
        tool_call_resp = _make_response("", tool_calls=[{
            "id": "call_xyz",
            "function": {"name": "calculator", "arguments": {"expression": "2+2"}},
        }])
        final_resp = _make_response("4")
        backend.chat = AsyncMock(side_effect=[tool_call_resp, final_resp])

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["calculator"],
        )
        req = _make_request("What is 2+2?")
        await handler.handle(req)

        # Check the second call had tool results with proper IDs
        second_call = backend.chat.call_args_list[1]
        messages = second_call[0][0].messages
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "call_xyz"
        assert "= 2+2" in tool_msgs[0].content

    @pytest.mark.asyncio
    async def test_assistant_message_appended_once_for_multiple_calls(self, backend, registry):
        """With multiple tool calls, assistant message should appear exactly once."""
        tool_call_resp = _make_response("", tool_calls=[
            {"id": "call_1", "function": {"name": "web_search", "arguments": {"query": "weather"}}},
            {"id": "call_2", "function": {"name": "calculator", "arguments": {"expression": "32*9/5+32"}}},
        ])
        final_resp = _make_response("Weather is 89.6F")
        backend.chat = AsyncMock(side_effect=[tool_call_resp, final_resp])

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search", "calculator"],
        )
        req = _make_request("Weather in F")
        await handler.handle(req)

        # Check message structure: user, assistant (once), tool1, tool2
        second_call = backend.chat.call_args_list[1]
        messages = second_call[0][0].messages
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(assistant_msgs) == 1  # NOT duplicated
        assert len(tool_msgs) == 2
        assert tool_msgs[0].tool_call_id == "call_1"
        assert tool_msgs[1].tool_call_id == "call_2"

    @pytest.mark.asyncio
    async def test_max_iterations_guard(self, backend, registry):
        """Handler stops after max iterations even if LLM keeps calling tools."""
        tool_call_resp = _make_response("", tool_calls=[{
            "id": "call_loop",
            "function": {"name": "web_search", "arguments": {"query": "loop"}},
        }])
        backend.chat = AsyncMock(return_value=tool_call_resp)

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )

        # The budget is resolved per-turn now (live setting + per-user
        # preference + per-request override) instead of being read from a
        # module constant at import time, so patch the resolver.
        with patch(
            "augmentum.modes.passthrough.handler._max_iterations",
            return_value=3,
        ):
            resp = await handler.handle(_make_request("Loop forever"))

        assert backend.chat.await_count == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_handled_gracefully(self, backend, registry):
        """If LLM calls a tool that doesn't exist, error message is injected."""
        tool_call_resp = _make_response("", tool_calls=[{
            "id": "call_bad",
            "function": {"name": "nonexistent_tool", "arguments": {}},
        }])
        final_resp = _make_response("Sorry, couldn't do that.")
        backend.chat = AsyncMock(side_effect=[tool_call_resp, final_resp])

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        resp = await handler.handle(_make_request("Do something"))

        second_call = backend.chat.call_args_list[1]
        messages = second_call[0][0].messages
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert any("Unknown tool" in m.content for m in tool_msgs)

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, backend, registry):
        """Tool execution failures are caught and injected as error messages."""
        async def failing_execute(self, **kwargs):
            raise RuntimeError("Connection refused")

        tool_call_resp = _make_response("", tool_calls=[{
            "id": "call_err",
            "function": {"name": "web_search", "arguments": {"query": "test"}},
        }])
        final_resp = _make_response("Search failed, here's my best guess.")
        backend.chat = AsyncMock(side_effect=[tool_call_resp, final_resp])

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )

        with patch.object(FakeSearchTool, "execute", failing_execute):
            resp = await handler.handle(_make_request("Search for something"))

        assert resp.message.content == "Search failed, here's my best guess."

    @pytest.mark.asyncio
    async def test_generated_tool_call_id_when_missing(self, backend, registry):
        """Handler generates a tool_call_id when the LLM doesn't provide one."""
        tool_call_resp = _make_response("", tool_calls=[{
            "function": {"name": "web_search", "arguments": {"query": "test"}},
            # No "id" field
        }])
        final_resp = _make_response("Done")
        backend.chat = AsyncMock(side_effect=[tool_call_resp, final_resp])

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        req = _make_request("test")
        await handler.handle(req)

        second_call = backend.chat.call_args_list[1]
        messages = second_call[0][0].messages
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id is not None
        assert tool_msgs[0].tool_call_id.startswith("call_")


# ---------------------------------------------------------------------------
# Streaming with tools — no double generation
# ---------------------------------------------------------------------------


class TestPassthroughStreamWithTools:
    """Native tier (cloud APIs + modern local models): stream-first path.

    The handler opens ``chat_stream`` from the very first call. Native
    tool calling guarantees content_delta and tool_calls don't
    interleave, so we forward content chunks immediately while
    accumulating tool_call deltas separately. After the stream ends,
    if tool calls accumulated we execute them and the next iteration
    streams the composed answer naturally.

    Backend fixture (line 86) uses ``api.openai.com`` so
    ``select_tier`` resolves to NATIVE.
    """

    @staticmethod
    def _make_native_stream(*responses):
        """Build a fake chat_stream that returns one of ``responses`` per call.

        Each response is a list of (kind, payload) tuples where kind is
        ``"content"``, ``"tool_calls"``, or ``"done"``. Yields one
        InternalStreamChunk per tuple, mimicking the SSE shape that
        OpenAIBackend.chat_stream produces.
        """
        call_count = {"i": 0}
        responses_list = list(responses)

        async def fake_stream(req):
            idx = call_count["i"]
            if idx >= len(responses_list):
                # Defensive: caller invoked chat_stream more times than
                # we have canned responses — yield an empty done chunk.
                yield InternalStreamChunk(content_delta="", done=True)
                return
            call_count["i"] += 1
            for kind, payload in responses_list[idx]:
                if kind == "content":
                    yield InternalStreamChunk(content_delta=payload)
                elif kind == "tool_calls":
                    yield InternalStreamChunk(
                        content_delta="",
                        augmentum={"tool_calls": payload},
                    )
                elif kind == "done":
                    yield InternalStreamChunk(
                        content_delta="",
                        done=True,
                        finish_reason=payload or "stop",
                    )

        return fake_stream

    @pytest.mark.asyncio
    async def test_stream_final_answer_after_tools(self, backend, registry):
        """After tool calls resolve, the composed final answer streams
        as separate content chunks (not a single dump).

        Native tier: the first chat_stream yields tool_call deltas (no
        content). After tools execute, the second chat_stream yields
        the composed answer progressively.
        """
        backend.chat_stream = self._make_native_stream(
            # First call: tool call request, no content
            [
                ("tool_calls", [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }]),
                ("done", "tool_calls"),
            ],
            # Second call: composed answer streams
            [
                ("content", "The answer "),
                ("content", "is 42."),
                ("done", "stop"),
            ],
        )

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        chunks = [c async for c in handler.handle_stream(_make_request("Test"))]

        # Native tier never calls chat() — everything goes through chat_stream.
        assert backend.chat.await_count == 0

        # Final content is progressively streamed (not a single dump).
        content_chunks = [c for c in chunks if c.content_delta]
        progressive_texts = [c.content_delta for c in content_chunks]
        assert "The answer " in progressive_texts
        assert "is 42." in progressive_texts
        # Verify they arrived as separate chunks (the streaming win).
        assert len([t for t in progressive_texts if t == "The answer "]) == 1
        assert len([t for t in progressive_texts if t == "is 42."]) == 1

    @pytest.mark.asyncio
    async def test_stream_emits_tool_status(self, backend, registry):
        """Streaming should emit tool status metadata chunks."""
        backend.chat_stream = self._make_native_stream(
            [
                ("tool_calls", [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }]),
                ("done", "tool_calls"),
            ],
            [
                ("content", "Done."),
                ("done", "stop"),
            ],
        )

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        chunks = [c async for c in handler.handle_stream(_make_request("Test"))]

        status_chunks = [
            c for c in chunks if c.augmentum and "tool_status" in c.augmentum
        ]
        assert len(status_chunks) >= 1
        assert "web_search" in status_chunks[0].augmentum["tool_names"]

    @pytest.mark.asyncio
    async def test_stream_direct_answer_streams_progressively(self, backend, registry):
        """Native tier + tools enabled + model answers directly (no tool
        calls): content streams as separate chunks, NOT as one giant
        dump.

        Regression for the original cloud-DeepSeek complaint — under
        the old peek-then-stream path, every direct answer arrived as
        a single chunk because we needed the full response to parse
        for tool calls first. Stream-first path forwards content
        immediately because native tool_calls land in a separate
        SSE field that doesn't interleave with content.
        """
        backend.chat_stream = self._make_native_stream(
            [
                ("content", "Two plus two "),
                ("content", "equals four."),
                ("done", "stop"),
            ],
        )

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search", "calculator"],  # tools enabled, won't be called
        )
        chunks = [c async for c in handler.handle_stream(_make_request("What's 2+2?"))]

        # Non-streaming chat is NEVER called in the native path.
        assert backend.chat.await_count == 0

        # Each content chunk arrives separately — the streaming win.
        content_chunks = [c.content_delta for c in chunks if c.content_delta]
        assert content_chunks.count("Two plus two ") == 1
        assert content_chunks.count("equals four.") == 1

    @pytest.mark.asyncio
    async def test_stream_emits_done_chunk(self, backend, registry):
        """Streaming final chunk should have done=True."""
        backend.chat_stream = self._make_native_stream(
            [
                ("tool_calls", [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"test"}'},
                }]),
                ("done", "tool_calls"),
            ],
            [
                ("content", "Done."),
                ("done", "stop"),
            ],
        )

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        chunks = [c async for c in handler.handle_stream(_make_request("Test"))]

        done_chunks = [c for c in chunks if c.done]
        assert len(done_chunks) >= 1

    @pytest.mark.asyncio
    async def test_stream_multi_chunk_tool_call_arguments(self, backend, registry):
        """Tool call arguments split across multiple SSE deltas should
        merge into a single tool call (native streaming spec)."""
        backend.chat_stream = self._make_native_stream(
            [
                # First chunk has name + id
                ("tool_calls", [{
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"que'},
                }]),
                # Second chunk has just arguments fragment
                ("tool_calls", [{
                    "index": 0,
                    "function": {"arguments": 'ry":"test"}'},
                }]),
                ("done", "tool_calls"),
            ],
            [
                ("content", "Found it."),
                ("done", "stop"),
            ],
        )

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        chunks = [c async for c in handler.handle_stream(_make_request("Test"))]

        # Tool was actually invoked (status emitted means parsing succeeded).
        status_chunks = [c for c in chunks if c.augmentum and "tool_status" in c.augmentum]
        assert len(status_chunks) == 1
        # Final answer came through.
        content_chunks = [c.content_delta for c in chunks if c.content_delta]
        assert "Found it." in content_chunks

    @pytest.mark.asyncio
    async def test_stream_no_tools_passes_through(self, backend, registry):
        """Without tools, streaming stays pure passthrough (no chunking hack).

        The no-tools branch emits a "thinking" status chunk first
        (signals to the UI that streaming is starting) followed by the
        backend's content chunks. The chunks themselves pass through
        unchanged — no buffering, no peek, no transformation.
        """
        async def fake_stream(req):
            yield InternalStreamChunk(content_delta="Token1 ")
            yield InternalStreamChunk(content_delta="Token2", done=True)

        backend.chat_stream = fake_stream
        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=[],  # no tools
        )
        chunks = [c async for c in handler.handle_stream(_make_request("Hi"))]
        # Status chunk + 2 content chunks
        assert len(chunks) == 3
        assert chunks[0].augmentum and chunks[0].augmentum.get("status") == "thinking"
        assert chunks[1].content_delta == "Token1 "
        assert chunks[2].content_delta == "Token2"


# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------


class TestToolResolution:
    def test_resolve_by_name(self, registry):
        handler = PassthroughHandler(
            backend=AsyncMock(),
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        tools = handler._resolve_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    def test_resolve_by_alias(self, registry):
        handler = PassthroughHandler(
            backend=AsyncMock(),
            tool_registry=registry,
            enabled_tools=["search"],
        )
        tools = handler._resolve_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    def test_resolve_unknown_tool_skipped(self, registry):
        handler = PassthroughHandler(
            backend=AsyncMock(),
            tool_registry=registry,
            enabled_tools=["web_search", "nonexistent"],
        )
        tools = handler._resolve_tools()
        assert len(tools) == 1

    def test_no_registry_returns_empty(self):
        handler = PassthroughHandler(
            backend=AsyncMock(),
            tool_registry=None,
            enabled_tools=["web_search"],
        )
        tools = handler._resolve_tools()
        assert tools == []

    def test_dedup_same_tool_via_alias(self, registry):
        """Resolve doesn't return the same tool twice even via aliases."""
        handler = PassthroughHandler(
            backend=AsyncMock(),
            tool_registry=registry,
            enabled_tools=["web_search", "search"],  # both resolve to web_search
        )
        tools = handler._resolve_tools()
        assert len(tools) == 1


# ---------------------------------------------------------------------------
# Text tier prompt building
# ---------------------------------------------------------------------------


class TestTextToolPrompt:
    def test_builds_prompt(self, registry):
        handler = PassthroughHandler(
            backend=AsyncMock(),
            tool_registry=registry,
            enabled_tools=["web_search", "calculator"],
        )
        tools = handler._resolve_tools()
        prompt = handler._build_text_tool_prompt(tools)
        assert "web_search" in prompt
        assert "calculator" in prompt
        assert "TOOL_CALL:" in prompt
        assert "TOOL_INPUT:" in prompt


# ---------------------------------------------------------------------------
# Message dataclass tool_call_id
# ---------------------------------------------------------------------------


class TestMessageToolCallId:
    def test_default_is_none(self):
        msg = Message(role="tool", content="result")
        assert msg.tool_call_id is None

    def test_can_set_tool_call_id(self):
        msg = Message(role="tool", content="result", tool_call_id="call_123")
        assert msg.tool_call_id == "call_123"


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------


class TestPassthroughMetrics:
    @pytest.mark.asyncio
    async def test_successful_tool_records_metric(self, backend, registry):
        tool_call_resp = _make_response("", tool_calls=[{
            "id": "call_m",
            "function": {"name": "web_search", "arguments": {"query": "test"}},
        }])
        final_resp = _make_response("Done")
        backend.chat = AsyncMock(side_effect=[tool_call_resp, final_resp])

        handler = PassthroughHandler(
            backend=backend,
            tool_registry=registry,
            enabled_tools=["web_search"],
        )
        await handler.handle(_make_request("test"))

        metrics = registry.metrics.snapshot()
        assert "web_search" in metrics
        assert metrics["web_search"]["calls"] == 1
        assert metrics["web_search"]["successes"] == 1


# ---------------------------------------------------------------------------
# Stale tool result condensation
# ---------------------------------------------------------------------------


class TestCondenseStaleToolResults:
    """Test _condense_stale_tool_results to prevent LLMs re-triggering tools."""

    def test_no_tool_messages_is_noop(self):
        """No tool messages — nothing to condense."""
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        PassthroughHandler._condense_stale_tool_results(msgs)
        assert msgs[0].content == "Hello"
        assert msgs[1].content == "Hi there"

    def test_current_turn_tool_results_preserved(self):
        """Tool results not yet followed by assistant are kept intact."""
        msgs = [
            Message(role="user", content="Search for cats"),
            Message(role="assistant", content="", tool_calls=[{
                "id": "c1", "function": {"name": "web", "arguments": "{}"},
            }]),
            Message(role="tool", content="A" * 500, tool_call_id="c1"),
        ]
        PassthroughHandler._condense_stale_tool_results(msgs)
        # Tool result should be untouched — no assistant response after it
        assert len(msgs[2].content) == 500

    def test_stale_tool_results_condensed(self):
        """Tool results from prior turns (followed by assistant) are condensed."""
        long_result = "Search result: " + "x" * 500
        msgs = [
            Message(role="user", content="What is Python?"),
            Message(role="assistant", content="", tool_calls=[{
                "id": "c1", "function": {"name": "web", "arguments": "{}"},
            }]),
            Message(role="tool", content=long_result, tool_call_id="c1"),
            Message(role="assistant", content="Python is a programming language."),
            Message(role="user", content="Tell me more about its history"),
        ]
        PassthroughHandler._condense_stale_tool_results(msgs)
        # The tool result at index 2 should be condensed
        assert "omitted" in msgs[2].content.lower()
        assert len(msgs[2].content) < 200
        # tool_call_id preserved
        assert msgs[2].tool_call_id == "c1"

    def test_short_tool_results_not_condensed(self):
        """Short tool results (<= 200 chars) are left alone even if stale."""
        msgs = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="", tool_calls=[{
                "id": "c1", "function": {"name": "calc", "arguments": "{}"},
            }]),
            Message(role="tool", content="= 4", tool_call_id="c1"),
            Message(role="assistant", content="The answer is 4."),
            Message(role="user", content="And 3+3?"),
        ]
        PassthroughHandler._condense_stale_tool_results(msgs)
        assert msgs[2].content == "= 4"

    def test_text_tier_tool_results_condensed(self):
        """Text-tier tool results (role=user with ## Tool Result) are condensed."""
        long_result = "## Tool Result (web_search)\n" + "x" * 500
        msgs = [
            Message(role="user", content="Search for cats"),
            Message(role="user", content=long_result),
            Message(role="assistant", content="Here are the results about cats."),
            Message(role="user", content="Now search for dogs"),
        ]
        PassthroughHandler._condense_stale_tool_results(msgs)
        assert "omitted" in msgs[1].content.lower()

    def test_multiple_turns_condensed(self):
        """Multiple prior turns with tool results are all condensed."""
        long = "R" * 300
        msgs = [
            Message(role="user", content="Q1"),
            Message(role="assistant", content="", tool_calls=[{"id": "c1", "function": {"name": "web", "arguments": "{}"}}]),
            Message(role="tool", content=long, tool_call_id="c1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
            Message(role="assistant", content="", tool_calls=[{"id": "c2", "function": {"name": "web", "arguments": "{}"}}]),
            Message(role="tool", content=long, tool_call_id="c2"),
            Message(role="assistant", content="A2"),
            Message(role="user", content="Q3"),
        ]
        PassthroughHandler._condense_stale_tool_results(msgs)
        assert "omitted" in msgs[2].content.lower()
        assert "omitted" in msgs[6].content.lower()

    def test_no_assistant_response_means_nothing_stale(self):
        """If there's no assistant response at all, nothing is stale."""
        msgs = [
            Message(role="user", content="Search"),
            Message(role="tool", content="A" * 500, tool_call_id="c1"),
        ]
        PassthroughHandler._condense_stale_tool_results(msgs)
        assert len(msgs[1].content) == 500

    def test_role_preserved_after_condensation(self):
        """Condensed messages keep their original role."""
        long = "X" * 300
        msgs = [
            Message(role="user", content="Q"),
            Message(role="tool", content=long, tool_call_id="c1"),
            Message(role="assistant", content="A"),
            Message(role="user", content="Q2"),
        ]
        PassthroughHandler._condense_stale_tool_results(msgs)
        assert msgs[1].role == "tool"


class TestEmptyHeaderIsExplicitNone:
    """The chat UI sends X-Augmentum-Tools: "" when the user toggles all
    tools off. That is an explicit choice — it must NOT fall back to
    config defaults or pull in ride-alongs (parity with analytical,
    which already treated an empty selector as none). Caught live
    2026-07-02: schedule tools appeared with tools fully disabled."""

    def test_empty_header_beats_config_defaults_and_ridealongs(self, registry):
        state = MagicMock()  # dispatcher attrs auto-truthy = substrate up
        state.tool_registry = registry
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search,calculator"
            result = _resolve_passthrough_tools(
                state, header_tools="", query="wake me at 9",
            )
        assert result == []

    def test_absent_header_still_uses_config_defaults(self, registry):
        # Headerless API clients keep the config-default behavior.
        state = MagicMock()
        state.companion_runtime = None
        state.scheduler_service = None
        with patch("augmentum.proxy.handler_factory.settings") as s:
            s.passthrough_tools = "web_search"
            state.tool_registry = registry
            result = _resolve_passthrough_tools(state, header_tools=None)
        assert "web_search" in result
