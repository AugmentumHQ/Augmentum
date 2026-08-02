"""Tests for the narrative recall tool-execution loop.

Verifies the orchestration contract:

* No tool_calls emitted → loop yields all buffered chunks and terminates
  on first iteration (zero overhead when the model doesn't reach for
  recall).
* One recall tool_call → dispatcher fires, ``tool`` message appears in
  the next inner request's history, second iteration's content reaches
  the consumer.
* Dispatcher exception → tool result becomes an error string, model
  sees it and can recover (no loop crash).
* Hard iter cap → loop force-terminates with a ``tool_loop_cap`` meta
  chunk instead of recursing forever on a stuck model.
* Non-recall tool_calls in the same request → passed through
  unmodified (the loop only owns the recall verbs).

Mock backend emits scripted streams keyed on iteration count, so we
can simulate the exact "tool-call → execute → final-content" sequence
without needing a live LLM.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalStreamChunk,
    Message,
)
from augmentum.modes.narrative.recall_loop import stream_with_recall_tools

# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class _ScriptedBackend:
    """Backend that yields a sequence of chunks per chat_stream call.

    Each call consumes one entry from ``scripts`` (a list of chunk
    lists). Records which inner requests it received so tests can
    assert message history correctness.
    """

    def __init__(self, scripts: list[list[InternalStreamChunk]]):
        self._scripts = list(scripts)
        self.calls_received: list[InternalChatRequest] = []

    async def chat_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        self.calls_received.append(request)
        if not self._scripts:
            return
        chunks = self._scripts.pop(0)
        for c in chunks:
            yield c


def _content_chunk(text: str, *, done: bool = False) -> InternalStreamChunk:
    return InternalStreamChunk(content_delta=text, done=done)


def _tool_call_chunk(
    *, idx: int = 0, tc_id: str = "call_1", name: str = "recall_entity",
    args_chunk: str = '',
) -> InternalStreamChunk:
    """Build one tool_calls-delta-bearing chunk."""
    delta: dict = {"index": idx}
    if tc_id:
        delta["id"] = tc_id
    fn: dict = {}
    if name:
        fn["name"] = name
    if args_chunk:
        fn["arguments"] = args_chunk
    if fn:
        delta["function"] = fn
    return InternalStreamChunk(
        content_delta="",
        augmentum={"tool_calls": [delta]},
    )


def _base_request() -> InternalChatRequest:
    return InternalChatRequest(
        model="mock-model",
        messages=[
            Message(role="system", content="You are a narrator."),
            Message(role="user", content="Where is Elena right now?"),
        ],
        tools=[{"type": "function", "function": {"name": "recall_entity"}}],
    )


# ---------------------------------------------------------------------------
# Loop behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tool_calls_passes_through_first_iteration():
    """No tool_calls emitted → first iteration's chunks reach the
    consumer verbatim, no extra backend calls."""
    backend = _ScriptedBackend([
        [_content_chunk("Elena is "), _content_chunk("kneeling in the south wing.", done=True)],
    ])
    dispatcher_calls: list = []

    async def dispatcher(name: str, args):
        dispatcher_calls.append((name, args))
        return "should not be called"

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    assert dispatcher_calls == []
    assert len(backend.calls_received) == 1
    visible = "".join(c.content_delta for c in received)
    assert "Elena is kneeling in the south wing." in visible


@pytest.mark.asyncio
async def test_one_recall_call_then_content():
    """Iter 1: tool_call + done. Iter 2: content + done. Verify
    dispatcher fires, tool message lands in iter-2 history, content
    reaches consumer."""
    backend = _ScriptedBackend([
        # Iter 1 — emit a tool_call across two deltas, then stop.
        [
            _tool_call_chunk(idx=0, tc_id="c1", name="recall_entity", args_chunk='{"na'),
            _tool_call_chunk(idx=0, tc_id="", name="", args_chunk='me":"Elena"}'),
            InternalStreamChunk(content_delta="", done=True, finish_reason="tool_calls"),
        ],
        # Iter 2 — model uses the tool result to produce content.
        [
            _content_chunk("Based on my notes, "),
            _content_chunk("Elena is in the south wing.", done=True),
        ],
    ])

    dispatcher_calls: list = []

    async def dispatcher(name: str, args):
        dispatcher_calls.append((name, args))
        return "Elena is in the south wing, kneeling."

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    # Dispatcher fired exactly once with the assembled args.
    assert dispatcher_calls == [("recall_entity", '{"name":"Elena"}')]
    # Both iterations ran.
    assert len(backend.calls_received) == 2
    # The iter-2 inner request must include the assistant+tool messages.
    iter2_msgs = backend.calls_received[1].messages
    roles = [m.role for m in iter2_msgs]
    assert roles[-2:] == ["assistant", "tool"]
    assert iter2_msgs[-1].tool_call_id == "c1"
    assert "south wing, kneeling" in iter2_msgs[-1].content
    # User-visible content is the iter-2 prose.
    visible = "".join(c.content_delta for c in received)
    assert "Based on my notes" in visible
    assert "south wing" in visible
    # And there must be a meta chunk so the UI can render a status hint.
    metas = [c for c in received if c.augmentum and c.augmentum.get("status") == "narrative_tool_used"]
    assert len(metas) == 1
    assert metas[0].augmentum["tool"] == "recall_entity"


@pytest.mark.asyncio
async def test_dispatcher_exception_becomes_tool_error_for_model():
    """A raise from the dispatcher must NOT crash the loop. The error
    text becomes the tool_result so the model can recover."""
    backend = _ScriptedBackend([
        [
            _tool_call_chunk(idx=0, tc_id="c1", name="recall_entity", args_chunk='{"name":"X"}'),
            InternalStreamChunk(content_delta="", done=True, finish_reason="tool_calls"),
        ],
        [_content_chunk("I'll improvise.", done=True)],
    ])

    async def dispatcher(name: str, args):
        raise RuntimeError("boom")

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    # Iter 2 must have seen the error string as the tool message body.
    iter2_msgs = backend.calls_received[1].messages
    last = iter2_msgs[-1]
    assert last.role == "tool"
    assert "recall_tool_error" in last.content
    assert "RuntimeError" in last.content
    # Meta chunk surfaced the dispatch failure.
    metas = [c for c in received if c.augmentum and c.augmentum.get("status") == "narrative_tool_used"]
    assert len(metas) == 1
    assert metas[0].augmentum["args"].get("_dispatch_error") is True


@pytest.mark.asyncio
async def test_max_iters_force_terminates_with_meta():
    """A stuck model that keeps calling recall but never produces
    content must hit the iter cap and surface a clear meta chunk."""
    # Emit the same tool_call every iteration, never finishing with
    # plain content. Loop should bail at max_iters.
    def _stuck_iter():
        return [
            _tool_call_chunk(idx=0, tc_id="c1", name="recall_entity",
                             args_chunk='{"name":"Elena"}'),
            InternalStreamChunk(content_delta="", done=True, finish_reason="tool_calls"),
        ]

    backend = _ScriptedBackend([_stuck_iter() for _ in range(5)])

    async def dispatcher(name: str, args):
        return "Some entity info."

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher, max_iters=2,
    ):
        received.append(chunk)

    # max_iters loop passes (2) + one forced tool-free synthesis pass (1).
    # The scripted backend ignores tool-stripping and keeps emitting a
    # tool_call, so the synthesis pass yields no prose and the loop falls
    # through to the terminal cap marker.
    assert len(backend.calls_received) == 3
    assert any(
        c.augmentum and c.augmentum.get("status") == "recall_tool_loop_synthesizing"
        for c in received
    )
    # Final chunk surfaces the cap (synthesis produced nothing here).
    last_meta = [c for c in received if c.augmentum and c.augmentum.get("status") == "recall_tool_loop_max_iters"]
    assert last_meta
    assert last_meta[-1].finish_reason == "tool_loop_cap"


@pytest.mark.asyncio
async def test_cap_synthesis_pass_writes_story_when_tools_drop():
    """When the model loops on reads to the cap, the forced tool-free
    synthesis pass must deliver prose (not an empty turn)."""
    def _stuck_iter():
        return [
            _tool_call_chunk(idx=0, tc_id="c1", name="recall_entity",
                             args_chunk='{"name":"Elena"}'),
            InternalStreamChunk(content_delta="", done=True, finish_reason="tool_calls"),
        ]

    def _synth_iter():
        return [
            InternalStreamChunk(content_delta="Elena steps into the hall."),
            InternalStreamChunk(content_delta="", done=True, finish_reason="stop"),
        ]

    backend = _ScriptedBackend([_stuck_iter(), _stuck_iter(), _synth_iter()])

    async def dispatcher(name: str, args):
        return "Some entity info."

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher, max_iters=2,
    ):
        received.append(chunk)

    # 2 loop passes + 1 synthesis pass; the synthesis pass had tools stripped.
    assert len(backend.calls_received) == 3
    assert backend.calls_received[-1].tools is None
    assert backend.calls_received[-1].tool_choice == "none"
    # Prose delivered; no terminal "empty" marker.
    content = "".join(c.content_delta for c in received if c.content_delta)
    assert "Elena steps into the hall." in content
    assert not any(
        c.augmentum and c.augmentum.get("status") == "recall_tool_loop_max_iters"
        for c in received
    )


@pytest.mark.asyncio
async def test_turn_summary_event_emits_with_full_shape(capsys):
    """The narrative_recall_turn log event is the measurement plan —
    it has to fire on every terminating turn (success OR cap) with
    the structured fields a downstream dashboard needs. Pin the shape
    so a future refactor can't silently drop a field."""
    # structlog routes through stdout in this codebase, not stdlib
    # logging, so capsys captures the event line directly. We assert
    # on the field-name strings to avoid parsing structlog's output
    # format (which changes between dev/prod renderers).
    backend = _ScriptedBackend([
        [
            _tool_call_chunk(idx=0, tc_id="c1", name="recall_entity",
                             args_chunk='{"name":"Elena"}'),
            InternalStreamChunk(content_delta="", done=True, finish_reason="tool_calls"),
        ],
        [_content_chunk("Elena is grieving.", done=True)],
    ])

    async def dispatcher(name: str, args):
        return "Elena state: grieving."

    received = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    out = capsys.readouterr().out  # structlog renders to stdout in this codebase
    # The required field-set — every one of these is consumed by a
    # downstream metric we promised in the audit. Assert the string
    # appears in the captured output to avoid coupling to renderer
    # format (structlog can switch JSON / KV / console).
    required_fields = [
        "iterations", "tool_calls", "tool_calls_per_name",
        "tool_latency_ms_avg", "tool_latency_ms_max",
        "tool_result_chars_total", "tool_dispatch_errors",
        "non_recall_dropped", "max_iters_hit", "finish_reason",
        "turn_total_ms",
    ]
    assert "narrative_recall_turn" in out, (
        f"summary event missing from captured output:\n{out}"
    )
    missing = [f for f in required_fields if f not in out]
    assert not missing, (
        f"narrative_recall_turn missing required fields: {missing}. "
        "Update _TurnMetrics.as_payload to include them."
    )


@pytest.mark.asyncio
async def test_content_streams_incrementally_not_buffered():
    """REGRESSION GUARD (2026-06-30 blob fix, amended 2026-07-15 for the
    pre-call gate). Content BEYOND the gate budget must reach the consumer
    AS the inner stream produces it — never buffered to end-of-stream (the
    blob the user saw). The gate is allowed to hold only the first
    ``_PREGATE_MAX_CHARS`` of an iteration; a first chunk that crosses the
    budget must be flushed IMMEDIATELY, before the backend produces the
    next chunk. Detected via production / consumption interleaving.
    """
    from augmentum.modes.narrative.recall_loop import _PREGATE_MAX_CHARS

    class _ProbeBackend:
        def __init__(self, chunks):
            self._chunks = chunks
            self.produced: list = []

        async def chat_stream(self, request):  # noqa: ANN001
            for c in self._chunks:
                self.produced.append(c)
                yield c

    opener = "one " * (_PREGATE_MAX_CHARS // 4 + 1)   # crosses the gate alone
    chunks = [
        _content_chunk(opener),
        _content_chunk("two "),
        _content_chunk("three", done=True),
    ]
    backend = _ProbeBackend(chunks)

    async def dispatcher(name: str, args):
        return ""

    gen = stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    )
    first = await gen.__anext__()

    assert len(backend.produced) == 1, (
        f"recall_loop buffered the inner stream "
        f"({len(backend.produced)} chunks produced before the consumer saw "
        "the first) — the blob regression is back. Content past the gate "
        "budget must stream live."
    )
    assert first.content_delta == opener

    rest = [c async for c in gen]
    visible = first.content_delta + "".join(c.content_delta for c in rest)
    assert opener + "two three" in visible


def test_narrative_engine_exposes_public_persistence_accessor():
    """The handler-side recall wiring used to access
    ``engine._persistence`` — a leading-underscore private attribute
    crossing module boundaries. The public ``persistence`` property
    on NarrativeEngine is the supported coupling; if a future refactor
    renames the private storage, this property keeps the contract.

    The test exists so a refactor that drops the property breaks
    here, not later in a recall-feature regression."""
    from augmentum.modes.narrative.engine import NarrativeEngine

    # Property must exist at the class level — not just the instance.
    assert isinstance(
        getattr(NarrativeEngine, "persistence", None), property
    ), (
        "NarrativeEngine.persistence property missing. The recall wiring "
        "in handler.py relies on this public accessor — don't remove it "
        "without updating the handler to a new public surface."
    )


@pytest.mark.asyncio
async def test_non_recall_tool_call_passes_through_meta():
    """If the model calls a tool name that isn't a recall verb, the
    loop must NOT execute it — surface a meta chunk so the outer
    handler / log sees it was ignored."""
    backend = _ScriptedBackend([
        [
            _tool_call_chunk(idx=0, tc_id="c1", name="some_other_tool",
                             args_chunk='{"x":1}'),
            InternalStreamChunk(content_delta="Final answer.", done=True),
        ],
    ])

    async def dispatcher(name: str, args):
        # Should never be invoked since this isn't a recall verb.
        raise AssertionError(f"dispatcher should not fire for {name}")

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    # Loop terminated on first iteration (no recall tool_calls present).
    assert len(backend.calls_received) == 1
    # The buffered final-iteration content was emitted to the consumer.
    visible = "".join(c.content_delta for c in received)
    assert "Final answer." in visible


# ---------------------------------------------------------------------------
# Pre-call gate tests (2026-07-15 — the DeepSeek tool-plan-preamble class)
# ---------------------------------------------------------------------------

_PREAMBLE = "Let me check whether there's a lorebook entry about the manor."
_STORY = (
    "The manor doors gave way with a groan, and Elena stepped into a hall "
    "that remembered music. Dust hung where chandeliers had burned, and "
    "somewhere above, a floorboard confessed to someone's weight."
)
# Long enough to cross _PREGATE_MAX_CHARS (600) in one piece.
_LONG_STORY = " ".join([_STORY] * 4)


@pytest.mark.asyncio
async def test_preamble_before_tool_call_is_suppressed():
    """Content emitted BEFORE an internal tool call is a plan preamble —
    it must not reach the visible stream, and the suppression must be
    surfaced via a meta chunk (never silent)."""
    backend = _ScriptedBackend([
        [
            _content_chunk(_PREAMBLE),
            _tool_call_chunk(idx=0, tc_id="c1", name="recall_entity",
                             args_chunk='{"name": "manor"}'),
        ],
        [_content_chunk(_STORY, done=True)],
    ])

    async def dispatcher(name: str, args):
        return "The manor: abandoned since the fire of 1802."

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    visible = "".join(c.content_delta for c in received)
    assert _PREAMBLE not in visible
    assert _STORY in visible
    metas = [c.augmentum for c in received if c.augmentum]
    assert any(
        m.get("status") == "tool_preamble_suppressed"
        and m.get("chars") == len(_PREAMBLE)
        for m in metas
    )
    # Working history stays provider-faithful: the assistant tool_call
    # turn keeps the preamble text the model actually produced.
    second_request = backend.calls_received[1]
    assistant_turns = [m for m in second_request.messages if m.role == "assistant"]
    assert any(_PREAMBLE in (m.content or "") for m in assistant_turns)


@pytest.mark.asyncio
async def test_short_no_tool_reply_is_fully_delivered():
    """A reply shorter than the gate budget with no tool calls must be
    flushed at stream end — nothing may be lost to the gate."""
    short = "She nodded once."
    backend = _ScriptedBackend([
        [_content_chunk(short, done=True)],
    ])

    async def dispatcher(name: str, args):
        raise AssertionError("no tools in this script")

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    visible = "".join(c.content_delta for c in received)
    assert visible == short


@pytest.mark.asyncio
async def test_long_story_passes_gate_and_streams_in_order():
    """Once cumulative content crosses the gate budget, everything
    streams live and in order — the anti-blob behavior is preserved."""
    pieces = [_LONG_STORY[i:i + 40] for i in range(0, len(_LONG_STORY), 40)]
    chunks = [_content_chunk(p) for p in pieces]
    chunks[-1] = _content_chunk(pieces[-1], done=True)
    backend = _ScriptedBackend([chunks])

    async def dispatcher(name: str, args):
        raise AssertionError("no tools in this script")

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    visible = "".join(c.content_delta for c in received)
    assert visible == _LONG_STORY


@pytest.mark.asyncio
async def test_thinking_passes_gate_live_and_survives_preamble_drop():
    """Thinking deltas stream immediately while the gate holds content,
    and thinking bundled INSIDE held content chunks is re-emitted when
    the preamble is dropped."""
    mixed = InternalStreamChunk(
        content_delta=_PREAMBLE, thinking_delta="planning the check",
    )
    backend = _ScriptedBackend([
        [
            InternalStreamChunk(thinking_delta="reading the scene"),
            mixed,
            _tool_call_chunk(idx=0, tc_id="c1", name="recall_entity",
                             args_chunk="{}"),
        ],
        [_content_chunk(_STORY, done=True)],
    ])

    async def dispatcher(name: str, args):
        return "entry text"

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    visible = "".join(c.content_delta for c in received)
    thinking = "".join(c.thinking_delta for c in received)
    assert _PREAMBLE not in visible
    assert _STORY in visible
    assert "reading the scene" in thinking
    assert "planning the check" in thinking


@pytest.mark.asyncio
async def test_post_gate_tool_call_keeps_streamed_prose():
    """Documented limitation: prose that crossed the gate BEFORE a tool
    call stays visible (no retraction of already-streamed text) — the
    conduct directive owns that case, not the gate."""
    backend = _ScriptedBackend([
        [
            _content_chunk(_LONG_STORY),   # > gate budget, streams live
            _tool_call_chunk(idx=0, tc_id="c1", name="recall_entity",
                             args_chunk="{}"),
        ],
        [_content_chunk(" The rest followed.", done=True)],
    ])

    async def dispatcher(name: str, args):
        return "entry text"

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(), backend=backend, dispatcher=dispatcher,
    ):
        received.append(chunk)

    visible = "".join(c.content_delta for c in received)
    assert _LONG_STORY in visible
    assert "The rest followed." in visible


@pytest.mark.asyncio
async def test_prose_before_write_call_is_flushed_not_dropped():
    """Read/write asymmetry: prose held by the gate is STORY when the
    pass ends in a mutating call (the model wrote the scene, then
    recorded the fact it established) — it must be flushed visible,
    with no suppression meta."""
    backend = _ScriptedBackend([
        [
            _content_chunk(_STORY),   # < gate budget, held
            _tool_call_chunk(idx=0, tc_id="c1", name="lorebook_create",
                             args_chunk='{"name": "manor fire"}'),
        ],
        [_content_chunk(" She wrote it down.", done=True)],
    ])

    async def dispatcher(name: str, args):
        return "created"

    received: list[InternalStreamChunk] = []
    async for chunk in stream_with_recall_tools(
        _base_request(),
        backend=backend,
        dispatcher=dispatcher,
        internal_tool_names=frozenset({"recall_entity", "lorebook_create"}),
        internal_write_names=frozenset({"lorebook_create"}),
    ):
        received.append(chunk)

    visible = "".join(c.content_delta for c in received)
    assert _STORY in visible
    assert "She wrote it down." in visible
    metas = [c.augmentum for c in received if c.augmentum]
    assert not any(
        m.get("status") == "tool_preamble_suppressed" for m in metas
    )
    # The flushed held chunks must not carry stream-end markers (the
    # loop continues into the tool round after the flush).
    story_chunks = [c for c in received if _STORY[:40] in (c.content_delta or "")]
    assert story_chunks and not any(c.done for c in story_chunks)
