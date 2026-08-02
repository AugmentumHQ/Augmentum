"""coder_runner — chunk translation + the loop_runner that drives CoderHandler."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("acp")

from augmentum.coder.acp_agent import AugmentumACPAgent  # noqa: E402
from augmentum.coder.coder_runner import (  # noqa: E402
    coder_chunk_to_events,
    make_coder_loop_runner,
)


def _chunk(**kw):
    kw.setdefault("thinking_delta", None)
    kw.setdefault("content_delta", None)
    kw.setdefault("done", False)
    return SimpleNamespace(**kw)


# -- pure translation --------------------------------------------------------

def test_chunk_thinking_then_content_order() -> None:
    ev = coder_chunk_to_events(_chunk(thinking_delta="hmm", content_delta="answer"))
    assert ev == [("thought", {"text": "hmm"}), ("text", {"text": "answer"})]


def test_chunk_content_only() -> None:
    assert coder_chunk_to_events(_chunk(content_delta="x")) == [("text", {"text": "x"})]


def test_chunk_empty_yields_nothing() -> None:
    assert coder_chunk_to_events(_chunk()) == []
    assert coder_chunk_to_events(_chunk(done=True)) == []


# -- the loop_runner over a fake handler -------------------------------------

class FakeStream:
    def __init__(self, chunks) -> None:
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


class FakeHandler:
    def __init__(self, chunks) -> None:
        self.stream = FakeStream(chunks)
        self.request = None

    def handle_stream(self, request):
        self.request = request
        return self.stream


@pytest.mark.asyncio
async def test_loop_runner_translates_and_stops_on_done() -> None:
    chunks = [
        _chunk(thinking_delta="plan"),
        _chunk(content_delta="hello "),
        _chunk(content_delta="world", done=True),
        _chunk(content_delta="AFTER-DONE"),  # must not be emitted
    ]
    handler = FakeHandler(chunks)
    runner = make_coder_loop_runner(
        build_handler=lambda sess: handler,
        build_request=lambda sess, text: {"prompt": text},
    )
    got = [ev async for ev in runner(SimpleNamespace(), "do it")]
    assert got == [
        ("thought", {"text": "plan"}),
        ("text", {"text": "hello "}),
        ("text", {"text": "world"}),
    ]
    assert handler.request == {"prompt": "do it"}
    assert handler.stream.closed is True  # stream closed after the turn


@pytest.mark.asyncio
async def test_end_to_end_agent_over_coder_runner() -> None:
    # AugmentumACPAgent + real make_coder_loop_runner + fake CoderHandler:
    # proves the editor agent path translates a coder stream into ACP updates.
    class Conn:
        def __init__(self):
            self.updates = []
        async def session_update(self, sid, update, **kw):
            self.updates.append(update)
        async def read_text_file(self, *a, **k):
            raise AssertionError

    handler = FakeHandler([
        _chunk(thinking_delta="thinking"),
        _chunk(content_delta="done answer", done=True),
    ])
    runner = make_coder_loop_runner(
        build_handler=lambda sess: handler,
        build_request=lambda sess, text: text,
    )
    conn = Conn()
    ag = AugmentumACPAgent(loop_runner=runner)
    ag.on_connect(conn)
    r = await ag.new_session(cwd="/workspace")
    resp = await ag.prompt(r.session_id, "go")
    assert resp.stop_reason == "end_turn"
    kinds = [type(u).__name__ for u in conn.updates]
    assert kinds == ["AgentThoughtChunk", "AgentMessageChunk"]
