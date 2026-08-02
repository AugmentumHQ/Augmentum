"""AugmentumACPAgent — session lifecycle, event translation, concurrency.

The coder loop is injected as a fake ``loop_runner`` so the agent is tested with
no running app and no real editor. A fake connection captures the session_update
calls the agent emits, letting us assert the native_loop -> ACP translation.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("acp")

from augmentum.coder.acp_agent import (  # noqa: E402
    AugmentumACPAgent,
    _extract_prompt_text,
)
from augmentum.coder.executors import RemoteEditorExecutor  # noqa: E402


class FakeConn:
    """Captures session_update calls (and satisfies channel construction)."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id, update, **kw):
        self.updates.append((session_id, update))

    # present so ACPEditorChannel construction is happy (never called here)
    async def read_text_file(self, *a, **k):
        raise AssertionError("not expected in these tests")


def _agent(runner, **kw) -> tuple[AugmentumACPAgent, FakeConn]:
    conn = FakeConn()
    ag = AugmentumACPAgent(loop_runner=runner, **kw)
    ag.on_connect(conn)
    return ag, conn


async def _empty_runner(sess, text):
    if False:
        yield  # make it an async generator


@pytest.mark.asyncio
async def test_new_session_builds_isolated_remote_executor() -> None:
    ag, _ = _agent(_empty_runner)
    r1 = await ag.new_session(cwd="/workspace/a")
    r2 = await ag.new_session(cwd="/workspace/b")
    s1 = ag._sessions[r1.session_id]
    s2 = ag._sessions[r2.session_id]
    assert r1.session_id != r2.session_id
    assert isinstance(s1.executor, RemoteEditorExecutor)
    # isolation: distinct executor, channel, and cancel objects per session
    assert s1.executor is not s2.executor
    assert s1.channel is not s2.channel
    assert s1.cancel is not s2.cancel


@pytest.mark.asyncio
async def test_prompt_translates_events_to_session_update() -> None:
    async def runner(sess, text):
        yield ("text", {"text": "thinking..."})
        yield ("tool_call", {"tool": "file_read", "args": {"path": "/x"}})
        yield ("tool_result", {"tool": "file_read", "ok": True, "snippet": "contents"})
        yield ("metrics", {"tokens": 5})  # dropped
        yield ("text", {"text": "done"})

    ag, conn = _agent(runner)
    r = await ag.new_session(cwd="/workspace")
    resp = await ag.prompt(r.session_id, "read the file")
    assert resp.stop_reason == "end_turn"

    # metrics dropped -> 4 updates: text, tool_call start, tool_call update, text
    kinds = [type(u).__name__ for _, u in conn.updates]
    assert kinds == [
        "AgentMessageChunk", "ToolCallStart", "ToolCallProgress", "AgentMessageChunk",
    ]
    # tool start/update share the same tool_call_id (correlated)
    start = conn.updates[1][1]
    prog = conn.updates[2][1]
    assert start.tool_call_id == prog.tool_call_id
    assert prog.status == "completed"


@pytest.mark.asyncio
async def test_tool_result_failure_maps_to_failed_status() -> None:
    async def runner(sess, text):
        yield ("tool_call", {"tool": "shell_exec"})
        yield ("tool_result", {"tool": "shell_exec", "ok": False, "snippet": "boom"})

    ag, conn = _agent(runner)
    r = await ag.new_session(cwd="/workspace")
    await ag.prompt(r.session_id, "run it")
    assert conn.updates[1][1].status == "failed"


@pytest.mark.asyncio
async def test_cancel_stops_the_turn_midstream() -> None:
    async def runner(sess, text):
        yield ("text", {"text": "first"})
        sess.cancel.set()  # simulate an ACP cancel arriving after the first event
        yield ("text", {"text": "second"})  # must NOT be emitted

    ag, conn = _agent(runner)
    r = await ag.new_session(cwd="/workspace")
    resp = await ag.prompt(r.session_id, "go")
    assert resp.stop_reason == "cancelled"
    # only the first event was emitted; "second" was cut off by the cancel check
    assert len(conn.updates) == 1
    assert type(conn.updates[0][1]).__name__ == "AgentMessageChunk"


@pytest.mark.asyncio
async def test_cancel_method_sets_session_event() -> None:
    ag, _ = _agent(_empty_runner)
    r = await ag.new_session(cwd="/workspace")
    assert not ag._sessions[r.session_id].cancel.is_set()
    await ag.cancel(r.session_id)
    assert ag._sessions[r.session_id].cancel.is_set()


@pytest.mark.asyncio
async def test_turn_semaphore_bounds_concurrency() -> None:
    # max 1 concurrent turn: a second prompt must wait for the first to release.
    gate = asyncio.Event()
    active = 0
    peak = 0

    async def runner(sess, text):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await gate.wait()  # hold the turn open
        active -= 1
        yield ("text", {"text": "x"})

    ag, _ = _agent(runner, max_concurrent_turns=1)
    r1 = await ag.new_session(cwd="/workspace")
    r2 = await ag.new_session(cwd="/workspace")
    t1 = asyncio.create_task(ag.prompt(r1.session_id, "a"))
    t2 = asyncio.create_task(ag.prompt(r2.session_id, "b"))
    await asyncio.sleep(0.05)
    assert peak == 1  # semaphore held the second turn out
    gate.set()
    await asyncio.gather(t1, t2)
    assert peak == 1


@pytest.mark.asyncio
async def test_unknown_session_prompt_raises() -> None:
    import acp
    ag, _ = _agent(_empty_runner)
    with pytest.raises(acp.RequestError):
        await ag.prompt("nope", "hi")


@pytest.mark.asyncio
async def test_initialize_returns_capabilities() -> None:
    import acp
    ag, _ = _agent(_empty_runner)
    resp = await ag.initialize(protocol_version=acp.PROTOCOL_VERSION)
    assert resp.protocol_version == acp.PROTOCOL_VERSION
    assert resp.agent_capabilities is not None


def test_extract_prompt_text_from_blocks_and_str() -> None:
    import acp
    assert _extract_prompt_text("plain") == "plain"
    blocks = [acp.text_block("hello"), acp.text_block("world")]
    assert _extract_prompt_text(blocks) == "hello\nworld"
    assert _extract_prompt_text([{"text": "d"}]) == "d"
