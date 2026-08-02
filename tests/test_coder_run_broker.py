"""Unit tests for ``augmentum.coder.run_broker``."""

from __future__ import annotations

import asyncio

import pytest

from augmentum.coder.run_broker import CoderRunBroker, sweep_orphan_running_runs
from augmentum.models.base import InternalStreamChunk
from augmentum.state.backends.sqlite import SQLiteBackend


def _chunk(content: str = "", *, done: bool = False, **aug) -> InternalStreamChunk:
    return InternalStreamChunk(
        content_delta=content,
        done=done,
        augmentum=dict(aug) if aug else None,
    )


@pytest.mark.asyncio
async def test_start_run_pumps_chunks_into_buffer():
    broker = CoderRunBroker()

    async def agent(_entry):
        yield _chunk("hello ")
        yield _chunk("world", done=True, status="complete")

    await broker.start_run(
        run_id="r1",
        user_id="alice",
        workspace_id="ws1",
        agent=agent,
    )

    # Drain via subscribe and confirm the broker pushed both chunks.
    received: list[str] = []
    async for buffered in broker.subscribe("r1", since_seq=0):
        received.append(buffered.chunk.content_delta)

    assert received == ["hello ", "world"]
    entry = broker.get("r1")
    assert entry.done is True
    assert entry.error == ""


@pytest.mark.asyncio
async def test_subscribe_replays_buffered_chunks_for_late_joiner():
    """A subscriber that attaches AFTER the agent already emitted some
    chunks must get them all (the mobile-screen-wake case)."""
    broker = CoderRunBroker()

    started = asyncio.Event()
    keep_going = asyncio.Event()

    async def agent(_entry):
        for i in range(3):
            yield _chunk(f"chunk-{i}", status="streaming")
        started.set()
        await keep_going.wait()
        yield _chunk("final", done=True, status="complete")

    await broker.start_run(
        run_id="r2", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await started.wait()
    # Let the broker actually push the three pre-wait chunks.
    await asyncio.sleep(0)

    # Late subscriber attaches now — should see the buffered prefix.
    async def _collect():
        out = []
        async for buf in broker.subscribe("r2", since_seq=0):
            out.append(buf.chunk.content_delta)
        return out

    collect_task = asyncio.create_task(_collect())
    await asyncio.sleep(0.01)
    keep_going.set()
    received = await collect_task

    assert received == ["chunk-0", "chunk-1", "chunk-2", "final"]


@pytest.mark.asyncio
async def test_subscribe_with_since_seq_skips_already_seen():
    broker = CoderRunBroker()

    async def agent(_entry):
        for i in range(5):
            yield _chunk(f"c{i}", status="streaming")
        yield _chunk("done", done=True, status="complete")

    await broker.start_run(
        run_id="r3", user_id="alice", workspace_id="ws1", agent=agent,
    )
    # Wait for completion so the buffer is stable.
    entry = broker.get("r3")
    while not entry.done:
        await asyncio.sleep(0.01)

    received = []
    async for buf in broker.subscribe("r3", since_seq=3):
        received.append(buf.chunk.content_delta)
    assert received == ["c3", "c4", "done"]


@pytest.mark.asyncio
async def test_cancel_stops_agent_and_records_cancelled():
    broker = CoderRunBroker()

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def agent(_entry):
        try:
            started.set()
            for i in range(1000):
                yield _chunk(f"c{i}")
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    await broker.start_run(
        run_id="r4", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await started.wait()
    assert broker.cancel("r4") is True
    # Wait for the task to actually finish unwinding.
    entry = broker.get("r4")
    while not entry.done:
        await asyncio.sleep(0.01)

    assert cancelled.is_set()
    assert entry.error == "cancelled"
    # Second cancel on a done run is a no-op.
    assert broker.cancel("r4") is False


@pytest.mark.asyncio
async def test_get_active_for_workspace_returns_only_running():
    broker = CoderRunBroker()
    block = asyncio.Event()

    async def agent_a(_entry):
        yield _chunk("running")
        await block.wait()

    async def agent_b(_entry):
        yield _chunk("done", done=True, status="complete")

    await broker.start_run(
        run_id="ra", user_id="alice", workspace_id="ws1", agent=agent_a,
    )
    await broker.start_run(
        run_id="rb", user_id="alice", workspace_id="ws2", agent=agent_b,
    )
    # Wait for rb to finish.
    entry_b = broker.get("rb")
    while not entry_b.done:
        await asyncio.sleep(0.01)

    active = broker.get_active_for_workspace(user_id="alice", workspace_id="ws1")
    assert active is not None and active.run_id == "ra"

    # ws2's only run is done → no active.
    assert broker.get_active_for_workspace(
        user_id="alice", workspace_id="ws2",
    ) is None
    # Other users can't see alice's run.
    assert broker.get_active_for_workspace(
        user_id="bob", workspace_id="ws1",
    ) is None

    block.set()
    await broker.shutdown()


@pytest.mark.asyncio
async def test_agent_exception_records_error_does_not_propagate():
    broker = CoderRunBroker()

    async def agent(_entry):
        yield _chunk("first")
        raise RuntimeError("boom")

    await broker.start_run(
        run_id="rerr", user_id="alice", workspace_id="ws1", agent=agent,
    )
    entry = broker.get("rerr")
    while not entry.done:
        await asyncio.sleep(0.01)
    assert "boom" in entry.error


@pytest.mark.asyncio
async def test_subscribe_buffer_overflow_emits_marker(tmp_path):
    # Tiny buffer so 3 chunks overflow it.
    broker = CoderRunBroker(buffer_cap=2)
    done = asyncio.Event()

    async def agent(_entry):
        for i in range(5):
            yield _chunk(f"c{i}", status="streaming")
        yield _chunk("end", done=True, status="complete")
        done.set()

    await broker.start_run(
        run_id="rov", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await done.wait()

    received_statuses = []
    async for buf in broker.subscribe("rov", since_seq=1):
        aug = buf.chunk.augmentum or {}
        received_statuses.append(aug.get("status") or "")

    # Since the run produced 6 chunks and cap=2, only the last 2 survive
    # in the buffer. The overflow marker must precede the replay.
    assert "buffer_overflow" in received_statuses


@pytest.mark.asyncio
async def test_subscribe_emits_terminal_error_chunk():
    """A run that died with an agent exception must end the subscription
    with a synthetic done=True chunk carrying the error — not a bare
    iterator exit that leaves the client thinking the stream just ended."""
    broker = CoderRunBroker()

    async def agent(_entry):
        yield _chunk("partial")
        raise RuntimeError("boom")

    await broker.start_run(
        run_id="rterm", user_id="alice", workspace_id="ws1", agent=agent,
    )
    entry = broker.get("rterm")
    while not entry.done:
        await asyncio.sleep(0.01)

    received = []
    async for buf in broker.subscribe("rterm", since_seq=0):
        received.append(buf)

    final = received[-1]
    aug = final.chunk.augmentum or {}
    assert final.chunk.done is True
    assert aug.get("status") == "error"
    assert "boom" in aug.get("error", "")
    assert aug.get("final_state") is True
    # Terminal seq sits past every real chunk so seq-deduping clients
    # keep it.
    assert final.seq == entry.seq + 1

    # A reconnect whose cursor is already past the terminal seq must
    # NOT get the terminal chunk again.
    again = [b async for b in broker.subscribe("rterm", since_seq=final.seq)]
    assert again == []


@pytest.mark.asyncio
async def test_subscribe_emits_terminal_cancelled_chunk():
    broker = CoderRunBroker()
    started = asyncio.Event()

    async def agent(_entry):
        started.set()
        for i in range(1000):
            yield _chunk(f"c{i}")
            await asyncio.sleep(0.01)

    await broker.start_run(
        run_id="rcan", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await started.wait()
    broker.cancel("rcan")
    entry = broker.get("rcan")
    while not entry.done:
        await asyncio.sleep(0.01)

    received = [b async for b in broker.subscribe("rcan", since_seq=0)]
    final = received[-1]
    assert final.chunk.done is True
    assert (final.chunk.augmentum or {}).get("status") == "cancelled"


@pytest.mark.asyncio
async def test_subscribe_overflow_marker_at_since_zero():
    """A fresh subscriber (since=0) attaching after the ring overflowed
    must still get the buffer_overflow marker — cursor 0 means 'from
    seq 1', which was evicted."""
    broker = CoderRunBroker(buffer_cap=2)
    done = asyncio.Event()

    async def agent(_entry):
        for i in range(5):
            yield _chunk(f"c{i}", status="streaming")
        yield _chunk("end", done=True, status="complete")
        done.set()

    await broker.start_run(
        run_id="rov0", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await done.wait()

    received = [b async for b in broker.subscribe("rov0", since_seq=0)]
    marker = received[0]
    aug = marker.chunk.augmentum or {}
    assert aug.get("status") == "buffer_overflow"
    assert aug.get("lost_from_seq") == 1
    # Marker seq must be unique in the stream (seq-deduping clients
    # must not drop it or any replayed chunk) and precede the replay.
    seqs = [b.seq for b in received]
    assert len(seqs) == len(set(seqs))
    assert marker.seq < received[1].seq


@pytest.mark.asyncio
async def test_subscribe_no_overflow_marker_when_buffer_intact():
    """since=0 on an un-overflowed buffer must NOT produce a marker."""
    broker = CoderRunBroker()

    async def agent(_entry):
        yield _chunk("a", status="streaming")
        yield _chunk("b", done=True, status="complete")

    await broker.start_run(
        run_id="rok", user_id="alice", workspace_id="ws1", agent=agent,
    )
    entry = broker.get("rok")
    while not entry.done:
        await asyncio.sleep(0.01)

    received = [b async for b in broker.subscribe("rok", since_seq=0)]
    statuses = [(b.chunk.augmentum or {}).get("status") for b in received]
    assert "buffer_overflow" not in statuses
    assert [b.seq for b in received] == [1, 2]


@pytest.mark.asyncio
async def test_sweep_orphan_running_runs_marks_cancelled(tmp_path):
    """Startup-time sweep: rows stuck in status='running' get cancelled."""
    backend = SQLiteBackend(str(tmp_path / "broker.db"))
    await backend.connect()
    conn = backend.conn
    # Insert a fake user + workspace so the FK constraints pass.
    await conn.execute(
        "INSERT INTO users (id, username, display_name, password_hash, role) "
        "VALUES ('alice', 'alice', 'Alice', 'pw', 'user')",
    )
    import time as _time

    await conn.execute(
        "INSERT INTO project_checkouts (id, name, status, created_at, user_id) "
        "VALUES ('ws1', 'ws1', 'running', ?, 'alice')",
        (_time.time(),),
    )
    await conn.execute(
        """
        INSERT INTO coder_turn_runs
            (id, user_id, project_id, session_id, strategy, model,
             provider, prompt_profile, tooling_profile, status,
             started_at, updated_at)
        VALUES ('stuck', 'alice', 'ws1', 'ws1', '', '', '', '', '',
                'running', ?, ?)
        """,
        (_time.time(), _time.time()),
    )
    await conn.commit()

    swept = await sweep_orphan_running_runs(conn)
    assert swept == 1
    cursor = await conn.execute(
        "SELECT status, finish_reason FROM coder_turn_runs WHERE id = ?",
        ("stuck",),
    )
    row = await cursor.fetchone()
    assert row[0] == "cancelled"
    assert row[1] == "server_restart"
    await backend.close()


@pytest.mark.asyncio
async def test_sweep_covers_whole_zombie_class(tmp_path):
    """Boot sweep marks non-terminal rows in ALL three run tables as
    interrupted, and never touches rows that finished legitimately."""
    import time as _time

    backend = SQLiteBackend(str(tmp_path / "broker.db"))
    await backend.connect()
    conn = backend.conn
    now = _time.time()
    await conn.execute(
        "INSERT INTO users (id, username, display_name, password_hash, role) "
        "VALUES ('alice', 'alice', 'Alice', 'pw', 'user')",
    )
    await conn.execute(
        "INSERT INTO project_checkouts (id, name, status, created_at, user_id) "
        "VALUES ('ws1', 'ws1', 'running', ?, 'alice')",
        (now,),
    )

    def _turn_run(run_id: str, status: str):
        return (
            "INSERT INTO coder_turn_runs (id, user_id, project_id, session_id, "
            "strategy, model, provider, prompt_profile, tooling_profile, "
            "status, started_at, updated_at) "
            f"VALUES ('{run_id}', 'alice', 'ws1', 'ws1', '', '', '', '', '', "
            f"'{status}', {now}, {now})"
        )

    await conn.execute(_turn_run("zombie_turn", "running"))
    await conn.execute(_turn_run("done_turn", "completed"))
    await conn.execute(_turn_run("err_turn", "error"))
    # Subagent breadcrumbs: one orphan, one legitimately finished.
    await conn.execute(
        "INSERT INTO coder_subagent_runs (subagent_id, user_id, role, "
        "started_at, stop_reason) VALUES ('sub_zombie', 'alice', 'tester', ?, 'running')",
        (int(now),),
    )
    await conn.execute(
        "INSERT INTO coder_subagent_runs (subagent_id, user_id, role, "
        "started_at, completed_at, stop_reason) "
        "VALUES ('sub_done', 'alice', 'tester', ?, ?, 'complete')",
        (int(now), int(now)),
    )
    # External-coder (Claude Code) runs: one orphan, one done.
    await conn.execute(
        "INSERT INTO claude_runs (id, user_id, workspace_id, task, status) "
        "VALUES ('cl_zombie', 'alice', 'ws1', 't', 'running')",
    )
    await conn.execute(
        "INSERT INTO claude_runs (id, user_id, workspace_id, task, status) "
        "VALUES ('cl_done', 'alice', 'ws1', 't', 'done')",
    )
    await conn.commit()

    swept = await sweep_orphan_running_runs(conn)
    assert swept == 3

    row = await (await conn.execute(
        "SELECT status, finish_reason FROM coder_turn_runs WHERE id='zombie_turn'",
    )).fetchone()
    assert row[0] == "cancelled" and row[1] == "server_restart"
    row = await (await conn.execute(
        "SELECT status FROM coder_turn_runs WHERE id='done_turn'",
    )).fetchone()
    assert row[0] == "completed"
    row = await (await conn.execute(
        "SELECT status FROM coder_turn_runs WHERE id='err_turn'",
    )).fetchone()
    assert row[0] == "error"

    row = await (await conn.execute(
        "SELECT stop_reason, stop_detail, completed_at "
        "FROM coder_subagent_runs WHERE subagent_id='sub_zombie'",
    )).fetchone()
    assert row[0] == "server_restart"
    assert row[1] == "interrupted (server restarted)"
    assert row[2] is not None
    row = await (await conn.execute(
        "SELECT stop_reason FROM coder_subagent_runs WHERE subagent_id='sub_done'",
    )).fetchone()
    assert row[0] == "complete"

    row = await (await conn.execute(
        "SELECT status, error FROM claude_runs WHERE id='cl_zombie'",
    )).fetchone()
    assert row[0] == "failed"
    assert row[1] == "interrupted (server restarted)"
    row = await (await conn.execute(
        "SELECT status FROM claude_runs WHERE id='cl_done'",
    )).fetchone()
    assert row[0] == "done"

    # Idempotent: a second sweep finds nothing.
    assert await sweep_orphan_running_runs(conn) == 0
    await backend.close()


@pytest.mark.asyncio
async def test_sweep_orphan_none_conn_is_noop():
    assert await sweep_orphan_running_runs(None) == 0
