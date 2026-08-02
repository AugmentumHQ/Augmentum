"""Tests for cooperative-turn inbox + pause/resume on the run broker."""

from __future__ import annotations

import asyncio

import pytest

from augmentum.coder.run_broker import _INBOX_CAP, CoderRunBroker
from augmentum.models.base import InternalStreamChunk
from augmentum.modes.coder.chat_egress import (
    _VALID_PHASES,
    _VALID_STATUSES,
    _validate_metadata,
)


class TestCoopMetadataIsRegistered:
    """Schema regression — the chunks the cooperative drain emits MUST
    pass ``_validate_metadata`` or they get caught by the handler's
    ``except Exception`` branch and re-raised as a turn error. Symptom
    when this drifts: queued messages stay 'queued' forever and the
    UI never auto-chains the next turn (observed live 2026-05-30).
    """

    def test_queue_followup_chunk_validates(self):
        # If this raises, my end-of-turn drain chunk would never reach
        # the frontend — the queue_followup stream event is what tells
        # the UI to auto-chain the next turn.
        _validate_metadata("completing", "queue_followup")

    def test_steer_delivered_chunk_validates(self):
        # If this raises, the iteration-boundary steer drain chunk
        # would never reach the frontend — queued/steering badges
        # would never flip to delivered.
        _validate_metadata("executing", "steer_delivered")

    def test_completing_phase_in_valid_phases_set(self):
        assert "completing" in _VALID_PHASES

    def test_cooperative_statuses_in_valid_statuses_set(self):
        assert "queue_followup" in _VALID_STATUSES
        assert "steer_delivered" in _VALID_STATUSES
        assert "queue_dropped" in _VALID_STATUSES

    def test_queue_dropped_chunk_validates(self):
        # The cancel/error inbox drain path emits this — without it
        # registered, every cancelled turn with queued messages
        # would secondarily error in the same ValueError-swallowed
        # way the original queue_followup bug did.
        _validate_metadata("executing", "queue_dropped")


def _quick_agent_factory(events: list[asyncio.Event] | None = None):
    """Build an agent factory that emits one chunk and finishes.

    Used to keep broker entries alive long enough to test pause /
    inbox operations without a real coder loop. ``events`` is an
    optional list of awaitables — the agent waits on each before
    emitting the matching chunk, giving the test control over the
    timing of iteration boundaries.
    """
    async def _agent(_entry):
        if events:
            for ev in events:
                await ev.wait()
                yield InternalStreamChunk(content_delta=".", done=False)
        yield InternalStreamChunk(content_delta="", done=True)
    return _agent


@pytest.mark.asyncio
async def test_inbox_enqueue_and_drain_by_mode():
    """Steer + queue entries drain separately by their mode filter."""
    broker = CoderRunBroker()
    gate = asyncio.Event()
    await broker.start_run(
        run_id="r-mode", user_id="u", workspace_id="ws",
        agent=_quick_agent_factory([gate]),
    )

    e1 = broker.enqueue_user_message("r-mode", content="steer-1", mode="steer")
    e2 = broker.enqueue_user_message("r-mode", content="queue-1", mode="queue")
    e3 = broker.enqueue_user_message("r-mode", content="steer-2", mode="steer")
    assert e1 and e2 and e3
    assert broker.inbox_depth("r-mode") == 3
    assert broker.inbox_depth("r-mode", mode="steer") == 2
    assert broker.inbox_depth("r-mode", mode="queue") == 1

    drained_steer = broker.drain_user_messages("r-mode", mode="steer")
    assert [m["content"] for m in drained_steer] == ["steer-1", "steer-2"]
    assert all(m["delivered_at"] is not None for m in drained_steer)
    # Queue entry is untouched.
    assert broker.inbox_depth("r-mode") == 1
    assert broker.inbox_depth("r-mode", mode="queue") == 1

    drained_queue = broker.drain_user_messages("r-mode", mode="queue")
    assert [m["content"] for m in drained_queue] == ["queue-1"]
    assert broker.inbox_depth("r-mode") == 0

    # Drain everything (after re-enqueue) — None mode drops the filter.
    broker.enqueue_user_message("r-mode", content="any", mode="queue")
    drained_all = broker.drain_user_messages("r-mode", mode=None)
    assert len(drained_all) == 1
    assert broker.inbox_depth("r-mode") == 0

    # Cleanup
    gate.set()
    async for _ in broker.subscribe("r-mode", since_seq=0):
        pass


@pytest.mark.asyncio
async def test_inbox_unknown_mode_falls_back_to_queue():
    """Defensive: garbage mode strings land as 'queue' (the safe default)."""
    broker = CoderRunBroker()
    gate = asyncio.Event()
    await broker.start_run(
        run_id="r-unknown", user_id="u", workspace_id="ws",
        agent=_quick_agent_factory([gate]),
    )
    entry = broker.enqueue_user_message(
        "r-unknown", content="oops", mode="banana",
    )
    assert entry is not None
    assert entry["mode"] == "queue"
    gate.set()
    async for _ in broker.subscribe("r-unknown", since_seq=0):
        pass


@pytest.mark.asyncio
async def test_inbox_rejects_when_unknown_or_finished():
    """Unknown run_id returns None; finished run returns None."""
    broker = CoderRunBroker()
    assert broker.enqueue_user_message("nope", content="x") is None

    async def _instant_done(_entry):
        yield InternalStreamChunk(content_delta="", done=True)

    await broker.start_run(
        run_id="r-done", user_id="u", workspace_id="ws", agent=_instant_done,
    )
    # Drain the run so entry.done is True.
    async for _ in broker.subscribe("r-done", since_seq=0):
        pass
    assert broker.get("r-done").done is True
    assert broker.enqueue_user_message("r-done", content="late") is None


@pytest.mark.asyncio
async def test_inbox_capacity_cap_rejects_overflow():
    """The 25-entry cap protects a wedged agent from unbounded backlog."""
    broker = CoderRunBroker()
    gate = asyncio.Event()
    await broker.start_run(
        run_id="r-cap", user_id="u", workspace_id="ws",
        agent=_quick_agent_factory([gate]),
    )
    # Fill to the cap.
    for i in range(_INBOX_CAP):
        assert broker.enqueue_user_message(
            "r-cap", content=f"msg-{i}", mode="queue",
        ) is not None
    assert broker.inbox_depth("r-cap") == _INBOX_CAP
    # Next enqueue is rejected.
    overflow = broker.enqueue_user_message(
        "r-cap", content="overflow", mode="queue",
    )
    assert overflow is None
    assert broker.inbox_depth("r-cap") == _INBOX_CAP
    gate.set()
    async for _ in broker.subscribe("r-cap", since_seq=0):
        pass


@pytest.mark.asyncio
async def test_pause_resume_gate_blocks_then_releases():
    """await_pause_gate blocks while paused and releases on resume."""
    broker = CoderRunBroker()
    gate = asyncio.Event()
    await broker.start_run(
        run_id="r-pause", user_id="u", workspace_id="ws",
        agent=_quick_agent_factory([gate]),
    )
    # Initial state: not paused. Gate returns immediately.
    await asyncio.wait_for(broker.await_pause_gate("r-pause"), timeout=0.1)

    # Pause and verify the gate blocks.
    assert broker.pause("r-pause") is True
    assert broker.get("r-pause").paused is True
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(broker.await_pause_gate("r-pause"), timeout=0.05)

    # Resume releases the awaiter.
    assert broker.resume("r-pause") is True
    assert broker.get("r-pause").paused is False
    await asyncio.wait_for(broker.await_pause_gate("r-pause"), timeout=0.1)

    # Idempotent: re-pause and re-resume return False/False
    # respectively when state is already correct.
    assert broker.pause("r-pause") is True
    assert broker.pause("r-pause") is False  # already paused
    assert broker.resume("r-pause") is True
    assert broker.resume("r-pause") is False  # already running

    gate.set()
    async for _ in broker.subscribe("r-pause", since_seq=0):
        pass


@pytest.mark.asyncio
async def test_end_of_turn_drain_promotes_undelivered_steers():
    """A steer that never hit an iteration boundary must not strand.

    Repro (reported 2026-07-03): the user interjects with mode="steer"
    while the model is writing its FINAL response — no further
    iteration boundary ever runs _coop_iteration_check, and the old
    end-of-turn drain filtered to mode="queue" only. The steer entry
    sat in the inbox until broker eviction: badge stuck on "steering"
    forever, content lost, user forced to retype. The natural-exit
    drain now pops the WHOLE inbox so undelivered steers ride the
    queue_followup chain into the next turn.
    """
    from types import SimpleNamespace

    from augmentum.modes.coder.handler import CoderHandler

    broker = CoderRunBroker()
    gate = asyncio.Event()
    await broker.start_run(
        run_id="r-promote", user_id="u", workspace_id="ws",
        agent=_quick_agent_factory([gate]),
    )
    broker.enqueue_user_message("r-promote", content="steer-late", mode="steer")
    broker.enqueue_user_message("r-promote", content="queued-followup", mode="queue")

    stub = SimpleNamespace(
        _coder_run_broker=broker,
        _current_broker_run_id=lambda: "r-promote",
    )
    drained = CoderHandler._coop_drain_queued_followups(stub)

    # FIFO order preserved, both modes present, delivery stamped,
    # inbox fully resolved — nothing left to strand.
    assert [m["content"] for m in drained] == ["steer-late", "queued-followup"]
    assert [m["mode"] for m in drained] == ["steer", "queue"]
    assert all(m["delivered_at"] is not None for m in drained)
    assert broker.inbox_depth("r-promote") == 0

    gate.set()
    async for _ in broker.subscribe("r-promote", since_seq=0):
        pass
