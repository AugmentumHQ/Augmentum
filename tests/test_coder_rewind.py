"""Tests for the coder rewind module + broker snapshot attachment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from augmentum.coder.rewind import (
    RewindOutcome,
    _pop_matching_turn_summary,
    _reset_per_request_state,
    rewind_last_turn,
)
from augmentum.coder.run_broker import CoderRunBroker
from augmentum.models.base import InternalStreamChunk

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

@dataclass
class _FakeSnapshot:
    """Stand-in for TurnSnapshot — captures restore() invocations."""

    turn_id: str = "t1"
    paths: list[str] = field(default_factory=lambda: ["/workspace/a.py", "/workspace/b.py"])
    fail_paths: list[str] = field(default_factory=list)
    restore_calls: list[list[str]] = field(default_factory=list)

    @property
    def touched_paths(self) -> list[str]:
        return list(self.paths)

    async def restore(self, paths: list[str]) -> list[str]:
        self.restore_calls.append(list(paths))
        return [p for p in paths if p in self.fail_paths]


class _FakeState:
    """Minimal CoderState surface for the rewind path."""

    def __init__(self, *, turn_summaries=None):
        self.turn_summaries = list(turn_summaries or [])
        self.plan = "old plan"
        self.plan_steps = ["step one", "step two"]
        self.current_step = 1
        self.step_outputs = {"0": "did stuff"}
        self.mission = ["some promise"]
        self.tasks = [{"content": "t", "activeForm": "T", "status": "pending"}]
        self.finish_requested = True
        self.finish_summary = "i think i'm done"
        self.recent_validation_errors = ["err"]
        self.recent_tool_calls = [{"tool": "file_read"}]
        self.consecutive_failures = 4
        self.error = "boom"
        self.current_intent = "DEBUG"
        self.cleared = False

    def clear_pending_objective_contract(self):
        self.cleared = True


class _FakeStateManager:
    """In-memory load/save backing for the rewind state path."""

    def __init__(self, state: _FakeState):
        self._state = state
        self.saved: _FakeState | None = None

    async def load_coder_state(self, session_id, *, user_id):  # noqa: ARG002
        return self._state

    async def save_coder_state(self, session_id, state, *, user_id):  # noqa: ARG002
        self.saved = state


class _FakeBundle:
    """Just enough surface to stand in for ReviewBundle."""

    def __init__(self, *, turn_id, workspace_id, session_id, snapshot, user_id, created_at):
        self.turn_id = turn_id
        self.workspace_id = workspace_id
        self.session_id = session_id
        self.snapshot = snapshot
        self.user_id = user_id
        self.created_at = created_at


class _FakeReviewRegistry:
    def __init__(self):
        self.bundles: list[_FakeBundle] = []
        self.resolved: list[tuple[str, str]] = []

    def pending_for(self, user_id):
        return [b for b in self.bundles if b.user_id == user_id]

    def resolve(self, turn_id, status):
        self.resolved.append((turn_id, status))


# ---------------------------------------------------------------------------
# _pop_matching_turn_summary
# ---------------------------------------------------------------------------

class TestPopMatchingTurnSummary:
    def test_pops_when_turn_id_matches(self):
        state = SimpleNamespace(turn_summaries=[
            {"turn_id": "older"},
            {"turn_id": "t1", "outcome": "done"},
        ])
        assert _pop_matching_turn_summary(state, "t1") is True
        assert state.turn_summaries == [{"turn_id": "older"}]

    def test_refuses_when_turn_id_differs(self):
        # Latest summary belongs to a DIFFERENT turn. Realistic case:
        # the rewound run was cancelled mid-flight and never wrote a
        # summary, so the last summary is from the previous turn —
        # popping it would be wrong.
        state = SimpleNamespace(turn_summaries=[
            {"turn_id": "previous"},
        ])
        assert _pop_matching_turn_summary(state, "t1") is False
        assert state.turn_summaries == [{"turn_id": "previous"}]

    def test_fallback_pops_when_no_turn_id_stamped(self):
        # Older persisted summaries from before the turn_id field
        # landed. Permissive fallback — pop the last one.
        state = SimpleNamespace(turn_summaries=[
            {"user_goal": "legacy"},
        ])
        assert _pop_matching_turn_summary(state, "t1") is True
        assert state.turn_summaries == []

    def test_no_summaries_returns_false(self):
        state = SimpleNamespace(turn_summaries=[])
        assert _pop_matching_turn_summary(state, "t1") is False


# ---------------------------------------------------------------------------
# _reset_per_request_state
# ---------------------------------------------------------------------------

class TestResetPerRequestState:
    def test_clears_plan_tasks_finish(self):
        state = _FakeState()
        _reset_per_request_state(state)
        assert state.plan == ""
        assert state.plan_steps == []
        assert state.current_step == 0
        assert state.step_outputs == {}
        assert state.mission == []
        assert state.tasks == []
        assert state.finish_requested is False
        assert state.finish_summary == ""
        assert state.consecutive_failures == 0
        assert state.error is None
        assert state.current_intent is None
        assert state.recent_validation_errors == []
        assert state.recent_tool_calls == []
        # Optional helper fires if present
        assert state.cleared is True


# ---------------------------------------------------------------------------
# Broker integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broker_attach_snapshot_binds_to_entry():
    broker = CoderRunBroker()

    async def _slow_agent(_entry):
        # Long-enough to outlive the attach call below.
        for _ in range(3):
            await asyncio.sleep(0.01)
            yield InternalStreamChunk(content_delta=".", done=False)
        yield InternalStreamChunk(content_delta="", done=True)

    await broker.start_run(
        run_id="r-attach",
        user_id="u",
        workspace_id="ws",
        agent=_slow_agent,
    )
    snap = _FakeSnapshot()
    assert broker.attach_snapshot("r-attach", snap) is True
    entry = broker.get("r-attach")
    assert entry.turn_snapshot is snap

    # Drain to let the task complete cleanly.
    async for _ in broker.subscribe("r-attach", since_seq=0):
        pass


@pytest.mark.asyncio
async def test_broker_attach_snapshot_returns_false_for_unknown():
    broker = CoderRunBroker()
    assert broker.attach_snapshot("nope", _FakeSnapshot()) is False


@pytest.mark.asyncio
async def test_broker_latest_for_workspace_picks_most_recent():
    broker = CoderRunBroker()

    async def _quick_agent(_entry):
        yield InternalStreamChunk(content_delta="", done=True)

    await broker.start_run(
        run_id="r-old", user_id="u", workspace_id="ws", agent=_quick_agent,
    )
    # Drain so the first one is done before we start the second.
    async for _ in broker.subscribe("r-old", since_seq=0):
        pass

    # Brief gap so the two run timestamps don't tie at sub-microsecond
    # resolution — production callers have user think-time between
    # turns; the test mimics that with a small sleep.
    await asyncio.sleep(0.01)

    await broker.start_run(
        run_id="r-new", user_id="u", workspace_id="ws", agent=_quick_agent,
    )
    async for _ in broker.subscribe("r-new", since_seq=0):
        pass

    latest = broker.latest_for_workspace(user_id="u", workspace_id="ws")
    assert latest is not None
    assert latest.run_id == "r-new"


# ---------------------------------------------------------------------------
# rewind_last_turn — end-to-end with fakes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rewind_returns_not_found_when_no_snapshot():
    """No broker entries, no review bundles → ok=False with a clear error."""
    app_state = SimpleNamespace(
        coder_run_broker=None,
        review_registry=None,
        state_manager=None,
    )
    outcome = await rewind_last_turn(
        user_id="u", workspace_id="ws", app_state=app_state,
    )
    assert isinstance(outcome, RewindOutcome)
    assert outcome.ok is False
    assert "Nothing to rewind" in outcome.error


@pytest.mark.asyncio
async def test_rewind_uses_review_bundle_when_broker_empty():
    """Snapshot via the review registry fallback path."""
    snap = _FakeSnapshot(turn_id="t-bundle")
    bundle = _FakeBundle(
        turn_id="t-bundle",
        workspace_id="ws",
        session_id="sess1",
        snapshot=snap,
        user_id="u",
        created_at=100.0,
    )
    registry = _FakeReviewRegistry()
    registry.bundles.append(bundle)

    state = _FakeState(turn_summaries=[{"turn_id": "t-bundle", "user_goal": "x"}])
    sm = _FakeStateManager(state)
    app_state = SimpleNamespace(
        coder_run_broker=None,
        review_registry=registry,
        state_manager=SimpleNamespace(
            backend=SimpleNamespace(conn=None),
            load_coder_state=sm.load_coder_state,
            save_coder_state=sm.save_coder_state,
        ),
    )

    outcome = await rewind_last_turn(
        user_id="u", workspace_id="ws", app_state=app_state,
    )
    assert outcome.ok is True
    assert outcome.run_id == "t-bundle"
    assert outcome.cancelled_in_flight is False
    assert set(outcome.restored_paths) == set(snap.paths)
    assert outcome.irreversible_paths == []
    assert outcome.turn_summary_popped is True
    # snapshot.restore was called once with all touched paths.
    assert len(snap.restore_calls) == 1
    assert set(snap.restore_calls[0]) == set(snap.paths)
    # state was saved with cleared scratchpads.
    assert sm.saved is state
    assert state.plan == ""
    assert state.turn_summaries == []
    # bundle was resolved as rewound.
    assert registry.resolved == [("t-bundle", "rewound")]


@pytest.mark.asyncio
async def test_rewind_mode_conv_skips_file_restore():
    """conv-only mode: snapshot.restore() is never called.

    The "poisoned context" path keeps file edits but drops the
    conversation/state. Verified by the absence of restore_calls on
    the fake snapshot.
    """
    snap = _FakeSnapshot()
    bundle = _FakeBundle(
        turn_id="t-conv",
        workspace_id="ws",
        session_id="sess1",
        snapshot=snap,
        user_id="u",
        created_at=100.0,
    )
    registry = _FakeReviewRegistry()
    registry.bundles.append(bundle)

    state = _FakeState(turn_summaries=[{"turn_id": "t-conv", "user_goal": "x"}])
    sm = _FakeStateManager(state)
    app_state = SimpleNamespace(
        coder_run_broker=None,
        review_registry=registry,
        state_manager=SimpleNamespace(
            backend=SimpleNamespace(conn=None),
            load_coder_state=sm.load_coder_state,
            save_coder_state=sm.save_coder_state,
        ),
    )

    outcome = await rewind_last_turn(
        user_id="u", workspace_id="ws", app_state=app_state, mode="conv",
    )
    assert outcome.ok is True
    assert outcome.mode == "conv"
    # Files were NOT restored — that's the whole point of conv mode.
    assert snap.restore_calls == []
    assert outcome.restored_paths == []
    # Conversation state WAS rolled back.
    assert outcome.turn_summary_popped is True
    assert state.plan == ""


@pytest.mark.asyncio
async def test_rewind_mode_files_keeps_conversation_state():
    """files-only mode: files restore but turn_summary stays put.

    The "edits were wrong but the chat is fine" path keeps the
    conversation history so the model remembers the discussion on
    the next turn.
    """
    snap = _FakeSnapshot()
    bundle = _FakeBundle(
        turn_id="t-files",
        workspace_id="ws",
        session_id="sess1",
        snapshot=snap,
        user_id="u",
        created_at=100.0,
    )
    registry = _FakeReviewRegistry()
    registry.bundles.append(bundle)

    initial_summary = {"turn_id": "t-files", "user_goal": "x"}
    state = _FakeState(turn_summaries=[initial_summary])
    sm = _FakeStateManager(state)
    app_state = SimpleNamespace(
        coder_run_broker=None,
        review_registry=registry,
        state_manager=SimpleNamespace(
            backend=SimpleNamespace(conn=None),
            load_coder_state=sm.load_coder_state,
            save_coder_state=sm.save_coder_state,
        ),
    )

    outcome = await rewind_last_turn(
        user_id="u", workspace_id="ws", app_state=app_state, mode="files",
    )
    assert outcome.ok is True
    assert outcome.mode == "files"
    # Files WERE restored.
    assert snap.restore_calls == [snap.paths]
    assert set(outcome.restored_paths) == set(snap.paths)
    # Conversation state stays intact — summary not popped, save not called.
    assert outcome.turn_summary_popped is False
    assert state.turn_summaries == [initial_summary]
    assert sm.saved is None


@pytest.mark.asyncio
async def test_rewind_surfaces_irreversible_paths():
    snap = _FakeSnapshot(
        turn_id="t-irrev",
        paths=["/a.py", "/b.bin"],
        fail_paths=["/b.bin"],
    )
    bundle = _FakeBundle(
        turn_id="t-irrev",
        workspace_id="ws",
        session_id="sess1",
        snapshot=snap,
        user_id="u",
        created_at=100.0,
    )
    registry = _FakeReviewRegistry()
    registry.bundles.append(bundle)

    app_state = SimpleNamespace(
        coder_run_broker=None,
        review_registry=registry,
        state_manager=None,
    )

    outcome = await rewind_last_turn(
        user_id="u", workspace_id="ws", app_state=app_state,
    )
    assert outcome.ok is True
    assert outcome.restored_paths == ["/a.py"]
    assert outcome.irreversible_paths == ["/b.bin"]
    # Side-effect warning is always present (operator notice).
    assert any("Workspace files restored" in w for w in outcome.warnings)
