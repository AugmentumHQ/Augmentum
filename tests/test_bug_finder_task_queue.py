"""TaskQueue tests — exercise the substrate for the Lead Agent.

Covers idempotent enqueue, priority-aware take_next, kind filtering,
status transitions (pending → in_progress → terminal), multi-tenant
scoping, parent-child task chains, resume safety
(``reset_in_progress``), the summary renderer, and the BugFinderTask
dataclass invariants.
"""

from __future__ import annotations

import time

import pytest

from augmentum.bug_finder.task_queue import (
    BugFinderTask,
    TaskKind,
    TaskQueue,
    TaskStatus,
    _task_id,
    render_queue_summary,
)
from augmentum.state.backends.sqlite import SQLiteBackend


@pytest.fixture
async def queue():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    yield TaskQueue(backend.conn)
    await backend.close()


# ---------------------------------------------------------------------------
# _task_id — stable hashing
# ---------------------------------------------------------------------------


def test_task_id_stable_for_same_inputs() -> None:
    a = _task_id("run-1", "detect", {"file": "x.py", "function": "foo"})
    b = _task_id("run-1", "detect", {"file": "x.py", "function": "foo"})
    assert a == b
    assert a.startswith("tsk_")


def test_task_id_changes_with_target() -> None:
    a = _task_id("run-1", "detect", {"file": "x.py", "function": "foo"})
    b = _task_id("run-1", "detect", {"file": "x.py", "function": "bar"})
    assert a != b


def test_task_id_changes_with_kind() -> None:
    a = _task_id("run-1", "detect", {"file": "x.py"})
    b = _task_id("run-1", "verify", {"file": "x.py"})
    assert a != b


def test_task_id_changes_with_run() -> None:
    a = _task_id("run-1", "detect", {"file": "x.py"})
    b = _task_id("run-2", "detect", {"file": "x.py"})
    assert a != b


def test_task_id_invariant_under_dict_order() -> None:
    """The hash must be invariant under JSON-key insertion order so two
    enqueues with the same content collapse regardless of how the
    target dict was built."""
    a = _task_id("r", "detect", {"file": "x.py", "function": "foo"})
    b = _task_id("r", "detect", {"function": "foo", "file": "x.py"})
    assert a == b


# ---------------------------------------------------------------------------
# enqueue — idempotency + basic shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_creates_a_pending_task(queue) -> None:
    task = await queue.enqueue(
        run_id="r1", user_id="u1",
        kind=TaskKind.DETECT,
        target={"file": "auth.py", "function": "login"},
        reason="user-supplied focus",
        priority=8,
        created_by="planner",
    )
    assert task.task_id.startswith("tsk_")
    assert task.kind == "detect"
    assert task.status == TaskStatus.PENDING.value
    assert task.priority == 8
    assert task.target == {"file": "auth.py", "function": "login"}
    assert task.reason == "user-supplied focus"
    assert task.created_by == "planner"


@pytest.mark.asyncio
async def test_enqueue_is_idempotent(queue) -> None:
    """Two enqueues with the same (run_id, kind, target) collapse to
    one row. First writer's metadata wins — repeat enqueue's priority
    + reason are ignored."""
    first = await queue.enqueue(
        run_id="r1", user_id="u1",
        kind=TaskKind.DETECT,
        target={"file": "x.py"}, reason="first", priority=3,
    )
    second = await queue.enqueue(
        run_id="r1", user_id="u1",
        kind=TaskKind.DETECT,
        target={"file": "x.py"}, reason="second", priority=9,
    )
    assert second.task_id == first.task_id
    assert second.reason == "first"     # first writer wins
    assert second.priority == 3
    tasks = await queue.list_tasks(run_id="r1", user_id="u1")
    assert len(tasks) == 1


@pytest.mark.asyncio
async def test_enqueue_clamps_priority_to_range(queue) -> None:
    low = await queue.enqueue(
        run_id="r1", user_id="u1",
        kind=TaskKind.DETECT, target={"file": "a"}, priority=-5,
    )
    high = await queue.enqueue(
        run_id="r1", user_id="u1",
        kind=TaskKind.DETECT, target={"file": "b"}, priority=999,
    )
    assert low.priority == 1
    assert high.priority == 10


# ---------------------------------------------------------------------------
# take_next — priority ordering + status transition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_take_next_returns_highest_priority_first(queue) -> None:
    for prio, fn in [(3, "low"), (8, "high"), (5, "mid")]:
        await queue.enqueue(
            run_id="r1", user_id="u1",
            kind=TaskKind.DETECT,
            target={"file": "x.py", "function": fn}, priority=prio,
        )
    t = await queue.take_next(run_id="r1", user_id="u1")
    assert t is not None
    assert t.target["function"] == "high"
    assert t.status == TaskStatus.IN_PROGRESS.value


@pytest.mark.asyncio
async def test_take_next_uses_creation_order_as_tiebreaker(queue) -> None:
    """Same priority → oldest first."""
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "first.py"}, priority=5,
    )
    # Small sleep to push the second creation_at past the first
    time.sleep(1.1)
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "second.py"}, priority=5,
    )
    t = await queue.take_next(run_id="r1", user_id="u1")
    assert t is not None
    assert t.target["file"] == "first.py"


@pytest.mark.asyncio
async def test_take_next_returns_none_when_empty(queue) -> None:
    t = await queue.take_next(run_id="nonexistent", user_id="u1")
    assert t is None


@pytest.mark.asyncio
async def test_take_next_filters_by_kind(queue) -> None:
    """The detector stage only takes DETECT tasks even if a higher-
    priority VERIFY is queued. Stage-bounded drain."""
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.VERIFY,
        target={"finding_id": "abc"}, priority=10,
    )
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "x.py"}, priority=5,
    )
    t = await queue.take_next(
        run_id="r1", user_id="u1", kinds=(TaskKind.DETECT,),
    )
    assert t is not None
    assert t.kind == TaskKind.DETECT.value


@pytest.mark.asyncio
async def test_take_next_skips_in_progress(queue) -> None:
    """Once a task is taken its status flips and the next take_next
    skips it. The substrate guarantees no double-dispatch."""
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "x.py"}, priority=5,
    )
    first = await queue.take_next(run_id="r1", user_id="u1")
    assert first is not None
    second = await queue.take_next(run_id="r1", user_id="u1")
    assert second is None


# ---------------------------------------------------------------------------
# Terminal transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_completed_records_summary(queue) -> None:
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "x.py"},
    )
    t = await queue.take_next(run_id="r1", user_id="u1")
    assert t is not None
    await queue.mark_completed(
        t.task_id, user_id="u1", result_summary="2 findings emitted",
    )
    tasks = await queue.list_tasks(
        run_id="r1", user_id="u1", status=TaskStatus.COMPLETED,
    )
    assert len(tasks) == 1
    assert tasks[0].result_summary == "2 findings emitted"
    assert tasks[0].completed_at > 0


@pytest.mark.asyncio
async def test_mark_dropped_reaches_terminal_state(queue) -> None:
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "x.py"},
    )
    t = await queue.take_next(run_id="r1", user_id="u1")
    await queue.mark_dropped(t.task_id, user_id="u1", reason="superseded")
    dropped = await queue.list_tasks(
        run_id="r1", user_id="u1", status=TaskStatus.DROPPED,
    )
    assert len(dropped) == 1
    assert dropped[0].result_summary == "superseded"


@pytest.mark.asyncio
async def test_mark_failed_records_reason(queue) -> None:
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "x.py"},
    )
    t = await queue.take_next(run_id="r1", user_id="u1")
    await queue.mark_failed(
        t.task_id, user_id="u1", reason="subagent crashed",
    )
    failed = await queue.list_tasks(
        run_id="r1", user_id="u1", status=TaskStatus.FAILED,
    )
    assert len(failed) == 1


# ---------------------------------------------------------------------------
# reset_in_progress — resume safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_in_progress_bounces_taken_tasks_back_to_pending(queue) -> None:
    """A container restart strands IN_PROGRESS tasks. The resume hook
    flips them back so the next take_next picks them up."""
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "x.py"},
    )
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "y.py"},
    )
    # Take two tasks → both IN_PROGRESS
    await queue.take_next(run_id="r1", user_id="u1")
    await queue.take_next(run_id="r1", user_id="u1")
    # Simulate restart
    n = await queue.reset_in_progress(run_id="r1", user_id="u1")
    assert n == 2
    pending = await queue.list_tasks(
        run_id="r1", user_id="u1", status=TaskStatus.PENDING,
    )
    assert len(pending) == 2


@pytest.mark.asyncio
async def test_reset_in_progress_leaves_terminal_tasks_alone(queue) -> None:
    """Completed/dropped tasks must not be revived on resume."""
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "x.py"},
    )
    t = await queue.take_next(run_id="r1", user_id="u1")
    await queue.mark_completed(t.task_id, user_id="u1")
    await queue.reset_in_progress(run_id="r1", user_id="u1")
    pending = await queue.list_tasks(
        run_id="r1", user_id="u1", status=TaskStatus.PENDING,
    )
    assert pending == []


# ---------------------------------------------------------------------------
# Multi-tenant scoping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_is_user_scoped(queue) -> None:
    """Two users with the same run_id mustn't see each other's tasks."""
    await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "u1.py"},
    )
    await queue.enqueue(
        run_id="r1", user_id="u2", kind=TaskKind.DETECT,
        target={"file": "u2.py"},
    )
    u1 = await queue.list_tasks(run_id="r1", user_id="u1")
    u2 = await queue.list_tasks(run_id="r1", user_id="u2")
    assert len(u1) == 1 and u1[0].target["file"] == "u1.py"
    assert len(u2) == 1 and u2[0].target["file"] == "u2.py"


@pytest.mark.asyncio
async def test_take_next_is_user_scoped(queue) -> None:
    """u1's queue shouldn't surface u2's pending task."""
    await queue.enqueue(
        run_id="r1", user_id="u2", kind=TaskKind.DETECT,
        target={"file": "u2.py"},
    )
    t = await queue.take_next(run_id="r1", user_id="u1")
    assert t is None


# ---------------------------------------------------------------------------
# Parent/child task chains (investigator branching)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parent_task_id_persists(queue) -> None:
    parent = await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.DETECT,
        target={"file": "auth.py", "function": "login"},
        priority=8,
    )
    child = await queue.enqueue(
        run_id="r1", user_id="u1", kind=TaskKind.INVESTIGATE,
        target={"thread_anchor": "auth.py:login"},
        priority=parent.priority + 1,   # investigation outranks parent
        parent_task_id=parent.task_id,
        created_by="investigator",
    )
    tasks = await queue.list_tasks(run_id="r1", user_id="u1")
    assert any(
        t.parent_task_id == parent.task_id for t in tasks
    )
    assert child.created_by == "investigator"


# ---------------------------------------------------------------------------
# Counts + summaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counts_by_status_reflects_lifecycle(queue) -> None:
    for fn in ("a", "b", "c"):
        await queue.enqueue(
            run_id="r1", user_id="u1", kind=TaskKind.DETECT,
            target={"file": "x.py", "function": fn},
        )
    counts = await queue.counts_by_status(run_id="r1", user_id="u1")
    assert counts.get("pending") == 3
    t = await queue.take_next(run_id="r1", user_id="u1")
    await queue.mark_completed(t.task_id, user_id="u1")
    counts = await queue.counts_by_status(run_id="r1", user_id="u1")
    assert counts.get("pending") == 2
    assert counts.get("completed") == 1


def test_render_queue_summary_returns_empty_on_empty_queue() -> None:
    out = render_queue_summary([])
    assert out == "(queue empty)"


def test_render_queue_summary_lists_pending_with_detail() -> None:
    now = int(time.time())
    tasks = [
        BugFinderTask(
            task_id="tsk_1", user_id="u", run_id="r", workspace_id="",
            kind="detect", target={"file": "auth.py", "function": "login"},
            reason="user focus", priority=9,
            status="pending", parent_task_id="", created_by="planner",
            result_summary="", created_at=now, completed_at=0,
        ),
        BugFinderTask(
            task_id="tsk_2", user_id="u", run_id="r", workspace_id="",
            kind="verify", target={"finding_id": "fnd_abc"},
            reason="finding from auth.py", priority=7,
            status="pending", parent_task_id="", created_by="lead",
            result_summary="", created_at=now, completed_at=0,
        ),
    ]
    out = render_queue_summary(tasks)
    assert "PENDING (2)" in out
    assert "p9" in out
    assert "detect" in out
    assert "auth.py" in out
    assert "fnd_abc" in out


def test_render_queue_summary_collapses_completed() -> None:
    """Completed tasks get a compact by-kind line, not full detail."""
    now = int(time.time())
    tasks = [
        BugFinderTask(
            task_id=f"tsk_{i}", user_id="u", run_id="r", workspace_id="",
            kind="detect", target={"file": "x"},
            reason="", priority=5,
            status="completed", parent_task_id="", created_by="planner",
            result_summary="", created_at=now, completed_at=now,
        )
        for i in range(4)
    ]
    out = render_queue_summary(tasks)
    assert "COMPLETED (4)" in out
    assert "4 detect" in out


# ---------------------------------------------------------------------------
# BugFinderTask dataclass invariants
# ---------------------------------------------------------------------------


def test_task_is_frozen() -> None:
    t = BugFinderTask(
        task_id="t", user_id="u", run_id="r", workspace_id="",
        kind="detect", target={}, reason="", priority=5,
        status="pending", parent_task_id="", created_by="planner",
        result_summary="", created_at=0, completed_at=0,
    )
    try:
        t.priority = 9  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("BugFinderTask should be frozen")


def test_is_pending_and_is_terminal_predicates() -> None:
    base = dict(
        task_id="t", user_id="u", run_id="r", workspace_id="",
        kind="detect", target={}, reason="", priority=5,
        parent_task_id="", created_by="planner",
        result_summary="", created_at=0, completed_at=0,
    )
    assert BugFinderTask(**base, status="pending").is_pending
    assert not BugFinderTask(**base, status="completed").is_pending
    assert BugFinderTask(**base, status="completed").is_terminal
    assert BugFinderTask(**base, status="dropped").is_terminal
    assert BugFinderTask(**base, status="failed").is_terminal
    assert not BugFinderTask(**base, status="pending").is_terminal
