"""Unit tests for the shared task-list plan-spine tracker.

2026-07-06: TaskSpineTracker (augmentum/coder/task_spine.py) is the
single implementation of the staleness nudge (factored out of hybrid's
inline copy) plus the new stop gate native uses. Integration coverage
for the hybrid wiring lives in test_coder_loop_wiring.py
(test_task_stale_nudge_*).
"""
from __future__ import annotations

import pytest

from augmentum.coder.task_spine import TaskSpineTracker
from augmentum.loops.breakers import TASK_STALE_NUDGE_AT


def _tasks(*statuses: str) -> list[dict]:
    return [
        {"content": f"step {i}", "activeForm": f"doing {i}", "status": s}
        for i, s in enumerate(statuses)
    ]


def test_mutation_resets_streak_and_marks_engaged():
    t = TaskSpineTracker.start([])
    assert not t.engaged_this_turn
    mutated, nudge = t.observe(_tasks("in_progress", "pending"))
    assert mutated and nudge == ""
    assert t.engaged_this_turn
    assert t.stale_streak == 0


def test_stale_nudge_fires_once_at_threshold():
    tasks = _tasks("in_progress", "pending")
    t = TaskSpineTracker.start(tasks)
    nudges = []
    for _ in range(TASK_STALE_NUDGE_AT + 3):
        _, n = t.observe(tasks)
        if n:
            nudges.append(n)
    assert len(nudges) == 1
    # in_progress item is named in the body.
    assert "step 0" in nudges[0]


def test_stale_nudge_rearms_after_mutation():
    tasks = _tasks("in_progress")
    t = TaskSpineTracker.start(tasks)
    for _ in range(TASK_STALE_NUDGE_AT):
        t.observe(tasks)
    assert t.stale_nudge_fired
    changed = _tasks("completed")  # model updated the list
    mutated, _ = t.observe(changed)
    assert mutated and not t.stale_nudge_fired


def test_no_streak_without_open_work():
    tasks = _tasks("completed", "completed")
    t = TaskSpineTracker.start(tasks)
    for _ in range(TASK_STALE_NUDGE_AT + 2):
        _, n = t.observe(tasks)
        assert n == ""
    assert t.stale_streak == 0


def test_nudge_disabled_flag_suppresses_body():
    tasks = _tasks("pending")
    t = TaskSpineTracker.start(tasks)
    for _ in range(TASK_STALE_NUDGE_AT + 2):
        _, n = t.observe(tasks, nudge_enabled=False)
        assert n == ""


def test_stop_gate_requires_engagement_this_turn():
    """A leftover list from a PRIOR turn must not block a stop."""
    stale_from_last_turn = _tasks("pending", "in_progress")
    t = TaskSpineTracker.start(stale_from_last_turn)
    t.observe(stale_from_last_turn)  # unchanged — never engaged
    assert t.stop_gate_nudge(stale_from_last_turn) == ""


def test_stop_gate_fires_once_with_open_engaged_work():
    t = TaskSpineTracker.start([])
    tasks = _tasks("completed", "in_progress", "pending")
    t.observe(tasks)  # engaged
    body = t.stop_gate_nudge(tasks)
    assert "2 unfinished" in body
    assert "step 1" in body and "step 2" in body
    # One-shot: second stop passes.
    assert t.stop_gate_nudge(tasks) == ""


def test_stop_gate_silent_when_all_completed():
    t = TaskSpineTracker.start([])
    tasks = _tasks("completed", "completed")
    t.observe(tasks)
    assert t.stop_gate_nudge(tasks) == ""


@pytest.mark.parametrize("bad", [None, "junk", 42])
def test_non_dict_items_ignored(bad):
    t = TaskSpineTracker.start([])
    tasks = [bad, {"content": "real", "status": "pending"}]
    mutated, _ = t.observe(tasks)
    assert mutated
    assert "real" in t.stop_gate_nudge(tasks)
