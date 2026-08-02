"""Duplicate-read ladder: tracker, context repair, lesson bodies.

Encodes the grep-carousel failure (ctr_21c9e2a7…: 101/146 calls =
code_grep cycling identical (pattern, path) pairs) and the 2026-07-06
direction: re-orient the model without the damage but maintaining the
lesson — repair, don't cancel.
"""
from __future__ import annotations

from augmentum.coder.duplicate_calls import (
    PRUNED_STUB,
    DuplicateCallTracker,
    duplicate_nudge_body,
    prune_duplicate_results,
    reorientation_body,
)
from augmentum.models.base import Message

TRACKED = frozenset({"code_grep", "file_read"})


def _tracker(**kw) -> DuplicateCallTracker:
    args = {"nudge_at": 4, "reorient_margin": 3, "tracked_tools": TRACKED}
    args.update(kw)
    return DuplicateCallTracker(**args)


def _obs(t, n, *, tool="code_grep", inp=None, out="same result", start=0):
    """Run n identical observations; return list of non-empty actions."""
    actions = []
    for i in range(n):
        a, _ = t.observe(
            tool_id=f"id{start + i}", tool=tool,
            tool_input=inp or {"pattern": "uOutline", "path": "a.js"},
            output=out,
        )
        if a:
            actions.append(a)
    return actions


# ---------------------------------------------------------------------------
# Tracker ladder
# ---------------------------------------------------------------------------

def test_below_threshold_no_action():
    assert _obs(_tracker(), 3) == []


def test_nudge_then_reorient_then_escalate():
    t = _tracker()
    actions = _obs(t, 10)
    # nudge at 4, reorient at 7, escalate at 10 (reorient + margin 3)
    assert actions == ["nudge", "reorient", "escalate"]


def test_cycle_of_distinct_calls_counted_per_key():
    # A,B,C,A,B,C… — the consecutive-identical blind spot. Each key
    # nudges independently when ITS count hits the threshold.
    t = _tracker()
    actions = []
    for i in range(12):
        inp = {"pattern": "x", "path": f"f{i % 3}.js"}
        a, _ = t.observe(tool_id=f"id{i}", tool="code_grep", tool_input=inp, output="r")
        if a:
            actions.append((a, i))
    # Each of the 3 keys reaches count 4 on rounds 9, 10, 11.
    assert [a for a, _ in actions] == ["nudge", "nudge", "nudge"]


def test_second_key_reaching_reorient_escalates():
    t = _tracker()
    assert _obs(t, 7, inp={"p": 1}) == ["nudge", "reorient"]
    # A different call now loops to the reorient rung — the window was
    # already repaired once; confirm upward instead of repairing again.
    assert _obs(t, 7, inp={"p": 2}, start=100) == ["nudge", "escalate"]


def test_untracked_and_mutating_tools_ignored():
    t = _tracker()
    for i in range(10):
        a, rec = t.observe(
            tool_id=f"id{i}", tool="code_edit",
            tool_input={"path": "a.py"}, output="ok",
        )
        assert a == "" and rec is None


def test_different_inputs_are_different_keys():
    t = _tracker()
    for i in range(10):
        a, _ = t.observe(
            tool_id=f"id{i}", tool="file_read",
            tool_input={"path": f"file{i}.py"}, output="content",
        )
        assert a == ""


# ---------------------------------------------------------------------------
# Context repair
# ---------------------------------------------------------------------------

def _history(rec_ids: list[str]) -> list:
    msgs = [Message(role="user", content="fix the outline shader")]
    for tid in rec_ids:
        msgs.append(Message(role="assistant", content=""))
        msgs.append(Message(role="tool", content=f"grep result for {tid}", tool_call_id=tid))
    return msgs


def test_prune_keeps_first_result_stubs_rest():
    t = _tracker()
    for i in range(7):
        action, rec = t.observe(
            tool_id=f"t{i}", tool="code_grep",
            tool_input={"pattern": "x"}, output="hit at line 3",
        )
    assert action == "reorient"
    msgs = _history([f"t{i}" for i in range(7)])
    pruned = prune_duplicate_results(msgs, rec)
    assert pruned == 6
    tool_msgs = [m for m in msgs if m.role == "tool"]
    assert tool_msgs[0].content == "grep result for t0"
    assert all(m.content == PRUNED_STUB for m in tool_msgs[1:])
    # Pairing untouched: every tool_call_id survives.
    assert [m.tool_call_id for m in tool_msgs] == [f"t{i}" for i in range(7)]
    # Idempotent — a second pass stubs nothing new.
    assert prune_duplicate_results(msgs, rec) == 0


def test_prune_leaves_unrelated_messages_alone():
    t = _tracker()
    for i in range(7):
        _, rec = t.observe(
            tool_id=f"t{i}", tool="code_grep",
            tool_input={"pattern": "x"}, output="r",
        )
    msgs = _history(["t0", "t1", "other_call", "t2"])
    prune_duplicate_results(msgs, rec)
    other = [m for m in msgs if getattr(m, "tool_call_id", "") == "other_call"]
    assert other[0].content == "grep result for other_call"
    assert msgs[0].content == "fix the outline shader"


# ---------------------------------------------------------------------------
# Lesson bodies
# ---------------------------------------------------------------------------

def test_reorientation_body_keeps_the_lesson():
    t = _tracker()
    for i in range(7):
        _, rec = t.observe(
            tool_id=f"t{i}", tool="code_grep",
            tool_input={"pattern": "uOutline"}, output="uOutlineThickness: line 42",
        )
    body = reorientation_body(rec)
    assert "code_grep" in body
    assert "7 times" in body
    assert "uOutlineThickness: line 42" in body        # ground truth kept
    assert "IDENTICAL every single time" in body       # sameness signal
    assert "EXHAUSTED" in body                         # the lesson
    assert "different next action" in body             # the redirection


def test_nudge_body_names_count_and_redirects():
    t = _tracker()
    for i in range(4):
        _, rec = t.observe(
            tool_id=f"t{i}", tool="file_read",
            tool_input={"path": "a.py"}, output="content",
        )
    body = duplicate_nudge_body(rec)
    assert "file_read" in body and "4 times" in body
    assert "DIFFERENT" in body
