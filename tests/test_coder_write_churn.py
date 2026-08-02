"""Unit tests for the same-file write-churn ladder.

2026-07-06: WriteChurnTracker (augmentum/coder/write_churn.py) — the
guard for the successful-rewrite spiral (a 9B native run rewrote one
file 20+ times; every write succeeded so no other breaker fired).
Wired into both native and hybrid loops in phase_act.py.
"""
from __future__ import annotations

from augmentum.coder.write_churn import WriteChurnTracker, churn_nudge_body


def _tracker(nudge_at: int = 3, break_at: int = 6) -> WriteChurnTracker:
    return WriteChurnTracker(nudge_at=nudge_at, break_at=break_at)


def test_nudge_fires_once_at_threshold():
    t = _tracker()
    actions = [t.observe(["/ws/a.py"])[0] for _ in range(5)]
    # Iterations 1-2 silent, 3 nudges, 4-5 silent (one-shot per path).
    assert actions == ["", "", "nudge", "", ""]


def test_break_fires_at_cap():
    t = _tracker(nudge_at=3, break_at=6)
    last = ("", "", 0)
    for _ in range(6):
        last = t.observe(["/ws/a.py"])
    assert last == ("break", "/ws/a.py", 6)


def test_break_wins_over_pending_nudge():
    """A path that jumps past both thresholds in one iteration breaks."""
    t = _tracker(nudge_at=2, break_at=3)
    action, path, n = t.observe(["/ws/a.py", "/ws/a.py", "/ws/a.py"])
    assert action == "break" and path == "/ws/a.py" and n == 3


def test_distinct_paths_tracked_independently():
    t = _tracker(nudge_at=3, break_at=6)
    for _ in range(2):
        assert t.observe(["/ws/a.py"])[0] == ""
        assert t.observe(["/ws/b.py"])[0] == ""
    # Third touch of each nudges per path.
    assert t.observe(["/ws/a.py"]) == ("nudge", "/ws/a.py", 3)
    assert t.observe(["/ws/b.py"]) == ("nudge", "/ws/b.py", 3)


def test_interleaved_reads_do_not_reset():
    """Counts are cumulative per turn — iterations that mutate nothing
    (empty path list) leave the counters untouched, mirroring the live
    failure where occasional shell probes interleaved the rewrites."""
    t = _tracker(nudge_at=3, break_at=6)
    t.observe(["/ws/a.py"])
    t.observe([])  # read/shell-only iteration
    t.observe(["/ws/a.py"])
    assert t.observe(["/ws/a.py"])[0] == "nudge"


def test_escalate_confirms_ignored_nudge():
    """Nudge at 3, escalate at nudge+margin — the post-nudge
    confirmation rung that hands the turn to the buddy model."""
    t = _tracker(nudge_at=3, break_at=15)
    t.escalate_margin = 3
    actions = [t.observe(["/ws/a.py"])[0] for _ in range(8)]
    assert actions == ["", "", "nudge", "", "", "escalate", "", ""]


def test_escalate_requires_prior_nudge():
    """A path that never crossed the nudge rung can't escalate (the
    nudge one-shot is the hypothesis; escalation is its confirmation)."""
    t = _tracker(nudge_at=0, break_at=15)  # nudge rung disabled
    t.escalate_margin = 3
    for _ in range(10):
        assert t.observe(["/ws/a.py"])[0] == ""


def test_reset_counts_gives_buddy_fresh_ladder():
    t = _tracker(nudge_at=3, break_at=6)
    for _ in range(5):
        t.observe(["/ws/a.py"])
    t.reset_counts()
    # Buddy edits the same file legitimately — no instant break/nudge.
    assert t.observe(["/ws/a.py"]) == ("", "", 0)
    assert t.observe(["/ws/a.py"]) == ("", "", 0)
    assert t.observe(["/ws/a.py"])[0] == "nudge"


def test_zero_thresholds_disable_rungs():
    t = _tracker(nudge_at=0, break_at=0)
    for _ in range(50):
        assert t.observe(["/ws/a.py"]) == ("", "", 0)


def test_nudge_body_is_prescriptive():
    body = churn_nudge_body("/ws/a.py", 5)
    assert "/ws/a.py" in body and "5" in body
    # The load-bearing prescriptions.
    assert "hypothesis" in body.lower()
    assert "code_edit" in body
    assert "file_read" in body
