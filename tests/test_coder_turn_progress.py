"""Unit tests for the coarse turn-progress ceiling.

2026-07-07: TurnProgressLedger (augmentum/coder/turn_progress.py) — the
superset backstop beneath the narrow breakers. No new file changed AND
no additional test passing for N iterations = measurably standing still
(the 147-150-iter Qwen3.6-35B runs touched the same files and re-ran the
same tests forever without either measure moving).
"""
from __future__ import annotations

from augmentum.coder.turn_progress import (
    TurnProgressLedger,
    progress_stall_nudge_body,
)

PASS_30 = "=== 30 passed in 0.4s ==="
PASS_31 = "=== 31 passed in 0.4s ==="


def test_new_file_change_counts_as_progress():
    led = TurnProgressLedger(stall_nudge=5, stall_break=8)
    for i in range(1, 20):
        # A brand-new file every iteration → never stalls.
        assert led.note(i, [f"/ws/file_{i}.py"], []) == ""


def test_more_tests_passing_counts_as_progress():
    led = TurnProgressLedger(stall_nudge=5, stall_break=8)
    assert led.note(1, [], [PASS_30]) == ""
    for i in range(2, 12):
        # Same file set, but the passing count keeps rising.
        sig = f"pytest:passed={30 + i}"
        assert led.note(i, [], [sig]) == ""


def test_stall_nudges_then_breaks():
    led = TurnProgressLedger(stall_nudge=5, stall_break=8)
    led.note(1, ["/ws/a.py"], [PASS_30])   # last progress = iter 1
    out = [led.note(i, ["/ws/a.py"], [PASS_30]) for i in range(2, 12)]
    # Re-touching the SAME file + SAME pass count = no progress.
    # Nudge fires when stalled reaches 5 (iter 6), break at 8 (iter 9).
    assert out[4] == "nudge"      # iteration 6
    assert out[7] == "break"      # iteration 9
    # No double-nudge before the break.
    assert out.count("nudge") == 1


def test_progress_rearms_the_nudge():
    led = TurnProgressLedger(stall_nudge=3, stall_break=99)
    led.note(1, ["/ws/a.py"], [])
    assert led.note(4, ["/ws/a.py"], []) == "nudge"   # stalled 3
    # A real step re-arms — a LATER stall can nudge again.
    assert led.note(5, ["/ws/b.py"], []) == ""        # new file → progress
    assert led.note(8, ["/ws/b.py"], []) == "nudge"   # stalled 3 again


def test_reset_after_handoff_rebaselines_clock_not_state():
    led = TurnProgressLedger(stall_nudge=4, stall_break=6)
    led.note(1, ["/ws/a.py"], [PASS_31])
    for i in range(2, 6):
        led.note(i, ["/ws/a.py"], [PASS_31])
    led.reset_after_handoff(6)   # buddy takes over at iter 6
    # Accumulated file/test state survives (re-touching a.py, same pass
    # count is still no-progress) but the stall clock restarts at 6.
    assert led.note(9, ["/ws/a.py"], [PASS_31]) == ""     # stalled 3 < 4
    assert led.note(10, ["/ws/a.py"], [PASS_31]) == "nudge"  # stalled 4


def test_body_is_prescriptive():
    body = progress_stall_nudge_body(30, 34)
    assert "30 iterations" in body and "stuck" in body
    assert "hand back to the user" in body
