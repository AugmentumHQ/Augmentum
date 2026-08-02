"""Unit tests for the command-carousel detector.

2026-07-07: CommandCarouselTracker (augmentum/coder/command_carousel.py)
— the guard for the test/probe re-run carousel (three Qwen3.6-35B runs
of 147-150 iterations re-running one pytest command dozens of ways,
varying only the output-shaping suffix). Wired into the native loop in
phase_act.py alongside the probe-signal and duplicate-read ladders.
"""
from __future__ import annotations

from augmentum.coder.command_carousel import (
    CommandCarouselTracker,
    carousel_nudge_body,
    carousel_reorientation_body,
    extract_signal,
    flaky_test_body,
    normalize_command,
)

PYTEST = "cd /workspace && python3 -m pytest ide/tests/ -v --tb=short"
OUT_RED = "collected 36 items\n...\n=== 34 passed, 2 failed in 0.41s ==="


# ── normalization ─────────────────────────────────────────────────────


def test_normalize_strips_output_shaping_pipe_tail():
    a = normalize_command(PYTEST + " 2>&1 | tail -80")
    b = normalize_command(PYTEST + " 2>&1 | tail -40")
    c = normalize_command(PYTEST + " | grep -E '(PASSED|FAILED)' | wc -l")
    assert a == b == c == PYTEST


def test_normalize_leaves_leading_grep_and_flags_intact():
    # A primary grep (no preceding pipe) is the real command — untouched.
    assert normalize_command("grep -rn foo src/") == "grep -rn foo src/"
    # Genuinely different invocations stay distinct.
    assert normalize_command("pytest a.py") != normalize_command("pytest b.py")


def test_normalize_collapses_redirections_and_whitespace():
    assert normalize_command("python3   x.py   2>/dev/null") == "python3 x.py"
    assert normalize_command("python3 x.py > out.txt") == "python3 x.py"


# ── signal extraction ─────────────────────────────────────────────────


def test_extract_signal_prefers_pytest_tally():
    assert extract_signal(OUT_RED) == "pytest:failed=2,passed=34"
    # A different tally is a different signal.
    assert extract_signal("=== 36 passed in 0.5s ===") != extract_signal(OUT_RED)


def test_extract_signal_falls_back_to_error_then_raw():
    err = extract_signal("Traceback...\nValueError: bad thing here")
    assert err.startswith("err:")
    raw = extract_signal("just some prose with no tally")
    assert raw.startswith("raw:")
    # Whitespace reflow doesn't change the raw signal.
    assert extract_signal("a  b\nc") == extract_signal("a b c  ")


# ── the spin ladder ───────────────────────────────────────────────────


def _rerun(t: CommandCarouselTracker, n: int, out: str, cmd: str = PYTEST):
    """n re-runs of the same command with the same output (no edits)."""
    return [
        t.observe(tool_id=f"tc_{i}", command=cmd, output=out)[0]
        for i in range(n)
    ]


def test_ladder_nudge_reorient_escalate_on_unmoving_signal():
    t = CommandCarouselTracker(nudge_at=4, reorient_margin=3, escalate_margin=3)
    # First run establishes the signal (improvement, no stale); runs
    # 2.. accrue stale. nudge at stale=4 (run 5), reorient at 7 (run 8),
    # escalate at 10 (run 11).
    actions = _rerun(t, 11, OUT_RED)
    assert actions[4] == "nudge"
    assert actions[7] == "reorient"
    assert actions[10] == "escalate"
    # Nothing fires before the nudge rung.
    assert actions[:4] == ["", "", "", ""]


def test_improvement_resets_the_stale_count():
    t = CommandCarouselTracker(nudge_at=3, reorient_margin=3)
    _rerun(t, 3, OUT_RED)  # stale climbing
    # A run where more tests pass is genuine progress → resets.
    green = "=== 36 passed in 0.5s ==="
    assert t.observe(tool_id="g", command=PYTEST, output=green)[0] == ""
    # Streak restarts from the new (better) baseline; re-running the
    # green result now accrues stale again but from zero.
    assert _rerun(t, 2, green) == ["", ""]


def test_fewer_failures_counts_as_progress():
    t = CommandCarouselTracker(nudge_at=2, reorient_margin=3)
    _rerun(t, 2, OUT_RED)  # 34 passed, 2 failed
    better = "=== 35 passed, 1 failed in 0.4s ==="
    assert t.observe(tool_id="b", command=PYTEST, output=better)[0] == ""


def test_distinct_commands_tracked_independently():
    t = CommandCarouselTracker(nudge_at=2, reorient_margin=3)
    acts = []
    # Interleave two different commands — each accrues its own stale
    # independent of the other's runs.
    for i in range(3):
        acts.append(t.observe(tool_id=f"a{i}", command="python3 a.py", output="same")[0])
        acts.append(t.observe(tool_id=f"b{i}", command="python3 b.py", output="same")[0])
    # a runs at acts[0,2,4] → stale 0,1,2 → nudge on its 3rd run.
    # b runs at acts[1,3,5] → same, independently. Interleaving doesn't
    # cross-contaminate the two commands' counts.
    assert acts[4] == "nudge"
    assert acts[5] == "nudge"


# ── flaky-test flag (#5) ──────────────────────────────────────────────


def test_flaky_flag_on_changed_result_without_edit():
    t = CommandCarouselTracker(nudge_at=99, reorient_margin=3)
    t.observe(tool_id="1", command=PYTEST, output="=== 34 passed, 2 failed ===")
    # Same command, DIFFERENT result, no note_mutations between → flaky.
    _, rec = t.observe(
        tool_id="2", command=PYTEST, output="=== 35 passed, 1 failed ===",
    )
    assert rec is not None and rec.just_flagged_flaky is True
    # One-shot: a further flap doesn't re-flag.
    _, rec2 = t.observe(
        tool_id="3", command=PYTEST, output="=== 33 passed, 3 failed ===",
    )
    assert rec2.just_flagged_flaky is False


def test_result_change_after_edit_is_not_flaky():
    t = CommandCarouselTracker(nudge_at=99, reorient_margin=3)
    t.observe(tool_id="1", command=PYTEST, output="=== 34 passed, 2 failed ===")
    t.note_mutations(1)  # an edit landed between the two runs
    _, rec = t.observe(
        tool_id="2", command=PYTEST, output="=== 35 passed, 1 failed ===",
    )
    assert rec.just_flagged_flaky is False


# ── disable / reset ───────────────────────────────────────────────────


def test_zero_threshold_and_empty_inputs_disable():
    t = CommandCarouselTracker(nudge_at=0, reorient_margin=3)
    assert _rerun(t, 6, OUT_RED) == [""] * 6
    t2 = CommandCarouselTracker(nudge_at=1, reorient_margin=3)
    assert t2.observe(tool_id="x", command="", output=OUT_RED) == ("", None)
    assert t2.observe(tool_id="y", command=PYTEST, output="   ") == ("", None)


def test_reset_rearms_for_buddy():
    t = CommandCarouselTracker(nudge_at=2, reorient_margin=3)
    acts = _rerun(t, 3, OUT_RED)
    assert "nudge" in acts
    t.reset()
    assert _rerun(t, 2, OUT_RED) == ["", ""]  # fresh budget, no nudge yet


# ── bodies ────────────────────────────────────────────────────────────


def test_bodies_are_prescriptive():
    t = CommandCarouselTracker(nudge_at=1, reorient_margin=1)
    t.observe(tool_id="1", command=PYTEST, output=OUT_RED)
    _, rec = t.observe(tool_id="2", command=PYTEST, output=OUT_RED)
    nudge = carousel_nudge_body(rec)
    assert "re-run" in nudge.lower() and "code has to change" in nudge
    reo = carousel_reorientation_body(rec)
    assert "<reorientation>" in reo and "EXHAUSTED" in reo
    flaky = flaky_test_body(rec)
    assert "non-deterministic" in flaky and "MOVE ON" in flaky
