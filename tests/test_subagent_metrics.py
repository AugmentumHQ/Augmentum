"""Phase-0 subagent metrics — the measurement foundation.

Pins ``augmentum/agents/metrics.py``, the scorecard every later phase of
the subagent-professionalization program is graded against
(``docs/superpowers/specs/2026-06-19-subagent-professionalization.md``).
Pure functions over the ``_row_to_dict`` run shape — no model, no DB.

The two load-bearing assertions: (1) tool-efficiency counts only
successful, non-empty, non-repeat calls as "new info"; (2) verification
grading surfaces the FALSE-POSITIVE rate (judge said passed, run actually
failed) — the over-validation error Phase 1's fail-closed change targets.
"""

from __future__ import annotations

import pytest

from augmentum.agents.metrics import (
    aggregate,
    grade_verification,
    reward_hacking_gap,
    tool_efficiency,
)


def _call(tool: str, *, outcome: str = "success", output_len: int = 100, args=None):
    return {
        "iteration": 1,
        "tool": tool,
        "args": args or {},
        "outcome": outcome,
        "reason": "",
        "output_len": output_len,
        "elapsed_ms": 5,
    }


def _run(role: str = "explore", **over):
    base = {
        "role": role,
        "stop_reason": "complete",
        "verification": "unchecked",
        "iterations": 3,
        "tokens_in": 1000,
        "tokens_out": 200,
        "wallclock_ms": 4000,
        "tool_call_log": [],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------- tool-efficiency


def test_tool_efficiency_none_when_no_calls():
    assert tool_efficiency([]) is None
    assert tool_efficiency(None) is None


def test_tool_efficiency_all_productive():
    log = [_call("read", args={"p": "a"}), _call("grep", args={"q": "x"})]
    assert tool_efficiency(log) == 1.0


def test_tool_efficiency_failed_and_empty_calls_dont_count():
    log = [
        _call("read", args={"p": "a"}),                 # new info
        _call("read", outcome="failure", args={"p": "b"}),  # dead
        _call("grep", output_len=0, args={"q": "z"}),       # succeeded but empty
    ]
    # 1 of 3 calls discovered new info
    assert tool_efficiency(log) == 1 / 3


def test_tool_efficiency_exact_repeat_is_not_new_info():
    log = [
        _call("read", args={"p": "same"}),  # new info
        _call("read", args={"p": "same"}),  # repeat → no new info
    ]
    assert tool_efficiency(log) == 0.5


# ---------------------------------------------------------------- verification grading


def test_grade_verification_empty():
    g = grade_verification([])
    assert g.n_labeled == 0
    assert g.accuracy is None


def test_grade_verification_false_positive_rate():
    # Two runs the eval says FAILED; the judge passed one of them (a false
    # positive) and correctly withheld pass on the other.
    graded = [
        (_run(verification="passed"), False),   # FP: judge said pass, actually fail
        (_run(verification="failed"), False),   # correct reject
        (_run(verification="passed"), True),    # correct pass
    ]
    g = grade_verification(graded)
    assert g.n_labeled == 3
    assert g.accuracy == 2 / 3
    # 1 false positive out of 2 actually-failing runs
    assert g.false_positive_rate == 0.5
    assert g.false_negative_rate == 0.0


def test_grade_verification_error_verdict_is_not_a_pass():
    # A fail-open "error" verdict must NOT count as a predicted pass — this
    # is the exact behavior Phase 1 hardens. If the run actually failed, an
    # "error" verdict that we treat as not-pass is the CORRECT reject.
    graded = [(_run(verification="error"), False)]
    g = grade_verification(graded)
    assert g.accuracy == 1.0
    assert g.false_positive_rate == 0.0


# ---------------------------------------------------------------- reward-hacking gap


def test_reward_hacking_gap_none_until_phase2():
    assert reward_hacking_gap(None, None) is None
    assert reward_hacking_gap(1.0, None) is None


def test_reward_hacking_gap_positive_signals_gaming():
    # Passes everything visible, fails everything held-out → gap 1.0
    assert reward_hacking_gap(1.0, 0.0) == 1.0
    assert reward_hacking_gap(0.9, 0.6) == pytest.approx(0.3)


# ---------------------------------------------------------------- aggregate scorecard


def test_aggregate_rolls_up_stop_reasons_and_per_role():
    runs = [
        _run("explore", stop_reason="complete",
             tool_call_log=[_call("read", args={"p": "a"})]),
        _run("explore", stop_reason="budget",
             tool_call_log=[_call("read", outcome="failure", args={"p": "b"})]),
        _run("review", stop_reason="complete", verification="passed"),
    ]
    card = aggregate(runs)
    assert card.n_runs == 3
    assert card.stop_reasons == {"complete": 2, "budget": 1}
    assert card.verification_counts["unchecked"] == 2
    assert card.verification_counts["passed"] == 1
    assert set(card.per_role) == {"explore", "review"}
    assert card.per_role["explore"]["n_runs"] == 2
    # explore: one run 1.0 efficiency, one run 0.0 → mean 0.5
    assert card.per_role["explore"]["mean_tool_efficiency"] == 0.5


def test_aggregate_with_ground_truth_grades_verification():
    runs = [_run(verification="passed"), _run(verification="passed")]
    graded = [(runs[0], True), (runs[1], False)]  # second is a false positive
    card = aggregate(runs, graded=graded)
    assert card.verification.n_labeled == 2
    assert card.verification.false_positive_rate == 1.0  # 1 FP / 1 actual-fail


def test_aggregate_to_dict_is_json_safe():
    import json

    card = aggregate([_run()])
    blob = json.dumps(card.to_dict())
    assert "n_runs" in blob
