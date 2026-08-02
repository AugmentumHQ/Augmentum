"""Unit tests for the always-green-probe detector.

2026-07-06: ProbeSignalTracker (augmentum/coder/probe_signal.py) — the
guard for the verification-probe-that-cannot-fail pattern (a 9B run
"verified" every edit with a print-script whose output never changed).
Wired into the native loop in phase_act.py.
"""
from __future__ import annotations

from augmentum.coder.probe_signal import (
    ProbeSignalTracker,
    probe_no_signal_nudge_body,
)

CMD = "cd /workspace/ide && python3 test_all.py"
OUT = "Testing all IDE commands...\n1. --help: OK\n2. explorer: OK\n"


def _edit_probe_cycles(t: ProbeSignalTracker, n: int, out: str = OUT) -> list[str]:
    """n cycles of (mutation lands) → (same probe, same output)."""
    actions = []
    for _ in range(n):
        t.note_mutations(1)
        actions.append(t.observe_probe(CMD, out))
    return actions


def test_nudges_after_threshold_unchanged_probes_across_edits():
    t = ProbeSignalTracker(nudge_at=3)
    t.observe_probe(CMD, OUT)  # baseline run
    actions = _edit_probe_cycles(t, 4)
    # Repeats 1-2 silent, 3rd unchanged-after-edit rerun nudges, then one-shot.
    assert actions == ["", "", "nudge", ""]


def test_changed_output_resets_the_count():
    t = ProbeSignalTracker(nudge_at=3)
    t.observe_probe(CMD, OUT)
    _edit_probe_cycles(t, 2)
    # The probe DID change once — it carries signal after all.
    t.note_mutations(1)
    assert t.observe_probe(CMD, OUT + "3. terminal: OK\n") == ""
    # Streak restarts from the new baseline.
    assert _edit_probe_cycles(t, 3, OUT + "3. terminal: OK\n") == ["", "", "nudge"]


def test_rerun_without_intervening_edit_is_neutral():
    """Plain re-runs (no mutations between) are the identical-call
    detector's shape, not ours — they prove nothing about the probe."""
    t = ProbeSignalTracker(nudge_at=2)
    t.observe_probe(CMD, OUT)
    for _ in range(10):
        assert t.observe_probe(CMD, OUT) == ""


def test_whitespace_reflow_still_matches():
    t = ProbeSignalTracker(nudge_at=1)
    t.observe_probe(CMD, OUT)
    t.note_mutations(2)
    assert t.observe_probe(CMD, OUT.replace("\n", "  \n ") + "\n\n") == "nudge"


def test_distinct_commands_tracked_independently():
    t = ProbeSignalTracker(nudge_at=1)
    t.observe_probe("python3 a.py", "ok")
    t.observe_probe("python3 b.py", "ok")
    t.note_mutations(1)
    assert t.observe_probe("python3 a.py", "ok") == "nudge"
    # One-shot per TURN (whole verification habit, not per command).
    assert t.observe_probe("python3 b.py", "ok") == ""


def test_reset_rearms_for_buddy():
    t = ProbeSignalTracker(nudge_at=1)
    t.observe_probe(CMD, OUT)
    t.note_mutations(1)
    assert t.observe_probe(CMD, OUT) == "nudge"
    t.reset()
    t.observe_probe(CMD, OUT)
    t.note_mutations(1)
    assert t.observe_probe(CMD, OUT) == "nudge"


def test_zero_threshold_and_empty_inputs_disable():
    t = ProbeSignalTracker(nudge_at=0)
    assert _edit_probe_cycles(t, 5) == [""] * 5
    t2 = ProbeSignalTracker(nudge_at=1)
    t2.observe_probe("", OUT)
    t2.observe_probe(CMD, "   ")
    t2.note_mutations(1)
    assert t2.observe_probe("", OUT) == ""


def test_nudge_body_is_prescriptive():
    body = probe_no_signal_nudge_body(CMD, 3)
    assert "test_run" in body and "assert" in body
    assert "cannot verify" in body or "not verification" in body
    assert CMD[:40] in body
