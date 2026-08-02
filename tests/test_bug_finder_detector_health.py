"""Detector-health gate (bug_finder/orchestrator.py::evaluate_detector_health).

Pins the trust-critical behavior this shipment introduced: a run whose
detector fan-out collapsed (most detectors stop with
``stop_reason="error"``) must be reported DEGRADED, not "complete / no
findings". Field data motivating it — 06-14 run bfr_f80deb251ce5:
396/399 detectors errored at iteration 0 (provider rate-limit under the
high-concurrency fan-out), yet the pipeline reported "complete / no
findings", which reads to the user as "your code is clean."

We test the pure decision function directly (the gate inside
``_build_report`` is a thin wrapper that rewrites stop_reason from it).
"""

from __future__ import annotations

from augmentum.bug_finder.orchestrator import (
    CostLedgerEntry,
    evaluate_detector_health,
)


def _entry(stage: str, stop_reason: str) -> CostLedgerEntry:
    return CostLedgerEntry(
        stage=stage,
        role=stage,
        model="m",
        instance_id="i",
        iterations=1,
        tokens_in=10,
        tokens_out=5,
        wallclock_ms=100,
        stop_reason=stop_reason,
    )


def _detectors(complete: int = 0, error: int = 0, budget: int = 0, stuck: int = 0):
    out = []
    out += [_entry("detector", "complete")] * complete
    out += [_entry("detector", "error")] * error
    out += [_entry("detector", "budget")] * budget
    out += [_entry("detector", "stuck")] * stuck
    return out


def test_healthy_run_not_degraded():
    h = evaluate_detector_health(_detectors(complete=10))
    assert h["detectors_total"] == 10
    assert h["detectors_errored"] == 0
    assert h["detector_error_rate"] == 0.0
    assert h["degraded"] is False


def test_mass_error_is_degraded():
    # 396/399 errored — the real field case.
    h = evaluate_detector_health(_detectors(complete=3, error=396))
    assert h["detectors_total"] == 399
    assert h["detectors_errored"] == 396
    assert h["detector_error_rate"] == round(396 / 399, 3)
    assert h["degraded"] is True


def test_no_detectors_never_degraded():
    """The no-chunks / empty-workspace path runs zero detectors. Zero
    errors out of zero is NOT degraded — that path has its own honest
    stop_detail ('no chunks'), and dividing by zero must not flip it."""
    h = evaluate_detector_health([])
    assert h["detectors_total"] == 0
    assert h["degraded"] is False
    # A ledger with only non-detector stages is likewise clean.
    h2 = evaluate_detector_health([_entry("planner", "complete"), _entry("comprehender", "budget")])
    assert h2["detectors_total"] == 0
    assert h2["degraded"] is False


def test_budget_and_stuck_do_not_count_as_errors():
    """Only stop_reason='error' (infrastructure failure) counts against
    health. budget = ran-and-hit-a-cap, stuck = ran-and-looped — both
    produced real work, neither is lost signal the way an error is."""
    h = evaluate_detector_health(_detectors(complete=2, budget=5, stuck=3))
    assert h["detectors_errored"] == 0
    assert h["detectors_budget"] == 5
    assert h["degraded"] is False


def test_threshold_boundary_inclusive():
    # Exactly at threshold trips (>=), just under does not.
    at = evaluate_detector_health(_detectors(complete=5, error=5), threshold=0.5)
    assert at["detector_error_rate"] == 0.5
    assert at["degraded"] is True
    under = evaluate_detector_health(_detectors(complete=6, error=4), threshold=0.5)
    assert under["detector_error_rate"] == 0.4
    assert under["degraded"] is False


def test_threshold_one_disables_gate_unless_all_errored():
    # threshold=1.0 → only an all-errored stage degrades.
    almost = evaluate_detector_health(_detectors(complete=1, error=99), threshold=1.0)
    assert almost["degraded"] is False  # 0.99 < 1.0
    allbad = evaluate_detector_health(_detectors(error=10), threshold=1.0)
    assert allbad["degraded"] is True


def test_single_errored_detector_is_degraded():
    """If the only detector that ran errored, 100% >= threshold → the
    run is degraded. One errored detector IS a failed scan."""
    h = evaluate_detector_health(_detectors(error=1))
    assert h["detector_error_rate"] == 1.0
    assert h["degraded"] is True


def test_verifier_errors_reported_but_do_not_gate():
    """Verifier error counts ride along for visibility, but an errored
    verifier doesn't (yet) trip the gate — the detector gate catches the
    dominant case, and a *refused* finding is a legit verdict."""
    ledger = _detectors(complete=5) + [
        _entry("verifier_repro", "error"),
        _entry("verifier_fix", "complete"),
    ]
    h = evaluate_detector_health(ledger)
    assert h["verifiers_total"] == 2
    assert h["verifiers_errored"] == 1
    assert h["degraded"] is False  # detectors were healthy
