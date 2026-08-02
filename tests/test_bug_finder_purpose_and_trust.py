"""P2 (purposeful-on-general-input) + P3 (verify calibration) + P4
(run-trust) for bug_finder.

- P2: ``derive_threat_model`` synthesizes a grounded threat model when the
  user gave none, and never overrides an explicit one.
- P3: an ERRORED verifier is marked distinctly (``ReproOutcome.errored``)
  and not conflated with a genuine refutation, so infra failure can't
  silently bury a real finding.
- P4: ``evaluate_detector_health`` reports a ``trustworthy`` verdict that
  folds in verifier collapse, while the hard ``degraded`` gate stays
  detector-only.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.bug_finder.orchestrator import (
    CostLedgerEntry,
    evaluate_detector_health,
)
from augmentum.bug_finder.scope_deriver import derive_threat_model
from augmentum.bug_finder.verifier import apply_repro_outcome, parse_repro_result


# ----------------------------------------------------------------------
# P2 — threat-model derivation
# ----------------------------------------------------------------------

def test_derive_respects_explicit_threat_model():
    tm, derived = derive_threat_model(existing_threat_model="Assets: X\nIn scope: Y")
    assert derived is False
    assert tm == "Assets: X\nIn scope: Y"


def test_derive_synthesizes_when_absent():
    tm, derived = derive_threat_model(
        existing_threat_model="",
        knowledge_brief="subsystems: auth, billing",
        detected_language="python",
        user_goal_description="audit the payment path",
    )
    assert derived is True
    assert "auto-derived" in tm
    assert "python" in tm
    assert "audit the payment path" in tm
    assert "In scope" in tm and "Out of scope" in tm
    # Anchors the hunt to the comprehension map when one exists.
    assert "structural map" in tm


def test_derive_minimal_default_without_brief():
    """Even with no comprehension brief and no stack, a purposeful frame
    beats an empty one — the In/Out-of-scope skeleton always lands."""
    tm, derived = derive_threat_model(existing_threat_model="")
    assert derived is True
    assert "In scope" in tm
    assert "Disproof discipline" in tm


def test_derive_treats_whitespace_threat_model_as_absent():
    tm, derived = derive_threat_model(existing_threat_model="   \n  ")
    assert derived is True


# ----------------------------------------------------------------------
# P3 — errored verifier is not a refutation
# ----------------------------------------------------------------------

@dataclass
class _Result:
    stop_reason: str
    output: str = ""
    stop_detail: str = ""


def test_errored_verifier_marked_distinctly():
    outcome = parse_repro_result(_Result(stop_reason="error", stop_detail="HTTP 503"))
    assert outcome.confirmed is False
    assert outcome.errored is True
    assert "503" in outcome.evidence


def test_budget_verifier_is_not_flagged_as_errored():
    outcome = parse_repro_result(_Result(stop_reason="budget"))
    assert outcome.confirmed is False
    assert outcome.errored is False  # ran-and-capped, not infra failure


def test_errored_verifier_note_says_not_judged():
    from augmentum.bug_finder.findings import Finding, FindingStatus

    finding = Finding(
        id="f1", file="a.py", function="g", claim="bug",
        claim_signature="sig", severity="high", evidence_paths=(),
    )
    outcome = parse_repro_result(_Result(stop_reason="error", stop_detail="boom"))
    updated = apply_repro_outcome(finding, outcome)
    # Status is unconfirmable (skips fix loop) BUT the note distinguishes
    # infra failure from a genuine "couldn't reproduce" so the finding
    # isn't read as refuted.
    assert updated.status == FindingStatus.UNCONFIRMABLE.value
    assert any("NOT JUDGED" in n for n in updated.notes)


def test_genuine_unconfirmable_note_distinct_from_errored():
    from augmentum.bug_finder.findings import Finding

    finding = Finding(
        id="f2", file="a.py", function="g", claim="bug",
        claim_signature="sig", severity="low", evidence_paths=(),
    )
    # Clean completion, verdict not "confirmed" → genuine unconfirmable.
    outcome = parse_repro_result(_Result(
        stop_reason="complete",
        output='{"result": "unconfirmed", "evidence": "no trigger found"}',
    ))
    assert outcome.errored is False
    updated = apply_repro_outcome(finding, outcome)
    assert any("unconfirmable" in n and "NOT JUDGED" not in n for n in updated.notes)


# ----------------------------------------------------------------------
# P4 — run-trust verdict
# ----------------------------------------------------------------------

def _det(stop_reason: str) -> CostLedgerEntry:
    return CostLedgerEntry("detector", "d", "m", "i", 1, 1, 1, 1, stop_reason)


def _ver(stop_reason: str) -> CostLedgerEntry:
    return CostLedgerEntry("verifier_repro", "v", "m", "i", 1, 1, 1, 1, stop_reason)


def test_trust_healthy_run():
    h = evaluate_detector_health([_det("complete")] * 10 + [_ver("complete")] * 5)
    assert h["trustworthy"] is True
    assert h["degraded"] is False


def test_trust_false_on_detector_collapse():
    h = evaluate_detector_health([_det("error")] * 8)
    assert h["degraded"] is True
    assert h["trustworthy"] is False


def test_trust_false_on_verifier_collapse_without_hard_degrade():
    """Verifier collapse marks the run untrustworthy but does NOT trip the
    hard detector gate (stop_reason stays whatever it was)."""
    h = evaluate_detector_health(
        [_det("complete")] * 10 + [_ver("error")] * 6,
    )
    assert h["degraded"] is False           # detectors were fine
    assert h["verifier_error_rate"] == 1.0
    assert h["trustworthy"] is False        # but verifiers collapsed


def test_trust_single_verifier_error_does_not_flip_trust():
    """Below the 4-sample verifier floor, one errored verifier doesn't
    flip the whole run's trust flag."""
    h = evaluate_detector_health([_det("complete")] * 10 + [_ver("error"), _ver("complete")])
    assert h["trustworthy"] is True
