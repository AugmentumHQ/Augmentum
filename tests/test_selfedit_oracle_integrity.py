"""Oracle Integrity (increment 1 of the SOTA verified-turn design).

Three load-bearing behaviors, each mapped to the 2026 reward-hacking research:

* FAIL-CLOSED COVERAGE (OpenEvolve cascade footgun): a verdict records which
  expected oracle actually RAN; an expected mechanical confirm that merely
  SKIPPED is a named coverage gap, never a silent clean pass, and can never
  auto-promote.
* HELD-OUT RETRODICTION (EvilGenie/PACE): a candidate grader is scored on the
  fresh history slice it couldn't have been tuned to; the visible-minus-held-out
  agreement gap is the reward-hacking alarm.
* THE TRUST RECORD: per-attempt + engine-wide integrity, with coverage gaps
  bucketed by class (the Oracle Foundry's demand seam).
"""

from __future__ import annotations

from augmentum.selfedit import trust
from augmentum.selfedit import verifier as V
from augmentum.selfedit.verifier import (
    ORACLE_MECHANICAL,
    PASS,
    SKIP,
    VerifierResult,
)


def _mech_confirm(status=PASS):
    return VerifierResult("test", ORACLE_MECHANICAL, status, confirms_intent=True,
                          score=1.0 if status == PASS else 0.0, required=True)


def _noregress(status=PASS):
    return VerifierResult("boot", ORACLE_MECHANICAL, status, confirms_intent=False,
                          score=1.0, required=True)


# ---------------------------------------------------------------------------
# fail-closed coverage
# ---------------------------------------------------------------------------

async def test_confirmable_intent_with_confirm_that_ran_is_covered_and_verified():
    v = await V.verify({}, intent_class="bugfix",
                       verifiers={},  # empty pool; inject the results directly
                       extra_results=[_noregress(), _mech_confirm(PASS)],
                       expect_mechanical_confirm=True)
    assert v.tier == V.TIER_VERIFIED
    assert v.coverage["complete"] is True
    assert v.coverage["mechanical_confirm_ran"] is True
    assert v.coverage_complete and v.auto_promotable


async def test_confirmable_intent_with_confirm_that_SKIPPED_is_a_named_gap():
    # the dangerous case: a mechanical confirm was expected but only skipped —
    # the verdict must NOT read as verified, must name the gap, must not auto-ship.
    v = await V.verify({}, intent_class="bugfix", verifiers={},
                       extra_results=[_noregress(PASS), _mech_confirm(SKIP)],
                       expect_mechanical_confirm=True)
    assert v.tier == V.TIER_HUMAN_REQUIRED       # green, but nothing confirmed intent
    assert v.coverage["complete"] is False
    assert v.coverage["gap"]                       # the honest "why not higher"
    assert not v.coverage_complete
    assert not v.auto_promotable                   # fail closed
    assert "COVERAGE GAP" in v.summary


async def test_unconfirmable_intent_is_complete_by_definition():
    # style/taste: no mechanical confirm is EXPECTED, so "no confirm ran" is not
    # a gap — human_required here is honest, not a coverage failure.
    v = await V.verify({}, intent_class="style", verifiers={},
                       extra_results=[_noregress(PASS)],
                       expect_mechanical_confirm=False)
    assert v.tier == V.TIER_HUMAN_REQUIRED
    assert v.coverage["complete"] is True and v.coverage["gap"] == ""


async def test_auto_promotable_requires_both_verified_and_coverage():
    # belt-and-suspenders: even a hand-tightened verified verdict can't auto-ship
    # if coverage is marked incomplete.
    v = V.Verdict(tier=V.TIER_VERIFIED, passed=True,
                  coverage={"complete": False, "gap": "x"})
    assert v.tier == V.TIER_VERIFIED
    assert v.auto_promotable is False


def test_handbuilt_verdict_defaults_to_covered():
    # backward-compat: verdicts constructed without coverage read as complete,
    # so existing callers (promote tests etc.) are unaffected.
    v = V.Verdict(tier=V.TIER_VERIFIED, passed=True)
    assert v.coverage_complete is True and v.auto_promotable is True


async def test_failed_verdict_still_carries_coverage():
    v = await V.verify({}, intent_class="bugfix", verifiers={},
                       extra_results=[_noregress(V.FAIL)],
                       expect_mechanical_confirm=True)
    assert v.tier == V.TIER_FAILED and v.auto_promotable is False
    assert "coverage" in v.to_dict()


# ---------------------------------------------------------------------------
# held-out retrodiction (reward-hacking detector)
# ---------------------------------------------------------------------------

def _case(i, *, kept, tier="verified", created="2026-01-01"):
    from augmentum.selfedit.retrodiction import ReplayCase
    return ReplayCase(
        attempt_id=f"a{i}", objective="o", surface="backend", tier="green",
        target="", files_changed=(), source="autonomous", stored_results=(),
        stored_tier=tier, human_kept=kept, created_at=created)


def test_split_holds_out_the_most_recent():
    from augmentum.selfedit.retrodiction import split_cases
    cases = [_case(i, kept=True, created=f"2026-01-{i:02d}") for i in range(1, 11)]
    visible, held = split_cases(cases, holdout_frac=0.3)
    assert len(visible) == 7 and len(held) == 3
    # held-out is the most-recent slice
    assert {c.attempt_id for c in held} == {"a8", "a9", "a10"}


def test_split_always_keeps_one_visible():
    from augmentum.selfedit.retrodiction import split_cases
    cases = [_case(i, kept=True) for i in range(2)]
    visible, held = split_cases(cases, holdout_frac=0.9)
    assert len(visible) >= 1 and len(held) >= 1


async def test_heldout_alarm_fires_when_grader_overfits_visible():
    # A grader that says "verified" for everything: on kept-only visible history
    # it agrees, but on reverted held-out history it CONTRADICTS — the alarm.
    from augmentum.selfedit.retrodiction import run_retrodiction_heldout

    def always_verified(_case):
        return V.TIER_VERIFIED

    # visible = 7 kept (grader agrees), held-out = 3 reverted (grader contradicts)
    cases = ([_case(i, kept=True, created=f"2026-01-{i:02d}") for i in range(1, 8)]
             + [_case(i, kept=False, created=f"2026-02-{i:02d}") for i in range(1, 4)])
    report = await run_retrodiction_heldout(always_verified, cases,
                                            holdout_frac=0.3, min_cases=5)
    assert report.visible.verdict == "pass"        # agreed with what it could see
    assert report.held_out.verdict == "fail"       # contradicted the fresh reverts
    assert report.reward_hack_alarm is True
    assert report.gap > 0


async def test_heldout_no_alarm_for_an_honest_grader():
    from augmentum.selfedit.retrodiction import run_retrodiction_heldout

    def honest(case):
        # follows the ground truth: verified when kept, failed when reverted
        return V.TIER_VERIFIED if case.human_kept else V.TIER_FAILED

    cases = ([_case(i, kept=True, created=f"2026-01-{i:02d}") for i in range(1, 8)]
             + [_case(i, kept=False, created=f"2026-02-{i:02d}") for i in range(1, 4)])
    report = await run_retrodiction_heldout(honest, cases, holdout_frac=0.3, min_cases=5)
    assert report.reward_hack_alarm is False
    assert report.held_out.verdict != "fail"


# ---------------------------------------------------------------------------
# the Trust Record
# ---------------------------------------------------------------------------

def _attempt(*, aid, tier, complete=True, gap="", surface="backend",
            intent_class="bugfix", expected=True):
    return {
        "id": aid, "surface": surface,
        "gate_verdict": {
            "tier": tier, "intent_class": intent_class,
            "coverage": {"complete": complete, "gap": gap,
                         "expected_mechanical_confirm": expected},
        },
    }


def test_attempt_trust_is_trusted_only_when_verified_and_covered():
    good = trust.attempt_trust(_attempt(aid="a1", tier="verified", complete=True))
    assert good["trusted"] is True and good["coverage_complete"] is True

    gapped = trust.attempt_trust(_attempt(aid="a2", tier="verified", complete=False,
                                          gap="oracle skipped"))
    assert gapped["trusted"] is False        # verified but a coverage gap → not trusted
    assert gapped["coverage_gap"] == "oracle skipped"

    hr = trust.attempt_trust(_attempt(aid="a3", tier="human_required", complete=True))
    assert hr["trusted"] is False            # not verified


def test_archive_trust_gauge_and_foundry_gap_seam():
    attempts = [
        _attempt(aid="a1", tier="verified", complete=True),
        _attempt(aid="a2", tier="verified", complete=True),
        _attempt(aid="a3", tier="human_required", complete=False, gap="skip",
                 intent_class="bugfix", surface="backend"),
        _attempt(aid="a4", tier="human_required", complete=False, gap="skip",
                 intent_class="bugfix", surface="backend"),
        {"id": "u1", "gate_verdict": {}},   # ungraded — excluded
    ]
    g = trust.archive_trust(attempts)
    assert g["graded"] == 4                  # the ungraded row is excluded
    assert g["fully_covered"] == 2 and g["coverage_gaps"] == 2
    assert g["coverage_rate"] == 0.5 and g["integrity"] == 0.5
    assert g["trusted_verified"] == 2
    # the coverage-gap class is surfaced as the foundry's demand signal
    assert g["gap_classes"][0]["class"] == "bugfix · backend"
    assert g["gap_classes"][0]["count"] == 2


def test_archive_trust_empty_is_perfect_integrity():
    g = trust.archive_trust([])
    assert g["graded"] == 0 and g["integrity"] == 1.0 and g["gap_classes"] == []
