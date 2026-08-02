"""Retrodiction tests — the archive as a labeled benchmark for graders.

The load-bearing honesty cases: a grader that would auto-promote what the human
reverted must FAIL (contradiction); one that turns settled verified-and-kept
history into ``failed`` must FAIL (flip); thin history is ``insufficient_history``,
never a pass; abstention (``human_required``) is honest and unscored.
"""

from __future__ import annotations

from augmentum.selfedit.retrodiction import (
    MIN_SETTLED_CASES,
    ReplayCase,
    benchmark_summary,
    case_from_attempt,
    cases_from_attempts,
    run_retrodiction,
    tier_policy_grader,
)
from augmentum.selfedit.verifier import (
    TIER_FAILED,
    TIER_HUMAN_REQUIRED,
    TIER_PROBABLE,
    TIER_VERIFIED,
    Verdict,
    VerifierResult,
)


def _attempt(i: int, *, status: str = "promoted", stored_tier: str = TIER_VERIFIED,
             source: str = "autonomous", results: list[dict] | None = None) -> dict:
    return {
        "id": f"a{i}", "objective": f"objective {i}", "surface": "frontend",
        "tier": "green", "target": "", "files_changed": ["ui/scripts/app.js"],
        "status": status, "source": source,
        "gate_verdict": {"tier": stored_tier, "passed": True,
                         "results": results or []},
    }


def _case(i: int, **kw) -> ReplayCase:
    return case_from_attempt(_attempt(i, **kw))


# --- case building ---------------------------------------------------------

def test_settled_statuses_become_cases():
    assert _case(1, status="promoted").human_kept is True
    assert _case(2, status="live").human_kept is True
    assert _case(3, status="rolled_back").human_kept is False


def test_unsettled_and_engine_verdicts_are_not_ground_truth():
    # pending states have no verdict; rejected/failed are the OLD grader's own
    # calls — using them to grade a NEW grader would be circular.
    for status in ("proposed", "editing", "gated", "rejected", "failed"):
        assert case_from_attempt(_attempt(9, status=status)) is None


def test_case_tolerates_malformed_gate_verdict():
    a = _attempt(1)
    a["gate_verdict"] = "not a dict"
    c = case_from_attempt(a)
    assert c is not None and c.stored_tier == "" and c.stored_results == ()


# --- the honesty gates -----------------------------------------------------

async def test_contradiction_fails_the_grader():
    cases = [_case(i) for i in range(MIN_SETTLED_CASES)]
    cases.append(_case(90, status="rolled_back"))
    report = await run_retrodiction(lambda c: TIER_VERIFIED, cases)
    assert report.verdict == "fail" and not report.passed
    assert len(report.contradictions) == 1
    assert report.contradictions[0]["attempt_id"] == "a90"


async def test_flip_on_settled_verified_history_fails():
    cases = [_case(i) for i in range(MIN_SETTLED_CASES + 1)]

    def grader(case: ReplayCase) -> str:
        return TIER_FAILED if case.attempt_id == "a0" else TIER_VERIFIED
    report = await run_retrodiction(grader, cases)
    assert report.verdict == "fail"
    assert len(report.flips) == 1 and report.flips[0]["attempt_id"] == "a0"


async def test_insufficient_history_is_never_a_pass():
    cases = [_case(1), _case(2)]
    report = await run_retrodiction(lambda c: TIER_VERIFIED, cases)
    assert report.verdict == "insufficient_history" and not report.passed


async def test_agreeing_grader_passes_and_upgrades_count():
    # 5 kept (one of them stored human_required — the interruption the candidate
    # grader would have saved) + 1 reverted, all graded in agreement.
    cases = [_case(i) for i in range(4)]
    cases.append(_case(50, stored_tier=TIER_HUMAN_REQUIRED))
    cases.append(_case(60, status="rolled_back"))

    def grader(case: ReplayCase) -> str:
        if not case.human_kept:
            return TIER_FAILED          # agrees: it was a mistake
        if case.attempt_id == "a50":
            return TIER_PROBABLE        # the upgrade
        return TIER_VERIFIED
    report = await run_retrodiction(grader, cases)
    assert report.passed, report.rationale
    assert report.agreement == 1.0
    assert report.upgrades == 1
    assert report.contradictions == [] and report.flips == []


async def test_universal_abstention_earns_nothing():
    cases = [_case(i) for i in range(MIN_SETTLED_CASES + 1)]
    report = await run_retrodiction(lambda c: TIER_HUMAN_REQUIRED, cases)
    assert report.verdict == "fail"
    assert report.n_abstained == len(cases) and report.n_directional == 0
    # settled verified coverage would now interrupt → all counted as downgrades
    assert report.downgrades == len(cases)


async def test_grader_crash_is_abstention_not_replay_crash():
    cases = [_case(i) for i in range(MIN_SETTLED_CASES + 1)]

    def grader(case: ReplayCase) -> str:
        if case.attempt_id == "a0":
            raise RuntimeError("unstable grader")
        return TIER_VERIFIED
    report = await run_retrodiction(grader, cases)
    assert report.n_abstained == 1
    assert report.n_directional == len(cases) - 1


async def test_async_grader_supported():
    async def grader(case: ReplayCase) -> str:
        return TIER_VERIFIED
    report = await run_retrodiction(grader, [_case(i) for i in range(6)])
    assert report.passed


# --- the first gradable grader-class: verdict-assembly policies -------------

async def test_tier_policy_grader_replays_stored_evidence():
    confirm = {"name": "audit", "oracle": "mechanical", "status": "pass",
               "confirms_intent": True, "score": 1.0, "confidence": 1.0,
               "required": True, "detail": ""}

    def policy(results: list[VerifierResult]) -> Verdict:
        ok = any(r.confirms_intent and r.status == "pass" and r.oracle == "mechanical"
                 for r in results)
        return Verdict(tier=TIER_VERIFIED if ok else TIER_HUMAN_REQUIRED,
                       passed=True, results=results)

    grader = tier_policy_grader(policy)
    with_evidence = _case(1, results=[confirm])
    without_evidence = _case(2, results=[])
    assert grader(with_evidence) == TIER_VERIFIED
    assert grader(without_evidence) == TIER_HUMAN_REQUIRED  # honest abstention

    cases = [_case(i, results=[confirm]) for i in range(6)]
    report = await run_retrodiction(grader, cases)
    assert report.passed


# --- the benchmark gauge ----------------------------------------------------

def test_benchmark_summary_composition():
    attempts = [
        _attempt(1, status="promoted"),
        _attempt(2, status="live", source="git"),
        _attempt(3, status="rolled_back", source="git"),
        _attempt(4, status="gated"),      # unsettled
        _attempt(5, status="rejected"),   # engine verdict, not ground truth
    ]
    b = benchmark_summary(attempts)
    assert b["settled_cases"] == 3 and b["unsettled_rows"] == 2
    assert b["kept"] == 2 and b["reverted"] == 1
    assert b["by_source"] == {"autonomous": 1, "git": 2}
    assert b["sufficient"] is False and b["min_settled_cases"] == MIN_SETTLED_CASES
    assert len(cases_from_attempts(attempts)) == 3
