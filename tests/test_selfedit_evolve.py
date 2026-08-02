"""Tests for the self-edit evolution core (dataset / rubric / gepa).

All external effects are injected, so these run with deterministic fakes — no
model, no DB, no RNG, no clock. The load-bearing tests:
  - evolve only ACCEPTS a variant that beats the baseline on the untrained
    holdout by the success threshold (honest improvement bar);
  - a no-improvement run is NOT accepted and keeps the baseline;
  - constraint gates reject ineligible variants before they can win.
"""

from __future__ import annotations

from augmentum.selfedit import verifier as V
from augmentum.selfedit.evolve import (
    Criterion,
    Rubric,
    build_from_sessions,
    build_synthetic,
    evolve,
    golden,
    length_penalty,
    max_size_constraint,
    merge,
    rubric_judgment_verifier,
    score_variant,
    similarity_constraint,
)
from augmentum.selfedit.evolve.dataset import (
    LABEL_FAILURE,
    LABEL_POSITIVE,
    EvalCase,
    EvalDataset,
)

# A rubric with one criterion the fakes can drive.
RUBRIC = Rubric(criteria=[Criterion("good", "Is it good?", 1.0)], name="t")

# A variant containing this marker is "improved" — the fake run/judge reward it.
GOOD = "IMPROVED"


async def _fake_run(variant: str, inp: str) -> str:
    # Output quality tracks whether the variant carries the marker.
    return f"{'great' if GOOD in variant else 'meh'}::{inp}"


async def _fake_judge(rubric: Rubric, inp: str, output: str) -> dict:
    return {"good": 1.0 if output.startswith("great") else 0.2}


def _ds(n: int = 30) -> EvalDataset:
    return EvalDataset(cases=[EvalCase(case_id=f"c{i}", inp=f"task {i}") for i in range(n)],
                       name="t")


# --- dataset ---------------------------------------------------------------

def test_split_is_deterministic_and_covers_all():
    ds = _ds(60)
    a1, b1, c1 = ds.split()
    a2, b2, c2 = ds.split()
    assert [x.case_id for x in a1] == [x.case_id for x in a2]  # stable
    assert len(a1) + len(b1) + len(c1) == 60                   # nothing dropped
    assert a1 and b1 and c1                                     # all buckets populated


async def test_build_synthetic_skips_empty_inputs():
    async def gen(_text, _n):
        return [{"input": "do x", "expectation": "x happens"}, {"input": "   "}]
    cases = await build_synthetic("artifact", generate=gen, n=2)
    assert len(cases) == 1
    assert cases[0].inp == "do x" and cases[0].source == "synthetic"


async def test_build_from_sessions_labels_by_score():
    rows = [{"id": "hi", "input": "a", "output": "ok"},
            {"id": "lo", "input": "b", "output": "bad"}]

    async def judge(row):
        return 0.9 if row["id"] == "hi" else 0.1
    cases = await build_from_sessions(rows, judge=judge, fail_below=0.5)
    by = {c.case_id.split("-")[-1]: c for c in cases}
    assert by["hi"].label == LABEL_POSITIVE
    assert by["lo"].label == LABEL_FAILURE
    assert by["lo"].reference_output == "bad"  # failure keeps output for reflection


def test_golden_and_merge_dedupe():
    g = golden([{"input": "p", "expectation": "q"}])
    ds = merge(g, g, name="m")  # same ids twice → dedupe
    assert len(ds) == 1 and ds.counts()["golden"] == 1


# --- rubric ----------------------------------------------------------------

def test_length_penalty_curve():
    assert length_penalty("x" * 10, 100, 200) == 1.0       # under soft cap
    assert length_penalty("x" * 300, 100, 200) == 0.0      # over hard cap
    assert length_penalty("x" * 150, 100, 200) == 0.5      # midpoint
    assert length_penalty("anything", 0, 0) == 1.0         # disabled


async def test_score_variant_aggregates_and_penalizes():
    cases = _ds(4).cases
    good = await score_variant(GOOD, cases, run=_fake_run, rubric=RUBRIC, judge=_fake_judge)
    bad = await score_variant("plain", cases, run=_fake_run, rubric=RUBRIC, judge=_fake_judge)
    assert good.mean > bad.mean
    # length penalty drags penalized_mean below the plain mean
    penalized = await score_variant(GOOD + "x" * 150, cases, run=_fake_run, rubric=RUBRIC,
                                    judge=_fake_judge, soft_cap=100, hard_cap=200)
    assert penalized.penalized_mean < penalized.mean


async def test_rubric_bridges_to_verifier_judgment_oracle():
    cases = _ds(3).cases
    ver = rubric_judgment_verifier("rub", RUBRIC, run=_fake_run, judge=_fake_judge,
                                   cases=cases, pass_floor=0.7)
    verdict = await V.verify({"variant": GOOD}, verifiers={ver.name: ver})
    assert verdict.tier == V.TIER_PROBABLE          # judgment oracle confirmed intent
    bad = await V.verify({"variant": "plain"}, verifiers={"rub": ver})
    assert bad.tier == V.TIER_HUMAN_REQUIRED        # judge below floor → unconfirmed


# --- gepa ------------------------------------------------------------------

async def _mutate_to_good(_current, _failures):
    return [GOOD]  # the reflective mutator finds the improved variant


async def _mutate_noop(current, _failures):
    return [current, ""]  # proposes nothing usable


async def test_evolve_accepts_real_improvement():
    res = await evolve("plain", _ds(40), mutate=_mutate_to_good, run=_fake_run,
                       rubric=RUBRIC, judge=_fake_judge, success_threshold=0.1)
    assert res.accepted is True
    assert res.best_variant == GOOD
    assert res.best_holdout > res.baseline_holdout


async def test_evolve_rejects_no_improvement():
    res = await evolve("plain", _ds(40), mutate=_mutate_noop, run=_fake_run,
                       rubric=RUBRIC, judge=_fake_judge)
    assert res.accepted is False
    assert res.best_variant == "plain"              # baseline kept
    assert res.improvement == 0.0


async def test_constraint_blocks_oversize_winner():
    big_good = GOOD + "x" * 5000

    async def mutate_big(_c, _f):
        return [big_good]
    res = await evolve("plain", _ds(40), mutate=mutate_big, run=_fake_run, rubric=RUBRIC,
                       judge=_fake_judge, constraints=[max_size_constraint(100)])
    assert res.accepted is False                     # the only candidate was gated out
    assert res.rejected_by_constraint >= 1
    assert res.best_variant == "plain"


async def test_similarity_constraint_blocks_drift():
    def sim(_a, _b):
        return 0.2  # everything is "too different"

    async def mutate_drift(_c, _f):
        return [GOOD]
    res = await evolve("plain", _ds(40), mutate=mutate_drift, run=_fake_run, rubric=RUBRIC,
                       judge=_fake_judge,
                       constraints=[similarity_constraint("plain", similarity=sim, floor=0.8)])
    assert res.accepted is False and res.rejected_by_constraint >= 1


async def test_evolve_failures_feed_the_mutator():
    seen: list[int] = []

    async def mutate_inspect(_current, failures):
        seen.append(len(failures))
        return [GOOD]
    await evolve("plain", _ds(40), mutate=mutate_inspect, run=_fake_run, rubric=RUBRIC,
                 judge=_fake_judge, n_failures=3, max_iterations=1)
    assert seen and seen[0] <= 3                      # mutator received the worst cases


async def test_samples_denoise_averages_a_noisy_judge():
    # A judge whose score oscillates each call — averaging over samples pulls the
    # measured score toward the true mean (0.5), instead of latching to one call.
    flip = {"n": 0}

    async def noisy_judge(_rubric, _inp, _output):
        flip["n"] += 1
        return {"good": 1.0 if flip["n"] % 2 else 0.0}
    cases = _ds(4).cases
    s = await score_variant("v", cases, run=_fake_run, rubric=RUBRIC, judge=noisy_judge, samples=4)
    assert 0.35 <= s.mean <= 0.65                     # denoised toward the 0.5 truth


async def test_finalist_runoff_accepts_holdout_best_among_many():
    # Two improved variants proposed; the hall-of-fame + holdout runoff still
    # surfaces a genuine winner that beats baseline on the untrained split.
    async def mutate_two(_c, _f):
        return [GOOD, GOOD + " v2"]
    res = await evolve("plain", _ds(60), mutate=mutate_two, run=_fake_run, rubric=RUBRIC,
                       judge=_fake_judge, success_threshold=0.1, finalists=3)
    assert res.accepted is True
    assert GOOD in res.best_variant and res.best_holdout > res.baseline_holdout
    # the runoff was recorded in history
    assert any("runoff" in h for h in res.history)
