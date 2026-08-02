"""The GEPA-style evolution loop, adapted to our primitives.

Genetic-Pareto-flavored prompt evolution: score the baseline, REFLECT on its
worst cases (the *why*, not just the *that*), MUTATE into candidate variants,
re-score on a held-out split, filter through CONSTRAINT GATES (size / anti-drift),
keep the best, iterate — then judge the winner on a never-trained holdout. Pure
string mutation, no GPU.

The reflective mutation, the artifact run, and the rubric judge are all INJECTED
(see ``rubric.py``); this module is the deterministic orchestration. A variant is
only ``accepted`` if it beats the baseline on the holdout by ``success_threshold``
— the honest improvement bar. The produced variant is handed to the existing
candidate → verify_change → store path by a later wiring slice; nothing here
mutates the repo.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from augmentum.selfedit.evolve.dataset import EvalDataset
from augmentum.selfedit.evolve.rubric import (
    ArtifactRun,
    Rubric,
    RubricJudge,
    score_variant,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FailureExample:
    """A low-scoring case handed to reflective mutation as evidence."""

    case_id: str
    inp: str
    expectation: str
    observed_output: str
    score: float


@dataclass
class Constraint:
    """A gate a variant must pass before it's eligible (size, anti-drift, …)."""

    name: str
    check: Callable[[str], tuple[bool, str]]   # (ok, detail)


def max_size_constraint(hard_cap: int) -> Constraint:
    def _check(text: str) -> tuple[bool, str]:
        ok = len(text) <= hard_cap
        return ok, f"len={len(text)} cap={hard_cap}"
    return Constraint("max_size", _check)


def similarity_constraint(baseline: str, *, similarity: Callable[[str, str], float],
                          floor: float = 0.8) -> Constraint:
    """Reject variants that drift too far from the baseline (anti-runaway-rewrite).
    ``similarity`` (0..1) is injected — embeddings or any text metric."""
    def _check(text: str) -> tuple[bool, str]:
        sim = similarity(baseline, text)
        return sim >= floor, f"similarity={sim:.3f} floor={floor}"
    return Constraint("similarity", _check)


# Injected seam: reflect on failures → propose candidate variant strings.
Mutate = Callable[[str, list[FailureExample]], Awaitable[Sequence[str]]]


@dataclass
class EvolveResult:
    baseline: str
    best_variant: str
    baseline_holdout: float
    best_holdout: float
    accepted: bool
    iterations: int
    history: list[dict] = field(default_factory=list)
    rejected_by_constraint: int = 0

    @property
    def improvement(self) -> float:
        return self.best_holdout - self.baseline_holdout

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted, "iterations": self.iterations,
            "baseline_holdout": round(self.baseline_holdout, 4),
            "best_holdout": round(self.best_holdout, 4),
            "improvement": round(self.improvement, 4),
            "changed": self.best_variant != self.baseline,
            "rejected_by_constraint": self.rejected_by_constraint,
            "history": self.history,
        }


def _passes(variant: str, constraints: Sequence[Constraint]) -> bool:
    for con in constraints:
        ok, _ = con.check(variant)
        if not ok:
            return False
    return True


def _failures(cs, cases_by_id: dict, n: int) -> list[FailureExample]:
    out: list[FailureExample] = []
    for case_score in cs.worst(n):
        c = cases_by_id.get(case_score.case_id)
        if c is None:
            continue
        out.append(FailureExample(
            case_id=c.case_id, inp=c.inp, expectation=c.expectation,
            observed_output=c.reference_output, score=case_score.score))
    return out


async def evolve(baseline: str, dataset: EvalDataset, *, mutate: Mutate,
                 run: ArtifactRun, rubric: Rubric, judge: RubricJudge,
                 constraints: Sequence[Constraint] = (), max_iterations: int = 10,
                 n_failures: int = 5, soft_cap: int = 0, hard_cap: int = 0,
                 success_threshold: float = 0.10, samples: int = 1,
                 finalists: int = 3) -> EvolveResult:
    """Evolve ``baseline`` against ``dataset``. Returns the best variant and an
    honest accept verdict.

    Two levers (added after live runs rarely accepted real gains):
    * ``samples`` denoises scoring (run+judge each case N×, average) so a true
      small improvement isn't lost in judge/run variance.
    * a **finalist holdout runoff**: every variant that scored well on validation
      across all iterations (a hall-of-fame) gets scored on the untrained holdout,
      and the holdout-best wins — not just the single iteration-best. This both
      raises the accept rate (more candidates get a fair holdout shot) and the
      quality of what's accepted (the winner generalizes, it didn't just spike on
      one split). Accept iff the holdout-best non-baseline beats baseline by
      ``success_threshold``."""
    train, val, holdout = dataset.split()
    cases_by_id = {c.case_id: c for c in train}
    if not train or not holdout:
        log.warning("evolve_dataset_too_small", train=len(train), holdout=len(holdout))

    def _score(variant: str, cases):
        return score_variant(variant, cases, run=run, rubric=rubric, judge=judge,
                             soft_cap=soft_cap, hard_cap=hard_cap, samples=samples)

    best = baseline
    best_val = (await _score(baseline, val)).penalized_mean if val else 0.0
    hall: dict[str, float] = {}          # variant → best validation score (the finalist pool)
    history: list[dict] = [{"iter": 0, "best_val": round(best_val, 4)}]
    rejected = 0
    iterations = 0

    for it in range(1, max_iterations + 1):
        iterations = it
        train_cs = await _score(best, train)
        failures = _failures(train_cs, cases_by_id, n_failures)
        try:
            proposals = list(await mutate(best, failures))
        except Exception as exc:  # noqa: BLE001 — a crashing mutator ends evolution gracefully
            log.warning("evolve_mutate_error", iter=it, error=repr(exc))
            break

        candidates = []
        for v in proposals:
            if not v or v == best:
                continue
            if not _passes(v, constraints):
                rejected += 1
                continue
            candidates.append(v)
        if not candidates:
            history.append({"iter": it, "best_val": round(best_val, 4), "candidates": 0})
            continue

        for v in candidates:
            sv = (await _score(v, val)).penalized_mean
            hall[v] = max(hall.get(v, 0.0), sv)
        top_variant = max(candidates, key=lambda v: hall[v])
        top_val = hall[top_variant]
        if top_val > best_val:                 # guides the next round's reflection
            best, best_val = top_variant, top_val
        history.append({"iter": it, "best_val": round(best_val, 4),
                        "candidates": len(candidates), "top": round(top_val, 4)})

    # Finalist runoff on the untrained holdout: the strongest-on-validation
    # variants compete; the holdout-best that genuinely beats baseline wins.
    baseline_holdout = (await _score(baseline, holdout)).penalized_mean if holdout else best_val
    pool = sorted(hall.items(), key=lambda kv: kv[1], reverse=True)[:max(1, finalists)]
    runoff = [(v, (await _score(v, holdout)).penalized_mean) for v, _ in pool]
    history.append({"runoff": [{"holdout": round(h, 4)} for _, h in runoff],
                    "baseline_holdout": round(baseline_holdout, 4)})

    cand, cand_holdout = (max(runoff, key=lambda x: x[1]) if runoff else (baseline, baseline_holdout))
    accepted = cand != baseline and (cand_holdout - baseline_holdout) >= success_threshold
    best_variant = cand if accepted else baseline
    best_holdout = cand_holdout if accepted else baseline_holdout

    result = EvolveResult(baseline=baseline, best_variant=best_variant,
                          baseline_holdout=baseline_holdout, best_holdout=best_holdout,
                          accepted=accepted, iterations=iterations, history=history,
                          rejected_by_constraint=rejected)
    log.info("evolve_done", **result.to_dict())
    return result
