"""Rubric-based fitness scoring for evolve candidates.

A rubric turns "is this output good?" into a structured 0..1 score so a candidate
variant can be ranked over a dataset. The actual judging is an LLM-as-judge call,
INJECTED as a callable — this module owns the rubric structure, the weighted
aggregation, and the GEPA length penalty (artifacts approaching a size cap score
lower so evolution doesn't win by bloating). It calls no model.

A bridge (``rubric_judgment_verifier``) exposes a rubric scorer as the existing
``Verifier`` judgment oracle, so the same rubric used to evolve an artifact can
also gate a candidate through ``verifier.verify`` — one judge path, not two.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from augmentum.selfedit.verifier import Verifier, judgment_verifier
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Criterion:
    name: str
    question: str
    weight: float = 1.0


@dataclass
class Rubric:
    criteria: list[Criterion] = field(default_factory=list)
    name: str = ""

    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria) or 1.0


@dataclass
class CaseScore:
    case_id: str
    score: float                                   # 0..1 weighted over criteria
    per_criterion: dict[str, float] = field(default_factory=dict)
    detail: str = ""


@dataclass
class CandidateScore:
    variant_id: str
    mean: float                                    # plain mean over cases
    penalized_mean: float                          # mean × length penalty
    case_scores: list[CaseScore] = field(default_factory=list)

    def worst(self, n: int) -> list[CaseScore]:
        """The n lowest-scoring cases — what reflective mutation should target."""
        return sorted(self.case_scores, key=lambda c: c.score)[:n]


def length_penalty(text: str, soft_cap: int, hard_cap: int) -> float:
    """1.0 up to ``soft_cap`` chars, decaying linearly to 0 at ``hard_cap``.
    Disabled (1.0) when either cap is non-positive."""
    if soft_cap <= 0 or hard_cap <= 0 or hard_cap <= soft_cap:
        return 1.0
    n = len(text)
    if n <= soft_cap:
        return 1.0
    if n >= hard_cap:
        return 0.0
    return 1.0 - (n - soft_cap) / (hard_cap - soft_cap)


# Injected seam: a judge scores one (rubric, input, output) → {criterion_name: 0..1}.
RubricJudge = Callable[[Rubric, str, str], Awaitable[dict]]
# Injected seam: run the artifact-under-test (a variant) on an input → output.
ArtifactRun = Callable[[str, str], Awaitable[str]]


def _aggregate(rubric: Rubric, per_criterion: dict) -> float:
    tw = rubric.total_weight()
    s = 0.0
    for c in rubric.criteria:
        v = float(per_criterion.get(c.name, 0.0))
        s += max(0.0, min(1.0, v)) * c.weight
    return s / tw


async def score_case(rubric: Rubric, inp: str, output: str, *, judge: RubricJudge,
                     case_id: str = "") -> CaseScore:
    """Score one produced output against the rubric via the injected judge."""
    try:
        per = await judge(rubric, inp, output)
    except Exception as exc:  # noqa: BLE001 — a crashing judge scores 0, never raises out
        log.warning("evolve_rubric_judge_error", case_id=case_id, error=repr(exc))
        return CaseScore(case_id=case_id, score=0.0, detail=f"judge error: {exc!r}")
    per = {k: float(v) for k, v in dict(per).items()}
    return CaseScore(case_id=case_id, score=_aggregate(rubric, per), per_criterion=per)


async def score_variant(variant: str, cases: Sequence, *, run: ArtifactRun,
                        rubric: Rubric, judge: RubricJudge, variant_id: str = "",
                        soft_cap: int = 0, hard_cap: int = 0, samples: int = 1) -> CandidateScore:
    """Run ``variant`` over every case, score each, and aggregate (with the
    length penalty applied to the variant text).

    ``samples`` > 1 runs+judges each case N times and averages — denoising the
    run/judge variance so a real small improvement isn't lost under noise (the
    failure mode seen in live runs). Costs N× the calls; default 1 (no change)."""
    n = max(1, samples)
    case_scores: list[CaseScore] = []
    weighted_sum = 0.0
    weight_total = 0.0
    for c in cases:
        acc = 0.0
        last: CaseScore | None = None
        for _ in range(n):
            output = await run(variant, c.inp)
            last = await score_case(rubric, c.inp, output, judge=judge, case_id=c.case_id)
            acc += last.score
        score = acc / n
        case_scores.append(CaseScore(case_id=c.case_id, score=score,
                                     per_criterion=(last.per_criterion if last else {})))
        w = getattr(c, "weight", 1.0) or 1.0
        weighted_sum += score * w
        weight_total += w
    mean = (weighted_sum / weight_total) if weight_total else 0.0
    penalized = mean * length_penalty(variant, soft_cap, hard_cap)
    return CandidateScore(variant_id=variant_id, mean=mean, penalized_mean=penalized,
                          case_scores=case_scores)


def rubric_judgment_verifier(name: str, rubric: Rubric, *, run: ArtifactRun,
                             judge: RubricJudge, cases: Sequence,
                             pass_floor: float = 0.7,
                             intent_classes: tuple[str, ...] = ("*",),
                             cost: int = 6) -> Verifier:
    """Expose a rubric scorer as the existing Verifier judgment oracle, so an
    evolved candidate can gate through ``verifier.verify``. The verdict's
    confidence IS the mean rubric score; it confirms intent when the mean clears
    ``pass_floor``."""
    async def _judge(ctx: dict) -> tuple[bool, float, str]:
        variant = str(ctx.get("variant", ""))
        cs = await score_variant(variant, cases, run=run, rubric=rubric, judge=judge)
        ok = cs.mean >= pass_floor
        return ok, cs.mean, f"rubric '{rubric.name}' mean={cs.mean:.3f} over {len(cases)} cases"

    return judgment_verifier(name, _judge, intent_classes=intent_classes, cost=cost)
