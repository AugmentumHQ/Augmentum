"""Eval-dataset construction for prompt/orchestration evolution.

You cannot evolve what you cannot score, and the surfaces we most want to improve
have no eval today. This module builds the dataset that scores them, from four
sources (after the GEPA recipe, adapted to our own data):

  A. synthetic — a strong model generates ``(input, expectation)`` cases from the
     artifact text. Bootstrap when there's no usage yet.
  B. mined     — REAL usage pulled from our own session tables (the richest
     signal we have; far more than a fresh agent). High-scoring rows become
     positive examples, low-scoring rows become failure cases for reflection.
  C. golden    — hand-curated high-value cases.
  D. auto      — cases whose success is mechanically checkable (a real oracle).

All external effects are injected: the caller fetches DB rows and supplies the
strong-model / judge callables. Nothing here touches a model or the DB. Splits
are deterministic (hash-bucketed by case id) so a run is reproducible and resumes
identically — no RNG.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

SOURCE_SYNTHETIC = "synthetic"
SOURCE_MINED = "mined"
SOURCE_GOLDEN = "golden"
SOURCE_AUTO = "auto"

LABEL_POSITIVE = "positive"
LABEL_FAILURE = "failure"


@dataclass
class EvalCase:
    """One scorable example for an artifact under evolution."""

    case_id: str
    inp: str                       # the input handed to the artifact (e.g. a user task)
    expectation: str = ""          # rubric / expected behavior in natural language
    source: str = SOURCE_SYNTHETIC
    label: str = ""                # positive | failure | "" (from mining)
    reference_output: str = ""     # the observed output (mined failures — fuel for reflection)
    mechanical_oracle: str = ""    # name of an auto-eval check, when one exists (source D)
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "inp": self.inp, "expectation": self.expectation,
            "source": self.source, "label": self.label,
            "reference_output": self.reference_output[:2000],
            "mechanical_oracle": self.mechanical_oracle, "weight": round(self.weight, 4),
        }


def _bucket(case_id: str) -> float:
    """Stable [0,1) bucket for a case id — deterministic split, no RNG."""
    h = hashlib.sha1(case_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0x100000000


@dataclass
class EvalDataset:
    cases: list[EvalCase] = field(default_factory=list)
    name: str = ""

    def __len__(self) -> int:
        return len(self.cases)

    def split(self, ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
              ) -> tuple[list[EvalCase], list[EvalCase], list[EvalCase]]:
        """Deterministic train/val/holdout split.

        Order cases by their hashed bucket (stable, no RNG) and slice by ratio —
        which, unlike a raw bucket-threshold, GUARANTEES a non-empty val + holdout
        whenever there are ≥3 cases (the small-N degeneracy that made live runs
        unscorable: 6 cases all hashing below 0.6 → empty val/holdout)."""
        ordered = sorted(self.cases, key=lambda c: _bucket(c.case_id))
        n = len(ordered)
        if n == 0:
            return [], [], []
        if n < 3:                       # too few to hold out — everything trains
            return ordered, [], []
        n_hold = max(1, round(n * ratios[2]))
        n_val = max(1, round(n * ratios[1]))
        if n_hold + n_val >= n:          # tiny n — keep ≥1 in train
            n_hold, n_val = 1, 1
        n_train = n - n_val - n_hold
        return (ordered[:n_train], ordered[n_train:n_train + n_val],
                ordered[n_train + n_val:])

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cases:
            out[c.source] = out.get(c.source, 0) + 1
        out["total"] = len(self.cases)
        return out


# Injected seam: a strong model turns the artifact into (input, expectation) pairs.
SyntheticGenerator = Callable[[str, int], Awaitable[Sequence[dict]]]
# Injected seam: a judge scores an observed (input, output) row 0..1.
RowJudge = Callable[[dict], Awaitable[float]]


async def build_synthetic(artifact_text: str, *, generate: SyntheticGenerator,
                          n: int = 20, id_prefix: str = "syn") -> list[EvalCase]:
    """Bootstrap cases by asking a strong model to read the artifact and emit
    ``{"input": ..., "expectation": ...}`` pairs (expectation is a rubric, not
    exact text). ``generate`` is injected — this never calls a model."""
    raw = await generate(artifact_text, n)
    cases: list[EvalCase] = []
    for i, item in enumerate(raw):
        inp = str(item.get("input", "")).strip()
        if not inp:
            continue
        cases.append(EvalCase(
            case_id=f"{id_prefix}-{i}", inp=inp,
            expectation=str(item.get("expectation", "")).strip(),
            source=SOURCE_SYNTHETIC,
        ))
    log.info("evolve_dataset_synthetic", requested=n, built=len(cases))
    return cases


async def build_from_sessions(rows: Iterable[dict], *, judge: RowJudge,
                              fail_below: float = 0.5, id_prefix: str = "mined",
                              ) -> list[EvalCase]:
    """Turn real session rows into labelled cases. Each row is
    ``{"id", "input", "output"}`` (the caller queries ui_sessions / coder /
    companion / claude_runs). ``judge`` scores the observed output 0..1; rows at
    or above ``fail_below`` become positive examples, below become failure cases
    (the fuel reflective mutation learns from)."""
    cases: list[EvalCase] = []
    for i, row in enumerate(rows):
        inp = str(row.get("input", "")).strip()
        if not inp:
            continue
        score = await judge(row)
        label = LABEL_POSITIVE if score >= fail_below else LABEL_FAILURE
        cases.append(EvalCase(
            case_id=f"{id_prefix}-{row.get('id', i)}", inp=inp,
            source=SOURCE_MINED, label=label,
            reference_output=str(row.get("output", "")),
        ))
    n_fail = sum(1 for c in cases if c.label == LABEL_FAILURE)
    log.info("evolve_dataset_mined", built=len(cases), failures=n_fail)
    return cases


def golden(items: Iterable[dict], *, id_prefix: str = "gold") -> list[EvalCase]:
    """Wrap hand-curated ``{"input", "expectation"}`` items as golden cases."""
    return [
        EvalCase(case_id=f"{id_prefix}-{i}", inp=str(it.get("input", "")).strip(),
                 expectation=str(it.get("expectation", "")).strip(),
                 source=SOURCE_GOLDEN, weight=float(it.get("weight", 1.0)))
        for i, it in enumerate(items) if str(it.get("input", "")).strip()
    ]


def merge(*case_lists: Iterable[EvalCase], name: str = "") -> EvalDataset:
    """Merge case lists into one dataset, de-duplicating by case id (last wins)."""
    by_id: dict[str, EvalCase] = {}
    for lst in case_lists:
        for c in lst:
            by_id[c.case_id] = c
    return EvalDataset(cases=list(by_id.values()), name=name)
