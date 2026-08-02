"""Self-edit *evolution* — eval-driven improvement of prompt/orchestration
artifacts (companion prompts, builder-flow templates, intent classifiers,
retrieval prompts).

This fills the 4th net-new gap from the grounding dig: the surfaces we most want
to improve have NO eval today, and you cannot evolve what you cannot score. The
approach is adapted from Nous' hermes-agent-self-evolution (DSPy+GEPA) — *mutate
and evaluate STRINGS via API calls, no GPU training* — translated onto our own
primitives:

  - the eval DATASET is built from four sources (synthetic / mined-from-our-own-
    session-tables / golden / mechanical-auto) — ``dataset.py``
  - fitness is a rubric-based score that also bridges to the existing Verifier's
    judgment oracle — ``rubric.py``
  - the GEPA-style loop (score → reflect on failures → mutate → re-score →
    constraint-gate → keep best) — ``gepa.py``

Everything external (model calls, DB queries, embeddings) is INJECTED, so this
package is pure and testable and calls neither a model nor the DB directly. A
later slice wires the produced variant through the existing
candidate → verify_change → store path; this layer only *finds* the better string.

See ``docs/superpowers/specs/2026-06-23-grows-with-user-hermes-build-reference.md``.
"""

from __future__ import annotations

from augmentum.selfedit.evolve.dataset import (
    EvalCase,
    EvalDataset,
    build_from_sessions,
    build_synthetic,
    golden,
    merge,
)
from augmentum.selfedit.evolve.gepa import (
    Constraint,
    EvolveResult,
    FailureExample,
    evolve,
    max_size_constraint,
    similarity_constraint,
)
from augmentum.selfedit.evolve.rubric import (
    CandidateScore,
    CaseScore,
    Criterion,
    Rubric,
    length_penalty,
    rubric_judgment_verifier,
    score_case,
    score_variant,
)
from augmentum.selfedit.evolve.session import run_evolve_session

__all__ = [
    "CandidateScore",
    "CaseScore",
    "Constraint",
    "Criterion",
    "EvalCase",
    "EvalDataset",
    "EvolveResult",
    "FailureExample",
    "Rubric",
    "build_from_sessions",
    "build_synthetic",
    "evolve",
    "golden",
    "length_penalty",
    "max_size_constraint",
    "merge",
    "rubric_judgment_verifier",
    "run_evolve_session",
    "score_case",
    "score_variant",
    "similarity_constraint",
]
