"""Augmentum self-editing — the foundation for the app improving itself.

Pillar: grow by remembering mistakes; the archive is sacred; rollback restores
code but never the lesson. Flow: propose → isolate a candidate (worktree) →
agent edits → fitness gate → promote | reject | rollback, every step recorded
permanently in ``self_edit_attempts``.

This package is the substrate (gate, candidate isolation, archive store). The
orchestration that wires it to the run engine, the gated-offer surface, and the
supervised-promotion/rollback layer builds on top of these primitives.
"""

from __future__ import annotations

from augmentum.selfedit import (
    activation,
    adapters,
    bootsmoke,
    candidate,
    debt,
    demand,
    foundry,
    gate,
    health,
    ingest,
    intent,
    loop,
    orchestrator,
    palate,
    probes,
    promote,
    retrodiction,
    rollback,
    runners,
    scanners,
    store,
    trust,
    verifier,
)
from augmentum.selfedit.activation import (
    ActivationScore,
    SkillGraph,
    build_graph,
    load_graph,
    query_atoms,
)
from augmentum.selfedit.adapters import verify_change
from augmentum.selfedit.bootsmoke import boot_smoke_verifier
from augmentum.selfedit.candidate import Candidate, create_candidate, remove_candidate
from augmentum.selfedit.debt import (
    DebtTarget,
    next_mechanical_objective,
    select_debt_targets,
    triage,
)
from augmentum.selfedit.foundry import (
    CoverageCell,
    coverage_map,
    coverage_summary,
    foundry_worklist,
    oracle_objective,
)
from augmentum.selfedit.gate import Check, CheckResult, GateVerdict, default_app_gate, run_gate
from augmentum.selfedit.health import (
    DimensionResult,
    HealthDelta,
    HealthReport,
    assess,
    compare,
    default_probes,
    register_probe,
)
from augmentum.selfedit.ingest import ingest_coder_turn, ingest_git_history
from augmentum.selfedit.intent import SelfEditIntent, classify_intent, classify_surface
from augmentum.selfedit.loop import DebtLoopReport, run_debt_loop
from augmentum.selfedit.orchestrator import (
    EditDriver,
    EditRequest,
    EditResult,
    SelfEditOutcome,
    run_self_edit,
)
from augmentum.selfedit.palate import (
    ChangeFeatures,
    Palate,
    PalateVerdict,
    build_palate,
    features_from_target,
    palate_profile,
)
from augmentum.selfedit.promote import (
    PromotionDecision,
    PromotionResult,
    decide_promotion,
    promote_attempt,
    revert_attempt,
)
from augmentum.selfedit.retrodiction import (
    HeldoutReport,
    ReplayCase,
    RetrodictionReport,
    benchmark_summary,
    case_from_attempt,
    cases_from_attempts,
    run_retrodiction,
    run_retrodiction_heldout,
    split_cases,
    tier_policy_grader,
)
from augmentum.selfedit.trust import archive_trust, attempt_trust
from augmentum.selfedit.verifier import Verdict, Verifier, verify

__all__ = [
    "activation", "adapters", "bootsmoke", "candidate", "debt", "gate", "health",
    "demand", "foundry", "ingest", "intent", "loop", "orchestrator", "palate",
    "probes", "promote",
    "retrodiction", "rollback", "runners", "scanners", "store", "trust", "verifier",
    "archive_trust", "attempt_trust",
    "ChangeFeatures", "Palate", "PalateVerdict", "build_palate",
    "features_from_target", "palate_profile",
    "CoverageCell", "coverage_map", "coverage_summary", "foundry_worklist",
    "oracle_objective",
    "ReplayCase", "RetrodictionReport", "HeldoutReport", "benchmark_summary",
    "case_from_attempt", "cases_from_attempts", "run_retrodiction",
    "run_retrodiction_heldout", "split_cases", "tier_policy_grader",
    "ingest_coder_turn", "ingest_git_history",
    "ActivationScore", "SkillGraph", "build_graph", "load_graph", "query_atoms",
    "DebtTarget", "next_mechanical_objective", "select_debt_targets", "triage",
    "DebtLoopReport", "run_debt_loop",
    "PromotionDecision", "PromotionResult", "decide_promotion", "promote_attempt",
    "revert_attempt",
    "Candidate", "create_candidate", "remove_candidate",
    "Check", "CheckResult", "GateVerdict", "default_app_gate", "run_gate",
    "DimensionResult", "HealthDelta", "HealthReport",
    "assess", "compare", "default_probes", "register_probe",
    "boot_smoke_verifier", "verify_change",
    "SelfEditIntent", "classify_intent", "classify_surface",
    "Verdict", "Verifier", "verify",
    "EditDriver", "EditRequest", "EditResult", "SelfEditOutcome", "run_self_edit",
]
