"""The Oracle Foundry — the archive's interruption map, folded into a worklist.

Move 2 of the recursion-maximization design (spec
``docs/superpowers/specs/2026-07-01-recursion-maximization-design.md``): every
``human_required`` verdict in the archive is a **human interruption** — a class
of change no mechanical oracle can confirm yet. Cluster those verdicts by
(surface × intent-class) and the densest clusters are exactly where authoring
ONE new oracle (a reproducing test, a health probe, a registered verifier)
permanently converts a whole class from ``human_required`` → ``verified``. The
interruption metric falls by construction, not by hope.

This module is the pure part: the coverage map (the first live gauge of
verification coverage V), the ranked worklist, and the composed self-edit
objective for authoring an oracle. It proposes **through the existing
capability lane** (``/api/selfedit/capability``) — one spine, never a second;
the foundry only writes the ask.

Goodhart discipline (the agent now authors its own examiners):
* Authored-oracle edits carry their own intent-class (``authored-oracle``,
  via the ``[authored-oracle]`` marker the classifier recognizes) and are
  red-tier in ``promote.decide_promotion`` — never auto-promoted, even when
  a mechanical check goes green, until the class itself graduates.
* The composed objective forbids touching scanners/suppressions/``.claude``
  (the orchestrator's tamper gate rejects those anyway) and requires the
  oracle to be demonstrably able to FAIL.
* Retrodiction (Move 1) is the harness a landed oracle is judged with over
  time; agreement there is necessary, never sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from augmentum.selfedit.intent import CLASS_AUTHORED_ORACLE, ORACLE_MARKER
from augmentum.selfedit.retrodiction import _KEPT_STATUSES, _REVERTED_STATUSES
from augmentum.selfedit.verifier import (
    TIER_FAILED,
    TIER_HUMAN_CONFIRMED,
    TIER_HUMAN_REQUIRED,
    TIER_PROBABLE,
    TIER_VERIFIED,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "CLASS_AUTHORED_ORACLE",
    "ORACLE_MARKER",
    "CoverageCell",
    "coverage_map",
    "foundry_worklist",
    "oracle_objective",
    "coverage_summary",
]

# Confirmation strength, weakest→strongest (failed is not coverage at all).
_TIER_STRENGTH = {
    TIER_FAILED: 0,
    "": 0,
    TIER_HUMAN_REQUIRED: 1,
    TIER_PROBABLE: 2,
    TIER_HUMAN_CONFIRMED: 3,
    TIER_VERIFIED: 4,
}

# A cluster smaller than this isn't worth an oracle yet (one-offs churn).
MIN_CLUSTER = 2

# Evidence objectives included in a composed ask (full text, never truncated —
# selection of WHICH rows, not cutting content within one).
_MAX_EVIDENCE = 5


@dataclass(frozen=True)
class CoverageCell:
    """One (surface × intent-class) cell of the coverage matrix."""

    surface: str
    intent_class: str
    total: int
    by_tier: tuple[tuple[str, int], ...]   # stored-tier histogram (hashable)
    best_tier: str                          # strongest confirmation ever achieved
    interruptions: int                      # human_required verdicts = human cost
    probables: int                          # judgment-tier confirmations (upgrade fodder)
    kept: int                               # settled: human kept it live
    reverted: int                           # settled: human backed it out
    evidence: tuple[str, ...] = field(default_factory=tuple)  # sample objectives

    @property
    def covered(self) -> bool:
        """A mechanical oracle already confirms this class."""
        return self.best_tier == TIER_VERIFIED

    @property
    def oracle_worthy(self) -> bool:
        """An authored oracle would convert real, recurring interruptions."""
        return not self.covered and (self.interruptions + self.probables) >= MIN_CLUSTER

    def to_dict(self) -> dict:
        return {
            "surface": self.surface, "intent_class": self.intent_class,
            "total": self.total, "by_tier": dict(self.by_tier),
            "best_tier": self.best_tier, "covered": self.covered,
            "interruptions": self.interruptions, "probables": self.probables,
            "kept": self.kept, "reverted": self.reverted,
            "oracle_worthy": self.oracle_worthy,
            "evidence": list(self.evidence),
        }


def coverage_map(attempts: list[dict]) -> list[CoverageCell]:
    """Fold archive rows into the (surface × intent-class) → best-oracle-tier
    matrix — the autonomy frontier, visible for the first time. Pure; same rows
    in, same matrix out. Rows without a stored verdict tier (e.g. git-ingested
    history, which carries no verifier trace by design) are excluded — absence
    of evidence is not an interruption."""
    buckets: dict[tuple[str, str], list[dict]] = {}
    for a in attempts:
        verdict = a.get("gate_verdict") or {}
        if not isinstance(verdict, dict):
            continue
        tier = str(verdict.get("tier", "")).strip().lower()
        if tier not in _TIER_STRENGTH or not tier:
            continue
        surface = str(a.get("surface", "") or "(none)")
        intent_class = str(verdict.get("intent_class", "") or "(unclassified)")
        buckets.setdefault((surface, intent_class), []).append(a)

    cells: list[CoverageCell] = []
    for (surface, intent_class), rows in buckets.items():
        hist: dict[str, int] = {}
        interruptions = probables = kept = reverted = 0
        best = ""
        evidence: list[str] = []
        for a in rows:
            tier = str((a.get("gate_verdict") or {}).get("tier", "")).strip().lower()
            hist[tier] = hist.get(tier, 0) + 1
            if _TIER_STRENGTH.get(tier, 0) > _TIER_STRENGTH.get(best, 0):
                best = tier
            if tier == TIER_HUMAN_REQUIRED:
                interruptions += 1
                obj = str(a.get("objective", "")).strip()
                if obj and len(evidence) < _MAX_EVIDENCE:
                    evidence.append(obj)
            elif tier == TIER_PROBABLE:
                probables += 1
            status = str(a.get("status", "")).strip().lower()
            if status in _KEPT_STATUSES:
                kept += 1
            elif status in _REVERTED_STATUSES:
                reverted += 1
        cells.append(CoverageCell(
            surface=surface, intent_class=intent_class, total=len(rows),
            by_tier=tuple(sorted(hist.items())), best_tier=best,
            interruptions=interruptions, probables=probables,
            kept=kept, reverted=reverted, evidence=tuple(evidence),
        ))
    # Densest human cost first — the foundry's natural reading order.
    cells.sort(key=lambda c: (-(c.interruptions + c.probables), -c.total,
                              c.surface, c.intent_class))
    return cells


def foundry_worklist(cells: list[CoverageCell], *,
                     min_cluster: int = MIN_CLUSTER) -> list[CoverageCell]:
    """The oracle-authoring candidates: uncovered cells whose recurring human
    cost clears the cluster floor, densest first. The threshold keeps one-off
    interruptions (not yet a *class*) off the worklist — they stay visible in
    the full matrix."""
    return [c for c in cells
            if not c.covered and (c.interruptions + c.probables) >= min_cluster]


def oracle_objective(cell: CoverageCell) -> str:
    """Compose the self-edit ask for authoring an oracle for this cell. Sent
    through the ordinary capability lane by the HUMAN (the Workshop button) —
    the foundry never launches an edit itself. The ``[authored-oracle]`` marker
    routes the attempt into its red-tier intent-class."""
    lines = [
        f"{ORACLE_MARKER} Author a mechanical oracle for the "
        f"'{cell.intent_class}' intent-class on the '{cell.surface}' surface.",
        "",
        f"This change class has required a human verdict {cell.interruptions} "
        f"time(s) (and a judgment-oracle guess {cell.probables} time(s)) because "
        "no mechanical check confirms its intent. Author ONE new check that "
        "does — whichever fits the class best:",
        "  * a reproducing/acceptance test under tests/,",
        "  * a health probe (augmentum/selfedit/health.py style), or",
        "  * a verifier registered via augmentum.selfedit.verifier."
        f"register_verifier(...) with intent_classes=(\"{cell.intent_class}\",) "
        "and confirms_intent=True (oracle type mechanical).",
        "",
        "Non-negotiable constraints:",
        "  * The oracle must be able to FAIL — show it distinguishes a good "
        "change of this class from a bad one (a check that always passes is "
        "reward-hacking, not coverage).",
        "  * Do NOT touch scanners, suppression files, or anything under "
        ".claude/ — the tamper gate rejects those paths.",
        "  * ADD coverage only; never weaken or remove an existing check.",
        "  * Keep the change small and reversible.",
    ]
    if cell.evidence:
        lines += ["", "Evidence — recent changes of this class that interrupted "
                      "the human:"]
        lines += [f"  * {obj}" for obj in cell.evidence]
    return "\n".join(lines)


def coverage_summary(attempts: list[dict]) -> dict:
    """The coverage matrix + worklist + the V gauge, shaped for
    ``GET /api/selfedit/coverage`` and the Workshop lane. Worklist entries carry
    their composed objective so the UI can hand it straight to the capability
    lane — the human reads it, picks, and fires; nothing is auto-selected."""
    cells = coverage_map(attempts)
    worklist = foundry_worklist(cells)
    graded = sum(c.total for c in cells)
    verified_attempts = sum(
        dict(c.by_tier).get(TIER_VERIFIED, 0) for c in cells)
    interruptions = sum(c.interruptions for c in cells)
    return {
        "cells": [c.to_dict() for c in cells],
        "worklist": [{**c.to_dict(), "oracle_objective": oracle_objective(c)}
                     for c in worklist],
        "gauge": {
            # V, first sliver: how much of graded history a mechanical oracle
            # confirmed, and how much still lands on the human.
            "graded_attempts": graded,
            "verified_attempts": verified_attempts,
            "verified_share": round(verified_attempts / graded, 4) if graded else 0.0,
            "interruptions": interruptions,
            "cells_total": len(cells),
            "cells_covered": sum(1 for c in cells if c.covered),
        },
        "min_cluster": MIN_CLUSTER,
    }
