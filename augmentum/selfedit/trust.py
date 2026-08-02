"""The Trust Record — the connective tissue of the verified self-edit turn.

The rigor techniques the loop uses (fail-closed oracle coverage, held-out
retrodiction, and — later — inline stakes, self-heal, thrash-guard) are each
small. What makes them a *system* rather than a bag of flags is that they all
answer one question a professional operator actually asks:

    "How much should I trust this verdict, and what did it cost to earn it?"

This module composes the INTEGRITY half of that answer from data the archive
already holds — per-attempt oracle coverage — into one legible record, at two
scopes:

* ``attempt_trust(attempt)`` — for one attempt: is its verdict fully covered,
  or did an expected oracle silently skip? The per-row column in the lineage
  and the theater panel.
* ``archive_trust(attempts)`` — the engine-wide integrity gauge: what fraction
  of graded verdicts are fully covered, and — the seam to Move 2 — which
  *confirmable* classes keep hitting coverage gaps. A class that repeatedly
  can't be mechanically confirmed is precisely the Oracle Foundry's demand
  signal: author the missing oracle there.

Pure, read-only, same rows in → same record out. No new table; coverage rides
on the archived ``gate_verdict`` (written by ``verifier.verify``).
"""

from __future__ import annotations

from augmentum.selfedit.verifier import TIER_VERIFIED
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["attempt_trust", "archive_trust"]


def _coverage_of(attempt: dict) -> dict:
    verdict = attempt.get("gate_verdict") or {}
    if not isinstance(verdict, dict):
        return {}
    cov = verdict.get("coverage") or {}
    return cov if isinstance(cov, dict) else {}


def attempt_trust(attempt: dict) -> dict:
    """The per-attempt integrity record: tier + whether the verdict was fully
    covered (no expected oracle silently skipped) + the honest gap text.

    ``trusted`` means the strongest claim (verified) is backed by a real oracle
    run — the fail-closed contract. A verdict with a coverage gap is never
    trusted as verified regardless of tier."""
    verdict = attempt.get("gate_verdict") or {}
    tier = str((verdict or {}).get("tier", "")).strip().lower() if isinstance(verdict, dict) else ""
    cov = _coverage_of(attempt)
    complete = bool(cov.get("complete", True))       # absent → treated complete (old rows)
    gap = str(cov.get("gap", "") or "")
    expected = bool(cov.get("expected_mechanical_confirm", False))
    return {
        "attempt_id": str(attempt.get("id", "")),
        "tier": tier,
        "coverage_complete": complete,
        "coverage_gap": gap,
        "expected_mechanical_confirm": expected,
        # Fully trustworthy = objectively verified AND nothing expected was skipped.
        "trusted": tier == TIER_VERIFIED and complete,
    }


def archive_trust(attempts: list[dict]) -> dict:
    """The engine-wide verification-integrity gauge + the foundry demand seam.

    Over every GRADED attempt (one that carries a real verdict tier), reports how
    many verdicts are fully covered, and buckets coverage GAPS by (surface ×
    intent-class) — the confirmable classes that keep failing to be mechanically
    confirmed, which is exactly where an authored oracle would pay off."""
    graded = 0
    complete = 0
    verified = 0
    gap_by_class: dict[str, int] = {}
    for a in attempts:
        verdict = a.get("gate_verdict") or {}
        if not isinstance(verdict, dict):
            continue
        tier = str(verdict.get("tier", "")).strip().lower()
        if not tier or tier == "":
            continue
        graded += 1
        cov = _coverage_of(a)
        if bool(cov.get("complete", True)):
            complete += 1
        else:
            surface = str(a.get("surface", "") or "(none)")
            intent_class = str(verdict.get("intent_class", "") or "(unclassified)")
            key = f"{intent_class} · {surface}"
            gap_by_class[key] = gap_by_class.get(key, 0) + 1
        if tier == TIER_VERIFIED and bool(cov.get("complete", True)):
            verified += 1
    # Densest gaps first — the foundry's reading order.
    ranked_gaps = sorted(gap_by_class.items(), key=lambda kv: -kv[1])
    return {
        "graded": graded,
        "fully_covered": complete,
        "coverage_gaps": graded - complete,
        "coverage_rate": round(complete / graded, 4) if graded else 1.0,
        "trusted_verified": verified,     # verified AND fully covered
        "gap_classes": [{"class": k, "count": n} for k, n in ranked_gaps],
        # A single legible headline for the theater/Workshop: how honest is the
        # engine's verification right now (1.0 = no expected oracle ever skipped).
        "integrity": round(complete / graded, 4) if graded else 1.0,
    }
