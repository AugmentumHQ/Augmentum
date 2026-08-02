"""Finding dataclass, dedup, parsing, ranking.

A *finding* is the unit of work that flows through the pipeline. Detectors
emit speculative findings; the verifier promotes them to ``confirmed`` (or
``unconfirmable``); fixers either deliver a patch (``fixed``) or give up
(``fix_failed``).

The detector subagent emits findings as JSON in its final response. This
module parses that JSON, dedupes across the N (default 3) detector runs
per chunk, ranks the result, and records ``runs_to_confirm`` —
the variance signal that Vidoc's pipeline never measured and that
distinguishes confirmed-3/3 gold from speculative-1/3 noise.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from augmentum.bug_finder.json_salvage import (
    salvage_json_array,
    salvage_json_object,
)


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEV_RANK = {
    Severity.INFO.value: 0,
    Severity.LOW.value: 1,
    Severity.MEDIUM.value: 2,
    Severity.HIGH.value: 3,
    Severity.CRITICAL.value: 4,
}


class FindingStatus(str, Enum):
    """Lifecycle states for a finding."""

    SPECULATIVE = "speculative"
    """Detected, but no repro has been built yet (or repro construction failed)."""

    CONFIRMED = "confirmed"
    """Verifier built a minimal repro and it triggered the bug."""

    UNCONFIRMABLE = "unconfirmable"
    """Verifier attempted but could not construct a triggering repro."""

    FIXED = "fixed"
    """Fixer produced a patch that makes the repro pass without regressing
    other tests; the verifier (in an isolated fork) confirmed both."""

    FIX_FAILED = "fix_failed"
    """Fixer exhausted attempts without producing a verifiable patch."""


_STATUS_RANK = {
    FindingStatus.SPECULATIVE.value: 0,
    FindingStatus.UNCONFIRMABLE.value: 1,
    FindingStatus.FIX_FAILED.value: 2,
    FindingStatus.CONFIRMED.value: 3,
    FindingStatus.FIXED.value: 4,
}


class ClaimSignature(str, Enum):
    """Semantic class of a bug claim.

    Used as a dedup-key component so the same lines flagged with two
    different causes count as two findings, not one. Detectors are
    expected to emit one of these values directly; ``classify_claim``
    is the keyword-based fallback when they don't.
    """

    NULL_DEREF = "null_deref"
    BOUNDS_CHECK = "bounds_check"
    RACE = "race"
    USE_AFTER_FREE = "use_after_free"
    INJECTION = "injection"
    MISSING_VALIDATION = "missing_validation"
    RESOURCE_LEAK = "resource_leak"
    DEADLOCK = "deadlock"
    AUTH_BYPASS = "auth_bypass"
    LOGIC_ERROR = "logic_error"
    TYPE_CONFUSION = "type_confusion"
    OTHER = "other"


_SIGNATURE_KEYWORDS: dict[ClaimSignature, tuple[str, ...]] = {
    ClaimSignature.NULL_DEREF: (
        "null deref", "nullptr", "null pointer", "dereference null",
        "none deref", "nonetype",
    ),
    ClaimSignature.BOUNDS_CHECK: (
        "bounds check", "out of bounds", "out-of-bounds", "buffer overflow",
        "index out of range", "off-by-one", "off by one", "stack overflow",
    ),
    ClaimSignature.RACE: (
        "race condition", "data race", "tochou", "tocttou", "time-of-check",
        "concurrent access",
    ),
    ClaimSignature.USE_AFTER_FREE: (
        "use after free", "use-after-free", "uaf", "double free",
        "after-free", "dangling pointer",
    ),
    ClaimSignature.INJECTION: (
        "sql injection", "command injection", "shell injection",
        "code injection", "xss", "cross-site scripting", "path traversal",
        "directory traversal",
    ),
    ClaimSignature.MISSING_VALIDATION: (
        "missing validation", "unvalidated input", "unchecked input",
        "no input validation",
    ),
    ClaimSignature.RESOURCE_LEAK: (
        "resource leak", "memory leak", "file descriptor leak", "fd leak",
        "unclosed", "not closed", "handle leak",
    ),
    ClaimSignature.DEADLOCK: (
        "deadlock", "livelock", "lock order",
    ),
    ClaimSignature.AUTH_BYPASS: (
        "auth bypass", "authentication bypass", "authorization bypass",
        "privilege escalation", "permission bypass", "missing auth",
    ),
    ClaimSignature.TYPE_CONFUSION: (
        "type confusion", "type punning", "incorrect cast",
    ),
    ClaimSignature.LOGIC_ERROR: (
        "logic error", "incorrect condition", "wrong condition",
        "missing case", "incorrect default",
    ),
}


def classify_claim(claim: str) -> str:
    """Map a free-text claim to a ``ClaimSignature`` value.

    Defensive fallback only — the detector subagent is prompted to emit a
    signature directly. When it doesn't (or emits one outside the enum),
    this scan picks the first keyword family that matches.
    """
    lowered = (claim or "").lower()
    for sig, kws in _SIGNATURE_KEYWORDS.items():
        for kw in kws:
            if kw in lowered:
                return sig.value
    return ClaimSignature.OTHER.value


@dataclass
class Finding:
    """A single bug claim, possibly verified, possibly fixed.

    ``id`` is derived deterministically from the dedup key so two runs
    of the pipeline against the same code produce stable IDs (useful
    for cross-run diffing).
    """

    id: str
    file: str
    function: str
    claim: str
    claim_signature: str
    severity: str
    evidence_paths: tuple[str, ...]
    suggested_repro: str = ""

    status: str = FindingStatus.SPECULATIVE.value
    runs_to_confirm: int = 0
    total_runs: int = 0

    # Cross-family confirmation (Anthropic ensemble pattern).
    # 0 = family tracking disabled (single-model run); >=1 = number of
    # distinct vendor families that flagged this finding. The denominator
    # `total_families` mirrors total_runs but counts unique families.
    families_to_confirm: int = 0
    total_families: int = 0

    # Populated by stage 5 (verify-is-real)
    repro_path: str = ""
    repro_command: str = ""
    repro_output: str = ""

    # Populated by stage 6 (fix)
    invariant: str = ""
    patch: str = ""
    fix_attempts: int = 0

    notes: list[str] = field(default_factory=list)


def _normalize_evidence(paths: Any) -> tuple[str, ...]:
    if not paths:
        return ()
    if isinstance(paths, str):
        return (paths.strip(),) if paths.strip() else ()
    if isinstance(paths, list | tuple):
        return tuple(sorted({str(p).strip() for p in paths if str(p).strip()}))
    return (str(paths).strip(),)


def _normalize_severity(value: Any) -> str:
    s = (str(value or "").strip().lower()) or Severity.MEDIUM.value
    return s if s in _SEV_RANK else Severity.MEDIUM.value


def _normalize_signature(value: Any, *, fallback_claim: str = "") -> str:
    s = (str(value or "").strip().lower())
    if s and s in {c.value for c in ClaimSignature}:
        return s
    return classify_claim(fallback_claim)


def _finding_id(
    file: str,
    function: str,
    signature: str,
    evidence: tuple[str, ...],
) -> str:
    """Stable hash of the dedup key. 16 hex chars is plenty — collisions
    require matching all four dimensions exactly."""
    blob = "|".join((file, function, signature, *evidence))
    return "fnd_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def finding_from_dict(d: dict[str, Any]) -> Finding | None:
    """Build a Finding from a detector-emitted dict. Returns ``None`` for
    obviously broken input (missing file, missing claim) — caller logs
    and skips."""
    if not isinstance(d, dict):
        return None
    file = str(d.get("file") or "").strip()
    claim = str(d.get("claim") or "").strip()
    if not file or not claim:
        return None
    function = str(d.get("function") or "").strip() or "<module>"
    evidence = _normalize_evidence(d.get("evidence_paths") or d.get("evidence"))
    signature = _normalize_signature(d.get("claim_signature"), fallback_claim=claim)
    severity = _normalize_severity(d.get("severity"))
    suggested = str(d.get("suggested_repro") or d.get("repro_hint") or "").strip()
    fid = _finding_id(file, function, signature, evidence)
    return Finding(
        id=fid,
        file=file,
        function=function,
        claim=claim,
        claim_signature=signature,
        severity=severity,
        evidence_paths=evidence,
        suggested_repro=suggested,
    )


def parse_detector_output(output: str) -> list[Finding]:
    """Extract findings from a detector subagent's final output.

    Accepts either ``{"findings": [...]}`` or a bare ``[...]``. Uses the
    salvage parser so a budget-TRUNCATED detector emit (no closing fence,
    cut mid-findings — the dominant failure on large codebases) still
    yields the findings that landed before the cut, instead of losing the
    whole set (audit 2026-06-17). Empty list = "no findings", not error.
    """
    if not output:
        return []

    items: list[Any] = []
    obj = salvage_json_object(output)
    if isinstance(obj, dict) and isinstance(obj.get("findings"), list):
        items = obj["findings"]
    else:
        arr = salvage_json_array(output)
        if isinstance(arr, list):
            items = arr

    out: list[Finding] = []
    for item in items:
        f = finding_from_dict(item)
        if f is not None:
            out.append(f)
    return out


def merge_runs(
    findings_per_run: list[list[Finding]],
    *,
    families: list[str] | None = None,
) -> list[Finding]:
    """Merge findings across N detector runs into a deduped list.

    Dedupe key: ``(file, function, claim_signature, sorted-evidence)``.

    Per-run dedupe: within a single run, multiple findings sharing the
    key collapse to one — runs can repeat themselves. The collapsed
    finding takes the longest ``claim`` and highest severity.

    Cross-run aggregation: ``runs_to_confirm`` counts *distinct runs*
    the key appeared in; ``total_runs = len(findings_per_run)``. Same
    finding flagged 3/3 runs is gold; 1/3 is noise.

    When ``families`` is provided (one entry per run, parallel to
    ``findings_per_run``), each merged Finding also gets
    ``families_to_confirm`` and ``total_families`` populated. Family-
    crossing confirmation is a stronger precision signal than raw
    run count because it breaks correlated-error patterns within one
    vendor's training data + RLHF.
    """
    n = len(findings_per_run)
    if n == 0:
        return []

    if families is not None and len(families) != n:
        raise ValueError(
            f"merge_runs: families={len(families)} must match runs={n}",
        )

    # Per-run dedupe first.
    per_run_keyed: list[dict[tuple[str, str, str, tuple[str, ...]], Finding]] = []
    for run in findings_per_run:
        keyed: dict[tuple[str, str, str, tuple[str, ...]], Finding] = {}
        for f in run:
            key = (f.file, f.function, f.claim_signature, f.evidence_paths)
            cur = keyed.get(key)
            if cur is None or len(f.claim) > len(cur.claim) or _SEV_RANK[f.severity] > _SEV_RANK[cur.severity]:
                keyed[key] = f
        per_run_keyed.append(keyed)

    # Aggregate across runs.
    all_keys: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for keyed in per_run_keyed:
        all_keys.update(keyed.keys())

    # Total distinct families (denominator) — set once across all runs.
    total_families = len(set(families)) if families is not None else 0

    merged_by_id: dict[str, Finding] = {}
    for key in all_keys:
        appearances_with_family: list[tuple[Finding, str | None]] = [
            (keyed[key], (families[idx] if families is not None else None))
            for idx, keyed in enumerate(per_run_keyed)
            if key in keyed
        ]
        appearances = [a for a, _ in appearances_with_family]
        # Pick the canonical text representation (longest claim, max severity).
        canon = max(appearances, key=lambda f: (_SEV_RANK[f.severity], len(f.claim)))
        sev = max(_SEV_RANK[a.severity] for a in appearances)
        sev_str = next(s for s, r in _SEV_RANK.items() if r == sev)
        suggested = max(
            (a.suggested_repro for a in appearances),
            key=lambda s: len(s or ""),
            default="",
        )
        if families is not None:
            fam_set = {fam for _, fam in appearances_with_family if fam}
            families_to_confirm = len(fam_set)
        else:
            families_to_confirm = 0
        merged_by_id[canon.id] = Finding(
            id=canon.id,
            file=canon.file,
            function=canon.function,
            claim=canon.claim,
            claim_signature=canon.claim_signature,
            severity=sev_str,
            evidence_paths=canon.evidence_paths,
            suggested_repro=suggested,
            runs_to_confirm=len(appearances),
            total_runs=n,
            families_to_confirm=families_to_confirm,
            total_families=total_families,
        )
    return list(merged_by_id.values())


def rank_findings(findings: list[Finding]) -> list[Finding]:
    """Order findings for report display.

    Sort priority:
      1. Status — FIXED > CONFIRMED > FIX_FAILED > UNCONFIRMABLE > SPECULATIVE
      2. Severity — CRITICAL > HIGH > MEDIUM > LOW > INFO
      3. Confirmation rate — runs_to_confirm / total_runs, descending
      4. File path — for stable ordering across runs
    """
    def _key(f: Finding) -> tuple[int, int, float, str, str]:
        conf_rate = f.runs_to_confirm / max(1, f.total_runs)
        return (
            -_STATUS_RANK.get(f.status, 0),
            -_SEV_RANK.get(f.severity, 0),
            -conf_rate,
            f.file,
            f.function,
        )

    return sorted(findings, key=_key)


def confirmation_histogram(findings: list[Finding]) -> dict[str, int]:
    """Return a ``{"k/n": count}`` histogram across all findings.

    Primary FP/variance signal in the run report — same metric users
    filter on for low-noise mode.
    """
    hist: dict[str, int] = defaultdict(int)
    for f in findings:
        key = "0/0" if f.total_runs == 0 else f"{f.runs_to_confirm}/{f.total_runs}"
        hist[key] += 1
    return dict(hist)
