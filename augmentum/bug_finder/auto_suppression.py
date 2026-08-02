"""Auto-suppression candidate surface.

Pattern memory accumulates ``(signature × file_pattern → hit_count)`` on
every scan. When a pattern keeps appearing but never gets confirmed as
a real bug, it's noise — and the right move is to suppress it so the
agent stops re-scoring it as a finding every run.

This module surfaces those patterns as ``SuppressionCandidate`` rows:

* **Strong evidence** — hit_count is the primary signal. Bandit B101
  showing up 4327 times in pydantic is a clear "library idiom" tell.
* **Zero confirmations** — if any past attempt to fix this pattern
  succeeded (``fix_count > 0``), the pattern is signal, not noise.
* **Severity-weighted thresholds** — security-class rules need MUCH
  more evidence before auto-suppression than style rules. We never
  want to auto-suppress SQL-injection or weak-crypto patterns at the
  same threshold as line-length warnings.

The substrate produces candidates; the human (or a higher-level
agent / lead) reviews and applies them. ``apply_candidate`` converts
a candidate into a real ``WorkspaceSuppression`` entry.

This is the "the bug_finder gets smarter at YOUR codebase" payoff:
patterns.json measures the noise, auto_suppression converts that
measurement into actionable cleanup, and the next run scans cleaner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from augmentum.bug_finder.workspace_substrate import (
    WorkspacePattern,
    WorkspaceSuppression,
    add_suppression,
    load_workspace_patterns,
    load_workspace_suppressions,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Severity / rule-class taxonomy
# ---------------------------------------------------------------------------


# Rules where any single occurrence might be a real exploit. We want
# overwhelming evidence before auto-suppressing these — and even then,
# the candidate is gated behind a manual confirmation step.
# NOTE: all entries are lowercase — lookups normalize the same way.
_DANGEROUS_RULES = frozenset(s.lower() for s in {
    # SQL / shell / code injection
    "bandit:B608", "ruff:S608",        # SQL string assembly
    "bandit:B602", "ruff:S602",        # subprocess with shell=True
    "bandit:B605", "ruff:S605",        # shell injection (os.system)
    "bandit:B609", "ruff:S609",        # shell command pipe
    "bandit:B611", "ruff:S611",        # RawSQL in Django
    # Auth / crypto / TLS
    "bandit:B501",                     # request with verify=False
    "bandit:B502", "ruff:S502",        # ssl insecure
    "bandit:B503", "ruff:S503",        # ssl bad protocol
    "bandit:B504", "ruff:S504",        # ssl no cert
    "bandit:B505", "ruff:S505",        # weak crypto key length
    "bandit:B324", "ruff:S324",        # weak hash MD5/SHA1
    # SSRF / deserialization
    "bandit:B411", "ruff:S411",        # xmlrpc untrusted
    "bandit:B301", "ruff:S301",        # pickle deserialize
    "bandit:B307", "ruff:S307",        # eval
    # XML
    "bandit:B313", "ruff:S313",
    "bandit:B314", "ruff:S314",
    "bandit:B319", "ruff:S319",
})

# Rules that are mostly style / convention. Auto-suppress at lower
# thresholds — these don't tend to hide real bugs.
_INERT_RULES = frozenset(s.lower() for s in {
    "ruff:E501",                       # line too long
    "ruff:E731",                       # lambda assignment
    "ruff:RUF100",                     # unused noqa
    "ruff:RUF005",                     # list concat could be unpacked
    "ruff:RUF012",                     # mutable class default
    "ruff:B018",                       # useless expression
})

# Rules that are mid-tier — common idioms in some frameworks (asserts
# in test/validation libraries, blind excepts in CLI tools). Default
# threshold applies.
_DEFAULT_HIT_THRESHOLD = 100
_DANGEROUS_HIT_THRESHOLD = 500
_INERT_HIT_THRESHOLD = 30


# ---------------------------------------------------------------------------
# Candidate shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuppressionCandidate:
    """One pattern surfaced as a suppression candidate.

    ``confidence`` is a coarse 0-1 score: 1.0 = "definitely suppress",
    < 0.5 = "borderline, review carefully". The score combines hit
    count, age (how long since last_seen), and rule-class danger.

    ``rule_id`` is the auto-generated identifier the suppression would
    receive if applied — ``auto_{scanner}_{category}_{file_hash}``.
    """

    signature: str          # "bandit:B101" or similar
    file_pattern: str       # the path or glob the pattern hit
    hit_count: int
    fix_count: int
    severity: str
    confidence: float
    rationale: str
    rule_id: str
    sample_claim: str = ""

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "file_pattern": self.file_pattern,
            "hit_count": self.hit_count,
            "fix_count": self.fix_count,
            "severity": self.severity,
            "confidence": round(self.confidence, 2),
            "rationale": self.rationale,
            "rule_id": self.rule_id,
            "sample_claim": self.sample_claim[:200],
        }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _rule_threshold(signature: str) -> int:
    """The minimum hit_count required to qualify as a candidate."""
    sig = signature.strip().lower()
    if sig in _DANGEROUS_RULES:
        return _DANGEROUS_HIT_THRESHOLD
    if sig in _INERT_RULES:
        return _INERT_HIT_THRESHOLD
    return _DEFAULT_HIT_THRESHOLD


def _confidence_score(pattern: WorkspacePattern) -> float:
    """Coarse 0-1 score combining evidence weight + rule class.

    A pattern with 1000 hits is more confident than one with 100. A
    pattern in the dangerous-rule set is *less* confident even with
    1000 hits — we want stronger evidence before suppressing security
    rules.
    """
    hits = pattern.hit_count
    sig = pattern.signature.strip().lower()

    if sig in _INERT_RULES:
        # Style rules: low evidence threshold.
        base = min(1.0, hits / 100.0)
        return min(1.0, base + 0.1)   # small bonus — style is style
    if sig in _DANGEROUS_RULES:
        # Security rules: high evidence threshold + ceiling.
        base = min(1.0, hits / 2000.0)
        return min(0.75, base)         # cap below 1.0 — never "definite" auto-suppress
    # Default mid-tier
    base = min(1.0, hits / 500.0)
    return min(0.95, base + 0.05)


def _rule_id_for(signature: str, file_pattern: str) -> str:
    """Stable, human-readable identifier for the generated rule."""
    safe = (
        signature.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )
    # Last path component keeps the rule id short + readable.
    tail = (file_pattern.rsplit("/", 1)[-1] or "all")[:32]
    return f"auto_{safe}__{tail}".lower()


def _rationale(pattern: WorkspacePattern) -> str:
    """Human-readable justification for the candidate."""
    sig = pattern.signature.strip().lower()
    parts = [f"{pattern.hit_count} hits"]
    if sig in _INERT_RULES:
        parts.append("style/style-adjacent rule")
    elif sig in _DANGEROUS_RULES:
        parts.append("SECURITY-class rule — review carefully")
    if pattern.fix_count == 0:
        parts.append("never confirmed as real")
    return "; ".join(parts)


def compute_suppression_candidates(
    workspace_root: Path,
    *,
    min_hits_override: int | None = None,
    require_zero_fixes: bool = True,
) -> list[SuppressionCandidate]:
    """Return candidates ordered by descending confidence.

    Filters:
      * Pattern's ``hit_count`` >= rule-class threshold (or override).
      * Pattern's ``fix_count`` == 0 (unless ``require_zero_fixes=False``).
      * Pattern not already covered by an existing suppression rule.
    """
    patterns = load_workspace_patterns(workspace_root)
    existing = load_workspace_suppressions(workspace_root)
    if not patterns:
        return []

    # Index existing suppressions by (scope, pattern) so we can skip
    # patterns already covered. Scope of an auto-suppression is
    # ``rule`` (per-category) by default.
    existing_index = {(s.scope, s.pattern.lower()) for s in existing}

    out: list[SuppressionCandidate] = []
    for p in patterns:
        sig = p.signature.strip()
        if require_zero_fixes and p.fix_count > 0:
            continue
        threshold = min_hits_override or _rule_threshold(sig)
        if p.hit_count < threshold:
            continue
        # Has the user already added a rule covering this category?
        # ``pattern`` in the suppression entry is matched against the
        # scanner's rule code (e.g. "B101"), which we derive by
        # stripping the scanner prefix.
        _, _, rule_code = sig.partition(":")
        if (
            ("rule", sig.lower()) in existing_index
            or ("rule", rule_code.lower()) in existing_index
            or ("category", rule_code.lower()) in existing_index
        ):
            continue

        out.append(SuppressionCandidate(
            signature=sig,
            file_pattern=p.file_pattern,
            hit_count=p.hit_count,
            fix_count=p.fix_count,
            severity=p.severity,
            confidence=_confidence_score(p),
            rationale=_rationale(p),
            rule_id=_rule_id_for(sig, p.file_pattern),
            sample_claim=p.sample_claim,
        ))

    out.sort(key=lambda c: (-c.confidence, -c.hit_count))
    return out


# ---------------------------------------------------------------------------
# Aggregation: project-wide candidates (collapses per-file rows)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregatedCandidate:
    """A signature-level summary across many files.

    The pattern memory records ``(signature × file)``, so the same
    signature can produce many candidate rows. This collapses them
    into one row per signature with the totals so the human reviewer
    sees "B101 has 4327 hits across 412 files, suppress globally"
    rather than 412 individual rows.
    """

    signature: str
    total_hits: int
    file_count: int
    confidence: float
    rationale: str
    rule_id: str
    severity: str
    top_files: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "total_hits": self.total_hits,
            "file_count": self.file_count,
            "confidence": round(self.confidence, 2),
            "rationale": self.rationale,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "top_files": list(self.top_files[:5]),
        }


def aggregate_candidates(
    candidates: list[SuppressionCandidate],
) -> list[AggregatedCandidate]:
    """Collapse per-file candidates into one row per signature.

    Use this when you already have per-file candidates (e.g. from
    ``compute_suppression_candidates``) and want a signature-level
    rollup. For a direct-from-patterns aggregate that doesn't lose
    signatures whose per-file counts are below the per-file
    threshold, use ``compute_aggregated_candidates``.
    """
    by_sig: dict[str, list[SuppressionCandidate]] = {}
    for c in candidates:
        by_sig.setdefault(c.signature, []).append(c)
    out: list[AggregatedCandidate] = []
    for sig, rows in by_sig.items():
        total = sum(r.hit_count for r in rows)
        files = [(r.file_pattern, r.hit_count) for r in rows]
        files.sort(key=lambda fc: -fc[1])
        rationale = (
            f"{total} total hits across {len(rows)} files; "
            + ("SECURITY rule — review " if sig.lower() in _DANGEROUS_RULES else "")
            + ("inert/style rule" if sig.lower() in _INERT_RULES else "")
        ).strip("; ")
        out.append(AggregatedCandidate(
            signature=sig,
            total_hits=total,
            file_count=len(rows),
            confidence=max(r.confidence for r in rows),
            rationale=rationale,
            rule_id=f"auto_{sig.replace(':','_').lower()}__global",
            severity=max(rows, key=lambda r: r.hit_count).severity,
            top_files=files[:5],
        ))
    out.sort(key=lambda r: (-r.confidence, -r.total_hits))
    return out


def compute_aggregated_candidates(
    workspace_root: Path,
    *,
    min_total_hits_override: int | None = None,
    require_zero_fixes: bool = True,
) -> list[AggregatedCandidate]:
    """Aggregate-first: compute candidates at the *signature* level,
    summing hit_count across all files.

    This is the common path — most real noise (B101 in pydantic, B101
    in celery) shows up as many small per-file hits that wouldn't
    individually qualify, but their sum is unmistakable signal.

    Thresholds apply to the *total* across files, not per-file:
      * inert rules: total >= ``_INERT_HIT_THRESHOLD``
      * security rules: total >= ``_DANGEROUS_HIT_THRESHOLD``
      * everything else: total >= ``_DEFAULT_HIT_THRESHOLD``

    Patterns with ANY confirmation across any file are skipped when
    ``require_zero_fixes=True`` — a single confirmed B101 anywhere
    in the repo means B101 isn't blanket noise.
    """
    patterns = load_workspace_patterns(workspace_root)
    existing = load_workspace_suppressions(workspace_root)
    if not patterns:
        return []

    existing_index = {(s.scope, s.pattern.lower()) for s in existing}

    by_sig: dict[str, list[WorkspacePattern]] = {}
    for p in patterns:
        by_sig.setdefault(p.signature.strip(), []).append(p)

    out: list[AggregatedCandidate] = []
    for sig, rows in by_sig.items():
        total = sum(r.hit_count for r in rows)
        total_fixes = sum(r.fix_count for r in rows)
        if require_zero_fixes and total_fixes > 0:
            continue
        threshold = min_total_hits_override or _rule_threshold(sig)
        if total < threshold:
            continue
        # Skip if a manually-added suppression already covers it
        _, _, rule_code = sig.partition(":")
        if (
            ("rule", sig.lower()) in existing_index
            or ("rule", rule_code.lower()) in existing_index
            or ("category", rule_code.lower()) in existing_index
        ):
            continue
        # Use the highest-hit row as the confidence basis
        proxy = max(rows, key=lambda r: r.hit_count)
        # ...but compute confidence using the TOTAL so a 4327-hit
        # signature across many files outranks a 100-hit single file.
        synthetic = WorkspacePattern(
            signature=sig,
            file_pattern=proxy.file_pattern,
            hit_count=total,
            fix_count=total_fixes,
            severity=proxy.severity,
        )
        conf = _confidence_score(synthetic)
        rationale_parts = [
            f"{total} total hits across {len(rows)} files",
        ]
        if sig.lower() in _DANGEROUS_RULES:
            rationale_parts.append("SECURITY rule — review carefully")
        elif sig.lower() in _INERT_RULES:
            rationale_parts.append("style/style-adjacent rule")
        if total_fixes == 0:
            rationale_parts.append("never confirmed as real")
        files = [(r.file_pattern, r.hit_count) for r in rows]
        files.sort(key=lambda fc: -fc[1])
        out.append(AggregatedCandidate(
            signature=sig,
            total_hits=total,
            file_count=len(rows),
            confidence=conf,
            rationale="; ".join(rationale_parts),
            rule_id=f"auto_{sig.replace(':','_').lower()}__global",
            severity=proxy.severity,
            top_files=files[:5],
        ))
    out.sort(key=lambda r: (-r.confidence, -r.total_hits))
    return out


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _lookup_or_synth(
    workspace_root: Path,
    *,
    rule_id: str,
    scope: str,
    pattern: str,
    reason: str,
) -> WorkspaceSuppression:
    """Read back the persisted suppression, or synthesize a stub if the
    add was a no-op (rule_id already existed). Returning a struct in
    both cases makes callers easier to chain."""
    for r in load_workspace_suppressions(workspace_root):
        if r.rule_id == rule_id:
            return r
    return WorkspaceSuppression(
        rule_id=rule_id, scope=scope, pattern=pattern, reason=reason,
    )


def apply_candidate(
    workspace_root: Path,
    candidate: SuppressionCandidate,
    *,
    reason: str = "",
    scope: str = "rule",
) -> WorkspaceSuppression:
    """Convert one ``SuppressionCandidate`` into a stored suppression.

    Default scope is ``rule`` (matches by scanner rule code, repo-
    wide). Pass ``scope="file"`` to limit the suppression to the
    pattern's file_pattern only.

    Returns the persisted ``WorkspaceSuppression`` so the caller can
    log + diff. Idempotent — re-applying the same candidate is a
    no-op write but still returns the existing record.
    """
    rule_code = candidate.signature.partition(":")[2] or candidate.signature
    pattern = rule_code if scope == "rule" else candidate.file_pattern
    note = reason or candidate.rationale
    add_suppression(
        workspace_root,
        rule_id=candidate.rule_id,
        scope=scope,
        pattern=pattern,
        reason=note,
    )
    return _lookup_or_synth(
        workspace_root,
        rule_id=candidate.rule_id, scope=scope, pattern=pattern, reason=note,
    )


def apply_aggregated(
    workspace_root: Path,
    candidate: AggregatedCandidate,
    *,
    reason: str = "",
) -> WorkspaceSuppression:
    """Apply an ``AggregatedCandidate`` repo-wide."""
    rule_code = candidate.signature.partition(":")[2] or candidate.signature
    note = reason or candidate.rationale
    add_suppression(
        workspace_root,
        rule_id=candidate.rule_id,
        scope="rule",
        pattern=rule_code,
        reason=note,
    )
    return _lookup_or_synth(
        workspace_root,
        rule_id=candidate.rule_id, scope="rule",
        pattern=rule_code, reason=note,
    )
