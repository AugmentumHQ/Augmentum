"""Codebase-agnostic detection stage.

Bridges the deterministic substrate (generic_scanners + custom checks +
workspace suppressions) into the bug-finder pipeline. Output is a list of
``Finding`` objects shaped exactly like the LLM-detector emits, so the
rest of the pipeline (ranking, dedup, fix, report) is oblivious to where
a finding came from.

Why this lives separately from ``dev_tools.py``:

* ``dev_tools.py`` wraps Augmentum-specific scanners (red_team_scan,
  security_check, runtime_checks) — only useful inside augmentum-shaped
  repos.
* ``generic_scanners.py`` wraps Bandit + Ruff — useful in any Python
  codebase.
* ``custom_check_runner.py`` runs codebase-specific AST checks the
  check-writer subagent generated on a prior run.

This stage glues all three to the per-workspace substrate so each
finding gets:

* filtered against ``suppressions.json`` (reviewed false positives)
* converted from ``ScannerFinding`` to ``Finding`` shape
* deduped against the LLM stream by the standard ``merge_runs`` key
* counted into ``patterns.json`` (signature × file_pattern → hit_count)

The orchestrator calls ``run_agnostic_stage`` after workspace prep and
before LLM detection. Confirmation counters get bumped post-verify via
``record_confirmation``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from augmentum.bug_finder.custom_check_runner import (
    collect_all_findings as run_custom_checks,
)
from augmentum.bug_finder.dev_tools import ScannerFinding
from augmentum.bug_finder.findings import (
    ClaimSignature,
    Finding,
    FindingStatus,
    classify_claim,
    finding_from_dict,
)
from augmentum.bug_finder.generic_scanners import run_generic_suite_timed
from augmentum.bug_finder.workspace_substrate import (
    ensure_substrate,
    is_suppressed,
    load_workspace_suppressions,
    upsert_pattern,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Severities that flow into the LLM pipeline as Findings. Lower
# severities are still counted (for dashboard / competency tracking)
# but don't compete for verify/fix attention. Bandit/Ruff at "info"
# level is mostly style nits.
_PIPELINE_SEVERITIES = frozenset({"critical", "high", "medium"})


# Cross-scanner duplicate map — Bandit and Ruff catch the same
# patterns under different rule codes. Without dedup, the same line
# gets seeded twice and the pattern memory double-counts. Keys are
# the canonical (scanner, category); values are equivalent
# (scanner, category) pairs scanners would also emit for that line.
_CROSS_SCANNER_EQUIVALENTS: dict[tuple[str, str], frozenset[tuple[str, str]]] = {
    # assert detected (production code)
    ("bandit", "B101"): frozenset({("ruff", "S101")}),
    ("ruff", "S101"):   frozenset({("bandit", "B101")}),
    # hardcoded password literal
    ("bandit", "B105"): frozenset({("ruff", "S105")}),
    ("ruff", "S105"):   frozenset({("bandit", "B105")}),
    # weak MD5/SHA1 hash
    ("bandit", "B324"): frozenset({("ruff", "S324")}),
    ("ruff", "S324"):   frozenset({("bandit", "B324")}),
    # subprocess shell injection / try/except/pass
    ("bandit", "B603"): frozenset({("ruff", "S603")}),
    ("ruff", "S603"):   frozenset({("bandit", "B603")}),
    ("bandit", "B110"): frozenset({("ruff", "S110")}),
    ("ruff", "S110"):   frozenset({("bandit", "B110")}),
}


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgnosticStageResult:
    """Aggregate output from one substrate sweep.

    ``seeded_findings`` is the Finding-shaped subset that proceeds into
    the LLM pipeline (typically medium+ severity, post-suppression).

    ``scanner_counts`` is the per-scanner raw count BEFORE suppression
    or severity filtering — useful for the dashboard / audit-history.

    ``suppressed_count`` is the number of findings filtered by the
    workspace's ``suppressions.json``.
    """

    seeded_findings: list[Finding]
    scanner_counts: dict[str, int]
    suppressed_count: int
    wallclock_seconds: float
    pattern_signatures: dict[str, int] = field(default_factory=dict)

    @property
    def total_raw(self) -> int:
        return sum(self.scanner_counts.values())

    def summary_line(self) -> str:
        """One-line text suitable for ``notes.append`` in the report."""
        parts = [
            f"{name}={count}"
            for name, count in sorted(self.scanner_counts.items())
        ]
        return (
            f"agnostic substrate: {self.total_raw} raw "
            f"({', '.join(parts)}); "
            f"{len(self.seeded_findings)} seeded into pipeline, "
            f"{self.suppressed_count} suppressed, "
            f"{self.wallclock_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# Scanner rule → claim_signature mapping
# ---------------------------------------------------------------------------


# Scanners emit text like "subprocess call with shell=True identified"
# that doesn't trigger ``classify_claim``'s keyword matcher. We map
# (scanner, category) directly to ClaimSignature for known rules,
# falling back to ``classify_claim`` for unknowns.
_RULE_SIGNATURE_MAP: dict[tuple[str, str], str] = {
    # Injection — SQL, shell, code
    ("bandit", "B102"):  ClaimSignature.INJECTION.value,    # exec
    ("bandit", "B307"):  ClaimSignature.INJECTION.value,    # eval
    ("ruff", "S307"):    ClaimSignature.INJECTION.value,
    ("bandit", "B301"):  ClaimSignature.INJECTION.value,    # pickle (deserialization RCE)
    ("ruff", "S301"):    ClaimSignature.INJECTION.value,
    ("bandit", "B602"):  ClaimSignature.INJECTION.value,    # subprocess shell=True
    ("ruff", "S602"):    ClaimSignature.INJECTION.value,
    ("bandit", "B604"):  ClaimSignature.INJECTION.value,    # shell=True via wrapper
    ("ruff", "S604"):    ClaimSignature.INJECTION.value,
    ("bandit", "B605"):  ClaimSignature.INJECTION.value,    # os.system / popen
    ("ruff", "S605"):    ClaimSignature.INJECTION.value,
    ("bandit", "B608"):  ClaimSignature.INJECTION.value,    # SQL string assembly
    ("ruff", "S608"):    ClaimSignature.INJECTION.value,
    ("bandit", "B609"):  ClaimSignature.INJECTION.value,    # shell wildcard injection
    ("ruff", "S609"):    ClaimSignature.INJECTION.value,
    ("bandit", "B611"):  ClaimSignature.INJECTION.value,    # Django RawSQL
    ("ruff", "S611"):    ClaimSignature.INJECTION.value,
    # XML — covered as injection-class
    ("bandit", "B313"):  ClaimSignature.INJECTION.value,
    ("bandit", "B314"):  ClaimSignature.INJECTION.value,
    ("bandit", "B319"):  ClaimSignature.INJECTION.value,
    ("ruff", "S313"):    ClaimSignature.INJECTION.value,
    ("ruff", "S314"):    ClaimSignature.INJECTION.value,
    ("ruff", "S319"):    ClaimSignature.INJECTION.value,
    # Auth / TLS bypass
    ("bandit", "B501"):  ClaimSignature.AUTH_BYPASS.value,  # verify=False
    ("bandit", "B502"):  ClaimSignature.AUTH_BYPASS.value,  # ssl insecure
    ("ruff", "S502"):    ClaimSignature.AUTH_BYPASS.value,
    ("bandit", "B503"):  ClaimSignature.AUTH_BYPASS.value,
    ("ruff", "S503"):    ClaimSignature.AUTH_BYPASS.value,
    ("bandit", "B504"):  ClaimSignature.AUTH_BYPASS.value,
    ("ruff", "S504"):    ClaimSignature.AUTH_BYPASS.value,
    # Hardcoded creds — not a bug class per se, but maps to AUTH_BYPASS
    # in spirit (use of bad credential).
    ("bandit", "B105"):  ClaimSignature.AUTH_BYPASS.value,
    ("ruff", "S105"):    ClaimSignature.AUTH_BYPASS.value,
    ("bandit", "B106"):  ClaimSignature.AUTH_BYPASS.value,
    ("ruff", "S106"):    ClaimSignature.AUTH_BYPASS.value,
    ("bandit", "B107"):  ClaimSignature.AUTH_BYPASS.value,
    ("ruff", "S107"):    ClaimSignature.AUTH_BYPASS.value,
    # Assert in production code — logic class, not security
    ("bandit", "B101"):  ClaimSignature.LOGIC_ERROR.value,
    ("ruff", "S101"):    ClaimSignature.LOGIC_ERROR.value,
    # Resource / cleanup
    ("ruff", "B017"):    ClaimSignature.MISSING_VALIDATION.value,
    ("ruff", "B007"):    ClaimSignature.LOGIC_ERROR.value,
    ("ruff", "B018"):    ClaimSignature.LOGIC_ERROR.value,
    ("ruff", "B023"):    ClaimSignature.LOGIC_ERROR.value,
    ("ruff", "B026"):    ClaimSignature.LOGIC_ERROR.value,
    ("ruff", "B904"):    ClaimSignature.LOGIC_ERROR.value,
    # Blind except — easy to miss bugs underneath
    ("ruff", "BLE001"):  ClaimSignature.OTHER.value,
    # Pyflakes
    ("ruff", "F401"):    ClaimSignature.OTHER.value,
    ("ruff", "F811"):    ClaimSignature.LOGIC_ERROR.value,
    ("ruff", "F823"):    ClaimSignature.LOGIC_ERROR.value,
    ("ruff", "F841"):    ClaimSignature.LOGIC_ERROR.value,
}


def _signature_for(sf: ScannerFinding) -> str:
    """Pick the best ClaimSignature for ``sf``.

    Precedence:
      1. Hard-coded (scanner, category) mapping (above).
      2. ``classify_claim()`` keyword matching on the message text.
      3. Falls through to ``OTHER`` via the standard normalization.
    """
    key = (sf.scanner.lower(), sf.category.upper() if sf.scanner != "ruff" else sf.category)
    # Ruff codes are case-sensitive (S608 not S608); bandit codes are
    # uppercase. The map uses each scanner's natural form.
    mapped = (
        _RULE_SIGNATURE_MAP.get((sf.scanner.lower(), sf.category))
        or _RULE_SIGNATURE_MAP.get((sf.scanner.lower(), sf.category.upper()))
    )
    if mapped:
        return mapped
    return classify_claim(sf.message)


# ---------------------------------------------------------------------------
# ScannerFinding → Finding adapter
# ---------------------------------------------------------------------------


def _scanner_to_finding(sf: ScannerFinding) -> Finding | None:
    """Convert a deterministic ScannerFinding into a Finding-dict and
    let ``finding_from_dict`` enforce the standard normalization."""
    if not sf.file or not sf.message:
        return None
    # The detector contract wants a non-empty ``function`` for grouping;
    # scanners don't know the function, so use the scanner+category as
    # a stable placeholder ("<scanner:category>"). Dedup remains
    # correct because ``claim_signature`` is also part of the key.
    function = f"<{sf.scanner}:{sf.category}>"
    # Scanners don't emit a fix-as-code; they emit either a hint
    # (Ruff `fix.message`) or nothing (Bandit). When present, append
    # it to the claim so the verifier sees the suggested remediation.
    claim = sf.message.strip()
    if sf.fix:
        claim = f"{claim}\nSuggested fix: {sf.fix.strip()}"
    payload = {
        "file": sf.file,
        "function": function,
        "claim": claim,
        "severity": sf.severity,
        "claim_signature": _signature_for(sf),
        # Evidence path = the file:line being flagged. Verifiers can
        # use this to anchor their repro attempt.
        "evidence_paths": [f"{sf.file}:{sf.line}"] if sf.line else [sf.file],
        "suggested_repro": "",
    }
    return finding_from_dict(payload)


# ---------------------------------------------------------------------------
# Stage driver
# ---------------------------------------------------------------------------


def run_agnostic_stage(
    workspace_root: Path,
    *,
    pipeline_severities: frozenset[str] = _PIPELINE_SEVERITIES,
    record_patterns: bool = True,
) -> AgnosticStageResult:
    """Run every deterministic source against ``workspace_root`` and
    return findings shaped for the bug-finder pipeline.

    Side effects (under ``record_patterns``):
      * Ensures ``.augmentum/bug_finder/`` exists.
      * Increments ``patterns.json`` hit_count for each
        (signature, file) pair surfaced this run.

    Suppressions from ``suppressions.json`` are applied before any
    pipeline-severity filtering — a suppressed finding is gone for the
    rest of the run regardless of its severity.
    """
    started = time.monotonic()
    if not workspace_root.is_dir():
        return AgnosticStageResult(
            seeded_findings=[],
            scanner_counts={},
            suppressed_count=0,
            wallclock_seconds=0.0,
        )

    ensure_substrate(workspace_root)
    suppressions = load_workspace_suppressions(workspace_root)

    # Bandit + Ruff sweep
    generic = run_generic_suite_timed(workspace_root)
    # Custom AST checks (writer-generated, codebase-specific)
    try:
        custom = run_custom_checks(workspace_root)
    except Exception as exc:  # noqa: BLE001 — substrate must be best-effort
        log.warning(
            "bug_finder_custom_check_collect_failed",
            workspace=str(workspace_root), error=str(exc),
        )
        custom = []

    # Flatten into one stream for processing
    all_raw: list[ScannerFinding] = []
    counts: dict[str, int] = {}
    for scanner_name, rows in generic.findings_by_scanner.items():
        all_raw.extend(rows)
        counts[scanner_name] = len(rows)
    # Custom checks are already per-finding; group counts by scanner tag.
    for sf in custom:
        all_raw.append(sf)
        counts[sf.scanner] = counts.get(sf.scanner, 0) + 1

    # Cross-scanner dedup — Bandit and Ruff catch the same patterns
    # under different rule codes (B101 vs S101 for asserts, etc.).
    # First scanner wins on (file, line). Without this, FastAPI's 198
    # bandit:B101 + 128 ruff:S101 both seed into the pipeline as
    # separate findings even though they're the same code.
    seen_by_position: set[tuple[str, int, tuple[str, str]]] = set()
    deduped: list[ScannerFinding] = []
    cross_dropped = 0
    for sf in all_raw:
        key_self = (sf.file, sf.line, (sf.scanner, sf.category))
        # Drop if we've already seen the same position via an equivalent
        # rule from another scanner.
        equivalents = _CROSS_SCANNER_EQUIVALENTS.get(
            (sf.scanner, sf.category), frozenset(),
        )
        if any(
            (sf.file, sf.line, eq) in seen_by_position
            for eq in equivalents
        ):
            cross_dropped += 1
            continue
        if key_self in seen_by_position:
            cross_dropped += 1
            continue
        seen_by_position.add(key_self)
        deduped.append(sf)
    if cross_dropped:
        counts["_cross_scanner_deduped"] = cross_dropped
    all_raw = deduped

    seeded: list[Finding] = []
    suppressed = 0
    pattern_hits: dict[str, int] = {}

    for sf in all_raw:
        rule_match = is_suppressed(
            suppressions,
            file=sf.file,
            category=sf.category,
            rule_id=sf.rule_id,
        )
        if rule_match:
            suppressed += 1
            continue

        # Pattern memory: count every (signature, file) appearance.
        # We use scanner+category as the signature to keep it stable
        # across runs even if the message text shifts.
        pattern_sig = f"{sf.scanner}:{sf.category}"
        if record_patterns:
            try:
                upsert_pattern(
                    workspace_root,
                    signature=pattern_sig,
                    file_pattern=sf.file,
                    sample_claim=sf.message[:200],
                    severity=sf.severity,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "bug_finder_pattern_upsert_failed",
                    error=str(exc), signature=pattern_sig,
                )
        pattern_hits[pattern_sig] = pattern_hits.get(pattern_sig, 0) + 1

        # Only medium+ flows into the LLM verify pipeline. Lower
        # severities still count in scanner_counts/patterns.json.
        if sf.severity not in pipeline_severities:
            continue
        finding = _scanner_to_finding(sf)
        if finding is None:
            continue
        # Mark these as scanner-seeded so the report can distinguish
        # deterministic sources from LLM-detected.
        finding.notes.append(f"source: scanner {sf.scanner}")
        # Scanners are deterministic — they always confirm themselves
        # on a re-run with the same code. Give them a single
        # confirmation up-front instead of leaving SPECULATIVE.
        finding.status = FindingStatus.SPECULATIVE.value
        finding.runs_to_confirm = 1
        finding.total_runs = 1
        seeded.append(finding)

    elapsed = time.monotonic() - started
    result = AgnosticStageResult(
        seeded_findings=seeded,
        scanner_counts=counts,
        suppressed_count=suppressed,
        wallclock_seconds=elapsed,
        pattern_signatures=pattern_hits,
    )
    log.info(
        "bug_finder_agnostic_stage_complete",
        workspace=str(workspace_root),
        raw=result.total_raw,
        seeded=len(seeded),
        suppressed=suppressed,
        wallclock=round(elapsed, 2),
    )
    return result


# ---------------------------------------------------------------------------
# Custom-checks-only sweep (used by the check-writer stage)
# ---------------------------------------------------------------------------


def run_custom_checks_stage(
    workspace_root: Path,
    *,
    only_checks: frozenset[str] | None = None,
    pipeline_severities: frozenset[str] = _PIPELINE_SEVERITIES,
    record_patterns: bool = True,
) -> AgnosticStageResult:
    """Run ONLY the workspace's custom AST checks — no Bandit/Ruff.

    The check-writer stage calls this right after it generates new
    checks, so their findings flow into the SAME run that wrote them
    without re-paying the (slow) generic scanner sweep. ``only_checks``
    limits the run to specific checks by name (filename without
    ``.py``); ``None`` runs every custom check.

    Output is identical in shape to :func:`run_agnostic_stage` —
    suppression-filtered, severity-gated, ``ScannerFinding`` →
    ``Finding`` converted, pattern-counted — so the orchestrator can
    extend ``scanner_seeded`` with the result and the rest of the
    pipeline is none the wiser.
    """
    from augmentum.bug_finder.custom_check_runner import run_all_custom_checks

    started = time.monotonic()
    if not workspace_root.is_dir():
        return AgnosticStageResult(
            seeded_findings=[], scanner_counts={},
            suppressed_count=0, wallclock_seconds=0.0,
        )

    ensure_substrate(workspace_root)
    suppressions = load_workspace_suppressions(workspace_root)

    all_raw: list[ScannerFinding] = []
    counts: dict[str, int] = {}
    for result in run_all_custom_checks(workspace_root):
        if not result.succeeded:
            continue
        if only_checks is not None and result.check_name not in only_checks:
            continue
        for sf in result.findings:
            all_raw.append(sf)
            counts[sf.scanner] = counts.get(sf.scanner, 0) + 1

    seeded: list[Finding] = []
    suppressed = 0
    pattern_hits: dict[str, int] = {}

    for sf in all_raw:
        if is_suppressed(
            suppressions, file=sf.file, category=sf.category,
            rule_id=sf.rule_id,
        ):
            suppressed += 1
            continue

        pattern_sig = f"{sf.scanner}:{sf.category}"
        if record_patterns:
            try:
                upsert_pattern(
                    workspace_root, signature=pattern_sig,
                    file_pattern=sf.file, sample_claim=sf.message[:200],
                    severity=sf.severity,
                )
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "bug_finder_pattern_upsert_failed",
                    error=str(exc), signature=pattern_sig,
                )
        pattern_hits[pattern_sig] = pattern_hits.get(pattern_sig, 0) + 1

        if sf.severity not in pipeline_severities:
            continue
        finding = _scanner_to_finding(sf)
        if finding is None:
            continue
        finding.notes.append(f"source: scanner {sf.scanner}")
        finding.status = FindingStatus.SPECULATIVE.value
        finding.runs_to_confirm = 1
        finding.total_runs = 1
        seeded.append(finding)

    elapsed = time.monotonic() - started
    return AgnosticStageResult(
        seeded_findings=seeded,
        scanner_counts=counts,
        suppressed_count=suppressed,
        wallclock_seconds=elapsed,
        pattern_signatures=pattern_hits,
    )


# ---------------------------------------------------------------------------
# Post-verify hook
# ---------------------------------------------------------------------------


def record_confirmation(
    workspace_root: Path,
    finding: Finding,
) -> None:
    """Bump the pattern's ``fix_count`` for a finding that survived
    verify/fix. Best-effort — never raises.

    Called by the orchestrator after the verify stage decides a finding
    is real (or after the fixer lands a patch).
    """
    if not workspace_root or not workspace_root.is_dir():
        return
    # The function placeholder ``<scanner:category>`` is the only place
    # we have the scanner signature post-conversion. Strip the angle
    # brackets back out.
    fn = finding.function or ""
    signature = (
        fn[1:-1] if fn.startswith("<") and fn.endswith(">")
        else finding.claim_signature
    )
    try:
        upsert_pattern(
            workspace_root,
            signature=signature,
            file_pattern=finding.file,
            sample_claim=finding.claim[:200],
            severity=finding.severity,
            confirmed=True,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "bug_finder_record_confirmation_failed",
            error=str(exc), signature=signature, file=finding.file,
        )
