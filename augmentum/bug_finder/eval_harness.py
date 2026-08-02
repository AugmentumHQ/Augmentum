"""Bug Finder self-evaluation harness.

Loads `tests/bug_finder_fixtures/` and scores a `BugFinderRunReport`
against an `expected.json` spec. Pure functions — no LLM calls, no
container management. The pytest harness in
`tests/test_bug_finder_eval.py` wraps these in smoke + live tiers.

The scoring model weights precision over recall: false-positive fatigue
is the dominant failure mode in AI security tooling (Anthropic, XBOW,
Semgrep all converged on this), so a pipeline that finds half the bugs
with no FPs scores higher than one that finds them all plus FPs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from augmentum.bug_finder.findings import (
    Finding,
    FindingStatus,
    Severity,
    _SEV_RANK,
    _STATUS_RANK,
)

# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedFinding:
    """One expected-finding row from `expected.json`."""

    signature: str
    file: str
    line_start: int
    line_end: int
    min_severity: str = Severity.INFO.value
    min_status: str = FindingStatus.SPECULATIVE.value
    must_build_poc: bool = False


@dataclass(frozen=True)
class Fixture:
    """One eval fixture loaded from disk."""

    fixture_id: str
    path: Path
    language: str
    kind: str  # "true_positive" | "fp_bait" | "red_herring"
    expected_findings: tuple[ExpectedFinding, ...]
    max_extra_findings: int
    notes: str = ""

    @property
    def source_files(self) -> list[Path]:
        """Every non-spec file inside the fixture directory."""
        return sorted(
            p for p in self.path.iterdir()
            if p.is_file() and p.name not in {"expected.json", "README.md"}
        )


def load_fixture(fixture_dir: Path) -> Fixture:
    """Parse one fixture directory. Raises ValueError on malformed input."""
    spec_path = fixture_dir / "expected.json"
    if not spec_path.exists():
        raise ValueError(f"fixture {fixture_dir.name}: missing expected.json")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"fixture {fixture_dir.name}: expected.json invalid: {e}") from e

    fid = spec.get("fixture_id") or fixture_dir.name
    if fid != fixture_dir.name:
        raise ValueError(
            f"fixture {fixture_dir.name}: fixture_id={fid!r} doesn't match directory name",
        )
    kind = spec.get("kind", "true_positive")
    if kind not in {"true_positive", "fp_bait", "red_herring"}:
        raise ValueError(f"fixture {fid}: unknown kind={kind!r}")

    expected_raw = spec.get("expected_findings") or []
    expected: list[ExpectedFinding] = []
    for row in expected_raw:
        if not isinstance(row, dict):
            raise ValueError(f"fixture {fid}: expected_findings entries must be objects")
        sig = str(row.get("signature") or "").strip()
        file = str(row.get("file") or "").strip()
        if not sig or not file:
            raise ValueError(f"fixture {fid}: signature + file are required on each expected finding")
        line_start = int(row.get("line_start") or 0)
        line_end = int(row.get("line_end") or line_start)
        if line_end < line_start:
            raise ValueError(f"fixture {fid}: line_end < line_start on expected finding")
        expected.append(ExpectedFinding(
            signature=sig,
            file=file,
            line_start=line_start,
            line_end=line_end,
            min_severity=str(row.get("min_severity") or Severity.INFO.value),
            min_status=str(row.get("min_status") or FindingStatus.SPECULATIVE.value),
            must_build_poc=bool(row.get("must_build_poc", False)),
        ))

    if kind != "true_positive" and expected:
        raise ValueError(f"fixture {fid}: {kind} fixtures must declare zero expected_findings")

    return Fixture(
        fixture_id=fid,
        path=fixture_dir,
        language=str(spec.get("language") or "python"),
        kind=kind,
        expected_findings=tuple(expected),
        max_extra_findings=int(spec.get("max_extra_findings", 0)),
        notes=str(spec.get("notes") or ""),
    )


def load_fixture_set(root: Path) -> list[Fixture]:
    """Load every fixture directory under `root`, sorted by id."""
    if not root.is_dir():
        raise FileNotFoundError(f"fixtures root not found: {root}")
    out: list[Fixture] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        out.append(load_fixture(child))
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class FindingMatch:
    """One alignment between an expected and observed finding."""

    expected: ExpectedFinding
    observed: Finding
    severity_ok: bool
    status_ok: bool
    poc_ok: bool

    @property
    def is_strong_match(self) -> bool:
        return self.severity_ok and self.status_ok and self.poc_ok


@dataclass
class FixtureScore:
    """Per-fixture verdict + the diagnostic breakdown for the report."""

    fixture_id: str
    kind: str
    expected_count: int
    matched_strong: int
    matched_weak: int       # alignment but failing severity/status/PoC gate
    unmatched_expected: list[ExpectedFinding]
    extra_findings: list[Finding]   # observed findings with no alignment
    extra_confirmed_findings: int    # subset of extra_findings with CONFIRMED+ status
    poc_required: int = 0            # expected findings with must_build_poc=True
    poc_built: int = 0               # of those, how many were strongly matched
    cost_tokens_in: int = 0
    cost_tokens_out: int = 0
    cost_wallclock_ms: int = 0

    @property
    def passed(self) -> bool:
        """Tightest definition: every TP recovered at min_status, every
        FP-bait/red-herring stayed clean.

        - true_positive: all expected_findings matched strongly + extras within tolerance
        - fp_bait / red_herring: zero confirmed-or-better findings
        """
        if self.kind == "true_positive":
            return (
                self.matched_strong == self.expected_count
                and not self.unmatched_expected
                and self.extra_confirmed_findings <= self.max_allowed_extra
            )
        # FP-bait / red herring: anything CONFIRMED is a fail
        return self.extra_confirmed_findings == 0

    max_allowed_extra: int = 0


@dataclass
class EvalReport:
    """Aggregate score across a full fixture set."""

    fixtures: list[FixtureScore]
    weights: dict[str, float] = field(default_factory=lambda: {
        "precision": 0.45,
        "fp_bait_survival": 0.30,
        "recall": 0.20,
        "poc_build_rate": 0.05,
    })

    @property
    def tp_fixtures(self) -> list[FixtureScore]:
        return [f for f in self.fixtures if f.kind == "true_positive"]

    @property
    def fp_bait_fixtures(self) -> list[FixtureScore]:
        return [f for f in self.fixtures if f.kind in {"fp_bait", "red_herring"}]

    @property
    def precision(self) -> float:
        """Confirmed-TP / (confirmed-TP + confirmed-FP).

        Confirmed-TP = strong matches across true_positive fixtures.
        Confirmed-FP = any CONFIRMED+ finding on a fp_bait/red_herring
        plus extra-confirmed findings on TP fixtures that don't align.
        """
        tp = sum(f.matched_strong for f in self.tp_fixtures)
        fp = sum(f.extra_confirmed_findings for f in self.fixtures)
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    @property
    def recall(self) -> float:
        """Strong-matched TPs / expected TPs."""
        expected = sum(f.expected_count for f in self.tp_fixtures)
        matched = sum(f.matched_strong for f in self.tp_fixtures)
        if expected == 0:
            return 0.0
        return matched / expected

    @property
    def fp_bait_survival(self) -> float:
        """Fraction of FP-bait + red-herring fixtures that produced zero
        confirmed findings."""
        fp_bait = self.fp_bait_fixtures
        if not fp_bait:
            return 1.0
        survived = sum(1 for f in fp_bait if f.extra_confirmed_findings == 0)
        return survived / len(fp_bait)

    @property
    def poc_build_rate(self) -> float:
        """Of eligible TP findings (must_build_poc=true), how many actually
        reached CONFIRMED. PoC construction is the disproof-oriented
        verifier's job, so this isolates that subsystem's health."""
        eligible = sum(f.poc_required for f in self.tp_fixtures)
        built = sum(f.poc_built for f in self.tp_fixtures)
        if eligible == 0:
            return 1.0
        return built / eligible

    @property
    def aggregate_score(self) -> float:
        """Weighted [0, 100] score. Precision and FP-bait survival dominate
        because FPs are the worst failure mode."""
        components = {
            "precision": self.precision,
            "fp_bait_survival": self.fp_bait_survival,
            "recall": self.recall,
            "poc_build_rate": self.poc_build_rate,
        }
        score = sum(self.weights[k] * v for k, v in components.items())
        return round(score * 100.0, 1)

    @property
    def passed_count(self) -> int:
        return sum(1 for f in self.fixtures if f.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixtures": [asdict(f) for f in self.fixtures],
            "weights": dict(self.weights),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "fp_bait_survival": round(self.fp_bait_survival, 4),
            "poc_build_rate": round(self.poc_build_rate, 4),
            "aggregate_score": self.aggregate_score,
            "passed_count": self.passed_count,
            "total_count": len(self.fixtures),
        }


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _line_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Lines in common between [a_start, a_end] and [b_start, b_end] (inclusive).
    Returns 0 when disjoint."""
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _signatures_compatible(expected: str, observed: str) -> bool:
    """Some signatures are interchangeable (injection ⇄ missing_validation
    for path traversal, etc.). Accept those families."""
    if expected == observed:
        return True
    interchangeable = {
        frozenset({"injection", "missing_validation"}),
        frozenset({"auth_bypass", "missing_validation"}),
        frozenset({"logic_error", "missing_validation"}),
    }
    return frozenset({expected, observed}) in interchangeable


def _match_findings(
    expected: tuple[ExpectedFinding, ...],
    observed: list[Finding],
    *,
    observed_lines: dict[str, tuple[int, int]] | None = None,
) -> tuple[list[FindingMatch], list[ExpectedFinding], list[Finding]]:
    """Greedy alignment of expected ↔ observed findings.

    `observed_lines` is an optional map from finding-id → (start, end) line
    range parsed from the detector's claim/evidence (some detectors emit
    a literal `line` field, others embed it in evidence paths like
    `bug.py:14`). When absent, we treat the observed line range as
    unbounded — file + signature must still align.

    Returns (matches, unmatched_expected, unmatched_observed).
    """
    matches: list[FindingMatch] = []
    used: set[str] = set()
    unmatched_expected: list[ExpectedFinding] = []
    observed_lines = observed_lines or {}

    for exp in expected:
        best: Finding | None = None
        for obs in observed:
            if obs.id in used:
                continue
            if Path(obs.file).name != Path(exp.file).name:
                continue
            if not _signatures_compatible(exp.signature, obs.claim_signature):
                continue
            # Optional line overlap check
            obs_range = observed_lines.get(obs.id)
            if obs_range is not None:
                if _line_overlap(exp.line_start, exp.line_end, obs_range[0], obs_range[1]) == 0:
                    continue
            best = obs
            break  # first compatible wins — fixtures have <= 1 expected per file
        if best is None:
            unmatched_expected.append(exp)
            continue
        used.add(best.id)
        severity_ok = _SEV_RANK.get(best.severity, 0) >= _SEV_RANK.get(exp.min_severity, 0)
        status_ok = _STATUS_RANK.get(best.status, 0) >= _STATUS_RANK.get(exp.min_status, 0)
        poc_ok = (not exp.must_build_poc) or bool(best.repro_path)
        matches.append(FindingMatch(
            expected=exp, observed=best,
            severity_ok=severity_ok, status_ok=status_ok, poc_ok=poc_ok,
        ))

    unmatched_observed = [o for o in observed if o.id not in used]
    return matches, unmatched_expected, unmatched_observed


def score_fixture(
    fixture: Fixture,
    observed: list[Finding],
    *,
    cost_tokens_in: int = 0,
    cost_tokens_out: int = 0,
    cost_wallclock_ms: int = 0,
    observed_lines: dict[str, tuple[int, int]] | None = None,
) -> FixtureScore:
    """Score a single run against its fixture."""
    matches, unmatched_expected, unmatched_observed = _match_findings(
        fixture.expected_findings, observed,
        observed_lines=observed_lines,
    )
    strong = sum(1 for m in matches if m.is_strong_match)
    weak = len(matches) - strong
    extra_confirmed = sum(
        1 for f in unmatched_observed
        if _STATUS_RANK.get(f.status, 0) >= _STATUS_RANK[FindingStatus.CONFIRMED.value]
    )
    poc_required = sum(1 for exp in fixture.expected_findings if exp.must_build_poc)
    poc_built = sum(
        1 for m in matches
        if m.expected.must_build_poc and m.is_strong_match
    )
    return FixtureScore(
        fixture_id=fixture.fixture_id,
        kind=fixture.kind,
        expected_count=len(fixture.expected_findings),
        matched_strong=strong,
        matched_weak=weak,
        unmatched_expected=list(unmatched_expected),
        extra_findings=list(unmatched_observed),
        extra_confirmed_findings=extra_confirmed,
        poc_required=poc_required,
        poc_built=poc_built,
        cost_tokens_in=cost_tokens_in,
        cost_tokens_out=cost_tokens_out,
        cost_wallclock_ms=cost_wallclock_ms,
        max_allowed_extra=fixture.max_extra_findings,
    )
