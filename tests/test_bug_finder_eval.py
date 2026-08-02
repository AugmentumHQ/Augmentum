"""Bug Finder self-evaluation pytest harness.

Three tiers:

  Tier 1 (always runs) — fixture well-formedness:
    Every fixture's expected.json parses; referenced source files exist;
    expected line ranges fall inside the source files; signatures are
    valid ClaimSignature values.

  Tier 2 (always runs) — scorer logic:
    Synthetic Finding objects exercise the matching + scoring math so
    regressions in the harness itself surface without LLM calls.

  Tier 3 (--run-live) — end-to-end orchestrator runs:
    Real ContainerManager + real LLM. Reports precision / recall /
    FP-bait survival / aggregate score. Skipped by default. Wire-up
    documented in augmentum/bug_finder/eval_runner.py — bring a model
    and Docker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from augmentum.bug_finder.eval_harness import (
    EvalReport,
    FixtureScore,
    load_fixture,
    load_fixture_set,
    score_fixture,
)
from augmentum.bug_finder.findings import (
    ClaimSignature,
    Finding,
    FindingStatus,
    Severity,
)

FIXTURES_ROOT = Path(__file__).parent / "bug_finder_fixtures"


# ---------------------------------------------------------------------------
# Tier 1 — fixture well-formedness
# ---------------------------------------------------------------------------


def test_smoke_fixtures_root_exists() -> None:
    assert FIXTURES_ROOT.is_dir(), f"fixtures root missing: {FIXTURES_ROOT}"
    # At least the documented 12 fixtures should be present.
    fixtures = load_fixture_set(FIXTURES_ROOT)
    assert len(fixtures) >= 12, (
        f"expected >= 12 fixtures, got {len(fixtures)} — did someone remove one?"
    )


def test_smoke_format_doc_present() -> None:
    assert (FIXTURES_ROOT / "_format.md").exists(), "_format.md is the schema reference"


@pytest.mark.parametrize("fixture_dir", sorted(
    p for p in FIXTURES_ROOT.iterdir()
    if p.is_dir() and not p.name.startswith(("_", "."))
))
def test_smoke_fixture_loads(fixture_dir: Path) -> None:
    """Every fixture parses without error."""
    fixture = load_fixture(fixture_dir)
    assert fixture.fixture_id == fixture_dir.name
    assert fixture.kind in {"true_positive", "fp_bait", "red_herring"}
    assert fixture.source_files, f"fixture {fixture.fixture_id} has no source files"


@pytest.mark.parametrize("fixture_dir", sorted(
    p for p in FIXTURES_ROOT.iterdir()
    if p.is_dir() and not p.name.startswith(("_", "."))
))
def test_smoke_expected_findings_reference_existing_lines(fixture_dir: Path) -> None:
    """expected.json line ranges must fall inside the actual source files."""
    fixture = load_fixture(fixture_dir)
    for exp in fixture.expected_findings:
        target = fixture.path / exp.file
        assert target.exists(), (
            f"fixture {fixture.fixture_id} references missing file {exp.file}"
        )
        line_count = len(target.read_text(encoding="utf-8").splitlines())
        assert 1 <= exp.line_start <= line_count, (
            f"{fixture.fixture_id}: line_start={exp.line_start} outside [1, {line_count}]"
        )
        assert exp.line_end <= line_count, (
            f"{fixture.fixture_id}: line_end={exp.line_end} exceeds file length {line_count}"
        )


@pytest.mark.parametrize("fixture_dir", sorted(
    p for p in FIXTURES_ROOT.iterdir()
    if p.is_dir() and not p.name.startswith(("_", "."))
))
def test_smoke_signatures_are_valid(fixture_dir: Path) -> None:
    """Every expected signature must be a known ClaimSignature."""
    fixture = load_fixture(fixture_dir)
    valid = {c.value for c in ClaimSignature}
    for exp in fixture.expected_findings:
        assert exp.signature in valid, (
            f"{fixture.fixture_id}: unknown signature {exp.signature!r}; "
            f"add it to ClaimSignature first"
        )


def test_smoke_fixture_ids_unique() -> None:
    fixtures = load_fixture_set(FIXTURES_ROOT)
    ids = [f.fixture_id for f in fixtures]
    assert len(ids) == len(set(ids)), f"duplicate fixture_ids in {ids}"


def test_smoke_fixture_set_balance() -> None:
    """Sanity: we have a mix of TP / FP-bait / red-herring fixtures.

    A pipeline can score 100% by finding every TP — but the FP-bait
    fixtures are what prevent gaming the score. If the set ever loses
    all FP-bait fixtures, the whole benchmark becomes worthless.
    """
    fixtures = load_fixture_set(FIXTURES_ROOT)
    by_kind = {}
    for f in fixtures:
        by_kind.setdefault(f.kind, []).append(f.fixture_id)
    assert by_kind.get("true_positive"), "no true_positive fixtures — score is meaningless"
    assert by_kind.get("fp_bait"), "no fp_bait fixtures — FPs aren't being measured"
    # red_herring is optional but should be present
    assert sum(len(v) for v in by_kind.values()) == len(fixtures)


# ---------------------------------------------------------------------------
# Tier 2 — scorer logic (synthetic findings, no LLM calls)
# ---------------------------------------------------------------------------


def _mk_finding(
    file: str,
    claim_signature: str,
    *,
    severity: str = Severity.HIGH.value,
    status: str = FindingStatus.CONFIRMED.value,
    repro_path: str = "/tmp/repro",
) -> Finding:
    return Finding(
        id=f"fnd_{file}_{claim_signature}",
        file=file,
        function="demo",
        claim=f"{claim_signature} in {file}",
        claim_signature=claim_signature,
        severity=severity,
        evidence_paths=(file,),
        status=status,
        runs_to_confirm=3,
        total_runs=3,
        repro_path=repro_path,
    )


def test_scorer_strong_match_against_true_positive() -> None:
    fx = load_fixture(FIXTURES_ROOT / "sql-injection-fstring")
    obs = [_mk_finding("bug.py", ClaimSignature.INJECTION.value)]
    score = score_fixture(fx, obs)
    assert score.matched_strong == 1
    assert score.matched_weak == 0
    assert not score.unmatched_expected
    assert score.passed


def test_scorer_severity_too_low_is_weak_match() -> None:
    fx = load_fixture(FIXTURES_ROOT / "sql-injection-fstring")
    # Expected severity high; observed low — alignment but failing gate.
    obs = [_mk_finding("bug.py", ClaimSignature.INJECTION.value, severity=Severity.LOW.value)]
    score = score_fixture(fx, obs)
    assert score.matched_strong == 0
    assert score.matched_weak == 1
    assert not score.passed


def test_scorer_speculative_status_is_weak_match() -> None:
    fx = load_fixture(FIXTURES_ROOT / "sql-injection-fstring")
    obs = [_mk_finding(
        "bug.py",
        ClaimSignature.INJECTION.value,
        status=FindingStatus.SPECULATIVE.value,
    )]
    score = score_fixture(fx, obs)
    assert score.matched_strong == 0
    assert score.matched_weak == 1


def test_scorer_missing_poc_when_required_is_weak_match() -> None:
    fx = load_fixture(FIXTURES_ROOT / "sql-injection-fstring")
    obs = [_mk_finding("bug.py", ClaimSignature.INJECTION.value, repro_path="")]
    score = score_fixture(fx, obs)
    assert score.matched_strong == 0
    assert score.matched_weak == 1


def test_scorer_fp_bait_with_confirmed_finding_fails() -> None:
    fx = load_fixture(FIXTURES_ROOT / "fp-safe-parameterized-query")
    # Detector wrongly flags the safe query.
    obs = [_mk_finding("bug.py", ClaimSignature.INJECTION.value)]
    score = score_fixture(fx, obs)
    assert score.extra_confirmed_findings == 1
    assert not score.passed


def test_scorer_fp_bait_with_speculative_only_passes() -> None:
    """A finding that never reaches CONFIRMED is the verifier doing its job.

    For FP-bait, we want zero CONFIRMED — speculative findings that the
    verifier failed to PoC are acceptable (they're the system saying
    "I considered this but couldn't prove it")."""
    fx = load_fixture(FIXTURES_ROOT / "fp-safe-parameterized-query")
    obs = [_mk_finding(
        "bug.py", ClaimSignature.INJECTION.value,
        status=FindingStatus.SPECULATIVE.value, repro_path="",
    )]
    score = score_fixture(fx, obs)
    assert score.extra_confirmed_findings == 0
    assert score.passed


def test_scorer_red_herring_with_confirmed_fails() -> None:
    fx = load_fixture(FIXTURES_ROOT / "red-herring-dead-code")
    obs = [_mk_finding("bug.py", ClaimSignature.INJECTION.value)]
    score = score_fixture(fx, obs)
    assert not score.passed


def test_scorer_compatible_signatures_align() -> None:
    """Path traversal often comes back as `missing_validation` rather than
    `injection` — both should align against the same expected entry."""
    fx = load_fixture(FIXTURES_ROOT / "path-traversal-user-input")
    obs = [_mk_finding("bug.py", ClaimSignature.MISSING_VALIDATION.value)]
    score = score_fixture(fx, obs)
    assert score.matched_strong == 1
    assert score.passed


def test_scorer_unrelated_finding_is_extra() -> None:
    fx = load_fixture(FIXTURES_ROOT / "sql-injection-fstring")
    expected_obs = _mk_finding("bug.py", ClaimSignature.INJECTION.value)
    extra_obs = _mk_finding("bug.py", ClaimSignature.RESOURCE_LEAK.value)
    score = score_fixture(fx, [expected_obs, extra_obs])
    assert score.matched_strong == 1
    assert score.extra_confirmed_findings == 1
    # max_extra_findings=1 in this fixture's expected.json, so this
    # still passes.
    assert score.passed


def test_scorer_aggregate_perfect_run() -> None:
    """An imaginary 100% pipeline: every TP matched strongly, every
    FP-bait / red-herring clean."""
    fixtures = load_fixture_set(FIXTURES_ROOT)
    scores: list[FixtureScore] = []
    for fx in fixtures:
        if fx.kind == "true_positive":
            obs = [
                _mk_finding(exp.file, exp.signature, severity=exp.min_severity,
                            status=exp.min_status,
                            repro_path="/tmp/repro" if exp.must_build_poc else "")
                for exp in fx.expected_findings
            ]
        else:
            obs = []  # clean run
        scores.append(score_fixture(fx, obs))
    report = EvalReport(fixtures=scores)
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.fp_bait_survival == 1.0
    assert report.aggregate_score >= 99.0


def test_scorer_aggregate_does_nothing_run() -> None:
    """An imaginary pipeline that finds nothing — recall=0, but FP-bait
    survival=100% because no FPs either. Precision is 0 (no TPs). Score
    should still be non-zero (FP-bait survival weight) but well below
    a real pipeline."""
    fixtures = load_fixture_set(FIXTURES_ROOT)
    scores = [score_fixture(fx, []) for fx in fixtures]
    report = EvalReport(fixtures=scores)
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.fp_bait_survival == 1.0
    # Default weights: 0.30 (fp_bait) + 0.05 (poc trivially 1.0) = 0.35.
    # Anything claiming "great pipeline" should beat this by a lot.
    assert 30.0 <= report.aggregate_score <= 40.0


def test_scorer_aggregate_finds_everything_plus_fps() -> None:
    """A noisy pipeline: finds every TP but also confirms every FP-bait.
    Should rank below the do-nothing pipeline because precision dominates."""
    fixtures = load_fixture_set(FIXTURES_ROOT)
    scores: list[FixtureScore] = []
    for fx in fixtures:
        if fx.kind == "true_positive":
            obs = [
                _mk_finding(exp.file, exp.signature, severity=exp.min_severity,
                            status=exp.min_status,
                            repro_path="/tmp/repro" if exp.must_build_poc else "")
                for exp in fx.expected_findings
            ]
        else:
            # Wrongly confirm something
            obs = [_mk_finding("bug.py", ClaimSignature.INJECTION.value)]
        scores.append(score_fixture(fx, obs))
    report = EvalReport(fixtures=scores)
    assert report.recall == 1.0
    assert report.fp_bait_survival == 0.0
    # Precision = 8 (TPs) / (8 + 4 FPs) = ~0.667
    assert 0.6 < report.precision < 0.75
    # Aggregate score should be meaningfully lower than the perfect run
    # because FP-bait survival dropped to 0.
    assert report.aggregate_score < 75.0


# ---------------------------------------------------------------------------
# Tier 3 — live orchestrator (--run-live)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_full_eval_placeholder() -> None:
    """Placeholder for the live eval path.

    Real wiring requires:
      - Docker (ContainerManager)
      - A model with API access (resolved via provider registry)
      - The runner from augmentum.bug_finder.eval_runner

    The runner is sketched in eval_runner.py but kept out of CI here
    because each run costs real tokens and ~5-15 minutes. Use:

        python -m augmentum.bug_finder.eval_runner --model claude-opus-4-7

    to drive it directly. The runner uses the same `score_fixture` /
    `EvalReport` machinery as this test, so any improvement to the
    Tier 2 logic flows through automatically.
    """
    pytest.skip("live eval — run via `python -m augmentum.bug_finder.eval_runner`")
