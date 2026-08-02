"""Contract tests for the agnostic_stage substrate + orchestrator wiring.

We can't drive the full ``run_bug_finder`` pipeline from a unit test —
it requires Docker containers, model backends, and live workspaces.
What we CAN pin is the contract the orchestrator depends on:

  * ``run_agnostic_stage`` returns ``Finding`` rows that match the
    LLM-detector shape closely enough that ``merge_runs`` /
    ``rank_findings`` won't choke on them.
  * Pattern memory is written to ``.augmentum/bug_finder/`` so a future
    run's ``familiarity score`` lookup finds the prior hits.
  * ``record_confirmation`` is callable on the seeded Finding objects
    without crashing — the orchestrator's post-verify hook depends on
    that staying true.
  * The orchestrator imports the agnostic_stage symbols it expects.

Bandit / Ruff binaries may not be installed in this environment, so the
test monkeypatches ``run_generic_suite_timed`` to inject controlled
scanner output — we're testing the wiring, not the underlying scanners.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from augmentum.bug_finder import agnostic_stage
from augmentum.bug_finder.agnostic_stage import (
    AgnosticStageResult,
    record_confirmation,
    run_agnostic_stage,
)
from augmentum.bug_finder.dev_tools import ScannerFinding
from augmentum.bug_finder.findings import Finding, FindingStatus
from augmentum.bug_finder.generic_scanners import GenericScannerSuiteResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_workspace(tmp_path: Path) -> Path:
    """A minimal real directory the agnostic_stage can scan against.

    Doesn't need actual buggy code — we're injecting scanner output via
    monkeypatch — but the path has to exist on disk so the
    ``.augmentum/`` substrate can be initialized inside it.
    """
    (tmp_path / "app.py").write_text(
        "def f():\n    eval('1+1')\n",
        encoding="utf-8",
    )
    return tmp_path


def _patch_scanners(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[ScannerFinding],
) -> None:
    """Override the generic-suite timed runner with a controlled stub."""
    grouped: dict[str, list[ScannerFinding]] = {}
    for sf in rows:
        grouped.setdefault(sf.scanner, []).append(sf)

    def _stub(_root: Path) -> GenericScannerSuiteResult:
        return GenericScannerSuiteResult(
            findings_by_scanner=grouped,
            wallclock_seconds=0.01,
        )

    monkeypatch.setattr(
        agnostic_stage, "run_generic_suite_timed", _stub,
    )

    # Custom checks pull from .augmentum/bug_finder/checks/ — stub
    # empty so the test doesn't depend on per-workspace check files.
    monkeypatch.setattr(
        agnostic_stage, "run_custom_checks", lambda _root: [],
    )


# ---------------------------------------------------------------------------
# run_agnostic_stage contract
# ---------------------------------------------------------------------------


def test_agnostic_stage_returns_finding_shaped_rows(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The output must look like ``Finding`` rows the orchestrator can
    pour into the same pipeline as LLM detector output."""
    _patch_scanners(monkeypatch, [
        ScannerFinding(
            scanner="bandit", severity="high", category="B307",
            file="app.py", line=2,
            message="Use of possibly insecure function: eval.",
        ),
    ])

    result = run_agnostic_stage(fake_workspace)

    assert isinstance(result, AgnosticStageResult)
    assert len(result.seeded_findings) == 1
    finding = result.seeded_findings[0]
    assert isinstance(finding, Finding)
    assert finding.file == "app.py"
    assert finding.severity == "high"
    # Scanner findings get a placeholder function so they don't collide
    # with LLM findings' real handler names.
    assert finding.function.startswith("<") and finding.function.endswith(">")
    # Source marker on notes is the signal the orchestrator's post-verify
    # hook will key off when deciding which findings to bump.
    assert any("source: scanner" in n for n in finding.notes)
    # Scanners self-confirm — Finding starts with one confirmation up-front
    # to differentiate from an LLM SPECULATIVE finding awaiting verify.
    assert finding.total_runs == 1


def test_agnostic_stage_filters_low_severity_from_pipeline(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``info`` / ``low`` scanner findings get counted but don't seed the
    LLM pipeline — they'd just burn verifier budget on style nits."""
    _patch_scanners(monkeypatch, [
        ScannerFinding(
            scanner="ruff", severity="info", category="F401",
            file="app.py", line=1, message="unused import",
        ),
        ScannerFinding(
            scanner="bandit", severity="medium", category="B608",
            file="app.py", line=2, message="possible SQL injection",
        ),
    ])

    result = run_agnostic_stage(fake_workspace)

    assert len(result.seeded_findings) == 1
    assert result.seeded_findings[0].claim_signature  # populated
    assert result.scanner_counts.get("ruff", 0) == 1
    assert result.scanner_counts.get("bandit", 0) == 1


def test_agnostic_stage_writes_pattern_memory(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``patterns.json`` is the per-workspace memory the lead's
    ``signature_familiarity_score`` lookup reads on subsequent runs.
    A first-time scanner hit must show up there."""
    _patch_scanners(monkeypatch, [
        ScannerFinding(
            scanner="bandit", severity="high", category="B307",
            file="app.py", line=2, message="eval is dangerous",
        ),
    ])

    run_agnostic_stage(fake_workspace)

    patterns_path = (
        fake_workspace / ".augmentum" / "bug_finder" / "patterns.json"
    )
    assert patterns_path.is_file()
    body = json.loads(patterns_path.read_text(encoding="utf-8"))
    # Schema is a list of pattern rows; one of them should match our
    # injected hit by signature.
    assert isinstance(body, list)
    sigs = {row.get("signature", "") for row in body}
    assert "bandit:B307" in sigs


def test_record_confirmation_is_safe_on_seeded_findings(
    fake_workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator's post-verify hook calls ``record_confirmation``
    on every scanner-sourced finding that survived verify. This must
    never raise — exceptions here would crash the run for a
    best-effort bookkeeping call."""
    _patch_scanners(monkeypatch, [
        ScannerFinding(
            scanner="bandit", severity="high", category="B307",
            file="app.py", line=2, message="eval",
        ),
    ])
    result = run_agnostic_stage(fake_workspace)
    finding = result.seeded_findings[0]
    finding.status = FindingStatus.CONFIRMED.value

    # Should not raise even though .augmentum/ already exists.
    record_confirmation(fake_workspace, finding)
    # Pattern memory should now show a non-zero fix_count for this sig.
    patterns_path = (
        fake_workspace / ".augmentum" / "bug_finder" / "patterns.json"
    )
    rows = json.loads(patterns_path.read_text(encoding="utf-8"))
    matched = [r for r in rows if r.get("signature") == "bandit:B307"]
    assert matched, "scanner signature missing from patterns after confirm"
    assert matched[0].get("fix_count", 0) >= 1


def test_agnostic_stage_handles_missing_workspace_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid workspace root yields an empty result instead of
    raising. The orchestrator wraps the call in try/except anyway, but
    the function itself shouldn't be the throwing layer for the easy
    cases."""
    bogus = tmp_path / "does_not_exist"
    result = run_agnostic_stage(bogus)
    assert result.seeded_findings == []
    assert result.total_raw == 0


# ---------------------------------------------------------------------------
# Orchestrator wiring contract
# ---------------------------------------------------------------------------


def test_orchestrator_imports_agnostic_symbols() -> None:
    """Pin the imports the orchestrator added. A future cleanup that
    deletes one of these would silently disable the substrate stage —
    the run still works but every bug_finder run loses Bandit/Ruff
    coverage."""
    from augmentum.bug_finder import orchestrator
    assert hasattr(orchestrator, "run_agnostic_stage")
    assert hasattr(orchestrator, "record_confirmation")
    assert hasattr(orchestrator, "AgnosticStageResult")


def test_orchestrator_source_references_substrate_stage() -> None:
    """The integration is a load-bearing block of code in ``run_bug_finder``.
    A regression that deletes the block (e.g. during a merge) would
    silently strip the substrate — find it by a stable anchor."""
    from augmentum.bug_finder import orchestrator
    import inspect
    src = inspect.getsource(orchestrator)
    assert "Stage 2.25: agnostic substrate" in src
    assert "run_agnostic_stage" in src
    assert "record_confirmation" in src


def test_orchestrator_loads_workspace_priors() -> None:
    """The compounding loop closes when patterns.json is READ at the
    start of a run and fed into the planner + lead. This pins the
    presence of the load + the two consumer sites — without these,
    patterns.json reverts to a write-only graveyard."""
    from augmentum.bug_finder import orchestrator
    import inspect
    src = inspect.getsource(orchestrator)
    assert "load_workspace_patterns" in src
    assert "render_pattern_priors" in src
    assert "workspace_priors_brief" in src
    # Planner must receive the brief
    assert "workspace_priors_brief=workspace_priors_brief" in src


def test_planner_signature_accepts_workspace_priors_brief() -> None:
    """A future refactor that drops the parameter from ``_run_planner``
    would silently disable workspace-pattern compounding for the
    planner. Lock the signature."""
    from augmentum.bug_finder import orchestrator
    import inspect
    sig = inspect.signature(orchestrator._run_planner)
    assert "workspace_priors_brief" in sig.parameters
