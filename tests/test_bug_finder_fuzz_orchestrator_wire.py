"""Orchestrator-side tests for the fuzz leg.

We unit-test the deterministic pieces that don't need a Docker container:

  * ``_merge_cross_modal`` — combining LLM + fuzz findings, bumping
    families_to_confirm on co-located hits, appending fuzz-only ones.
  * Verifier short-circuit — ``_run_verify_is_real`` should leave
    ``status=CONFIRMED`` findings untouched (the fuzz crash IS the
    PoC; no point asking the verifier to rebuild one).

Full end-to-end requires a real container and is exercised separately
via the smoke script.
"""

from __future__ import annotations

import pytest

from augmentum.bug_finder.findings import (
    ClaimSignature,
    Finding,
    FindingStatus,
    Severity,
)
from augmentum.bug_finder.orchestrator import _merge_cross_modal


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _llm_finding(
    *,
    file: str = "x.py",
    function: str = "parse",
    families: int = 1,
    total_families: int = 1,
) -> Finding:
    return Finding(
        id=f"fnd_llm_{file}_{function}",
        file=file,
        function=function,
        claim="LLM-detected speculative bug",
        claim_signature=ClaimSignature.MISSING_VALIDATION.value,
        severity=Severity.MEDIUM.value,
        evidence_paths=(f"{file}:5",),
        status=FindingStatus.SPECULATIVE.value,
        runs_to_confirm=2,
        total_runs=3,
        families_to_confirm=families,
        total_families=total_families,
    )


def _fuzz_finding(
    *,
    file: str = "x.py",
    function: str = "parse",
    claim: str = "Fuzzer triggered AttributeError: NoneType",
) -> Finding:
    return Finding(
        id=f"fnd_fuzz_{file}_{function}",
        file=file,
        function=function,
        claim=claim,
        claim_signature=ClaimSignature.NULL_DEREF.value,
        severity=Severity.MEDIUM.value,
        evidence_paths=(f"{file}:parse",),
        status=FindingStatus.CONFIRMED.value,
        runs_to_confirm=1,
        total_runs=1,
        families_to_confirm=1,
        total_families=1,
        repro_path="/workspace/.augmentum/fuzz/x/artifacts/crash-aabb",
        repro_command="python3 fuzz_parse.py crash-aabb",
        repro_output="AttributeError: 'NoneType' object has no attribute 'split'",
    )


# ---------------------------------------------------------------------------
# _merge_cross_modal
# ---------------------------------------------------------------------------


def test_merge_no_fuzz_findings_returns_llm_unchanged() -> None:
    llm = [_llm_finding()]
    merged = _merge_cross_modal(llm, [])
    assert merged == llm


def test_merge_no_llm_findings_returns_fuzz_as_is() -> None:
    fuzz = [_fuzz_finding()]
    merged = _merge_cross_modal([], fuzz)
    assert merged == fuzz


def test_merge_bumps_families_when_sites_match() -> None:
    """LLM + fuzz at the same (file, function): the LLM finding wins
    as canonical, with families_to_confirm bumped to reflect the
    cross-modal evidence."""
    llm = _llm_finding(families=1, total_families=1)
    fuzz = _fuzz_finding()
    merged = _merge_cross_modal([llm], [fuzz])

    assert len(merged) == 1
    out = merged[0]
    # LLM identity preserved
    assert out.id == llm.id
    assert out.claim == llm.claim
    # Family bump reflects cross-modal agreement
    assert out.families_to_confirm == 2
    assert out.total_families >= 2
    # Fuzz crash details flow in as a note + repro hints
    assert any("Cross-modal confirmation" in n for n in out.notes)
    assert out.repro_path.startswith("/workspace/.augmentum/fuzz")


def test_merge_fuzz_only_finding_appended_standalone() -> None:
    """A fuzz crash at a site no LLM finding mentioned still surfaces."""
    llm = _llm_finding(file="a.py", function="foo")
    fuzz = _fuzz_finding(file="b.py", function="bar")
    merged = _merge_cross_modal([llm], [fuzz])

    assert len(merged) == 2
    ids = {f.id for f in merged}
    assert llm.id in ids
    assert fuzz.id in ids


def test_merge_preserves_max_severity() -> None:
    """If the fuzz finding has higher severity than the LLM one, the
    merged finding takes the higher value — losing severity on merge
    would be a regression."""
    llm = _llm_finding()  # medium
    fuzz = _fuzz_finding()
    object.__setattr__(fuzz, "severity", Severity.HIGH.value)
    merged = _merge_cross_modal([llm], [fuzz])
    assert merged[0].severity == Severity.HIGH.value


def test_merge_only_first_fuzz_per_site_wins() -> None:
    """If two fuzz findings somehow target the same site, only the
    first matches the LLM finding. Belt-and-suspenders against
    accidental double-merge."""
    llm = _llm_finding()
    fuzz1 = _fuzz_finding(claim="first crash")
    fuzz2 = _fuzz_finding(claim="second crash")
    object.__setattr__(fuzz2, "id", "fnd_fuzz_2")
    merged = _merge_cross_modal([llm], [fuzz1, fuzz2])
    # 1 from LLM (enriched) + 0 fuzz appended (both matched same site)
    # The implementation tracks matched_sites set so once a site is
    # claimed, neither fuzz finding is appended standalone.
    assert len(merged) == 1
    assert merged[0].families_to_confirm == 2
    # The first cross-modal note is "first crash"
    assert any("first crash" in n for n in merged[0].notes)


# ---------------------------------------------------------------------------
# Verifier short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_skips_already_confirmed_findings() -> None:
    """Verifier shouldn't waste a subagent run on a finding that's
    already CONFIRMED (e.g. a fuzz crash carrying its own PoC).

    We construct a CONFIRMED finding and one SPECULATIVE finding, mock
    the resolver to fail loudly if asked, and verify only the
    SPECULATIVE one would have triggered backend resolution. Concretely:
    a CONFIRMED finding alone never calls resolve_backend.
    """
    from augmentum.bug_finder.orchestrator import _run_verify_is_real

    async def resolve_backend(_model: str):
        raise AssertionError(
            "verifier asked for a backend — should not happen with "
            "CONFIRMED-only findings",
        )

    confirmed = _fuzz_finding()

    # Stub objects sufficient to satisfy the type signature without
    # actually executing any path.
    class _StubWorkspace:
        workspace_id = "ws_test"

    class _StubCM:
        async def run_command(self, *a, **kw):
            return ""

    class _StubConfig:
        verifier_budget = None
        role_models = type("RM", (), {"verifier": "anthropic:claude-x"})()
        intake = type("In", (), {"threat_model": ""})()

    out = await _run_verify_is_real(
        [confirmed],
        config=_StubConfig(),  # type: ignore[arg-type]
        cm=_StubCM(),  # type: ignore[arg-type]
        workspace=_StubWorkspace(),  # type: ignore[arg-type]
        resolve_backend=resolve_backend,
        user_id="",
        ledger=[],
    )
    assert len(out) == 1
    assert out[0].id == confirmed.id
    assert out[0].status == FindingStatus.CONFIRMED.value
