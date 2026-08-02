"""Bug Finder normalized-findings store tests.

Covers the migration-225 projection from BugFinderRunReport → per-finding
rows: idempotency, user scoping, signature-based queries, recurrence
aggregation, and the backfill path that re-derives rows from blobs.

Integration-level test (real aiosqlite, real schema). Uses
SQLiteBackend so all the connection-level safety wiring
(install_safe_rollback etc.) is in place. The fixtures live entirely in
the test; no fixture-repo dependencies — that's a different harness.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from augmentum.bug_finder.findings import (
    ClaimSignature,
    Finding,
    FindingStatus,
    Severity,
)
from augmentum.bug_finder.orchestrator import BugFinderRunReport
from augmentum.bug_finder.store import BugFinderRunStore
from augmentum.bug_finder.workspace import WorkspaceBaseline
from augmentum.state.backends.sqlite import SQLiteBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> Any:
    """Fresh in-memory DB with full schema applied + store wired up.

    SQLiteBackend.connect() runs all migrations automatically."""
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    yield BugFinderRunStore(backend.conn)
    await backend.close()


def _mk_report(
    *,
    run_id: str = "bfr_test",
    workspace_id: str = "ws_demo",
    findings: list[Finding] | None = None,
    started_at: float | None = None,
    completed_at: float | None = None,
    stop_reason: str = "complete",
) -> BugFinderRunReport:
    now = time.time()
    return BugFinderRunReport(
        run_id=run_id,
        started_at=started_at or now,
        completed_at=completed_at or now + 60,
        intake={
            "workspace_id": workspace_id,
            "focus_paths": [],
            "threat_model": "",
        },
        workspace_id=workspace_id,
        baseline=WorkspaceBaseline(),
        findings=findings or [],
        confirmation_hist={},
        cost_ledger=[],
        stop_reason=stop_reason,
        stop_detail="",
        same_model_self_verification=True,
        notes=[],
    )


def _mk_finding(
    fid: str,
    *,
    file: str = "app.py",
    function: str = "handler",
    signature: str = ClaimSignature.INJECTION.value,
    severity: str = Severity.HIGH.value,
    status: str = FindingStatus.CONFIRMED.value,
    evidence: tuple[str, ...] = ("app.py:12",),
    repro_path: str = "/repros/poc.py",
    patch: str = "",
    runs_to_confirm: int = 3,
    total_runs: int = 3,
) -> Finding:
    return Finding(
        id=fid,
        file=file,
        function=function,
        claim=f"{signature} in {file}",
        claim_signature=signature,
        severity=severity,
        evidence_paths=evidence,
        suggested_repro="run app.py with crafted input",
        status=status,
        runs_to_confirm=runs_to_confirm,
        total_runs=total_runs,
        repro_path=repro_path,
        patch=patch,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_run_writes_normalized_findings(store: BugFinderRunStore) -> None:
    report = _mk_report(findings=[
        _mk_finding("fnd_a", file="auth.py", evidence=("auth.py:23-30",)),
        _mk_finding("fnd_b", file="upload.py",
                    signature=ClaimSignature.MISSING_VALIDATION.value,
                    severity=Severity.MEDIUM.value),
    ])
    await store.complete_run(report, user_id="user_alice")

    rows = await store.workspace_finding_history(
        "ws_demo", user_id="user_alice",
    )
    assert len(rows) == 2
    by_id = {r["finding_id"]: r for r in rows}
    assert by_id["fnd_a"]["file"] == "auth.py"
    assert by_id["fnd_a"]["line_start"] == 23
    assert by_id["fnd_a"]["line_end"] == 30
    assert by_id["fnd_b"]["claim_signature"] == "missing_validation"


@pytest.mark.asyncio
async def test_line_range_parses_single_line(store: BugFinderRunStore) -> None:
    report = _mk_report(findings=[
        _mk_finding("fnd_single", evidence=("app.py:42",)),
    ])
    await store.complete_run(report, user_id="user_alice")
    rows = await store.workspace_finding_history("ws_demo", user_id="user_alice")
    assert rows[0]["line_start"] == 42
    assert rows[0]["line_end"] == 42


@pytest.mark.asyncio
async def test_line_range_absent_when_evidence_has_no_lines(store: BugFinderRunStore) -> None:
    report = _mk_report(findings=[
        _mk_finding("fnd_no_lines", evidence=("app.py",)),
    ])
    await store.complete_run(report, user_id="user_alice")
    rows = await store.workspace_finding_history("ws_demo", user_id="user_alice")
    assert rows[0]["line_start"] is None
    assert rows[0]["line_end"] is None


@pytest.mark.asyncio
async def test_complete_run_is_idempotent(store: BugFinderRunStore) -> None:
    """Re-completing the same run replaces the rows, doesn't accumulate."""
    report = _mk_report(findings=[_mk_finding("fnd_a"), _mk_finding("fnd_b")])
    await store.complete_run(report, user_id="user_alice")
    await store.complete_run(report, user_id="user_alice")  # second call
    rows = await store.workspace_finding_history("ws_demo", user_id="user_alice")
    assert len(rows) == 2  # not 4


@pytest.mark.asyncio
async def test_complete_run_drops_findings_that_disappear(store: BugFinderRunStore) -> None:
    """If a re-run produces fewer findings (rare but possible — different
    runs_to_confirm, lower variance), the dropped ones must vanish from
    the normalized table too."""
    report = _mk_report(findings=[_mk_finding("fnd_a"), _mk_finding("fnd_b")])
    await store.complete_run(report, user_id="user_alice")
    # Second run finds only one
    report2 = _mk_report(findings=[_mk_finding("fnd_a")])
    await store.complete_run(report2, user_id="user_alice")
    rows = await store.workspace_finding_history("ws_demo", user_id="user_alice")
    assert {r["finding_id"] for r in rows} == {"fnd_a"}


@pytest.mark.asyncio
async def test_user_scoping_prevents_cross_tenant_read(store: BugFinderRunStore) -> None:
    """User-scope invariant: alice's findings never visible to bob."""
    report_alice = _mk_report(
        run_id="bfr_alice", workspace_id="ws_alice",
        findings=[_mk_finding("fnd_a", file="alice.py")],
    )
    report_bob = _mk_report(
        run_id="bfr_bob", workspace_id="ws_bob",
        findings=[_mk_finding("fnd_b", file="bob.py")],
    )
    await store.complete_run(report_alice, user_id="user_alice")
    await store.complete_run(report_bob, user_id="user_bob")

    alice_findings = await store.list_findings_by_signature(
        ClaimSignature.INJECTION.value, user_id="user_alice",
    )
    bob_findings = await store.list_findings_by_signature(
        ClaimSignature.INJECTION.value, user_id="user_bob",
    )
    assert {f["finding_id"] for f in alice_findings} == {"fnd_a"}
    assert {f["finding_id"] for f in bob_findings} == {"fnd_b"}
    # Cross-tenant: alice asks for bob's workspace — should be empty
    cross = await store.workspace_finding_history("ws_bob", user_id="user_alice")
    assert cross == []


@pytest.mark.asyncio
async def test_list_findings_by_signature_filters(store: BugFinderRunStore) -> None:
    report = _mk_report(findings=[
        _mk_finding("fnd_a", signature=ClaimSignature.INJECTION.value),
        _mk_finding("fnd_b", signature=ClaimSignature.RACE.value),
        _mk_finding("fnd_c",
                    signature=ClaimSignature.INJECTION.value,
                    status=FindingStatus.SPECULATIVE.value),
    ])
    await store.complete_run(report, user_id="user_alice")

    injections = await store.list_findings_by_signature(
        ClaimSignature.INJECTION.value, user_id="user_alice",
    )
    assert {f["finding_id"] for f in injections} == {"fnd_a", "fnd_c"}

    confirmed_only = await store.list_findings_by_signature(
        ClaimSignature.INJECTION.value, user_id="user_alice",
        status=FindingStatus.CONFIRMED.value,
    )
    assert {f["finding_id"] for f in confirmed_only} == {"fnd_a"}


@pytest.mark.asyncio
async def test_signature_recurrence_aggregates(store: BugFinderRunStore) -> None:
    """Same signature+file across two separate runs → hit_count=2."""
    report1 = _mk_report(
        run_id="bfr_1",
        findings=[
            _mk_finding("fnd_a", file="risky.py",
                        signature=ClaimSignature.INJECTION.value),
            _mk_finding("fnd_b", file="safe.py",
                        signature=ClaimSignature.NULL_DEREF.value),
        ],
    )
    report2 = _mk_report(
        run_id="bfr_2",
        findings=[
            _mk_finding("fnd_a2", file="risky.py",
                        signature=ClaimSignature.INJECTION.value),
        ],
    )
    await store.complete_run(report1, user_id="user_alice")
    await store.complete_run(report2, user_id="user_alice")

    recurrences = await store.signature_recurrence(
        user_id="user_alice", min_hits=2,
    )
    assert len(recurrences) == 1
    row = recurrences[0]
    assert row["claim_signature"] == "injection"
    assert row["file"] == "risky.py"
    assert row["hit_count"] == 2


@pytest.mark.asyncio
async def test_backfill_from_runs(store: BugFinderRunStore) -> None:
    """If the normalized table got wiped (e.g., from a manual recovery),
    backfill_findings_from_runs should rebuild it from the blob source."""
    report = _mk_report(findings=[
        _mk_finding("fnd_a", file="x.py"),
        _mk_finding("fnd_b", file="y.py"),
    ])
    await store.complete_run(report, user_id="user_alice")

    # Wipe the normalized table to simulate a recovery scenario
    async with store._conn.execute("DELETE FROM bug_finder_findings"):
        pass
    await store._conn.commit()
    rows = await store.workspace_finding_history("ws_demo", user_id="user_alice")
    assert rows == []

    # Backfill
    count = await store.backfill_findings_from_runs(user_id="user_alice")
    assert count == 1  # one run backfilled
    rows_after = await store.workspace_finding_history("ws_demo", user_id="user_alice")
    assert {r["finding_id"] for r in rows_after} == {"fnd_a", "fnd_b"}


@pytest.mark.asyncio
async def test_no_findings_run_writes_zero_rows(store: BugFinderRunStore) -> None:
    """A clean run (no findings) is the happy path; should write zero rows
    without raising."""
    report = _mk_report(findings=[])
    await store.complete_run(report, user_id="user_alice")
    rows = await store.workspace_finding_history("ws_demo", user_id="user_alice")
    assert rows == []
