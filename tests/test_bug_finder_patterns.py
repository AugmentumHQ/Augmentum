"""Cross-run pattern memory tests.

Covers PatternStore (write + read) plus the planner-brief renderer.
Integration-level — real schema, real :memory: DB.
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
from augmentum.bug_finder.patterns import (
    PatternStore,
    pattern_to_dict,
    render_pattern_brief,
)
from augmentum.state.backends.sqlite import SQLiteBackend


@pytest.fixture
async def store() -> Any:
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    yield PatternStore(backend.conn)
    await backend.close()


def _mk_finding(
    *,
    fid: str = "fnd_demo",
    file: str = "app.py",
    signature: str = ClaimSignature.INJECTION.value,
    severity: str = Severity.HIGH.value,
    status: str = FindingStatus.CONFIRMED.value,
    claim: str = "SQL injection via f-string interpolation",
) -> Finding:
    return Finding(
        id=fid,
        file=file,
        function="handler",
        claim=claim,
        claim_signature=signature,
        severity=severity,
        evidence_paths=(f"{file}:14",),
        status=status,
        runs_to_confirm=3,
        total_runs=3,
        repro_path="/repros/poc.py",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_run_creates_pattern(store: PatternStore) -> None:
    findings = [_mk_finding()]
    affected = await store.update_from_findings(
        findings, run_id="bfr_1", user_id="user_a", workspace_id="ws_x",
    )
    assert affected == 1
    patterns = await store.list_patterns(user_id="user_a", workspace_id="ws_x")
    assert len(patterns) == 1
    p = patterns[0]
    assert p.hit_count == 1
    assert p.fix_count == 0
    assert p.speculative_count == 0
    assert p.claim_signature == "injection"
    assert p.file == "app.py"
    assert p.last_severity == "high"
    assert p.sample_claim.startswith("SQL injection")


@pytest.mark.asyncio
async def test_repeated_runs_increment_hit_count(store: PatternStore) -> None:
    """Same finding across two runs → hit_count=2, sample_claim stable."""
    f1 = _mk_finding(claim="SQL injection via f-string interpolation")
    await store.update_from_findings(
        [f1], run_id="bfr_1", user_id="user_a", workspace_id="ws_x",
    )
    # Second run — same key, slightly different claim text. The sample
    # should stick with the first one (stable text helps the planner).
    f2 = _mk_finding(claim="String-formatted SQL — same bug, different wording")
    await store.update_from_findings(
        [f2], run_id="bfr_2", user_id="user_a", workspace_id="ws_x",
    )
    patterns = await store.list_patterns(user_id="user_a", workspace_id="ws_x")
    assert len(patterns) == 1
    assert patterns[0].hit_count == 2
    assert patterns[0].sample_claim.startswith("SQL injection")
    assert patterns[0].last_run_id == "bfr_2"


@pytest.mark.asyncio
async def test_fix_increments_fix_count(store: PatternStore) -> None:
    """A finding that reaches FIXED bumps fix_count separately."""
    await store.update_from_findings(
        [_mk_finding(status=FindingStatus.CONFIRMED.value)],
        run_id="bfr_1", user_id="user_a", workspace_id="ws_x",
    )
    # Re-audit, this time the bug is fixed:
    await store.update_from_findings(
        [_mk_finding(status=FindingStatus.FIXED.value)],
        run_id="bfr_2", user_id="user_a", workspace_id="ws_x",
    )
    patterns = await store.list_patterns(user_id="user_a", workspace_id="ws_x")
    assert patterns[0].hit_count == 2
    assert patterns[0].fix_count == 1
    assert not patterns[0].is_unresolved


@pytest.mark.asyncio
async def test_unresolved_only_filter(store: PatternStore) -> None:
    """The "stuff that keeps coming back unfixed" filter."""
    # ws_x: confirmed, never fixed
    await store.update_from_findings(
        [_mk_finding(file="risky.py")],
        run_id="bfr_1", user_id="user_a", workspace_id="ws_x",
    )
    # ws_x: confirmed once, fixed once
    await store.update_from_findings(
        [_mk_finding(file="ok.py", fid="fnd_b")],
        run_id="bfr_2", user_id="user_a", workspace_id="ws_x",
    )
    await store.update_from_findings(
        [_mk_finding(file="ok.py", fid="fnd_b", status=FindingStatus.FIXED.value)],
        run_id="bfr_3", user_id="user_a", workspace_id="ws_x",
    )
    unresolved = await store.list_patterns(
        user_id="user_a", workspace_id="ws_x", unresolved_only=True,
    )
    assert len(unresolved) == 1
    assert unresolved[0].file == "risky.py"


@pytest.mark.asyncio
async def test_signature_filter(store: PatternStore) -> None:
    """Filter patterns by claim_signature."""
    await store.update_from_findings(
        [
            _mk_finding(file="a.py", signature=ClaimSignature.INJECTION.value),
            _mk_finding(file="b.py", fid="fnd_b",
                        signature=ClaimSignature.RACE.value),
            _mk_finding(file="c.py", fid="fnd_c",
                        signature=ClaimSignature.INJECTION.value),
        ],
        run_id="bfr_1", user_id="user_a", workspace_id="ws_x",
    )
    inj = await store.list_patterns(
        user_id="user_a", workspace_id="ws_x",
        signature=ClaimSignature.INJECTION.value,
    )
    assert {p.file for p in inj} == {"a.py", "c.py"}


@pytest.mark.asyncio
async def test_user_scoping_prevents_cross_tenant_read(store: PatternStore) -> None:
    await store.update_from_findings(
        [_mk_finding(file="alice.py")],
        run_id="bfr_a", user_id="user_alice", workspace_id="ws_alice",
    )
    await store.update_from_findings(
        [_mk_finding(file="bob.py", fid="fnd_b")],
        run_id="bfr_b", user_id="user_bob", workspace_id="ws_bob",
    )
    alice = await store.list_patterns(user_id="user_alice")
    bob = await store.list_patterns(user_id="user_bob")
    assert {p.file for p in alice} == {"alice.py"}
    assert {p.file for p in bob} == {"bob.py"}


@pytest.mark.asyncio
async def test_forget_pattern(store: PatternStore) -> None:
    """User-initiated forget removes the row entirely."""
    await store.update_from_findings(
        [_mk_finding()],
        run_id="bfr_1", user_id="user_a", workspace_id="ws_x",
    )
    patterns = await store.list_patterns(user_id="user_a")
    assert len(patterns) == 1
    pid = patterns[0].pattern_id

    deleted = await store.forget_pattern(pid, user_id="user_a")
    assert deleted is True
    assert await store.list_patterns(user_id="user_a") == []

    # Forget what no longer exists: returns False, doesn't raise.
    deleted_again = await store.forget_pattern(pid, user_id="user_a")
    assert deleted_again is False


@pytest.mark.asyncio
async def test_annotate_attaches_note(store: PatternStore) -> None:
    """User annotates a pattern they want to flag as intentional."""
    await store.update_from_findings(
        [_mk_finding(file="cache.py",
                     signature=ClaimSignature.OTHER.value,
                     claim="pickle.loads on internal cache")],
        run_id="bfr_1", user_id="user_a", workspace_id="ws_x",
    )
    pid = (await store.list_patterns(user_id="user_a"))[0].pattern_id
    updated = await store.annotate(
        pid, "intentional — internal trust boundary, not user-controlled",
        user_id="user_a",
    )
    assert updated is True
    [p] = await store.list_patterns(user_id="user_a")
    assert p.note.startswith("intentional")


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def test_render_pattern_brief_empty_returns_blank() -> None:
    assert render_pattern_brief([]) == ""


def test_render_pattern_brief_includes_signal_columns() -> None:
    """The brief must contain enough info for the planner to triage:
    file, signature, hit count, fix count, severity, and any user note."""
    from augmentum.bug_finder.patterns import Pattern
    ts = int(time.time())
    patterns = [
        Pattern(
            pattern_id="pat_a",
            user_id="user_a",
            workspace_id="ws_x",
            claim_signature="injection",
            file="auth.py",
            first_seen_at=ts - 86400,
            last_seen_at=ts,
            last_run_id="bfr_5",
            hit_count=4,
            fix_count=1,
            speculative_count=1,
            sample_claim="SQL injection",
            last_severity="high",
            note="watched closely after the May incident",
        ),
        Pattern(
            pattern_id="pat_b",
            user_id="user_a",
            workspace_id="ws_x",
            claim_signature="race",
            file="upload.py",
            first_seen_at=ts - 86400,
            last_seen_at=ts,
            last_run_id="bfr_5",
            hit_count=1,
            fix_count=0,
            speculative_count=0,
            sample_claim="TOCTOU between stat and open",
            last_severity="medium",
            note="",
        ),
    ]
    brief = render_pattern_brief(patterns)
    assert "auth.py" in brief
    assert "injection" in brief
    assert "watched closely" in brief
    assert "upload.py" in brief
    assert "race" in brief
    # The fallback to sample_claim when no user note:
    assert "TOCTOU between stat and open" in brief


def test_render_pattern_brief_truncates_long_lists() -> None:
    from augmentum.bug_finder.patterns import Pattern
    ts = int(time.time())
    patterns = [
        Pattern(
            pattern_id=f"pat_{i}",
            user_id="user_a", workspace_id="ws_x",
            claim_signature="injection",
            file=f"file_{i}.py",
            first_seen_at=ts, last_seen_at=ts, last_run_id="bfr_1",
            hit_count=1, fix_count=0, speculative_count=0,
            sample_claim="bug", last_severity="medium", note="",
        ) for i in range(20)
    ]
    brief = render_pattern_brief(patterns, max_lines=5)
    assert "and 15 more pattern(s) not shown" in brief
    # Files 0-4 shown, 5+ hidden
    assert "file_0.py" in brief
    assert "file_4.py" in brief
    assert "file_15.py" not in brief


# ---------------------------------------------------------------------------
# Orchestrator integration check (signature only — full integration is
# verified by the bug_finder_run handler tests further down the road)
# ---------------------------------------------------------------------------


def test_prefix_patterns_helper_prepends_brief() -> None:
    from augmentum.bug_finder.orchestrator import _prefix_patterns
    brief = "## Patterns observed in prior runs of this workspace\n\nfoo"
    sys_prompt = "You are a security planner.\n..."
    combined = _prefix_patterns(sys_prompt, brief)
    assert combined.startswith("## Patterns observed")
    assert "You are a security planner" in combined
    # Empty brief returns the original unchanged.
    assert _prefix_patterns(sys_prompt, "") == sys_prompt
    assert _prefix_patterns(sys_prompt, "   ") == sys_prompt


def test_pattern_to_dict_serializable() -> None:
    """Routes serialize Pattern instances to JSON — dict shape must work."""
    from augmentum.bug_finder.patterns import Pattern
    p = Pattern(
        pattern_id="pat_x", user_id="user_a", workspace_id="ws",
        claim_signature="injection", file="a.py",
        first_seen_at=1, last_seen_at=2, last_run_id="r1",
        hit_count=3, fix_count=1, speculative_count=0,
        sample_claim="claim", last_severity="high", note="n",
    )
    d = pattern_to_dict(p)
    assert d["pattern_id"] == "pat_x"
    assert d["hit_count"] == 3
    import json as _json
    # Roundtrip JSON — every value must be JSON-serializable
    assert _json.loads(_json.dumps(d)) == d
