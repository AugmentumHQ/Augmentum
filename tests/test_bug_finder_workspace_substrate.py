"""Tests for the per-workspace learning substrate."""

from __future__ import annotations

import json
import time
from pathlib import Path

from augmentum.bug_finder.workspace_substrate import (
    Competency,
    WorkspacePattern,
    WorkspaceSuppression,
    add_suppression,
    append_audit_history,
    custom_checks_dir,
    ensure_substrate,
    familiar_workspace_patterns,
    is_suppressed,
    load_audit_history,
    load_competency,
    load_workspace_patterns,
    load_workspace_suppressions,
    render_pattern_priors,
    save_competency,
    substrate_dir,
    unresolved_workspace_patterns,
    upsert_pattern,
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_substrate_dir_returns_canonical_path(tmp_path: Path) -> None:
    out = substrate_dir(tmp_path)
    assert out == tmp_path / ".augmentum" / "bug_finder"


def test_custom_checks_dir_under_substrate(tmp_path: Path) -> None:
    assert custom_checks_dir(tmp_path) == (
        tmp_path / ".augmentum" / "bug_finder" / "custom_checks"
    )


def test_ensure_substrate_is_idempotent(tmp_path: Path) -> None:
    a = ensure_substrate(tmp_path)
    b = ensure_substrate(tmp_path)
    assert a == b
    assert a.is_dir()
    assert custom_checks_dir(tmp_path).is_dir()


def test_ensure_substrate_writes_gitignore(tmp_path: Path) -> None:
    ensure_substrate(tmp_path)
    gi = substrate_dir(tmp_path) / ".gitignore"
    assert gi.is_file()
    content = gi.read_text(encoding="utf-8")
    assert "audit_history.jsonl" in content


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------


def test_load_patterns_empty_when_missing(tmp_path: Path) -> None:
    assert load_workspace_patterns(tmp_path) == []


def test_upsert_pattern_writes_new_entry(tmp_path: Path) -> None:
    p = upsert_pattern(
        tmp_path,
        signature="sql_injection",
        file_pattern="augmentum/auth/store.py",
        sample_claim="f-string in execute()",
        severity="high",
    )
    assert p.hit_count == 1
    assert p.fix_count == 0
    loaded = load_workspace_patterns(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].signature == "sql_injection"
    assert loaded[0].file_pattern == "augmentum/auth/store.py"


def test_upsert_pattern_increments_hit_count(tmp_path: Path) -> None:
    """Repeat upsert against the same (signature, file_pattern) bumps
    the hit_count rather than creating a duplicate row."""
    upsert_pattern(tmp_path, signature="sql_injection", file_pattern="x.py")
    upsert_pattern(tmp_path, signature="sql_injection", file_pattern="x.py")
    upsert_pattern(tmp_path, signature="sql_injection", file_pattern="x.py")
    patterns = load_workspace_patterns(tmp_path)
    assert len(patterns) == 1
    assert patterns[0].hit_count == 3


def test_upsert_pattern_distinguishes_distinct_files(tmp_path: Path) -> None:
    upsert_pattern(tmp_path, signature="sql_injection", file_pattern="a.py")
    upsert_pattern(tmp_path, signature="sql_injection", file_pattern="b.py")
    patterns = load_workspace_patterns(tmp_path)
    assert len(patterns) == 2


def test_upsert_pattern_tracks_fixes(tmp_path: Path) -> None:
    upsert_pattern(
        tmp_path, signature="x", file_pattern="x.py",
        confirmed=True,
    )
    upsert_pattern(
        tmp_path, signature="x", file_pattern="x.py",
        confirmed=True,
    )
    patterns = load_workspace_patterns(tmp_path)
    assert patterns[0].fix_count == 2


# ---------------------------------------------------------------------------
# Suppressions
# ---------------------------------------------------------------------------


def test_add_suppression_writes_entry(tmp_path: Path) -> None:
    add_suppression(
        tmp_path,
        rule_id="md5_cache_key",
        scope="rule",
        pattern="B324",
        reason="MD5 used for cache key, not security",
    )
    rules = load_workspace_suppressions(tmp_path)
    assert len(rules) == 1
    assert rules[0].rule_id == "md5_cache_key"
    assert rules[0].pattern == "B324"


def test_add_suppression_is_idempotent_by_rule_id(tmp_path: Path) -> None:
    """Adding the same rule_id twice should not duplicate the entry."""
    add_suppression(tmp_path, rule_id="x", scope="rule", pattern="B324")
    add_suppression(tmp_path, rule_id="x", scope="rule", pattern="B324")
    rules = load_workspace_suppressions(tmp_path)
    assert len(rules) == 1


def test_is_suppressed_matches_file_scope() -> None:
    rules = [
        WorkspaceSuppression(
            rule_id="r1", scope="file", pattern="tests/",
        ),
    ]
    assert is_suppressed(rules, file="src/tests/x.py", category="X") == "r1"
    assert is_suppressed(rules, file="src/other.py", category="X") == ""


def test_is_suppressed_matches_category_scope() -> None:
    rules = [
        WorkspaceSuppression(
            rule_id="r2", scope="category", pattern="hardcoded_password",
        ),
    ]
    assert is_suppressed(
        rules, file="x.py", category="hardcoded_password_literal",
    ) == "r2"
    assert is_suppressed(rules, file="x.py", category="other") == ""


def test_is_suppressed_matches_rule_scope() -> None:
    rules = [
        WorkspaceSuppression(rule_id="r3", scope="rule", pattern="S105"),
    ]
    assert is_suppressed(
        rules, file="x.py", category="S105", rule_id="S105",
    ) == "r3"


def test_is_suppressed_returns_empty_for_no_match() -> None:
    assert is_suppressed([], file="x.py", category="X") == ""


# ---------------------------------------------------------------------------
# Competency
# ---------------------------------------------------------------------------


def test_competency_defaults_zero(tmp_path: Path) -> None:
    c = load_competency(tmp_path)
    assert c.audit_count == 0
    assert c.total_findings_ever == 0
    assert c.precision() == 0.0


def test_competency_save_and_reload(tmp_path: Path) -> None:
    c = Competency(
        audit_count=5,
        total_findings_ever=120,
        total_confirmed=80,
        total_fixed=40,
        last_run_at=int(time.time()),
        scanner_effectiveness={"bandit": 0.6, "ruff": 0.4},
    )
    save_competency(tmp_path, c)
    loaded = load_competency(tmp_path)
    assert loaded.audit_count == 5
    assert loaded.total_confirmed == 80
    assert loaded.scanner_effectiveness == {"bandit": 0.6, "ruff": 0.4}


def test_competency_precision_handles_zero_total(tmp_path: Path) -> None:
    c = Competency(audit_count=1, total_confirmed=5)
    assert c.precision() == 0.0   # division-by-zero guarded


def test_competency_precision_computes_ratio(tmp_path: Path) -> None:
    c = Competency(total_findings_ever=200, total_confirmed=80)
    assert c.precision() == 0.4


# ---------------------------------------------------------------------------
# Audit history
# ---------------------------------------------------------------------------


def test_append_history_creates_file(tmp_path: Path) -> None:
    append_audit_history(
        tmp_path,
        run_id="r1",
        duration_seconds=42.5,
        findings_by_severity={"high": 3, "medium": 10},
        confirmed=2,
        mode="explore",
    )
    rows = load_audit_history(tmp_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["findings_by_severity"] == {"high": 3, "medium": 10}


def test_append_history_is_append_only(tmp_path: Path) -> None:
    for i in range(3):
        append_audit_history(
            tmp_path, run_id=f"r{i}",
            duration_seconds=1.0,
            findings_by_severity={},
        )
    rows = load_audit_history(tmp_path)
    assert len(rows) == 3
    assert [r["run_id"] for r in rows] == ["r0", "r1", "r2"]


def test_load_history_respects_limit(tmp_path: Path) -> None:
    for i in range(20):
        append_audit_history(
            tmp_path, run_id=f"r{i}",
            duration_seconds=1.0,
            findings_by_severity={},
        )
    rows = load_audit_history(tmp_path, limit=5)
    assert len(rows) == 5
    # Tail-window of latest entries
    assert rows[-1]["run_id"] == "r19"


# ---------------------------------------------------------------------------
# Pattern priors — the read-side of the compounding loop
# ---------------------------------------------------------------------------


def _pattern(
    sig: str = "x", file: str = "f.py",
    hits: int = 1, fixes: int = 0,
    last: int = 0, sev: str = "medium", claim: str = "",
) -> WorkspacePattern:
    return WorkspacePattern(
        signature=sig, file_pattern=file,
        hit_count=hits, fix_count=fixes,
        last_seen_at=last, severity=sev, sample_claim=claim,
    )


def test_workspace_pattern_is_unresolved_property() -> None:
    """``is_unresolved`` is the load-bearing predicate for the prior
    brief — any pattern that's been seen but never fixed is "still
    likely present"."""
    assert _pattern(hits=1, fixes=0).is_unresolved
    assert _pattern(hits=5, fixes=0).is_unresolved
    assert not _pattern(hits=0, fixes=0).is_unresolved
    assert not _pattern(hits=3, fixes=1).is_unresolved


def test_unresolved_workspace_patterns_filters_and_sorts() -> None:
    """Returns only unresolved patterns, sorted by recency."""
    patterns = [
        _pattern(sig="old_unresolved", hits=1, fixes=0, last=10),
        _pattern(sig="new_unresolved", hits=2, fixes=0, last=100),
        _pattern(sig="resolved", hits=3, fixes=2, last=500),  # excluded
    ]
    out = unresolved_workspace_patterns(patterns)
    assert [p.signature for p in out] == [
        "new_unresolved", "old_unresolved",
    ]


def test_familiar_workspace_patterns_threshold() -> None:
    """Only patterns above the recurrence threshold count as familiar."""
    patterns = [
        _pattern(sig="seen_once", hits=1),
        _pattern(sig="seen_thrice", hits=3),
        _pattern(sig="seen_many", hits=15),
    ]
    out = familiar_workspace_patterns(patterns, min_hit_count=3)
    assert [p.signature for p in out] == ["seen_many", "seen_thrice"]


def test_render_pattern_priors_empty_input_returns_empty_string() -> None:
    """First-contact workspace: no priors → no prompt change. Critical
    for not biasing first audits with empty headers."""
    assert render_pattern_priors([]) == ""


def test_render_pattern_priors_includes_unresolved_section() -> None:
    out = render_pattern_priors([
        _pattern(sig="auth_bypass", file="auth/middleware.py",
                 hits=2, fixes=0, last=200, sev="high",
                 claim="cached User stale on is_active flip"),
    ])
    assert "Unresolved" in out
    assert "auth_bypass" in out
    assert "auth/middleware.py" in out
    assert "cached User stale" in out
    # Anti-bias framing must remain visible — the brief must NOT read
    # as a license to confirm priors.
    assert "WHERE-to-look" in out


def test_render_pattern_priors_deduplicates_unresolved_from_familiar() -> None:
    """A pattern listed as unresolved must not also appear in the
    familiar section — that'd be one row twice and waste token budget."""
    p = _pattern(sig="dup", file="x.py", hits=10, fixes=0)
    out = render_pattern_priors([p])
    # Should appear once (in unresolved) since fix_count == 0
    assert out.count("`dup`") == 1


def test_render_pattern_priors_includes_familiar_section_when_present() -> None:
    out = render_pattern_priors([
        _pattern(sig="ruff:F401", file="proxy/server.py",
                 hits=10, fixes=3, last=50, sev="low"),
    ])
    assert "Familiar" in out
    assert "ruff:F401" in out
