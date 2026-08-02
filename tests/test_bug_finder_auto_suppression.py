"""Auto-suppression candidate tests.

Cover the load-bearing rules:

* Empty pattern memory → no candidates.
* Pattern with hits >= default threshold + zero fixes → candidate.
* Pattern with any confirmation (fix_count > 0) → NOT a candidate.
* Security-class rules need a much higher hit threshold.
* Inert/style rules surface at a much lower threshold.
* Already-suppressed patterns don't surface twice.
* ``apply_candidate`` persists a real WorkspaceSuppression.
* Aggregation collapses per-file candidates into one signature row.
"""

from __future__ import annotations

from pathlib import Path

from augmentum.bug_finder.auto_suppression import (
    aggregate_candidates,
    apply_aggregated,
    apply_candidate,
    compute_aggregated_candidates,
    compute_suppression_candidates,
)
from augmentum.bug_finder.workspace_substrate import (
    add_suppression,
    load_workspace_suppressions,
    upsert_pattern,
)

# ---------------------------------------------------------------------------
# Empty / no-candidate paths
# ---------------------------------------------------------------------------


def test_empty_workspace_returns_no_candidates(tmp_path: Path) -> None:
    assert compute_suppression_candidates(tmp_path) == []


def test_low_hit_pattern_is_not_candidate(tmp_path: Path) -> None:
    # Default threshold is 100; 50 hits should not surface.
    for _ in range(50):
        upsert_pattern(tmp_path, signature="ruff:B028",
                       file_pattern="src/x.py", severity="medium")
    assert compute_suppression_candidates(tmp_path) == []


def test_confirmed_pattern_is_not_candidate(tmp_path: Path) -> None:
    """Any past confirmation (fix_count > 0) excludes the pattern."""
    for _ in range(150):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/x.py", severity="medium")
    # One confirmation is enough to disqualify
    upsert_pattern(tmp_path, signature="bandit:B101",
                   file_pattern="src/x.py", severity="medium",
                   confirmed=True)
    assert compute_suppression_candidates(tmp_path) == []


# ---------------------------------------------------------------------------
# Threshold tiers
# ---------------------------------------------------------------------------


def test_default_rule_surfaces_above_100_hits(tmp_path: Path) -> None:
    for _ in range(105):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/x.py", severity="medium")
    candidates = compute_suppression_candidates(tmp_path)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.signature == "bandit:B101"
    assert c.hit_count == 105
    assert c.confidence > 0.0


def test_inert_rule_surfaces_at_lower_threshold(tmp_path: Path) -> None:
    """E501 (line too long) should surface at the 30-hit threshold,
    not 100."""
    for _ in range(35):
        upsert_pattern(tmp_path, signature="ruff:E501",
                       file_pattern="src/x.py", severity="low")
    candidates = compute_suppression_candidates(tmp_path)
    assert any(c.signature == "ruff:E501" for c in candidates)


def test_dangerous_rule_requires_high_threshold(tmp_path: Path) -> None:
    """SQL-injection-class rule should NOT surface at 100 hits — it
    needs >=500 before the substrate proposes auto-suppression."""
    for _ in range(150):
        upsert_pattern(tmp_path, signature="bandit:B608",
                       file_pattern="src/db.py", severity="high")
    assert compute_suppression_candidates(tmp_path) == []


def test_dangerous_rule_at_high_threshold_surfaces_with_low_confidence(
    tmp_path: Path,
) -> None:
    for _ in range(550):
        upsert_pattern(tmp_path, signature="bandit:B608",
                       file_pattern="src/db.py", severity="high")
    candidates = compute_suppression_candidates(tmp_path)
    assert len(candidates) == 1
    c = candidates[0]
    # Security rules are capped below 0.75 even with overwhelming hits
    assert c.confidence <= 0.75
    assert "security" in c.rationale.lower()


# ---------------------------------------------------------------------------
# Existing-suppression filter
# ---------------------------------------------------------------------------


def test_already_suppressed_rule_skipped(tmp_path: Path) -> None:
    for _ in range(150):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/x.py", severity="medium")
    # Manually suppressed via standard substrate helper
    add_suppression(tmp_path, rule_id="manual_B101",
                    scope="rule", pattern="B101",
                    reason="library idiom")
    assert compute_suppression_candidates(tmp_path) == []


# ---------------------------------------------------------------------------
# apply_candidate persists
# ---------------------------------------------------------------------------


def test_apply_candidate_writes_suppression(tmp_path: Path) -> None:
    for _ in range(120):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/x.py", severity="medium")
    candidates = compute_suppression_candidates(tmp_path)
    assert candidates
    applied = apply_candidate(tmp_path, candidates[0],
                              reason="confirmed library idiom")
    rules = load_workspace_suppressions(tmp_path)
    assert any(r.rule_id == applied.rule_id for r in rules)
    rule = next(r for r in rules if r.rule_id == applied.rule_id)
    assert rule.scope == "rule"
    assert rule.pattern == "B101"   # scanner prefix stripped


def test_apply_candidate_file_scope(tmp_path: Path) -> None:
    for _ in range(120):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/x.py", severity="medium")
    candidate = compute_suppression_candidates(tmp_path)[0]
    apply_candidate(tmp_path, candidate, scope="file")
    rules = load_workspace_suppressions(tmp_path)
    rule = next(r for r in rules if r.rule_id == candidate.rule_id)
    assert rule.scope == "file"
    assert rule.pattern == "src/x.py"


def test_apply_candidate_dedups_via_substrate(tmp_path: Path) -> None:
    """apply_candidate uses ``add_suppression`` which is idempotent by
    rule_id, so re-applying the same candidate doesn't duplicate."""
    for _ in range(120):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/x.py", severity="medium")
    candidate = compute_suppression_candidates(tmp_path)[0]
    apply_candidate(tmp_path, candidate)
    apply_candidate(tmp_path, candidate)
    assert len(load_workspace_suppressions(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_collapses_per_file_rows(tmp_path: Path) -> None:
    """Per-file candidates for the same signature roll up into one
    aggregated row with cumulative hit counts."""
    for f, hits in [("src/a.py", 150), ("src/b.py", 200), ("src/c.py", 120)]:
        for _ in range(hits):
            upsert_pattern(tmp_path, signature="bandit:B101",
                           file_pattern=f, severity="medium")
    candidates = compute_suppression_candidates(tmp_path)
    assert len(candidates) == 3
    agg = aggregate_candidates(candidates)
    assert len(agg) == 1
    row = agg[0]
    assert row.signature == "bandit:B101"
    assert row.total_hits == 150 + 200 + 120
    assert row.file_count == 3
    # Top-files sorted by hit count desc
    assert row.top_files[0] == ("src/b.py", 200)


def test_aggregate_orders_by_confidence_then_volume(tmp_path: Path) -> None:
    """Aggregator orders by (confidence desc, hits desc).

    Inert rules (E501) intentionally get a confidence bump because
    suppressing style rules is safer than suppressing assert-detection.
    So an inert rule above its threshold can outrank a medium-tier
    rule even at lower hit counts — that's the substrate saying
    "suppress the safe stuff first."
    """
    for _ in range(80):
        upsert_pattern(tmp_path, signature="ruff:E501",
                       file_pattern="src/a.py")
    for _ in range(150):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/b.py")
    candidates = compute_suppression_candidates(tmp_path)
    agg = aggregate_candidates(candidates)
    # E501 wins because inert rules carry higher confidence per-hit
    assert agg[0].signature == "ruff:E501"
    # Order is by confidence — both should surface
    assert {a.signature for a in agg} == {"ruff:E501", "bandit:B101"}


# ---------------------------------------------------------------------------
# Direct-aggregate computation (no per-file threshold pre-filter)
# ---------------------------------------------------------------------------


def test_compute_aggregated_sums_across_files_below_per_file_threshold(
    tmp_path: Path,
) -> None:
    """The headline case: B101 spread across 50 files with 5 hits
    each. Per-file would reject (5 < 100); aggregate accepts (250 > 100)."""
    for i in range(50):
        for _ in range(5):
            upsert_pattern(tmp_path, signature="bandit:B101",
                           file_pattern=f"src/file_{i}.py")
    # Per-file computation returns nothing
    assert compute_suppression_candidates(tmp_path) == []
    # Direct-aggregate surfaces the cumulative pattern
    agg = compute_aggregated_candidates(tmp_path)
    assert len(agg) == 1
    row = agg[0]
    assert row.signature == "bandit:B101"
    assert row.total_hits == 250
    assert row.file_count == 50


def test_compute_aggregated_skips_when_any_file_confirmed(
    tmp_path: Path,
) -> None:
    for _ in range(150):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/a.py")
    upsert_pattern(tmp_path, signature="bandit:B101",
                   file_pattern="src/b.py", confirmed=True)
    assert compute_aggregated_candidates(tmp_path) == []


def test_compute_aggregated_respects_existing_suppression(
    tmp_path: Path,
) -> None:
    for _ in range(200):
        upsert_pattern(tmp_path, signature="bandit:B101",
                       file_pattern="src/a.py")
    add_suppression(tmp_path, rule_id="manual_B101", scope="rule",
                    pattern="B101", reason="library idiom")
    assert compute_aggregated_candidates(tmp_path) == []


def test_apply_aggregated_persists(tmp_path: Path) -> None:
    for f in ("src/a.py", "src/b.py"):
        for _ in range(120):
            upsert_pattern(tmp_path, signature="bandit:B101",
                           file_pattern=f)
    candidates = compute_suppression_candidates(tmp_path)
    agg = aggregate_candidates(candidates)
    apply_aggregated(tmp_path, agg[0], reason="library idiom")
    rules = load_workspace_suppressions(tmp_path)
    assert len(rules) == 1
    assert rules[0].pattern == "B101"
    assert rules[0].scope == "rule"
