"""Debt-paydown triage tests — the discipline that keeps the loop from Goodharting.

Locks the two load-bearing rules: mechanical findings (silent-catch, dead CSS, a
blocking call, a missing test) go to the auto-lane; structural ones that need a
human decision — and crucially a missing-CSS class, which is *taste* — never do.
"""

from __future__ import annotations

import json

from augmentum.selfedit import debt
from augmentum.selfedit.scanners import parse_audit_json

# Realistic full audit shape — every metric name matches what audit.py emits.
_REPORT = parse_audit_json(json.dumps({
    "score": 80.0,
    "metrics": {
        "code_quality": {
            "silent_catches": 4, "console_log": 2, "dead_css": 9, "ws_gaps": 1,
            "mixed_errors": 2, "missing_css": 110, "tech_debt": 17, "model_map_misuse": 1,
        },
        "async_blocking": {"errors": 3, "warnings": 5},
        "runtime": {"errors": 12, "warnings": 225},
        "wiring": {"errors": 1, "warnings": 31},
        "coverage": {"coverage_gaps": 6, "modules_total": 1112},
        "dead_code": {"orphaned_endpoints": 22, "ghost_calls": 1, "dependency_drift": 7},
        "doc_facts": {"doc_inaccuracies": 1},
        "exceptions": {"stale_entries": 1},
        "security": {"low": 12, "medium": 4, "total": 16},
        "red_team": {"total": 2, "high": 2},
        "deps": {"vulnerabilities": 1},
        "db_safety": {"warnings": 5, "errors": 0},
        "registry": {"drift": 3},
        "unknown_scanner": {"made_up_metric": 99},
    },
}))


def _by_metric(targets):
    return {(t.scanner, t.metric): t for t in targets}


def test_mechanical_targets_are_the_safe_set():
    keys = set(_by_metric(debt.select_debt_targets(_REPORT, kinds=(debt.KIND_MECHANICAL,))))
    # the audit-confirmable auto-lane (incl. the new markers)
    for k in [("code_quality", "silent_catches"), ("code_quality", "console_log"),
              ("code_quality", "dead_css"), ("code_quality", "ws_gaps"),
              ("code_quality", "mixed_errors"), ("async_blocking", "errors"),
              ("async_blocking", "warnings"), ("runtime", "errors"),
              ("wiring", "errors"), ("doc_facts", "doc_inaccuracies"),
              ("exceptions", "stale_entries"), ("coverage", "coverage_gaps")]:
        assert k in keys, k
    # none of the human-judgment findings sneak into the auto-lane
    for k in [("dead_code", "orphaned_endpoints"), ("code_quality", "missing_css"),
              ("dead_code", "ghost_calls"), ("security", "medium"),
              ("red_team", "total"), ("deps", "vulnerabilities")]:
        assert k not in keys, k


def test_console_log_metric_name_matches_audit():
    # regression guard: the metric is `console_log` (singular), as audit.py emits.
    assert ("code_quality", "console_log") in debt._CATALOG
    assert ("code_quality", "console_logs") not in debt._CATALOG


def test_missing_css_is_structural_taste():
    css = _by_metric(debt.select_debt_targets(_REPORT, kinds=(debt.KIND_STRUCTURAL,))).get(
        ("code_quality", "missing_css"))
    assert css is not None
    assert css.confirms_via == debt.CONFIRM_HUMAN  # styling is taste, never auto
    assert "taste" in css.note


def test_security_schema_and_adversarial_are_structural_red_tier():
    s = _by_metric(debt.select_debt_targets(_REPORT, kinds=(debt.KIND_STRUCTURAL,)))
    assert s[("security", "low")].confirms_via == debt.CONFIRM_HUMAN
    assert "red-tier" in s[("security", "medium")].note
    assert "red-tier" in s[("red_team", "total")].note      # adversarial
    assert "red-tier" in s[("db_safety", "warnings")].note
    assert s[("deps", "vulnerabilities")].confirms_via == debt.CONFIRM_HUMAN
    # red_team.high is NOT catalogued separately → no double-count with .total
    assert ("red_team", "high") not in s


def test_ghost_call_is_structural_real_bug():
    g = _by_metric(debt.select_debt_targets(_REPORT, kinds=(debt.KIND_STRUCTURAL,))).get(
        ("dead_code", "ghost_calls"))
    assert g is not None and "real" in g.note  # weight-1.0 user-facing break, judgment to fix


def test_zero_count_metric_is_skipped():
    keys = set(_by_metric(debt.select_debt_targets(_REPORT)))
    assert ("db_safety", "errors") not in keys  # count 0 → not a target


def test_unknown_non_problem_metric_is_skipped():
    keys = set(_by_metric(debt.select_debt_targets(_REPORT)))
    assert ("unknown_scanner", "made_up_metric") not in keys  # not problem-shaped → silent


def test_new_problem_metric_auto_surfaces_as_structural():
    # a scanner/metric the catalog has never seen, but clearly a problem count,
    # auto-appears in needs-you (structural, never auto-lane) — the app can grow
    # scanners without manual cataloguing.
    rep = parse_audit_json(json.dumps({"score": 90.0, "metrics": {
        "new_scanner": {"secret_leaks": 4, "modules_total": 999, "fancy_score": 12},
    }}))
    by = _by_metric(debt.select_debt_targets(rep))
    leak = by.get(("new_scanner", "secret_leaks"))
    assert leak is not None and leak.kind == debt.KIND_STRUCTURAL and leak.discovered is True
    assert ("new_scanner", "modules_total") not in by   # informational/aggregate → skipped
    assert ("new_scanner", "fancy_score") not in by     # not problem-shaped → skipped
    # auto-surfaced findings never enter the mechanical auto-lane
    mech = _by_metric(debt.select_debt_targets(rep, kinds=(debt.KIND_MECHANICAL,)))
    assert ("new_scanner", "secret_leaks") not in mech


def test_mechanical_first_then_by_count():
    targets = debt.select_debt_targets(_REPORT)
    assert targets[0].kind == debt.KIND_MECHANICAL  # auto-lane leads
    mech_counts = [t.count for t in targets if t.kind == debt.KIND_MECHANICAL]
    assert mech_counts == sorted(mech_counts, reverse=True)  # larger piles first


def test_next_mechanical_objective_is_the_loops_pick():
    target = debt.next_mechanical_objective(_REPORT)
    assert target is not None and target.kind == debt.KIND_MECHANICAL
    assert target.objective and target.confirms_via in (debt.CONFIRM_SCANNER, debt.CONFIRM_TEST)


def test_next_mechanical_objective_none_when_clean():
    clean = parse_audit_json(json.dumps({"score": 100.0, "metrics": {}}))
    assert debt.next_mechanical_objective(clean) is None


def test_triage_splits_and_counts_consistently():
    t = debt.triage(_REPORT)
    assert t.mechanical and t.structural
    assert all(x.kind == debt.KIND_MECHANICAL for x in t.mechanical)
    assert all(x.kind == debt.KIND_STRUCTURAL for x in t.structural)
    # count helpers are self-consistent with the target counts
    assert t.mechanical_count == sum(x.count for x in t.mechanical)
    assert t.structural_count == sum(x.count for x in t.structural)
