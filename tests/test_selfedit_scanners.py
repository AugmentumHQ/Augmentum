"""augmentum-dev audit → Verifier bridge tests."""

from __future__ import annotations

import json

from augmentum.selfedit import scanners as S
from augmentum.selfedit import verifier as V

# Realistic audit JSON (shape from the live audit.py --format=json).
_BASE = {
    "score": 86.1,
    "metrics": {
        "async_blocking": {"errors": 0, "warnings": 5},
        "code_quality": {"silent_catches": 4, "missing_css": 110, "tech_debt": 17},
        "coverage": {"coverage_gaps": 3, "modules_covered": 1109, "modules_total": 1112},
        "security": {"total": 0, "critical": 0, "high": 0},
        "db_safety": {"errors": 0, "warnings": 5},
    },
    "regressions": [], "improvements": [], "smoke_errors": [], "tool_failures": [],
}


def _mut(**over):
    d = json.loads(json.dumps(_BASE))
    for path, val in over.items():
        scanner, key = path.split(".")
        d["metrics"][scanner][key] = val
    return d


def test_parse_real_shape():
    r = S.parse_audit_json(json.dumps(_BASE))
    assert r.score == 86.1
    assert r.metrics["code_quality"]["silent_catches"] == 4
    assert r.smoke_errors == []


def test_parse_bad_json_raises():
    import pytest
    with pytest.raises(ValueError):
        S.parse_audit_json("not json")


def test_delta_clean_is_not_regression():
    base = S.parse_audit_json(json.dumps(_BASE))
    cand = S.parse_audit_json(json.dumps(_BASE))
    d = S.audit_delta(cand, base)
    assert d.regressed is False and d.worsened == [] and d.broke_boot is False


def test_delta_new_debt_is_worsened_and_score_drop_regresses():
    base = S.parse_audit_json(json.dumps(_BASE))
    worse = _mut(**{"code_quality.silent_catches": 6})
    worse["score"] = 84.0  # score dropped
    cand = S.parse_audit_json(json.dumps(worse))
    d = S.audit_delta(cand, base)
    assert d.regressed is True
    assert any("silent_catches: 4->6" in w for w in d.worsened)


def test_delta_informational_increase_is_not_debt():
    base = S.parse_audit_json(json.dumps(_BASE))
    more_modules = _mut(**{"coverage.modules_covered": 1200})  # higher coverage = fine
    cand = S.parse_audit_json(json.dumps(more_modules))
    d = S.audit_delta(cand, base)
    assert not any("modules_covered" in w for w in d.worsened)


def test_delta_broke_boot_regresses():
    base = S.parse_audit_json(json.dumps(_BASE))
    broken = json.loads(json.dumps(_BASE))
    broken["smoke_errors"] = ["create_app import failed"]
    cand = S.parse_audit_json(json.dumps(broken))
    d = S.audit_delta(cand, base)
    assert d.broke_boot is True and d.regressed is True


async def test_audit_verifier_pass_and_fail():
    base = S.parse_audit_json(json.dumps(_BASE))

    async def clean_run(_dir):
        return json.dumps(_BASE)
    v = S.audit_verifier(run_audit=clean_run, baseline=base)
    r = await v.run({"candidate_dir": "/x"})
    assert r.oracle == V.ORACLE_MECHANICAL and r.confirms_intent is False
    assert r.status == V.PASS

    async def broken_run(_dir):
        bad = json.loads(json.dumps(_BASE))
        bad["score"] = 70.0
        bad["smoke_errors"] = ["boom"]
        return json.dumps(bad)
    rf = await S.audit_verifier(run_audit=broken_run, baseline=base).run({"candidate_dir": "/x"})
    assert rf.status == V.FAIL and "BOOT" in rf.detail


async def test_audit_verifier_unavailable_skips():
    base = S.parse_audit_json(json.dumps(_BASE))

    async def crash_run(_dir):
        raise RuntimeError("audit missing")
    r = await S.audit_verifier(run_audit=crash_run, baseline=base).run({})
    assert r.status == V.SKIP  # unavailable → skip, never a false fail


async def test_audit_verifier_makes_human_required_not_verified_in_router():
    # The honesty check: a clean audit is no-regression (confirms_intent=False),
    # so on its own it yields human_required, never verified.
    base = S.parse_audit_json(json.dumps(_BASE))

    async def clean_run(_dir):
        return json.dumps(_BASE)
    pool = {"audit": S.audit_verifier(run_audit=clean_run, baseline=base)}
    verdict = await V.verify({"candidate_dir": "/x"}, verifiers=pool)
    assert verdict.passed is True
    assert verdict.tier == V.TIER_HUMAN_REQUIRED  # didn't break ≠ good


def test_select_scanners_incremental():
    assert "db_safety" in S.select_scanners(["augmentum/state/migrations/289_x.sql"])
    css = S.select_scanners(["ui/styles/coder.css"])
    assert "code_quality" in css
    routes = S.select_scanners(["augmentum/proxy/foo_routes.py"])
    assert {"wiring", "dead_code"} <= routes
    assert S.select_scanners(["README.md"]) == set()  # nothing relevant
