"""P1 verification harness — bootsmoke, intent, adapters, verify_change.

These lock the *honest verdict* contract: "didn't break" is never "good"
(→ human_required), a fatal boot short-circuits to failed, only a confirm
oracle (a reproducing test that passes, a behavior gate) reaches verified, a
model judge reaches at most probable, and a migration surface strips every
confirm oracle (schema corruption is never auto-confirmed).
"""

from __future__ import annotations

import json

from augmentum.selfedit import adapters as A
from augmentum.selfedit import bootsmoke as B
from augmentum.selfedit import gate as G
from augmentum.selfedit import health as H
from augmentum.selfedit import intent as I
from augmentum.selfedit import scanners as S
from augmentum.selfedit import verifier as V

_AUDIT_BASE = S.parse_audit_json(
    '{"score": 86.1, "metrics": {"code_quality": {"silent_catches": 4}}, '
    '"regressions": [], "smoke_errors": [], "tool_failures": []}'
)


def _bugfix() -> I.SelfEditIntent:
    return I.classify_intent("fix the crash in the upload handler",
                             changed_paths=["augmentum/proxy/upload_routes.py"])


# ---------------------------------------------------------------------------
# bootsmoke
# ---------------------------------------------------------------------------

async def test_bootsmoke_clean_passes():
    async def ok_run(_dir):
        return B.BootResult(ok=True, failures=[])
    r = await B.boot_smoke_verifier(boot_runner=ok_run).run({"candidate_dir": "/x"})
    assert r.status == V.PASS and r.oracle == V.ORACLE_MECHANICAL
    assert r.confirms_intent is False  # booting ≠ did-what-was-asked


async def test_bootsmoke_broken_fails():
    async def broken(_dir):
        return B.BootResult(ok=False, failures=["import create_app: NameError x"])
    r = await B.boot_smoke_verifier(boot_runner=broken).run({"candidate_dir": "/x"})
    assert r.status == V.FAIL and "BOOT BROKE" in r.detail


async def test_bootsmoke_unlaunchable_skips_not_fails():
    async def cant_launch(_dir):
        return B.BootResult(ok=False, failures=["could not launch interpreter"], launched=False)
    r = await B.boot_smoke_verifier(boot_runner=cant_launch).run({"candidate_dir": "/x"})
    assert r.status == V.SKIP  # infra hiccup never reads as a code regression


# ---------------------------------------------------------------------------
# intent classifier
# ---------------------------------------------------------------------------

def test_intent_bugfix_is_mechanically_confirmable():
    it = _bugfix()
    assert it.intent_class == I.CLASS_BUGFIX
    assert it.mechanically_confirmable is True
    assert it.surface == I.SURFACE_BACKEND


def test_intent_style_is_not_confirmable():
    it = I.classify_intent("make the Agents panel cleaner, nicer spacing",
                           changed_paths=["ui/styles/coder.css"])
    assert it.intent_class == I.CLASS_STYLE
    assert it.mechanically_confirmable is False  # the "moved the button" case
    assert it.surface == I.SURFACE_FRONTEND


def test_intent_debt_detected():
    it = I.classify_intent("wire the missing setting layer and add the missing test")
    assert it.intent_class == I.CLASS_DEBT
    assert it.mechanically_confirmable is True


def test_intent_feature_from_implement():
    it = I.classify_intent("add a new endpoint that returns the run history")
    assert it.intent_class == I.CLASS_FEATURE


def test_surface_migration_dominates():
    s = I.classify_surface(["augmentum/state/migrations/289_x.sql", "ui/scripts/x.js"])
    assert s == I.SURFACE_MIGRATION
    assert I.classify_surface([]) == I.SURFACE_NONE


async def test_derive_spec_no_backend_is_noop():
    it = _bugfix()
    out = await I.derive_spec(it, request="fix it", backend=None, model="")
    assert out.behaviors == []


# ---------------------------------------------------------------------------
# adapters — Check→Verifier, pytest confirm, health, behavior gate, goal judge
# ---------------------------------------------------------------------------

async def test_verifier_from_check_maps_status():
    async def passing():
        return ("pass", "ok")
    v = A.verifier_from_check(G.Check("c", passing), confirms_intent=True)
    r = await v.run({})
    assert r.status == V.PASS and r.confirms_intent is True


async def test_pytest_confirm_skips_without_tests():
    v = A.pytest_confirm_verifier([], cwd="/x")
    r = await v.run({})
    assert r.status == V.SKIP


def test_behaviors_passed_helper():
    assert A.behaviors_passed([{"status": "pass"}, {"status": "pass"}]) == (True, 2, 2)
    assert A.behaviors_passed([{"status": "pass"}, {"status": "fail"}]) == (False, 1, 2)
    assert A.behaviors_passed([{"status": "untested"}]) == (False, 0, 0)


async def test_behavior_gate_verifier_pass_fail_skip():
    rp = await A.behavior_gate_verifier([{"status": "pass"}]).run({})
    assert rp.status == V.PASS and rp.confirms_intent is True
    rf = await A.behavior_gate_verifier([{"status": "pass"}, {"status": "fail"}]).run({})
    assert rf.status == V.FAIL
    rs = await A.behavior_gate_verifier([{"status": "untested"}]).run({})
    assert rs.status == V.SKIP


async def test_health_verifier_regression_fails():
    base = H.HealthReport(score=1.0, ok=True,
                          dimensions=[H.DimensionResult("compile", ok=True, score=1.0)])
    async def regressed(_dir):
        return H.HealthReport(score=0.5, ok=False,
                              dimensions=[H.DimensionResult("compile", ok=False, score=0.0)])
    r = await A.health_verifier(run_health=regressed, baseline=base).run({"candidate_dir": "/x"})
    assert r.status == V.FAIL and "new failures" in r.detail


async def test_goal_judge_none_signal_skips():
    class _V:
        ok = None
        reason = "backend down"
    async def judge(**_):
        return _V()
    v = A.goal_judge_verifier(judge)
    r = await v.run({})
    assert r.status == V.SKIP  # no-signal judge never counts as a failed intent


# ---------------------------------------------------------------------------
# verify_change — the honest router end to end
# ---------------------------------------------------------------------------

async def _clean_audit(_dir):
    return json.dumps(_AUDIT_BASE.raw)


async def _boot_ok(_dir):
    return B.BootResult(ok=True, failures=[])


async def test_verify_change_clean_no_confirm_is_human_required():
    # Boots clean + audit clean, but nothing confirmed the intent → not "good".
    verdict = await A.verify_change(
        candidate_dir="/x", intent=_bugfix(),
        baseline_audit=_AUDIT_BASE, run_audit=_clean_audit, boot_runner=_boot_ok,
    )
    assert verdict.passed is True
    assert verdict.tier == V.TIER_HUMAN_REQUIRED


async def test_verify_change_boot_break_is_failed():
    async def boot_bad(_dir):
        return B.BootResult(ok=False, failures=["import create_app: SyntaxError"])
    verdict = await A.verify_change(
        candidate_dir="/x", intent=_bugfix(),
        baseline_audit=_AUDIT_BASE, run_audit=_clean_audit, boot_runner=boot_bad,
    )
    assert verdict.passed is False and verdict.tier == V.TIER_FAILED


async def test_verify_change_confirm_oracle_reaches_verified():
    # A passing behavior gate (mechanical confirm) on a bugfix → verified.
    gate = A.behavior_gate_verifier([{"status": "pass"}, {"status": "pass"}])
    verdict = await A.verify_change(
        candidate_dir="/x", intent=_bugfix(),
        baseline_audit=_AUDIT_BASE, run_audit=_clean_audit, boot_runner=_boot_ok,
        extra_verifiers=[gate],
    )
    assert verdict.tier == V.TIER_VERIFIED and verdict.auto_promotable is True


async def test_verify_change_judge_reaches_probable_not_verified():
    class _V:
        ok = True
        reason = "looks done"
    async def judge(**_):
        return _V()
    gj = A.goal_judge_verifier(judge)
    verdict = await A.verify_change(
        candidate_dir="/x", intent=_bugfix(),
        baseline_audit=_AUDIT_BASE, run_audit=_clean_audit, boot_runner=_boot_ok,
        extra_verifiers=[gj],
    )
    assert verdict.tier == V.TIER_PROBABLE  # a model judge never reaches verified


async def test_verify_change_migration_strips_confirm_oracles():
    # Even a passing confirm oracle can't auto-confirm a migration change.
    it = I.classify_intent("add a column", changed_paths=["augmentum/state/migrations/289_x.sql"])
    assert it.surface == I.SURFACE_MIGRATION
    gate = A.behavior_gate_verifier([{"status": "pass"}])
    verdict = await A.verify_change(
        candidate_dir="/x", intent=it,
        baseline_audit=_AUDIT_BASE, run_audit=_clean_audit, boot_runner=_boot_ok,
        extra_verifiers=[gate], test_paths=["tests/test_x.py"],
    )
    assert verdict.tier == V.TIER_HUMAN_REQUIRED  # corruption is never auto-confirmed
