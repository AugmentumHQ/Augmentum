"""Orchestrator tests — the full propose→isolate→edit→verify→record loop.

The end-to-end tests run against a REAL temporary git repo with a fake edit
driver: only the agent is swapped (a direct file write instead of Claude), so the
whole pipeline — worktree isolation, commit, verify_change, the archive store —
is exercised for real. Boot-smoke is the one injected oracle (the temp repo isn't
Augmentum, so the real importer would always fail); everything else is genuine.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import aiosqlite
import pytest

from augmentum.selfedit import bootsmoke as B
from augmentum.selfedit import orchestrator as O
from augmentum.selfedit import store
from augmentum.selfedit import verifier as V

_MIGRATION = (
    pathlib.Path(__file__).resolve().parent.parent
    / "augmentum" / "state" / "migrations" / "288_self_edit_attempts.sql"
)

_HAS_GIT = shutil.which("git") is not None
_needs_git = pytest.mark.skipif(not _HAS_GIT, reason="git not available")


async def _db():
    from augmentum.selfedit.growth_db import _ensure_columns
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)")
    await conn.executescript(_MIGRATION.read_text())
    # mirror the live growth-DB open path: additive columns landed after 288 are
    # ALTER-ed in idempotently (so the test table matches what create_attempt writes).
    await _ensure_columns(conn)
    await conn.commit()
    return conn


def _git_repo(tmp_path) -> str:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    def g(*a):
        subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, env=env)
    g("init", "-b", "main")
    g("config", "user.email", "t@t.t")
    g("config", "user.name", "t")
    (repo / "seed.py").write_text("x = 1\n")
    g("add", "-A")
    g("commit", "-m", "seed")
    return str(repo)


def _writer_driver(filename: str, content: str) -> O.EditDriver:
    """A fake agent that writes one file into the candidate worktree."""
    async def _drive(req: O.EditRequest) -> O.EditResult:
        (pathlib.Path(req.candidate.path) / filename).write_text(content)
        return O.EditResult(ok=True, run_id="run-fake", final_text="edited")
    return _drive


async def _boot_ok(_dir):
    return B.BootResult(ok=True, failures=[])


async def _boot_broken(_dir):
    return B.BootResult(ok=False, failures=["import create_app: SyntaxError"])


# ---------------------------------------------------------------------------
# resolve_terminal — the pure honest mapping
# ---------------------------------------------------------------------------

def _verdict(tier: str, passed: bool) -> V.Verdict:
    return V.Verdict(tier=tier, passed=passed, summary=f"{tier} summary")


def test_resolve_terminal_agent_failure():
    s, outcome, lesson = O.resolve_terminal(
        edit=O.EditResult(ok=False, error="boom"), has_changes=False, verdict=None)
    assert s == "failed" and "boom" in lesson


def test_resolve_terminal_no_changes():
    s, _, lesson = O.resolve_terminal(
        edit=O.EditResult(ok=True), has_changes=False, verdict=None)
    assert s == "rejected" and "no edits" in lesson


def test_resolve_terminal_verify_failed():
    s, _, _ = O.resolve_terminal(
        edit=O.EditResult(ok=True), has_changes=True, verdict=_verdict(V.TIER_FAILED, False))
    assert s == "rejected"


def test_resolve_terminal_passing_rests_at_gated():
    for tier in (V.TIER_VERIFIED, V.TIER_PROBABLE, V.TIER_HUMAN_REQUIRED):
        s, _, _ = O.resolve_terminal(
            edit=O.EditResult(ok=True), has_changes=True, verdict=_verdict(tier, True))
        assert s == "gated", tier


def test_resolve_terminal_edit_then_error_still_verifies():
    # The agent made a real change then errored late (ok=False) — the verifier is
    # the arbiter: a verified change rests at gated, not thrown away as failed.
    s, outcome, _ = O.resolve_terminal(
        edit=O.EditResult(ok=False, error="model turn failed"),
        has_changes=True, verdict=_verdict(V.TIER_VERIFIED, True))
    assert s == "gated" and "verified" in outcome


def test_resolve_terminal_edit_then_error_bad_change_rejected():
    # ...but if that change fails verification, it's rejected (not gated).
    s, _, _ = O.resolve_terminal(
        edit=O.EditResult(ok=False, error="boom"),
        has_changes=True, verdict=_verdict(V.TIER_FAILED, False))
    assert s == "rejected"


def test_resolve_terminal_lesson_distinguishes_tier():
    _, _, verified = O.resolve_terminal(
        edit=O.EditResult(ok=True), has_changes=True, verdict=_verdict(V.TIER_VERIFIED, True))
    _, _, human = O.resolve_terminal(
        edit=O.EditResult(ok=True), has_changes=True, verdict=_verdict(V.TIER_HUMAN_REQUIRED, True))
    assert "auto-promote" in verified and "only you" in human


def test_tier_for_surface():
    assert O._tier_for_surface("migration") == "red"
    assert O._tier_for_surface("backend") == "yellow"
    assert O._tier_for_surface("frontend") == "green"


# ---------------------------------------------------------------------------
# end-to-end against a real temp git repo
# ---------------------------------------------------------------------------

@_needs_git
async def test_e2e_passing_edit_rests_at_gated(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="add a helper module", user_id="u1", conn=conn,
            driver=_writer_driver("helper.py", "def h():\n    return 1\n"),
            worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_ok,
        )
        assert out.status == "gated"
        assert out.verdict.tier == V.TIER_HUMAN_REQUIRED  # boots, but no confirm oracle
        assert "helper.py" in out.files_changed
        # archived, never pruned
        row = await store.get_attempt(conn, attempt_id=out.attempt_id, user_id="u1")
        assert row["status"] == "gated" and row["gate_passed"] is True
        assert row["run_id"] == "run-fake" and row["lesson"]
        # worktree dir cleaned up, branch (lineage) kept
        assert not os.path.exists(out.candidate.path)
    finally:
        await conn.close()


@_needs_git
async def test_e2e_broken_boot_is_rejected(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="fix the thing", user_id="u1", conn=conn,
            driver=_writer_driver("bad.py", "def x(:\n"),
            worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_broken,
        )
        assert out.status == "rejected"
        row = await store.get_attempt(conn, attempt_id=out.attempt_id, user_id="u1")
        assert row["status"] == "rejected" and row["gate_passed"] is False
        assert row["lesson"]  # the failure taught something (anti-Westworld)
    finally:
        await conn.close()


@_needs_git
async def test_e2e_audit_tamper_refused(tmp_path):
    # The agent edited the JUDGE (a scanner suppressions file) to make a finding
    # 'disappear' — a real run did exactly this. It must be refused BEFORE verify,
    # never cherry-picked, whatever the driver.
    repo = _git_repo(tmp_path)
    conn = await _db()

    async def _tamper_driver(req: O.EditRequest) -> O.EditResult:
        p = (pathlib.Path(req.candidate.path)
             / ".claude/skills/augmentum-dev/scripts/runtime_suppressions.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"silent_exceptions": ["augmentum/x.py:1"]}\n')
        return O.EditResult(ok=True, run_id="run-tamper", final_text="suppressed it")

    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="fix a runtime error", user_id="u1", conn=conn,
            driver=_tamper_driver, worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_ok)
        assert out.status == "rejected"
        assert "audit infra" in out.outcome
        row = await store.get_attempt(conn, attempt_id=out.attempt_id, user_id="u1")
        assert row["status"] == "rejected" and "judge" in (row["lesson"] or "")
    finally:
        await conn.close()


def test_touches_audit_infra():
    assert O._touches_audit_infra(".claude/skills/x/runtime_suppressions.json")
    assert O._touches_audit_infra("augmentum/x/.claude/foo.py")
    assert O._touches_audit_infra("ui/quality_suppressions.json")
    assert not O._touches_audit_infra("augmentum/proxy/server.py")
    assert not O._touches_audit_infra("ui/scripts/workshop.js")


@_needs_git
async def test_e2e_no_op_edit_is_rejected(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="do nothing", user_id="u1", conn=conn,
            driver=O.null_edit_driver, worktrees_dir=str(tmp_path / "wt"),
            boot_runner=_boot_ok,
        )
        assert out.status == "rejected" and out.files_changed == []
    finally:
        await conn.close()


@_needs_git
async def test_e2e_agent_failure_is_failed(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    async def _broken_driver(req):
        return O.EditResult(ok=False, error="container died")
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="x", user_id="u1", conn=conn,
            driver=_broken_driver, worktrees_dir=str(tmp_path / "wt"),
        )
        assert out.status == "failed"
        row = await store.get_attempt(conn, attempt_id=out.attempt_id, user_id="u1")
        assert "container died" in row["lesson"]
    finally:
        await conn.close()


@_needs_git
async def test_e2e_confirm_oracle_reaches_verified(tmp_path):
    from augmentum.selfedit.adapters import behavior_gate_verifier
    repo = _git_repo(tmp_path)
    conn = await _db()
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="add a feature", user_id="u1", conn=conn,
            driver=_writer_driver("feature.py", "FEATURE = True\n"),
            worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_ok,
            extra_verifiers=[behavior_gate_verifier([{"status": "pass"}, {"status": "pass"}])],
        )
        assert out.status == "gated" and out.verdict.tier == V.TIER_VERIFIED
        assert out.verdict.auto_promotable is True
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# self-heal — the cheap repair rung below escalation (2026-07-02)
# ---------------------------------------------------------------------------

def _healing_driver(fname: str):
    """Writes BROKEN code first; on a repair pass (prior_context is set), writes
    fixed code — the fake analogue of a model told exactly what it broke."""
    async def _drive(req: O.EditRequest) -> O.EditResult:
        p = pathlib.Path(req.candidate.path) / fname
        if req.prior_context and "verification" in req.prior_context.lower():
            p.write_text("ok = 1  # repaired\n")
        else:
            p.write_text("broken = (  # BROKEN syntax\n")
        return O.EditResult(ok=True, run_id="run-fake", final_text="edited")
    return _drive


def _stubborn_broken_driver(fname: str):
    """Writes DIFFERENT code each call (so there's always a new edit to verify)
    but always broken the SAME way — the model keeps trying yet never fixes the
    failure. Must stagnation-break on the unchanged failure signature, not loop."""
    calls = {"n": 0}
    async def _drive(req: O.EditRequest) -> O.EditResult:
        calls["n"] += 1
        (pathlib.Path(req.candidate.path) / fname).write_text(
            f"broken_{calls['n']} = (  # BROKEN\n")
        return O.EditResult(ok=True, run_id="run-fake", final_text="edited")
    return _drive


def _boot_from_file(fname: str):
    """A boot-runner whose verdict reflects the candidate file — fails while the
    code is BROKEN, passes once repaired. Mechanical + required = a FAILED verdict
    that self-heal is allowed to repair."""
    async def _boot(target_dir: str) -> B.BootResult:
        try:
            content = (pathlib.Path(target_dir) / fname).read_text()
        except OSError:
            return B.BootResult(ok=True, failures=[])
        if "BROKEN" in content:
            return B.BootResult(ok=False, failures=["import create_app: IndentationError: broken"])
        return B.BootResult(ok=True, failures=[])
    return _boot


@_needs_git
async def test_self_heal_repairs_a_fixable_break(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="add mod", user_id="u1", conn=conn,
            driver=_healing_driver("mod.py"), boot_runner=_boot_from_file("mod.py"),
            worktrees_dir=str(tmp_path / "wt"), max_heal_attempts=2,
        )
        # broke first, then self-healed → lands at gated, not rejected
        assert out.status == "gated", (out.status, out.outcome)
        assert out.heals == 1
        assert out.verdict is not None and out.verdict.tier != V.TIER_FAILED
    finally:
        await conn.close()


@_needs_git
async def test_self_heal_off_by_default_rejects_the_break(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="add mod", user_id="u1", conn=conn,
            driver=_healing_driver("mod.py"), boot_runner=_boot_from_file("mod.py"),
            worktrees_dir=str(tmp_path / "wt"),  # max_heal_attempts defaults to 0
        )
        assert out.status == "rejected" and out.heals == 0
    finally:
        await conn.close()


@_needs_git
async def test_self_heal_stops_on_stagnation(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="add mod", user_id="u1", conn=conn,
            driver=_stubborn_broken_driver("mod.py"), boot_runner=_boot_from_file("mod.py"),
            worktrees_dir=str(tmp_path / "wt"), max_heal_attempts=3,
        )
        # same break after a repair → stop (don't burn all 3), reject for escalation
        assert out.status == "rejected"
        assert out.heals == 1, "should stagnation-break after one no-progress repair"
    finally:
        await conn.close()


@_needs_git
async def test_self_heal_keeps_one_promotable_commit(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    try:
        out = await O.run_self_edit(
            repo_dir=repo, objective="add mod", user_id="u1", conn=conn,
            driver=_healing_driver("mod.py"), boot_runner=_boot_from_file("mod.py"),
            worktrees_dir=str(tmp_path / "wt"), max_heal_attempts=2, keep_worktree=True,
        )
        assert out.status == "gated" and out.candidate is not None
        # the branch carries exactly ONE commit over main (amend, not stacked) so a
        # promote cherry-pick of the tip lands the whole repaired change
        n = subprocess.run(["git", "-C", repo, "rev-list", "--count",
                            f"main..{out.candidate.branch}"],
                           capture_output=True, text=True).stdout.strip()
        assert n == "1", f"expected 1 commit on the branch, got {n}"
    finally:
        await conn.close()
