"""Promotion (P4, oracle-tier-gated) + the rollback floor (P5 L1/L2) tests.

Locks the corrected predicate — auto-promote keys on ORACLE TIER, not surface —
and the parachute logic (snapshot/restore + crash-loop counter) that guarantees a
bad backend edit can't permanently brick the box.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import aiosqlite
import pytest

from augmentum.selfedit import promote, rollback, store
from augmentum.selfedit import verifier as V

_MIGRATION = (
    pathlib.Path(__file__).resolve().parent.parent
    / "augmentum" / "state" / "migrations" / "288_self_edit_attempts.sql"
)
_HAS_GIT = shutil.which("git") is not None
_needs_git = pytest.mark.skipif(not _HAS_GIT, reason="git not available")


def _verdict(tier, passed=True):
    return V.Verdict(tier=tier, passed=passed, summary=tier)


# ---------------------------------------------------------------------------
# decide_promotion — the honest predicate (pure)
# ---------------------------------------------------------------------------

def test_verified_auto_promotes_only_when_opted_in():
    v = _verdict(V.TIER_VERIFIED)
    # default posture = propose: even verified waits
    decide = promote.decide_promotion(v, surface="frontend")
    assert decide.auto is False and "propose" in decide.reason
    # opted in: verified auto-promotes
    d2 = promote.decide_promotion(v, surface="frontend", autonomy_level=promote.AUTONOMY_AUTO_VERIFIED)
    assert d2.auto is True


def test_human_required_never_auto_even_when_opted_in():
    d = promote.decide_promotion(_verdict(V.TIER_HUMAN_REQUIRED), surface="frontend",
                                 autonomy_level=promote.AUTONOMY_AUTO_VERIFIED)
    assert d.auto is False  # green ≠ confirmed intent — the CSS-button case


def test_probable_is_not_auto():
    d = promote.decide_promotion(_verdict(V.TIER_PROBABLE), surface="backend",
                                 autonomy_level=promote.AUTONOMY_AUTO_VERIFIED)
    assert d.auto is False  # a model judge → propose, not auto


def test_migration_is_never_auto_even_if_verified():
    d = promote.decide_promotion(_verdict(V.TIER_VERIFIED), surface="migration",
                                 autonomy_level=promote.AUTONOMY_AUTO_VERIFIED)
    assert d.auto is False and "red-tier" in d.reason


def test_verified_backend_auto_promotes_when_opted_in():
    # surface does NOT block backend — it only chooses the revert mechanism
    d = promote.decide_promotion(_verdict(V.TIER_VERIFIED), surface="backend",
                                 autonomy_level=promote.AUTONOMY_AUTO_VERIFIED)
    assert d.auto is True


def test_restart_needed_by_surface():
    assert promote.restart_needed("backend") is True
    assert promote.restart_needed("frontend") is False  # served live


def test_failed_verdict_never_promotes():
    d = promote.decide_promotion(_verdict(V.TIER_FAILED, passed=False), surface="frontend",
                                 autonomy_level=promote.AUTONOMY_AUTO_VERIFIED)
    assert d.auto is False


# ---------------------------------------------------------------------------
# git promote / revert against a real repo
# ---------------------------------------------------------------------------

def _repo_with_candidate(tmp_path):
    """A repo on main + a selfedit branch carrying one commit (the candidate)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    def g(*a):
        return subprocess.run(["git", "-C", str(repo), *a], check=True,
                              capture_output=True, text=True, env=env)
    g("init", "-b", "main")
    g("config", "user.email", "t@t.t")
    g("config", "user.name", "t")
    (repo / "app.py").write_text("v = 1\n")
    g("add", "-A")
    g("commit", "-m", "base")
    g("checkout", "-b", "selfedit/att1")
    (repo / "app.py").write_text("v = 2  # candidate\n")
    g("add", "-A")
    g("commit", "-m", "candidate edit")
    sha = g("rev-parse", "HEAD").stdout.strip()
    g("checkout", "main")
    return str(repo), sha


async def _db():
    from augmentum.selfedit.growth_db import _ensure_columns
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)")
    await conn.executescript(_MIGRATION.read_text())
    await _ensure_columns(conn)  # mirror the live growth-DB open (post-288 columns)
    await conn.commit()
    return conn


@_needs_git
async def test_promote_attempt_cherry_picks_and_records(tmp_path):
    repo, cand_sha = _repo_with_candidate(tmp_path)
    conn = await _db()
    try:
        await store.create_attempt(conn, attempt_id="att1", user_id="u1",
                                   objective="bump v", surface="backend")
        res = await promote.promote_attempt(
            conn=conn, repo_dir=repo, attempt_id="att1", user_id="u1",
            candidate_sha=cand_sha, verdict=_verdict(V.TIER_VERIFIED), surface="backend",
            autonomy_level=promote.AUTONOMY_AUTO_VERIFIED,
        )
        assert res.promoted is True and res.promoted_commit and res.needs_restart is True
        # the live tree (main) now carries the candidate's change
        assert "candidate" in (pathlib.Path(repo) / "app.py").read_text()
        row = await store.get_attempt(conn, attempt_id="att1", user_id="u1")
        assert row["status"] == "promoted" and row["promoted_commit"] == res.promoted_commit
    finally:
        await conn.close()


@_needs_git
async def test_git_promote_recovers_from_a_dirty_clone(tmp_path):
    # The live-found bug: the managed clone had a dirty working tree (leftover
    # deletions + untracked scaffolding), so EVERY cherry-pick refused with
    # "local changes would be overwritten" and promotes silently no-op'd. The
    # promote must clean the disposable clone first and then apply cleanly.
    repo, cand_sha = _repo_with_candidate(tmp_path)
    p = pathlib.Path(repo)
    # dirty the main tree the two ways we saw live: a tracked deletion + untracked cruft
    (p / "app.py").unlink()                      # tracked file deleted (uncommitted)
    (p / "_verify.py").write_text("# agent scaffolding cruft\n")  # untracked
    new_sha = await promote.git_promote(repo, cand_sha)
    assert new_sha                                # it applied instead of raising
    text = (p / "app.py").read_text()
    assert "candidate" in text                   # the candidate change landed
    # tree is clean afterwards (the promote committed; nothing left dangling)
    status = subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                            capture_output=True, text=True).stdout.strip()
    assert status == ""


@_needs_git
async def test_propose_leaves_gated_no_git_change(tmp_path):
    repo, cand_sha = _repo_with_candidate(tmp_path)
    before = (pathlib.Path(repo) / "app.py").read_text()
    conn = await _db()
    try:
        await store.create_attempt(conn, attempt_id="att1", user_id="u1", objective="x")
        # human_required → never auto, even opted in
        res = await promote.promote_attempt(
            conn=conn, repo_dir=repo, attempt_id="att1", user_id="u1",
            candidate_sha=cand_sha, verdict=_verdict(V.TIER_HUMAN_REQUIRED), surface="frontend",
            autonomy_level=promote.AUTONOMY_AUTO_VERIFIED,
        )
        assert res.promoted is False
        assert (pathlib.Path(repo) / "app.py").read_text() == before  # live tree untouched
    finally:
        await conn.close()


@_needs_git
async def test_revert_attempt_restores_and_keeps_history(tmp_path):
    repo, cand_sha = _repo_with_candidate(tmp_path)
    conn = await _db()
    try:
        await store.create_attempt(conn, attempt_id="att1", user_id="u1", objective="x")
        res = await promote.promote_attempt(
            conn=conn, repo_dir=repo, attempt_id="att1", user_id="u1",
            candidate_sha=cand_sha, verdict=_verdict(V.TIER_VERIFIED), surface="backend",
            autonomy_level=promote.AUTONOMY_AUTO_VERIFIED,
        )
        revert_sha = await promote.revert_attempt(
            conn=conn, repo_dir=repo, attempt_id="att1", user_id="u1",
            promoted_commit=res.promoted_commit, reason="regressed",
        )
        assert revert_sha
        # code restored…
        assert "v = 1" in (pathlib.Path(repo) / "app.py").read_text()
        row = await store.get_attempt(conn, attempt_id="att1", user_id="u1")
        assert row["status"] == "rolled_back"
        # …but the promoted commit is still in history (lesson never erased)
        log = subprocess.run(["git", "-C", repo, "log", "--oneline"],
                             capture_output=True, text=True).stdout
        assert "candidate edit" in log and "Revert" in log
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# rollback floor (L1 ref + L2 snapshot/counter)
# ---------------------------------------------------------------------------

def test_boot_counter_climbs_and_resets(tmp_path):
    data = str(tmp_path / "data")
    assert rollback.boot_attempts(data) == 0
    assert rollback.record_boot_attempt(data) == 1
    assert rollback.record_boot_attempt(data) == 2
    assert rollback.should_rollback(data, threshold=3) is False
    rollback.record_boot_attempt(data)
    assert rollback.should_rollback(data, threshold=3) is True
    # a healthy startup resets the counter — so it means CONSECUTIVE failed boots
    rollback.mark_boot_healthy(data, ref="abc123")
    assert rollback.boot_attempts(data) == 0
    assert rollback.should_rollback(data, threshold=3) is False
    assert rollback.read_last_good_ref(data) == "abc123"
    assert rollback.last_healthy_at(data) > 0


def test_snapshot_and_restore_roundtrip(tmp_path):
    data = str(tmp_path / "data")
    src = tmp_path / "augmentum"
    (src / "sub").mkdir(parents=True)
    (src / "a.py").write_text("good = 1\n")
    (src / "sub" / "b.py").write_text("good = 2\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "junk.pyc").write_text("nope")

    assert rollback.has_snapshot(data) is False
    assert rollback.snapshot_tree(str(src), data) is True
    assert rollback.has_snapshot(data) is True

    # the live tree gets corrupted by a bad edit…
    (src / "a.py").write_text("broken(\n")
    # …the parachute restores known-good
    assert rollback.restore_tree(data, str(src)) is True
    assert (src / "a.py").read_text() == "good = 1\n"
    assert (src / "sub" / "b.py").read_text() == "good = 2\n"
    # __pycache__ is not snapshotted
    assert not (pathlib.Path(rollback._snapshot_dir(data)) / "__pycache__").exists()


def test_restore_without_snapshot_is_noop(tmp_path):
    data = str(tmp_path / "data")
    dst = tmp_path / "augmentum"
    dst.mkdir()
    assert rollback.restore_tree(data, str(dst)) is False


def test_snapshot_missing_source_is_safe(tmp_path):
    data = str(tmp_path / "data")
    assert rollback.snapshot_tree(str(tmp_path / "nope"), data) is False
