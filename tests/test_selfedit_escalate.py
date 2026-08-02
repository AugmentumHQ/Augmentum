"""Escalation-ladder tests — local groundwork → frontier, context carried up.

The end-to-end tests run against a REAL temp git repo with fake per-rung drivers
(only the agent is swapped), so the whole climb — worktree per rung, verify,
archive, the stop-at-first-gated rule, and the prior-context handoff — is genuine.
Boot-smoke is the one injected oracle (the temp repo isn't Augmentum).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import aiosqlite
import pytest

from augmentum.selfedit import bootsmoke as B
from augmentum.selfedit import escalate as E
from augmentum.selfedit import orchestrator as O
from augmentum.selfedit import store

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
    await _ensure_columns(conn)  # mirror the live growth-DB open (post-288 columns)
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


async def _boot_ok(_dir):
    return B.BootResult(ok=True, failures=[])


def _noop_driver(calls: list) -> O.EditDriver:
    async def _drive(req: O.EditRequest) -> O.EditResult:
        calls.append(("noop", req.prior_context))
        return O.EditResult(ok=True, run_id="noop-run", final_text="explored, no edit")
    return _drive


def _writer_driver(filename: str, content: str, calls: list) -> O.EditDriver:
    async def _drive(req: O.EditRequest) -> O.EditResult:
        calls.append(("writer", req.prior_context))
        (pathlib.Path(req.candidate.path) / filename).write_text(content)
        return O.EditResult(ok=True, run_id="writer-run", final_text="edited")
    return _drive


# --- summarizer --------------------------------------------------------------

async def test_summarize_handoff_builds_brief(monkeypatch):
    async def fake_get_run(conn, *, run_id, user_id, include_raw=False):
        return {"events": [
            {"kind": "message", "text": "I searched for .foo to confirm it is unused"},
            {"kind": "tool_call", "tool": "search", "path": ""},
            {"kind": "file_change", "tool": "edit_file", "path": "a.css"},
        ]}
    monkeypatch.setattr(E.run_store, "get_run", fake_get_run)
    brief = await E.summarize_run_for_handoff(
        conn=None, run_id="r", user_id="u", label="Qwen-27B",
        status="rejected", lesson="capped at the step budget")
    assert "Qwen-27B" in brief
    assert "capped at the step budget" in brief
    assert "I searched for .foo" in brief
    assert "search" in brief and "a.css" in brief
    assert "VERIFY" in brief  # trust-but-verify, not blind-trust the failed rung's notes


async def test_summarize_handoff_carries_diff(monkeypatch):
    async def fake_get_run(conn, *, run_id, user_id, include_raw=False):
        return {"events": [{"kind": "message", "text": "removed .foo"}]}
    monkeypatch.setattr(E.run_store, "get_run", fake_get_run)
    diff = "--- a/ui/styles/x.css\n+++ b/ui/styles/x.css\n@@ -1,3 +1,0 @@\n-.foo { color: red; }"
    brief = await E.summarize_run_for_handoff(
        conn=None, run_id="r", user_id="u", label="local", status="rejected",
        lesson="verify failed", diff=diff)
    assert "```diff" in brief and ".foo { color: red; }" in brief  # actual patch carried
    assert "DISCARD" in brief  # told to drop inappropriate bits


async def test_summarize_handoff_survives_missing_run(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("no such run")
    monkeypatch.setattr(E.run_store, "get_run", boom)
    brief = await E.summarize_run_for_handoff(
        conn=None, run_id="x", user_id="u", label="local", status="failed", lesson="oops")
    assert "PRIOR ATTEMPT" in brief and "local" in brief  # head still rendered


# --- build_ladder ------------------------------------------------------------

async def test_build_ladder_skips_frontier_when_not_allowed():
    rungs = await E.build_ladder(
        None, None, [E.RungSpec(model="deepseek", frontier=True)],
        allow_frontier=False)
    assert rungs == []  # cost-gated rung never built


async def test_build_ladder_drops_unavailable_rung():
    # native engine with no registry / no native_loop → driver is None → dropped.
    rungs = await E.build_ladder(
        None, None, [E.RungSpec(model="local-x")], allow_frontier=True)
    assert rungs == []


# --- the climb ---------------------------------------------------------------

@_needs_git
async def test_escalation_stops_at_first_gated(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    calls: list = []
    try:
        rungs = [
            E.LadderRung("local", _writer_driver("helper.py", "def h():\n    return 1\n", calls)),
            E.LadderRung("frontier", _noop_driver(calls), frontier=True),
        ]
        out = await E.run_self_edit_escalating(
            repo_dir=repo, objective="add a helper", user_id="u1", conn=conn,
            rungs=rungs, worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_ok)
        assert out is not None and out.status == "gated"
        assert [c[0] for c in calls] == ["writer"]  # frontier rung never ran
    finally:
        await conn.close()


@_needs_git
async def test_escalation_climbs_and_carries_context(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    calls: list = []
    try:
        rungs = [
            E.LadderRung("local", _noop_driver(calls)),
            E.LadderRung("frontier", _writer_driver("fix.py", "y = 2\n", calls), frontier=True),
        ]
        out = await E.run_self_edit_escalating(
            repo_dir=repo, objective="make the change", user_id="u1", conn=conn,
            rungs=rungs, worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_ok)
        assert out is not None and out.status == "gated"
        assert [c[0] for c in calls] == ["noop", "writer"]   # climbed
        assert calls[0][1] == ""                              # local got no prior context
        assert "PRIOR ATTEMPT" in calls[1][1]                 # frontier got the handoff
        assert "local" in calls[1][1]
        # BOTH rungs archived (accountability — never pruned)
        rows = await store.list_attempts(conn, user_id="u1")
        assert len(rows) == 2
    finally:
        await conn.close()


@_needs_git
async def test_start_index_skips_cheap_rung(tmp_path):
    # the skill-graph routing hint: a failure-prone region starts higher, so the
    # cheap rung never runs — the stronger one lands it directly.
    repo = _git_repo(tmp_path)
    conn = await _db()
    calls: list = []
    try:
        rungs = [
            E.LadderRung("cheap", _noop_driver(calls)),
            E.LadderRung("stronger", _writer_driver("fix.py", "z = 3\n", calls)),
        ]
        out = await E.run_self_edit_escalating(
            repo_dir=repo, objective="x", user_id="u1", conn=conn, rungs=rungs,
            start_index=1, worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_ok)
        assert out is not None and out.status == "gated"
        assert [c[0] for c in calls] == ["writer"]   # cheap rung skipped entirely
        # only the rung that actually ran is archived
        rows = await store.list_attempts(conn, user_id="u1")
        assert len(rows) == 1
    finally:
        await conn.close()


@_needs_git
async def test_start_index_clamped_so_top_rung_always_runs(tmp_path):
    # an over-large hint can't skip the whole ladder — the last rung still runs.
    repo = _git_repo(tmp_path)
    conn = await _db()
    calls: list = []
    try:
        rungs = [E.LadderRung("a", _noop_driver(calls)),
                 E.LadderRung("b", _noop_driver(calls))]
        out = await E.run_self_edit_escalating(
            repo_dir=repo, objective="x", user_id="u1", conn=conn, rungs=rungs,
            start_index=99, worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_ok)
        assert out is not None
        assert [c[0] for c in calls] == ["noop"]   # only the last rung ran
    finally:
        await conn.close()


@_needs_git
async def test_escalation_all_fail_returns_last(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()
    calls: list = []
    try:
        rungs = [E.LadderRung("local", _noop_driver(calls)),
                 E.LadderRung("stronger", _noop_driver(calls))]
        out = await E.run_self_edit_escalating(
            repo_dir=repo, objective="x", user_id="u1", conn=conn,
            rungs=rungs, worktrees_dir=str(tmp_path / "wt"), boot_runner=_boot_ok)
        assert out is not None and out.status == "rejected"  # last attempt
        assert [c[0] for c in calls] == ["noop", "noop"]     # tried every rung
    finally:
        await conn.close()
