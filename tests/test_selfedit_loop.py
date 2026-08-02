"""Subprocess command_runner + debt-paydown loop tests.

The runner is the real dev-bind execution path (no container): it runs an agent
argv with cwd = the candidate worktree and streams stdout. The loop ties audit →
triage → orchestrated attempts; both are exercised here with a real temp git repo,
a synthetic audit, and a fake agent.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import aiosqlite
import pytest

from augmentum.selfedit import loop as L
from augmentum.selfedit import runners, store
from augmentum.selfedit.candidate import Candidate
from augmentum.selfedit.orchestrator import EditRequest

_MIGRATION = (
    pathlib.Path(__file__).resolve().parent.parent
    / "augmentum" / "state" / "migrations" / "288_self_edit_attempts.sql"
)
_HAS_GIT = shutil.which("git") is not None
_needs_git = pytest.mark.skipif(not _HAS_GIT, reason="git not available")

# A synthetic audit with mechanical + structural debt.
_AUDIT = {
    "score": 80.0,
    "metrics": {
        "code_quality": {"silent_catches": 5, "dead_css": 2, "missing_css": 99},
        "coverage": {"coverage_gaps": 3},
        "dead_code": {"orphaned_endpoints": 10},
    },
    "regressions": [], "smoke_errors": [], "tool_failures": [],
}


# ---------------------------------------------------------------------------
# subprocess_command_runner — the real dev-bind path
# ---------------------------------------------------------------------------

async def test_subprocess_runner_streams_stdout(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    cand = Candidate(name="a", path=str(wt), branch="selfedit/a", base_ref="HEAD", base_sha="x")
    chunks: list[bytes] = []
    async def on_chunk(b: bytes) -> None:
        chunks.append(b)
    # a fake "agent": prints two lines, proving streaming + cwd plumbing
    argv = [sys.executable, "-c", "print('line-1'); print('line-2')"]
    runner = runners.make_subprocess_runner()
    await runner(request=EditRequest(cand, "obj", "a", "u1"), argv=argv,
                 on_chunk=on_chunk, environment={})
    out = b"".join(chunks).decode()
    assert "line-1" in out and "line-2" in out


async def test_subprocess_runner_runs_in_candidate_cwd(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "marker.txt").write_text("here")
    cand = Candidate(name="a", path=str(wt), branch="selfedit/a", base_ref="HEAD", base_sha="x")
    chunks: list[bytes] = []
    async def on_chunk(b: bytes) -> None:
        chunks.append(b)
    argv = [sys.executable, "-c", "import os; print(os.listdir('.'))"]
    await runners.subprocess_command_runner(
        request=EditRequest(cand, "o", "a", "u1"), argv=argv, on_chunk=on_chunk, environment={})
    assert "marker.txt" in b"".join(chunks).decode()


async def test_subprocess_runner_missing_binary_raises(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    cand = Candidate(name="a", path=str(wt), branch="selfedit/a", base_ref="HEAD", base_sha="x")
    async def on_chunk(_b):  # pragma: no cover - never called
        pass
    runner = runners.make_subprocess_runner()
    with pytest.raises(runners.RunnerError):
        await runner(request=EditRequest(cand, "o", "a", "u1"),
                     argv=["definitely-not-a-real-binary-xyz"], on_chunk=on_chunk, environment={})


async def test_subprocess_runner_passes_environment(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    cand = Candidate(name="a", path=str(wt), branch="selfedit/a", base_ref="HEAD", base_sha="x")
    chunks: list[bytes] = []
    async def on_chunk(b):
        chunks.append(b)
    argv = [sys.executable, "-c", "import os; print(os.environ.get('SELFEDIT_TOK'))"]
    await runners.subprocess_command_runner(
        request=EditRequest(cand, "o", "a", "u1"), argv=argv, on_chunk=on_chunk,
        environment={"SELFEDIT_TOK": "secret123"})
    assert "secret123" in b"".join(chunks).decode()


# ---------------------------------------------------------------------------
# run_debt_loop
# ---------------------------------------------------------------------------

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


async def _audit_runner(_dir):
    return json.dumps(_AUDIT)


async def test_loop_dry_run_returns_plan_without_editing():
    conn = await _db()
    try:
        rep = await L.run_debt_loop(
            repo_dir=".", user_id="u1", conn=conn,
            live_audit_runner=_audit_runner, dry_run=True, max_attempts=2,
        )
        assert rep.dry_run is True and rep.attempted == []
        # mechanical targets surfaced; missing_css/orphaned are structural, not here
        metrics = {(t.scanner, t.metric) for t in rep.targets}
        assert ("code_quality", "silent_catches") in metrics
        assert ("code_quality", "missing_css") not in metrics
        assert any(t.metric == "missing_css" for t in rep.structural)
        assert rep.baseline_score == 80.0
    finally:
        await conn.close()


@_needs_git
async def test_loop_attempts_mechanical_targets(tmp_path):
    repo = _git_repo(tmp_path)
    conn = await _db()

    async def boot_ok(_dir):
        from augmentum.selfedit.bootsmoke import BootResult
        return BootResult(ok=True, failures=[])

    async def fake_driver(req: EditRequest):
        # the "agent" edits one file in the candidate
        (pathlib.Path(req.candidate.path) / "fix.py").write_text("# fixed\n")
        from augmentum.selfedit.orchestrator import EditResult
        return EditResult(ok=True, run_id="r", final_text="fixed")

    try:
        rep = await L.run_debt_loop(
            repo_dir=repo, user_id="u1", conn=conn, driver=fake_driver,
            live_audit_runner=_audit_runner, candidate_audit_runner=_audit_runner,
            boot_runner=boot_ok, worktrees_dir=str(tmp_path / "wt"), max_attempts=2,
        )
        assert len(rep.attempted) == 2  # capped at max_attempts
        assert rep.deferred >= 1        # more mechanical targets than the cap
        # each attempt landed in the archive
        for o in rep.attempted:
            row = await store.get_attempt(conn, attempt_id=o.attempt_id, user_id="u1")
            assert row is not None and row["status"] in ("gated", "rejected")
    finally:
        await conn.close()
