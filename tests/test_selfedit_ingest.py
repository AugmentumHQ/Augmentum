"""Ingest-all-work tests — git history + coder turns become archive rows.

Load-bearing honesty cases: a reverted commit lands as ``rolled_back`` (a real
mistake, witnessed) and the revert commit itself is skipped; re-runs are
idempotent; ambiguous coder outcomes get a verdict-free status that moves no
activation weight; ingested provenance is damped in the fold, never full-trust.
"""

from __future__ import annotations

import pathlib
import subprocess

import aiosqlite

from augmentum.selfedit import store
from augmentum.selfedit.activation import build_graph, modulation_for_attempt
from augmentum.selfedit.ingest import (
    _surface_for_files,
    ingest_coder_turn,
    ingest_git_history,
)

_MIGRATION = (
    pathlib.Path(__file__).resolve().parent.parent
    / "augmentum" / "state" / "migrations" / "288_self_edit_attempts.sql"
)


async def _db():
    from augmentum.selfedit.growth_db import _ensure_columns
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, description TEXT)"
    )
    await conn.executescript(_MIGRATION.read_text())
    await _ensure_columns(conn)  # mirror the live growth-DB open (source, target)
    await conn.commit()
    return conn


def _git(repo: pathlib.Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _make_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    return repo


def _commit(repo: pathlib.Path, relpath: str, content: str, message: str) -> str:
    p = repo / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


# --- store.ingest_attempt ---------------------------------------------------

async def test_ingest_attempt_roundtrip_and_idempotency():
    conn = await _db()
    try:
        wrote = await store.ingest_attempt(
            conn, attempt_id="git:abc", user_id="u1", objective="fix the panel",
            source="git", status="live", surface="frontend",
            files_changed=["ui/scripts/app.js"], outcome="kept in live history",
            promoted_commit="abc", created_at="2025-01-02 03:04:05",
        )
        assert wrote is True
        again = await store.ingest_attempt(
            conn, attempt_id="git:abc", user_id="u1", objective="OVERWRITE?",
            source="git", status="rolled_back",
        )
        assert again is False  # existing row untouched
        got = await store.get_attempt(conn, attempt_id="git:abc", user_id="u1")
        assert got["status"] == "live" and got["source"] == "git"
        assert got["objective"] == "fix the panel"
        assert got["created_at"] == "2025-01-02 03:04:05"  # true work timestamp
        assert got["files_changed"] == ["ui/scripts/app.js"]
    finally:
        await conn.close()


async def test_engine_rows_default_to_autonomous_source():
    conn = await _db()
    try:
        await store.create_attempt(conn, attempt_id="a1", user_id="u1",
                                   objective="engine's own attempt")
        got = await store.get_attempt(conn, attempt_id="a1", user_id="u1")
        assert got["source"] == "autonomous"
    finally:
        await conn.close()


# --- git history ingestion --------------------------------------------------

async def test_git_history_ingest_with_revert(tmp_path):
    repo = _make_repo(tmp_path)
    sha_a = _commit(repo, "ui/styles/app.css", "body{}", "style the app shell")
    sha_b = _commit(repo, "augmentum/util.py", "x = 1\n", "add util constant")
    _git(repo, "revert", "--no-edit", sha_b)

    conn = await _db()
    try:
        result = await ingest_git_history(conn, repo_dir=str(repo), user_id="u1")
        assert result["ok"] is True
        assert result["ingested"] == 2          # A + B; the revert commit is skipped
        assert result["skipped_reverts"] == 1
        assert result["marked_rolled_back"] == 1

        kept = await store.get_attempt(conn, attempt_id=f"git:{sha_a}", user_id="u1")
        assert kept["status"] == "live" and kept["source"] == "git"
        assert kept["surface"] == "frontend"
        assert kept["objective"] == "style the app shell"
        assert kept["promoted_commit"] == sha_a

        reverted = await store.get_attempt(conn, attempt_id=f"git:{sha_b}", user_id="u1")
        assert reverted["status"] == "rolled_back"  # the mistake, witnessed
        assert reverted["surface"] == "backend"

        # idempotent: a re-run ingests nothing new
        rerun = await ingest_git_history(conn, repo_dir=str(repo), user_id="u1")
        assert rerun["ingested"] == 0 and rerun["existing"] == 2
    finally:
        await conn.close()


async def test_git_ingest_reports_failure_honestly(tmp_path):
    conn = await _db()
    try:
        result = await ingest_git_history(
            conn, repo_dir=str(tmp_path / "not-a-repo"), user_id="u1")
        assert result["ok"] is False and "error" in result
    finally:
        await conn.close()


def test_surface_for_files_vocabulary():
    assert _surface_for_files(["ui/scripts/app.js"]) == "frontend"
    assert _surface_for_files(["augmentum/proxy/server.py"]) == "backend"
    assert _surface_for_files(["augmentum/state/migrations/306_x.sql"]) == "migration"
    assert _surface_for_files(["ui/app.js", "augmentum/x.py"]) == "mixed"
    assert _surface_for_files(["scripts/build.sh"]) == "config"
    assert _surface_for_files([]) == ""


# --- coder turn ingestion ----------------------------------------------------

async def test_coder_turn_statuses_are_honest():
    conn = await _db()
    try:
        wrote = await ingest_coder_turn(
            conn, user_id="u1", turn_id="t1", user_goal="wire the toggle",
            outcome="done", files_edited=[{"path": "src/app.ts", "summary": "edited"}],
            workspace_id="ws1",
        )
        assert wrote is True
        done = await store.get_attempt(conn, attempt_id="coder:t1", user_id="u1")
        assert done["status"] == "live" and done["source"] == "coder"
        assert done["files_changed"] == ["src/app.ts"]

        await ingest_coder_turn(
            conn, user_id="u1", turn_id="t2", user_goal="hard refactor",
            outcome="incomplete", files_edited=["src/a.ts"])
        ambiguous = await store.get_attempt(conn, attempt_id="coder:t2", user_id="u1")
        assert ambiguous["status"] == "ingested"  # no verdict → moves no weights
        assert modulation_for_attempt(ambiguous) == 0.0

        await ingest_coder_turn(
            conn, user_id="u1", turn_id="t3", user_goal="x",
            outcome="stopped (tool errors)", files_edited=["src/a.ts"])
        stopped = await store.get_attempt(conn, attempt_id="coder:t3", user_id="u1")
        assert stopped["status"] == "failed"

        # read-only turns and missing identity are not work units
        assert await ingest_coder_turn(conn, user_id="u1", turn_id="t4",
                                       user_goal="look around", outcome="done",
                                       files_edited=[]) is False
        assert await ingest_coder_turn(conn, user_id="u1", turn_id="",
                                       user_goal="x", outcome="done",
                                       files_edited=["a"]) is False
        # idempotent
        assert await ingest_coder_turn(conn, user_id="u1", turn_id="t1",
                                       user_goal="x", outcome="done",
                                       files_edited=["a"]) is False
    finally:
        await conn.close()


# --- provenance damping in the activation fold -------------------------------

def test_source_weights_damp_ingested_verdicts():
    assert modulation_for_attempt({"status": "promoted"}) == 1.0
    assert modulation_for_attempt({"status": "promoted", "source": "autonomous"}) == 1.0
    assert modulation_for_attempt({"status": "live", "source": "git"}) == 0.6
    assert modulation_for_attempt({"status": "rolled_back", "source": "git"}) == -0.6
    assert modulation_for_attempt({"status": "live", "source": "coder"}) == 0.25
    # unknown provenance is conservative, never silent full-trust
    assert modulation_for_attempt({"status": "live", "source": "mystery"}) == 0.5
    # no verdict → no weight, whatever the source
    assert modulation_for_attempt({"status": "ingested", "source": "coder"}) == 0.0


async def test_load_attempts_carries_source_and_target():
    """The regression guard for the severed-signal bug: graphs built off the
    LIVE archive must fold `target:` atoms (load_attempts previously never
    selected the column, so the debt loop's per-class transfer signal was dead
    on the wired path while passing in dict-based tests)."""
    from augmentum.selfedit.activation import load_attempts
    conn = await _db()
    try:
        await store.create_attempt(
            conn, attempt_id="a1", user_id="u1", objective="fix dead css",
            surface="frontend", target="code_quality.dead_css")
        await store.set_gate(conn, attempt_id="a1", user_id="u1", passed=True,
                             verdict={}, files_changed=["ui/styles/app.css"])
        await store.finalize(conn, attempt_id="a1", user_id="u1", status="promoted")

        rows = await load_attempts(conn, user_id="u1")
        assert rows[0]["target"] == "code_quality.dead_css"
        assert rows[0]["source"] == "autonomous"
        graph = build_graph(rows)
        assert "target:code_quality.dead_css" in graph.activity
    finally:
        await conn.close()
