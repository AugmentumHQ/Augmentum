"""Candidate-isolation tests — git worktree lifecycle on a throwaway repo."""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from augmentum.selfedit import candidate as C


def _git(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True,
                   capture_output=True, text=True)


def _make_repo(d: str) -> str:
    repo = os.path.join(d, "repo")
    os.makedirs(repo)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.test")
    _git(repo, "config", "user.name", "Test")
    with open(os.path.join(repo, "a.txt"), "w") as f:
        f.write("base\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)
async def test_candidate_full_lifecycle():
    with tempfile.TemporaryDirectory() as d:
        repo = _make_repo(d)
        wt = os.path.join(d, "wt")

        cand = await C.create_candidate(repo, name="t1", worktrees_dir=wt)
        assert os.path.isdir(cand.path)
        assert cand.branch == "selfedit/t1"
        assert len(cand.base_sha) >= 7

        # the live tree is untouched: editing in the candidate doesn't change repo/a.txt
        with open(os.path.join(cand.path, "b.txt"), "w") as f:
            f.write("candidate edit\n")
        changes = await C.candidate_changes(cand)
        assert "b.txt" in changes
        assert not os.path.exists(os.path.join(repo, "b.txt"))  # main worktree clean

        sha = await C.commit_candidate(cand, "candidate work")
        assert len(sha) >= 7

        listed = await C.list_candidates(repo)
        assert any(e.get("branch", "").endswith("selfedit/t1") for e in listed)

        await C.remove_candidate(repo, cand)
        assert not os.path.isdir(cand.path)
        listed_after = await C.list_candidates(repo)
        assert not any(e.get("branch", "").endswith("selfedit/t1") for e in listed_after)


async def test_create_candidate_bad_repo_raises():
    with tempfile.TemporaryDirectory() as d, pytest.raises(C.GitError):
        await C.create_candidate(d, name="x", worktrees_dir=os.path.join(d, "wt"))


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True).returncode != 0,
    reason="git not available",
)
async def test_commit_excludes_agent_scaffolding():
    # The editing agent leaves throwaway test-runner scaffolding at the repo root
    # (_verify.py/_run_test.py/...). It must NOT enter the candidate commit — else
    # a promote drags junk into the real tree (observed live 2026-07-02).
    with tempfile.TemporaryDirectory() as d:
        repo = _make_repo(d)
        wt = os.path.join(d, "wt")
        cand = await C.create_candidate(repo, name="scaf", worktrees_dir=wt)
        # a genuine edit + typical agent scaffolding
        with open(os.path.join(cand.path, "real.py"), "w") as f:
            f.write("x = 1\n")
        for junk in ("_verify.py", "_run_test.py", "verify.py", "run_test.sh"):
            with open(os.path.join(cand.path, junk), "w") as f:
                f.write("# throwaway\n")
        await C.commit_candidate(cand, "work + scaffolding")
        # the committed tree carries the real edit but none of the scaffolding
        files = subprocess.run(["git", "-C", cand.path, "show", "--name-only",
                                "--pretty=format:", "HEAD"],
                               capture_output=True, text=True).stdout.split()
        assert "real.py" in files
        for junk in ("_verify.py", "_run_test.py", "verify.py", "run_test.sh"):
            assert junk not in files, f"{junk} leaked into the commit"
