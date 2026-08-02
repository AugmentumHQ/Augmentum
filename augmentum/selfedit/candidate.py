"""Candidate isolation for self-editing — a git worktree the agent edits safely.

The agent never touches the live working tree. Instead each self-edit gets a
*candidate*: a fresh ``git worktree`` on a new branch off a base commit. The
agent edits there, the fitness gate validates there, and only a passing
candidate is promoted. A rejected candidate is removed without ever having
touched the running code.

Worktrees share the repo's ``.git`` but have an independent working directory
and branch, so edits land on the candidate branch — the main branch and working
tree are untouched throughout. (The grounding dig found no worktree support in
the codebase; this adds it.)

All operations are async subprocess calls to ``git``; none of them run
``git stash`` (which is hook-blocked repo-wide anyway).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass


class GitError(RuntimeError):
    """A git command exited non-zero."""


@dataclass
class Candidate:
    name: str          # caller-chosen id (e.g. the self_edit_attempts.id)
    path: str          # the worktree working directory
    branch: str        # the candidate branch (selfedit/<name>)
    base_ref: str      # what the caller asked to branch from
    base_sha: str      # the resolved commit it actually branched from


async def _git(repo_dir: str, *args: str, timeout: float = 120.0) -> tuple[int, str]:
    """Run ``git -C <repo_dir> <args>``; return (exit_code, combined_output).

    ``-c safe.directory=*`` is prepended for every call: in dev-bind the repo is
    a host-owned mount (``/host-augmentum-src``) while the app process runs as a
    different uid, so bare git refuses with "detected dubious ownership". This
    trusts the dir for THIS invocation only (no global config write); candidate
    worktrees under /tmp are process-owned and unaffected either way."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "-c", "safe.directory=*", "-C", repo_dir, *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
    except Exception as exc:  # noqa: BLE001
        return 127, f"could not launch git: {exc!r}"
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        return 124, "git timed out"
    return (proc.returncode or 0), (out or b"").decode("utf-8", errors="replace").strip()


async def _git_ok(repo_dir: str, *args: str, timeout: float = 120.0) -> str:
    code, out = await _git(repo_dir, *args, timeout=timeout)
    if code != 0:
        raise GitError(f"git {' '.join(args)} failed ({code}): {out[-600:]}")
    return out


def default_worktrees_dir() -> str:
    """Where candidate worktrees live by default — OUTSIDE the repo tree (so they
    never appear in the live working tree or get committed)."""
    return os.path.join(tempfile.gettempdir(), "augmentum-selfedit")


def _git_dir_writable(repo_dir: str) -> bool:
    """True iff we can actually WRITE into ``repo_dir/.git`` — probed by a real
    write (not ``os.access``, which root bypasses even on a read-only mount)."""
    probe = os.path.join(repo_dir, ".git", ".selfedit_wprobe")
    try:
        with open(probe, "w") as f:
            f.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


async def _ensure_safe_directory(*dirs: str) -> None:
    """Add each dir to the app user's GLOBAL git ``safe.directory`` (idempotent).

    The source repo is a foreign-owned (root) bind mount while the app runs as a
    non-root uid, so git's dubious-ownership guard blocks clone/fetch/worktree.
    The ``*`` wildcard needs git ≥ 2.35.2; the bundled git is 2.34.1, and ``-c``
    on the command line doesn't cover the clone-source check — but a GLOBAL
    exact-path entry (what git itself recommends) is honored everywhere, including
    the audit subprocess that later runs git inside the worktree."""
    existing = ""
    with contextlib.suppress(Exception):
        proc = await asyncio.create_subprocess_exec(
            "git", "config", "--global", "--get-all", "safe.directory",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await proc.communicate()
        existing = (out or b"").decode("utf-8", "replace")
    have = set(existing.split("\n"))
    for d in dirs:
        if d and d not in have:
            with contextlib.suppress(Exception):
                proc = await asyncio.create_subprocess_exec(
                    "git", "config", "--global", "--add", "safe.directory", d,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await proc.communicate()


async def prepare_writable_repo(source_dir: str, work_dir: str, *,
                                timeout: float = 600.0) -> str:
    """Return a repo dir with a WRITABLE ``.git`` for candidate worktrees.

    Dev-bind mounts the source repo read-only (``/host-augmentum-src``) — git can
    read it but ``git worktree add`` needs to write a branch ref + worktree
    metadata into ``.git``. So when the source is read-only we maintain a
    ``git clone --shared --no-checkout`` at ``work_dir`` (under the writable
    ``/data`` volume): objects ALTERNATE to the ro source (no copy, no drift on the
    sacred object store), while refs/worktrees live in the writable clone. The
    agent's candidate commits land as objects in the clone, never touching the
    source. HEAD is refreshed from the source each call so candidates branch off
    the latest commit. If the source's ``.git`` is already writable, it's used
    directly (no clone)."""
    if _git_dir_writable(source_dir):
        return source_dir
    if not os.path.isdir(os.path.join(source_dir, ".git")):
        return source_dir  # not a git repo; caller falls back to dry-run
    # Trust the foreign-owned source so clone/fetch/worktree/audit-git all work.
    await _ensure_safe_directory(source_dir, os.path.join(source_dir, ".git"))
    # A pre-existing clone we can't write into (e.g. left by a root shell, while
    # the app runs as a non-root uid) is useless — rebuild it as OUR uid.
    if os.path.isdir(os.path.join(work_dir, ".git")) and not _git_dir_writable(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
    if not os.path.isdir(os.path.join(work_dir, ".git")):
        os.makedirs(os.path.dirname(work_dir) or ".", exist_ok=True)
        try:
            # The source mount is foreign-owned (root) while we run as a non-root
            # uid → git's dubious-ownership guard. Trust the source EXPLICITLY by
            # path: the "*" wildcard only works in git ≥ 2.35.2, but exact-path
            # safe.directory is honored everywhere (the bundled git is 2.34.1).
            proc = await asyncio.create_subprocess_exec(
                "git", "-c", f"safe.directory={source_dir}",
                "-c", f"safe.directory={source_dir}/.git",
                "-c", "safe.directory=*", "clone", "--shared",
                "--no-checkout", source_dir, work_dir,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except (OSError, TimeoutError) as exc:
            raise GitError(f"selfedit clone failed: {exc!r}") from exc
        if (proc.returncode or 0) != 0:
            raise GitError(f"selfedit clone failed: "
                           f"{(out or b'').decode('utf-8', 'replace')[-400:]}")
    else:  # refresh the clone's HEAD to the source's current commit
        await _git(work_dir, "fetch", "--quiet", source_dir, "HEAD")
        await _git(work_dir, "update-ref", "HEAD", "FETCH_HEAD")
    return work_dir


async def create_candidate(
    repo_dir: str, *, name: str, base_ref: str = "HEAD",
    worktrees_dir: str | None = None,
) -> Candidate:
    """Create a worktree on a new branch ``selfedit/<name>`` off ``base_ref``."""
    worktrees_dir = worktrees_dir or default_worktrees_dir()
    os.makedirs(worktrees_dir, exist_ok=True)
    branch = f"selfedit/{name}"
    path = os.path.join(worktrees_dir, name)
    from augmentum.utils.logging import get_logger
    get_logger(__name__).info("selfedit_create_candidate", repo_dir=repo_dir,
                              git_writable=_git_dir_writable(repo_dir),
                              worktree_path=path)
    base_sha = await _git_ok(repo_dir, "rev-parse", base_ref)
    await _git_ok(repo_dir, "worktree", "add", "-b", branch, path, base_sha)
    return Candidate(name=name, path=path, branch=branch, base_ref=base_ref, base_sha=base_sha)


async def candidate_changes(candidate: Candidate) -> list[str]:
    """Paths changed in the candidate worktree (porcelain), relative to repo root."""
    code, out = await _git(candidate.path, "status", "--porcelain")
    if code != 0 or not out:
        return []
    return [line[3:] for line in out.splitlines() if line.strip()]


# Throwaway test-runner scaffolding the editing agent writes at the repo root to
# run its own checks. It must never enter the candidate commit — otherwise a
# promote carries this junk into the real source tree (observed live: promoted
# changes dragging in _verify.py/_run_test.py). A genuine change never adds these
# root-level helper names, so unstaging them is safe.
_SCAFFOLD_FILES = (
    "verify.py", "_verify.py", "run_verify.py", "_run_verify.py",
    "run_test.sh", "_run_test.sh", "run_test.py", "_run_test.py",
)


async def commit_candidate(candidate: Candidate, message: str, *,
                           amend: bool = False) -> str:
    """Stage all edits and commit them on the candidate branch. Returns the SHA.
    Raises if there is nothing to commit (caller can treat as a no-op edit).

    ``amend`` folds the new edits into the branch's EXISTING single commit
    (``commit --amend``) instead of adding a second one. The self-heal loop needs
    this: a promote cherry-picks the branch TIP only, so a heal must keep the
    change as ONE commit or the initial edit would be lost on promote."""
    await _git_ok(candidate.path, "add", "-A")
    # Drop agent scaffolding from the staged set so it never lands on promote.
    for name in _SCAFFOLD_FILES:
        if os.path.exists(os.path.join(candidate.path, name)):
            with contextlib.suppress(Exception):
                await _git(candidate.path, "reset", "-q", "--", name)
    # Pass an author identity inline: the app runs as a non-root uid with no git
    # identity in the /data clone, so a bare commit dies with "Author identity
    # unknown". Surgical (-c) — no global config write.
    commit_args = ["commit", "-m", message]
    if amend:
        commit_args = ["commit", "--amend", "--no-edit"]
    code, out = await _git(candidate.path,
                           "-c", "user.email=selfedit@augmentum.local",
                           "-c", "user.name=Augmentum Self-Edit",
                           *commit_args)
    if code != 0:
        raise GitError(f"candidate commit failed: {out[-400:]}")
    return await _git_ok(candidate.path, "rev-parse", "HEAD")


async def candidate_diff(candidate: Candidate, *, max_chars: int = 8000) -> str:
    """The committed patch of the candidate (base → HEAD) — what the agent actually
    changed. Carried up the escalation ladder so a stronger model repairs the real
    diff instead of re-deriving it from prose notes. Truncated for prompt budgets."""
    code, out = await _git(candidate.path, "diff", candidate.base_sha, "HEAD")
    if code != 0 or not out.strip():
        code, out = await _git(candidate.path, "diff")  # fall back to uncommitted
    if code != 0:
        return ""
    return out[:max_chars] + ("\n…(diff truncated)" if len(out) > max_chars else "")


async def remove_candidate(
    repo_dir: str, candidate: Candidate, *, delete_branch: bool = True,
) -> None:
    """Tear down a candidate: remove the worktree (force, since it may carry
    uncommitted edits) and optionally delete its branch. Best-effort."""
    await _git(repo_dir, "worktree", "remove", "--force", candidate.path)
    # Prune any stale worktree metadata even if the dir was already gone.
    await _git(repo_dir, "worktree", "prune")
    if delete_branch:
        await _git(repo_dir, "branch", "-D", candidate.branch)


async def list_candidates(repo_dir: str) -> list[dict]:
    """All selfedit/* worktrees currently registered."""
    code, out = await _git(repo_dir, "worktree", "list", "--porcelain")
    if code != 0:
        return []
    entries: list[dict] = []
    cur: dict = {}
    for line in out.splitlines():
        if line.startswith("worktree "):
            cur = {"path": line[len("worktree "):]}
        elif line.startswith("branch "):
            cur["branch"] = line[len("branch "):]
        elif line == "" and cur:
            entries.append(cur)
            cur = {}
    if cur:
        entries.append(cur)
    return [e for e in entries if "selfedit/" in e.get("branch", "")]
