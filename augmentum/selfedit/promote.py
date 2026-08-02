"""Promotion + revert — taking a gated candidate live, reversibly.

The orchestrator leaves a passing candidate at ``gated`` with its oracle tier in
the verdict; this module decides whether it may go live and applies it.

**The corrected P4 predicate (key on ORACLE TIER, not surface).** A change may
auto-promote iff an oracle confirmed its *intent* (``verified`` = a mechanical
oracle, or ``human_confirmed`` = the user kept it) AND it's reversible — never
just because it "didn't break." Surface (frontend/backend/migration) decides the
ROLLBACK MECHANISM and the post-promote action, NOT whether to auto-ship: a
frontend CSS change is the canonical ``human_required`` case (compiles, health
green, still wrong). A migration is red-tier — human only, full stop.

Promotion is a git cherry-pick of the candidate's commit onto the live branch, so
frontend and backend promote uniformly and the live ``.git`` keeps the full
lineage. Revert is a ``git revert`` (a new commit, history preserved) — the
anti-Westworld rule: rollback restores the code, it never erases the record.

Recovery from a *boot-fatal* promotion is the separate, lower-level concern of
``rollback`` (the entrypoint parachute); this module is the normal, healthy
promote/propose/revert path.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from augmentum.selfedit import store as _store
from augmentum.selfedit.candidate import _git, _git_ok
from augmentum.selfedit.intent import (
    CLASS_AUTHORED_ORACLE,
    SURFACE_FRONTEND,
    SURFACE_MIGRATION,
)
from augmentum.selfedit.verifier import TIER_HUMAN_CONFIRMED, TIER_VERIFIED, Verdict
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Autonomy posture (the setting `selfedit_autonomy_level`).
AUTONOMY_PROPOSE = "propose"            # never auto — every change waits for a human
AUTONOMY_AUTO_VERIFIED = "auto_verified"  # auto-promote oracle-confirmed changes


@dataclass
class PromotionDecision:
    auto: bool
    reason: str


def decide_promotion(verdict: Verdict | None, *, surface: str,
                     autonomy_level: str = AUTONOMY_PROPOSE) -> PromotionDecision:
    """The honest auto-promote predicate. Pure. Auto iff the intent was confirmed
    by an oracle AND the change is reversible AND the operator opted into
    auto-promotion — surface only ever *blocks* (migration), never *grants*."""
    if verdict is None or not verdict.passed:
        return PromotionDecision(False, "no passing verdict")
    if surface == SURFACE_MIGRATION:
        return PromotionDecision(False, "migration is red-tier — human only")
    if verdict.intent_class == CLASS_AUTHORED_ORACLE:
        # The engine authoring its own examiner: even a green mechanical check
        # never auto-ships an oracle — retrodiction agreement is necessary, the
        # human endorsement is the sufficiency, until the class graduates.
        return PromotionDecision(False, "authored-oracle is red-tier — human only")
    if autonomy_level != AUTONOMY_AUTO_VERIFIED:
        return PromotionDecision(False, "autonomy=propose — awaiting a human verdict")
    if verdict.tier in (TIER_VERIFIED, TIER_HUMAN_CONFIRMED):
        return PromotionDecision(True, f"{verdict.tier} + reversible → auto-promote")
    # probable / human_required: green, but intent not objectively confirmed
    return PromotionDecision(False, f"{verdict.tier} — needs a human verdict, not auto")


def restart_needed(surface: str) -> bool:
    """Frontend is served live (hard-refresh picks it up); a backend change needs
    a restart to take effect. The surface's only role post-promote."""
    return surface != SURFACE_FRONTEND


async def git_promote(repo_dir: str, candidate_sha: str) -> str:
    """Cherry-pick the candidate's commit onto the live branch. Returns the new
    HEAD sha. On conflict, aborts cleanly and raises (never leaves a half-applied
    cherry-pick on the live tree).

    The promote target is a MANAGED, disposable clone (``prepare_writable_repo``)
    — the source of truth is the read-only host mount plus the candidate branches,
    never this clone's working tree. Cherry-pick refuses on a dirty tree ("local
    changes would be overwritten"), and ordinary self-edit operations can leave
    the clone dirty (leftover deletions/untracked scaffolding), which was silently
    failing every promote. So we make the working tree match HEAD first — logged,
    never silent, and safe: it touches only the working tree, never a branch or a
    candidate worktree (those live in their own registered directories)."""
    from augmentum.selfedit.candidate import GitError
    dirty_code, dirty_out = await _git(repo_dir, "status", "--porcelain")
    if dirty_code == 0 and dirty_out.strip():
        log.warning("selfedit_promote_cleaning_dirty_tree",
                    n=len(dirty_out.strip().splitlines()),
                    sample=dirty_out.strip().splitlines()[:8])
        # restore tracked changes (the "would be overwritten" blocker) and sweep
        # untracked cruft (e.g. agent scaffolding _verify.py/_run_test.py that
        # would otherwise collide with a candidate that also created them)
        await _git(repo_dir, "reset", "--hard", "HEAD")
        await _git(repo_dir, "clean", "-fd")
    code, out = await _git(repo_dir, "cherry-pick", "-x", candidate_sha)
    if code != 0:
        with contextlib.suppress(Exception):
            await _git(repo_dir, "cherry-pick", "--abort")
        raise GitError(f"promote cherry-pick failed (aborted): {out[-400:]}")
    return await _git_ok(repo_dir, "rev-parse", "HEAD")


async def git_revert(repo_dir: str, sha: str) -> str:
    """Revert a promoted commit with a NEW revert commit (history preserved — the
    lesson is never erased). Returns the new HEAD sha."""
    if await _is_merge(repo_dir, sha):
        await _git_ok(repo_dir, "revert", "--no-edit", "-m", "1", sha)
    else:
        await _git_ok(repo_dir, "revert", "--no-edit", sha)
    return await _git_ok(repo_dir, "rev-parse", "HEAD")


async def _is_merge(repo_dir: str, sha: str) -> bool:
    code, out = await _git(repo_dir, "rev-list", "--parents", "-n", "1", sha)
    return code == 0 and len(out.split()) > 2  # sha + >1 parent


@dataclass
class PromotionResult:
    promoted: bool
    reason: str
    promoted_commit: str = ""
    needs_restart: bool = False


async def promote_attempt(
    *, conn: Any, repo_dir: str, attempt_id: str, user_id: str,
    candidate_sha: str, verdict: Verdict | None, surface: str,
    autonomy_level: str = AUTONOMY_PROPOSE,
) -> PromotionResult:
    """Apply the predicate; on auto, cherry-pick + finalize ``promoted``; else
    leave the attempt at ``gated`` for the human (the proposal queue). Never
    raises — a failed cherry-pick is recorded, not thrown."""
    decision = decide_promotion(verdict, surface=surface, autonomy_level=autonomy_level)
    if not decision.auto:
        log.info("selfedit_promote_proposed", attempt_id=attempt_id, reason=decision.reason)
        return PromotionResult(False, decision.reason)

    try:
        new_sha = await git_promote(repo_dir, candidate_sha)
    except Exception as exc:  # noqa: BLE001 — a failed promote is a recorded outcome
        log.warning("selfedit_promote_failed", attempt_id=attempt_id, error=repr(exc))
        with contextlib.suppress(Exception):
            await _store.finalize(
                conn, attempt_id=attempt_id, user_id=user_id, status="gated",
                outcome="auto-promote failed (conflict) — needs a human",
                lesson=f"cherry-pick onto live failed: {exc!r}",
            )
        return PromotionResult(False, f"promote failed: {exc!r}")

    needs_restart = restart_needed(surface)
    with contextlib.suppress(Exception):
        await _store.finalize(
            conn, attempt_id=attempt_id, user_id=user_id, status="promoted",
            outcome=f"auto-promoted ({decision.reason})", promoted_commit=new_sha,
            lesson="verified change promoted live — last-good can advance",
        )
    log.info("selfedit_promoted", attempt_id=attempt_id, commit=new_sha,
             needs_restart=needs_restart)
    return PromotionResult(True, decision.reason, promoted_commit=new_sha,
                           needs_restart=needs_restart)


async def revert_attempt(
    *, conn: Any, repo_dir: str, attempt_id: str, user_id: str,
    promoted_commit: str, reason: str = "",
) -> str:
    """Revert a promoted attempt (history-preserving) and mark it ``rolled_back``.
    Returns the revert commit sha. The lesson is kept (anti-Westworld)."""
    revert_sha = await git_revert(repo_dir, promoted_commit)
    with contextlib.suppress(Exception):
        await _store.finalize(
            conn, attempt_id=attempt_id, user_id=user_id, status="rolled_back",
            outcome=f"reverted: {reason}" if reason else "reverted",
            lesson="promoted change reverted — code restored, the record kept",
        )
    log.info("selfedit_reverted", attempt_id=attempt_id, revert=revert_sha)
    return revert_sha
