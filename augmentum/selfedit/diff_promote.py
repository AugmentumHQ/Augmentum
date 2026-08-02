"""Apply-diff promotion — land a VERIFIED candidate by applying its file changes
(and removals) onto the live tree, snapshot-first, instead of a git merge.

⚠️ SINGLE-WRITER NOTE (reconciled 2026-06-23): the *wired, user-facing* path that
takes collected self-edits live is ``selfedit/apply.py`` (the "Go live" lane:
collect kept edits in the clone → checkpoint → write to the live tree → restart,
with per-file revert). That is THE live writer. THIS module is the lower-level,
*per-candidate immediate-promote* primitive (apply ONE candidate worktree's diff
straight to live), kept for a future auto-promote-on-verify flow and for
git-less deployments. **Do not wire both as live writers at once** — they'd fight
over the same tree/baseline. They are consistent on the floor: both snapshot via
``rollback.snapshot_tree`` before writing (see ``apply.py::apply_pending``), so the
L2 parachute (``rollback.py`` + the entrypoint) recovers either path. The only
real difference is the SOURCE: this reads a candidate worktree's files on disk;
``apply.py`` reads blobs from the ``--no-checkout`` clone via ``git show``.

The refinement: the workspace is the agent's edit + verify + track-changes
sandbox; promoting to live = **apply the diff** (write the changed files, delete
the removed ones), NOT a 3-way merge. Conflict-free for the serialized self-edit
loop (seed from live → verify in the WS → apply back promptly), and SYMMETRIC with
the file-based rollback floor:

    promote  = apply the candidate's files onto live   (this module / apply.py)
    rollback = restore the /data snapshot onto live     (rollback.py)

No git on the live tree — the one place the app container can't do git anyway. The
snapshot is taken (via ``rollback.snapshot_tree``) BEFORE anything lands, so the
existing L2 parachute restores it.

Pure file operations — testable with temp dirs. ``classify_porcelain`` splits a
``git status --porcelain`` listing into changed-vs-removed; ``compute_changes``
wraps it around a candidate worktree.

CAVEAT (documented, not a bug): apply-diff assumes the candidate was seeded from
the CURRENT live state and applied promptly (baseline == live — the serialized
single-writer loop). If live drifts in between, re-seed/re-verify; this does not do
3-way conflict resolution (that's git_promote's job for git-tree deployments).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any

from augmentum.selfedit import rollback
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _safe_rel(rel: str) -> bool:
    """Reject path-traversal / absolute paths so an apply can't escape the tree."""
    n = os.path.normpath(rel)
    return not (os.path.isabs(n) or n == ".." or n.startswith(".." + os.sep))


def classify_porcelain(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split ``git status --porcelain`` lines into (changed_or_added, removed).
    Pure. Handles rename arrows (``old -> new`` → the new path is 'changed')."""
    changed: list[str] = []
    removed: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        xy, path = line[:2], line[3:].strip()
        if "->" in path:                       # rename: take the destination
            path = path.split("->")[-1].strip()
        path = path.strip('"')
        if not _safe_rel(path):
            log.warning("selfedit_apply_unsafe_path", path=path)
            continue
        (removed if "D" in xy else changed).append(path)
    return changed, removed


@dataclass
class ApplyResult:
    applied: bool
    written: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    snapshotted: bool = False
    detail: str = ""

    def to_dict(self) -> dict:
        return {"applied": self.applied, "written": len(self.written),
                "removed": len(self.removed), "snapshotted": self.snapshotted,
                "detail": self.detail}


def apply_diff(candidate_dir: str, live_dir: str, *, changed: list[str],
               removed: list[str], data_dir: str = "", snapshot: bool = True) -> ApplyResult:
    """Apply the candidate's changed/removed files onto ``live_dir`` (the
    snapshot/apply scope, e.g. the ``augmentum`` package). Snapshots ``live_dir``
    first (the parachute) when ``data_dir`` is given. Paths are relative to both
    roots. Never raises — a failed apply is recorded so the caller can restore."""
    snapped = False
    if snapshot and data_dir:
        snapped = rollback.snapshot_tree(live_dir, data_dir)

    written: list[str] = []
    removed_done: list[str] = []
    try:
        for rel in changed:
            if not _safe_rel(rel):
                continue
            src = os.path.join(candidate_dir, rel)
            if not os.path.isfile(src):        # dir entries / vanished files — skip safely
                continue
            dst = os.path.join(live_dir, rel)
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.copy2(src, dst)
            written.append(rel)
        for rel in removed:
            if not _safe_rel(rel):
                continue
            dst = os.path.join(live_dir, rel)
            if os.path.isfile(dst):
                os.remove(dst)
                removed_done.append(rel)
    except Exception as exc:  # noqa: BLE001 — a failed apply is recorded, snapshot lets caller restore
        log.warning("selfedit_apply_diff_failed", error=repr(exc))
        return ApplyResult(False, written, removed_done, snapped, f"apply failed: {exc!r}")

    log.info("selfedit_apply_diff", written=len(written), removed=len(removed_done),
             snapshot=snapped)
    return ApplyResult(True, written, removed_done, snapped, "applied")


async def compute_changes(candidate: Any) -> tuple[list[str], list[str]]:
    """Changed-vs-removed paths for a candidate worktree (``git status
    --porcelain``), via the candidate's own git helper."""
    from augmentum.selfedit.candidate import _git  # the candidate's git runner

    code, out = await _git(candidate.path, "status", "--porcelain")
    if code != 0 or not out:
        return [], []
    return classify_porcelain(out.splitlines())
