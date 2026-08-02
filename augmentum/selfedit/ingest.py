"""Ingest-all-work — the archive learns from every stream of real work.

The never-pruned archive only grew from the engine's own (rare) autonomous
attempts, while the overwhelmingly larger stream of propose→edit→verify→
human-verdict cycles — ordinary git commits on the live repo, applied
coder-mode turns — never became rows. Doctrine #3 says hook paths that already
run; this module is those hooks, shipped as DATA on the existing spine (new
rows in ``self_edit_attempts`` with a ``source`` tag, no new tables, no new
loop). The activation fold weights each source (``activation._SOURCE_WEIGHT``)
and the retrodiction benchmark reports composition by source, so provenance
stays legible to every consumer.

Honest-verdict mapping (the part that must not lie):
* A git commit that is still in history → ``live`` (the human kept it — the
  repo IS the endorsement). A commit later reverted → ``rolled_back`` (a real
  mistake, witnessed). The revert commit itself is a verdict-carrier, not a
  unit of work — skipped, counted.
* A coder turn: ``done`` → ``live`` (applied and concluded; implicit keep,
  damped hard by its source weight), agent-stopped/errored → ``failed``;
  anything ambiguous (``incomplete`` …) → ``ingested`` — a status with NO
  modulation, so it moves no weights and simply enriches the corpus.

Idempotent throughout: deterministic ids (``git:<sha>``, ``coder:<turn_id>``)
+ ``store.ingest_attempt``'s existing-row check make re-runs safe.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from augmentum.selfedit import store
from augmentum.selfedit.candidate import _git
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_REVERTS_RE = re.compile(r"This reverts commit ([0-9a-f]{7,40})", re.IGNORECASE)

# record/field separators for the one-pass `git log` parse (never appear in
# commit messages in practice; a pathological message degrades one record, not
# the run).
_RS, _FS = "\x1e", "\x1f"


def _surface_for_files(paths: list[str]) -> str:
    """The same coarse surface vocabulary the orchestrator uses, derived from
    paths alone (an ingested commit has no request text to classify)."""
    kinds: set[str] = set()
    for p in paths:
        q = (p or "").replace("\\", "/").lstrip("./")
        if not q:
            continue
        if q.startswith("augmentum/state/migrations/"):
            kinds.add("migration")
        elif q.startswith("ui/"):
            kinds.add("frontend")
        elif q.startswith("augmentum/") or q.endswith(".py"):
            kinds.add("backend")
        else:
            kinds.add("config")
    if not kinds:
        return ""
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


async def ingest_git_history(
    conn: Any, *, repo_dir: str, user_id: str, limit: int = 2000,
    branch: str = "HEAD",
) -> dict:
    """Backfill the live repo's commit history into the archive (source=``git``).

    One ``git log`` pass (no per-commit subprocesses); merges excluded (they are
    integration events, not units of work). Returns honest counts — including
    what was skipped and whether the scan hit ``limit`` (truncated) — so a
    partial backfill never reads as a complete one."""
    limit = max(1, min(int(limit or 2000), 20000))
    code, out = await _git(
        repo_dir, "log", branch, f"--max-count={limit}", "--no-merges",
        f"--pretty=format:{_RS}%H{_FS}%at{_FS}%s{_FS}%B{_FS}", "--name-only",
    )
    if code != 0:
        log.warning("selfedit_ingest_git_log_failed", code=code, detail=out[:400])
        return {"ok": False, "error": f"git log failed ({code})", "detail": out[:400]}

    # parse: RS-delimited records of  sha FS unix-ts FS subject FS raw-body FS files
    records: list[dict] = []
    reverted: set[str] = set()
    for rec in out.split(_RS):
        parts = rec.split(_FS)
        if len(parts) != 5:
            continue
        sha, ts_raw, subject, body, files_blob = parts
        sha = sha.strip()
        if not sha:
            continue
        try:
            ts = int(ts_raw.strip() or "0")
        except ValueError:
            ts = 0
        files = [ln.strip() for ln in files_blob.splitlines() if ln.strip()]
        is_revert = bool(_REVERTS_RE.search(body)) or subject.startswith('Revert "')
        for m in _REVERTS_RE.finditer(body):
            reverted.add(m.group(1))
        records.append({"sha": sha, "ts": ts, "subject": subject.strip(),
                        "files": files, "is_revert": is_revert})

    counts = {"ok": True, "scanned": len(records), "ingested": 0,
              "existing": 0, "skipped_reverts": 0, "skipped_no_files": 0,
              "marked_rolled_back": 0, "truncated": len(records) >= limit}
    # full-SHA revert targets may be abbreviated in the message — match by prefix.
    def _was_reverted(sha: str) -> bool:
        return any(sha.startswith(r) or r.startswith(sha) for r in reverted)

    for rec in records:
        if rec["is_revert"]:
            counts["skipped_reverts"] += 1
            continue
        if not rec["files"]:
            counts["skipped_no_files"] += 1
            continue
        rolled_back = _was_reverted(rec["sha"])
        status = "rolled_back" if rolled_back else "live"
        wrote = await store.ingest_attempt(
            conn, attempt_id=f"git:{rec['sha']}", user_id=user_id,
            objective=rec["subject"] or "(no commit message)", source="git",
            status=status, surface=_surface_for_files(rec["files"]),
            files_changed=rec["files"],
            outcome=("reverted in later history" if rolled_back
                     else "kept in live history"),
            promoted_commit=rec["sha"],
            created_at=_utc(rec["ts"]) if rec["ts"] else "",
        )
        if wrote:
            counts["ingested"] += 1
            counts["marked_rolled_back"] += 1 if rolled_back else 0
        else:
            counts["existing"] += 1

    log.info("selfedit_ingest_git_done", **{k: v for k, v in counts.items()
                                            if k != "ok"})
    return counts


def _coder_status(outcome: str) -> str:
    o = (outcome or "").strip().lower()
    if o == "done":
        return "live"
    if "stopped" in o or "error" in o or "failed" in o:
        return "failed"
    return "ingested"  # ambiguous (e.g. "incomplete"): corpus, never a verdict


async def ingest_coder_turn(
    conn: Any, *, user_id: str, turn_id: str, user_goal: str, outcome: str,
    files_edited: list[Any], workspace_id: str = "",
) -> bool:
    """Mirror one applied coder-mode turn into the archive (source=``coder``).
    Called from the coder handler's turn-archive hook, flag-gated by
    ``selfedit_ingest_coder_enabled``. Only turns that actually edited files
    are work units; a read-only turn is not an attempt."""
    if not turn_id or not user_id:
        return False
    files: list[str] = []
    for f in files_edited or []:
        path = f.get("path") if isinstance(f, dict) else f
        if path:
            files.append(str(path))
    if not files:
        return False
    return await store.ingest_attempt(
        conn, attempt_id=f"coder:{turn_id}", user_id=user_id,
        objective=(user_goal or "").strip() or "(coder turn)", source="coder",
        status=_coder_status(outcome), surface=_surface_for_files(files),
        files_changed=files, outcome=(outcome or "").strip(),
        base_ref=workspace_id,
    )
