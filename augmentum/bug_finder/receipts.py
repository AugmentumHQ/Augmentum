"""Per-action receipts — the evidence trail behind every claimed fix.

Each fixer attempt produces one or more ``Receipt`` rows: what was
attempted, against which finding, by which model, what the pre/post
file state looked like, and whether SWD verified or rolled it back.

Receipts are append-only JSONL at
``.augmentum/bug_finder/receipts.jsonl`` — same shape as
``audit_history.jsonl`` so future trend tooling can read them with one
loader.

This is the substrate-level upgrade for ``record_confirmation``:
instead of just bumping ``patterns.json::fix_count``, we keep the full
trust record. ``patterns.json`` answers "has this pattern been fixed
here before?"; receipts answer "what exactly was done, by whom, when,
and is the file still in the state we left it?".

Why separate from patterns:

* Patterns aggregate (signature × file → counts). Receipts are
  per-event.
* Patterns are user-visible knowledge (committed to git via the
  substrate ``.gitignore``); receipts are run-local evidence.
* Trend tooling wants receipts indexed by ``finding_id`` and
  ``run_id``; patterns are indexed by signature.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from augmentum.bug_finder.swd import ActionResult, SWDRunResult
from augmentum.bug_finder.workspace_substrate import (
    ensure_substrate,
    substrate_dir,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Receipt shape
# ---------------------------------------------------------------------------


@dataclass
class Receipt:
    """One verified (or rejected) file action.

    Field choices: everything you'd want to answer "did the fix that
    closed finding X actually land, by which model, and is the file
    still in the state the fixer left it?".
    """

    # What — links back to the finding/run
    finding_id: str
    run_id: str
    op: str                    # FileAction.op
    path: str                  # repo-relative
    intent: str                # FileAction.intent

    # How it went
    status: str                # ActionStatus value
    error: str = ""

    # Evidence — hash trail (no raw content kept)
    pre_hash: str = ""
    post_hash: str = ""
    pre_existed: bool = False
    post_existed: bool = False
    pre_size: int = 0
    post_size: int = 0

    # Who/when
    model_id: str = ""
    provider: str = ""
    git_head: str = ""
    ts: int = 0
    claim_signature: str = ""
    reason: str = ""           # FileAction.reason — the model's justification

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _receipts_path(workspace_root: Path) -> Path:
    return substrate_dir(workspace_root) / "receipts.jsonl"


def append_receipt(workspace_root: Path, receipt: Receipt) -> None:
    """Append one receipt. Best-effort; never raises.

    If ``ts`` is zero, fill from the current clock. Caller-supplied
    timestamps win so receipts written from a job-restart replay keep
    their original event time.
    """
    ensure_substrate(workspace_root)
    if receipt.ts == 0:
        receipt.ts = int(time.time())
    try:
        with _receipts_path(workspace_root).open(
            "a", encoding="utf-8",
        ) as fp:
            fp.write(json.dumps(receipt.to_dict(), ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning(
            "bug_finder_receipt_append_failed",
            workspace=str(workspace_root), error=str(exc),
        )


def append_receipts(
    workspace_root: Path,
    receipts: list[Receipt],
) -> None:
    """Append many receipts in one open/write/close cycle."""
    if not receipts:
        return
    ensure_substrate(workspace_root)
    now = int(time.time())
    try:
        with _receipts_path(workspace_root).open(
            "a", encoding="utf-8",
        ) as fp:
            for r in receipts:
                if r.ts == 0:
                    r.ts = now
                fp.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning(
            "bug_finder_receipts_batch_append_failed",
            workspace=str(workspace_root), error=str(exc),
        )


def load_receipts(
    workspace_root: Path,
    *,
    limit: int = 200,
) -> list[Receipt]:
    """Load the last ``limit`` receipts in append order. Returns
    empty list on any read error — receipts are best-effort."""
    p = _receipts_path(workspace_root)
    if not p.is_file():
        return []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[Receipt] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if not isinstance(d, dict):
            continue
        out.append(_dict_to_receipt(d))
    return out


def receipts_for_finding(
    workspace_root: Path,
    finding_id: str,
    *,
    limit: int = 50,
) -> list[Receipt]:
    """All receipts for a specific finding — its full fix history.

    Useful for the orchestrator's "have we tried fixing this before?"
    branch, or for the UI to show "this finding was previously
    addressed and the fix is still in place"."""
    if not finding_id:
        return []
    out: list[Receipt] = []
    for r in load_receipts(workspace_root, limit=10_000):
        if r.finding_id == finding_id:
            out.append(r)
            if len(out) >= limit:
                break
    return out


def receipts_for_path(
    workspace_root: Path,
    path: str,
    *,
    limit: int = 50,
) -> list[Receipt]:
    """All receipts touching ``path`` — file's audit trail."""
    if not path:
        return []
    norm = path.replace("\\", "/")
    out: list[Receipt] = []
    for r in load_receipts(workspace_root, limit=10_000):
        if r.path.replace("\\", "/") == norm:
            out.append(r)
            if len(out) >= limit:
                break
    return out


def _dict_to_receipt(d: dict[str, Any]) -> Receipt:
    return Receipt(
        finding_id=str(d.get("finding_id") or ""),
        run_id=str(d.get("run_id") or ""),
        op=str(d.get("op") or ""),
        path=str(d.get("path") or ""),
        intent=str(d.get("intent") or ""),
        status=str(d.get("status") or ""),
        error=str(d.get("error") or ""),
        pre_hash=str(d.get("pre_hash") or ""),
        post_hash=str(d.get("post_hash") or ""),
        pre_existed=bool(d.get("pre_existed") or False),
        post_existed=bool(d.get("post_existed") or False),
        pre_size=int(d.get("pre_size") or 0),
        post_size=int(d.get("post_size") or 0),
        model_id=str(d.get("model_id") or ""),
        provider=str(d.get("provider") or ""),
        git_head=str(d.get("git_head") or ""),
        ts=int(d.get("ts") or 0),
        claim_signature=str(d.get("claim_signature") or ""),
        reason=str(d.get("reason") or ""),
    )


# ---------------------------------------------------------------------------
# Adapter: SWDRunResult → list[Receipt]
# ---------------------------------------------------------------------------


def receipts_from_swd_result(
    result: SWDRunResult,
    *,
    run_id: str,
    model_id: str = "",
    provider: str = "",
    git_head: str = "",
    claim_signature: str = "",
) -> list[Receipt]:
    """Convert one SWD batch outcome into receipts. Run/model context
    is provided by the caller — the engine itself is provenance-free
    so it can be unit-tested without that scaffolding."""
    out: list[Receipt] = []
    now = int(time.time())
    for ar in result.results:
        out.append(_action_result_to_receipt(
            ar, run_id=run_id, model_id=model_id, provider=provider,
            git_head=git_head, claim_signature=claim_signature, ts=now,
        ))
    return out


def _action_result_to_receipt(
    ar: ActionResult,
    *,
    run_id: str,
    model_id: str,
    provider: str,
    git_head: str,
    claim_signature: str,
    ts: int,
) -> Receipt:
    pre = ar.pre
    post = ar.post
    return Receipt(
        finding_id=ar.action.finding_id,
        run_id=run_id,
        op=ar.action.op,
        path=ar.action.path,
        intent=ar.action.intent,
        status=ar.status,
        error=ar.error,
        pre_hash=pre.sha256 if pre else "",
        post_hash=post.sha256 if post else "",
        pre_existed=bool(pre and pre.exists),
        post_existed=bool(post and post.exists),
        pre_size=pre.size if pre else 0,
        post_size=post.size if post else 0,
        model_id=model_id,
        provider=provider,
        git_head=git_head,
        ts=ts,
        claim_signature=claim_signature,
        reason=ar.action.reason,
    )


# ---------------------------------------------------------------------------
# Trust queries — "is this fix still in place?"
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustStatus:
    """Quick assertion about a previously-applied fix.

    ``in_place`` means: the file's current SHA256 matches the
    ``post_hash`` recorded in the most recent VERIFIED receipt for
    this finding+path. False means the file has drifted (someone
    edited it, or it was reverted).
    """

    finding_id: str
    path: str
    in_place: bool
    last_post_hash: str = ""
    current_hash: str = ""
    receipt_ts: int = 0


def check_fix_still_in_place(
    workspace_root: Path,
    finding_id: str,
    path: str,
) -> TrustStatus:
    """Compare the file on disk now against what we last verified."""
    from augmentum.bug_finder.swd import (
        resolve_safe_path,
        snapshot_file,
    )

    recs = receipts_for_finding(workspace_root, finding_id)
    last_verified = next(
        (r for r in reversed(recs)
         if r.path.replace("\\", "/") == path.replace("\\", "/")
         and r.status == "verified"),
        None,
    )
    if last_verified is None:
        return TrustStatus(finding_id=finding_id, path=path, in_place=False)
    abs_path = resolve_safe_path(workspace_root, path)
    if abs_path is None:
        return TrustStatus(
            finding_id=finding_id, path=path,
            in_place=False, last_post_hash=last_verified.post_hash,
            receipt_ts=last_verified.ts,
        )
    current = snapshot_file(abs_path, path, keep_content=False)
    return TrustStatus(
        finding_id=finding_id,
        path=path,
        in_place=current.sha256 == last_verified.post_hash,
        last_post_hash=last_verified.post_hash,
        current_hash=current.sha256,
        receipt_ts=last_verified.ts,
    )
