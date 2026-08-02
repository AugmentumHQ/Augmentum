"""Strict Write Discipline — verify-before-trust for fixer-stage writes.

The bug-finder's fixer claims it patched a file. The trust gap: did it?

* The hash before vs after the edit tells us whether the file actually
  changed.
* The model's stated **intent** (MUTATE / NOOP / UNKNOWN) lets us catch
  hallucinated mutations — "I patched the SQL injection" → hash
  identical → reject.
* A pre-snapshot stash lets us **rollback** any operation that fails
  verification, so a botched batch leaves the workspace clean.

Borrowed pattern from ``thewaltero/mythos-router`` (see memory
``mythos-router-research``). Augmented with our claim-signature link so
each ``ActionReceipt`` ties back to the finding it was meant to fix.

The engine is provider-agnostic. It accepts a list of ``FileAction``
objects from anywhere — LLM tool calls, ``[FILE_ACTION]`` text blocks,
even a deterministic patch script.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------


# Paths the engine refuses to touch regardless of action. Conservative
# defaults; project-policy overlays can extend.
_SENSITIVE_FILE_NAMES = frozenset({
    ".env", ".env.local", ".env.production",
    "id_rsa", "id_ed25519",
    "secrets.toml", "credentials.json",
})
_SENSITIVE_PATH_PREFIXES = (".git/", ".ssh/", "node_modules/")
_SENSITIVE_SUFFIXES = (".key", ".pem", ".p12", ".pfx")

# Default cap for files SWD will snapshot + rollback. Anything larger
# means we can't safely undo a botched write — reject at preflight.
_DEFAULT_MAX_SNAPSHOT_BYTES = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Enums + dataclasses
# ---------------------------------------------------------------------------


class ActionOp(str, Enum):
    """File operations SWD understands."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    READ   = "read"


class ActionIntent(str, Enum):
    """The model's stated effect of this action on disk.

    MUTATE means "I am changing this file" — verification fails if the
    hash is unchanged after apply.

    NOOP means "I read or referenced this file but didn't change it" —
    verification fails if the hash IS changed (concurrent write).

    UNKNOWN waives intent enforcement; only structural checks apply
    (path exists/doesn't, op succeeded).
    """

    MUTATE = "mutate"
    NOOP = "noop"
    UNKNOWN = "unknown"


class ActionStatus(str, Enum):
    VERIFIED = "verified"      # passed all checks
    NOOP = "noop"              # NOOP intent confirmed (no change)
    DRIFT = "drift"            # hash differs from expected (concurrent or hallucinated)
    FAILED = "failed"          # op never landed (exception / refused)
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class FileSnapshot:
    """Captured state of a file at a single point in time.

    ``content`` is held only long enough for rollback; receipts persist
    only the hash + summary fields, never the raw bytes.
    """

    path: str
    exists: bool
    size: int = 0
    mtime: float = 0.0
    sha256: str = ""
    content: bytes | None = None

    def summary(self) -> FileSnapshotSummary:
        return FileSnapshotSummary(
            path=self.path, exists=self.exists,
            size=self.size, mtime=self.mtime, sha256=self.sha256,
        )


@dataclass(frozen=True)
class FileSnapshotSummary:
    """Hash-only summary — what receipts serialize."""

    path: str
    exists: bool
    size: int = 0
    mtime: float = 0.0
    sha256: str = ""


@dataclass(frozen=True)
class FileAction:
    """One file operation the engine should execute and verify."""

    op: str                    # one of ActionOp values
    path: str                  # repo-relative path
    intent: str = ActionIntent.UNKNOWN.value
    content: bytes | None = None   # required for CREATE / MODIFY
    reason: str = ""           # human-readable claim (for receipts)
    finding_id: str = ""       # ties the action back to a Finding


@dataclass
class ActionResult:
    """Outcome of one FileAction, including diagnostic fields."""

    action: FileAction
    status: str = ActionStatus.FAILED.value
    pre: FileSnapshotSummary | None = None
    post: FileSnapshotSummary | None = None
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status in (
            ActionStatus.VERIFIED.value,
            ActionStatus.NOOP.value,
        )


@dataclass
class SWDRunResult:
    """Aggregate outcome from one ``SWDEngine.run`` invocation."""

    results: list[ActionResult] = field(default_factory=list)
    rolled_back: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return (
            not self.rolled_back
            and all(r.succeeded for r in self.results)
        )


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _is_sensitive(rel_path: str) -> bool:
    """Refuse paths that match the built-in sensitive-file rules."""
    rp = rel_path.replace("\\", "/")
    if any(rp.startswith(prefix) for prefix in _SENSITIVE_PATH_PREFIXES):
        return True
    if any(rp.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES):
        return True
    name = rp.rsplit("/", 1)[-1]
    return name in _SENSITIVE_FILE_NAMES


def resolve_safe_path(root: Path, rel_path: str) -> Path | None:
    """Resolve ``rel_path`` against ``root``, refusing anything that
    escapes the root or matches a sensitive-file rule.

    Returns ``None`` on rejection — the caller logs and produces a
    ``FAILED`` result. We do NOT raise here so that one bad action in
    a batch can't cascade-cancel the good ones (the engine's loop
    handles that explicitly).
    """
    rp = rel_path.strip().replace("\\", "/")
    if not rp or rp.startswith("/"):
        return None
    if _is_sensitive(rp):
        return None
    try:
        candidate = (root / rp).resolve()
        root_resolved = root.resolve()
    except OSError:
        return None
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot_file(
    abs_path: Path,
    rel_path: str,
    *,
    max_bytes: int = _DEFAULT_MAX_SNAPSHOT_BYTES,
    keep_content: bool = True,
) -> FileSnapshot:
    """Snapshot the file at ``abs_path``. Returns a snapshot describing
    "doesn't exist" rather than raising for missing files — that's a
    valid state SWD needs to track."""
    try:
        st = abs_path.stat()
    except FileNotFoundError:
        return FileSnapshot(path=rel_path, exists=False)
    except OSError as exc:
        return FileSnapshot(path=rel_path, exists=False, sha256=f"err:{exc}")
    if st.st_size > max_bytes:
        # Track existence + size but don't read content; the engine
        # rejects modify/delete on these but allows read intent.
        return FileSnapshot(
            path=rel_path, exists=True,
            size=st.st_size, mtime=st.st_mtime,
            sha256="oversize", content=None,
        )
    try:
        data = abs_path.read_bytes()
    except OSError as exc:
        return FileSnapshot(
            path=rel_path, exists=True, size=st.st_size,
            mtime=st.st_mtime, sha256=f"err:{exc}",
        )
    h = hashlib.sha256(data).hexdigest()
    return FileSnapshot(
        path=rel_path, exists=True,
        size=st.st_size, mtime=st.st_mtime, sha256=h,
        content=data if keep_content else None,
    )


def _hash_bytes(b: bytes | None) -> str:
    if b is None:
        return ""
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


@dataclass
class SWDEngine:
    """Verify-before-trust engine for file mutations.

    Lifecycle: ``run(actions)`` does
    ``Plan → Snapshot_Before → Execute → Snapshot_After → Verify →
    Commit/Rollback``. No iteration; if anything fails verification in
    strict mode, the whole batch rolls back and the caller decides
    whether to re-prompt the model.
    """

    workspace_root: Path
    max_snapshot_bytes: int = _DEFAULT_MAX_SNAPSHOT_BYTES
    strict: bool = True       # rollback the whole batch on any failure

    # ---- public ----------------------------------------------------------

    def run(self, actions: list[FileAction]) -> SWDRunResult:
        result = SWDRunResult()
        if not actions:
            return result
        if not self.workspace_root.is_dir():
            result.errors.append("workspace_root does not exist")
            return result

        # Plan + Snapshot_Before. We capture original state for every
        # path the batch touches BEFORE any apply runs, so a later
        # action in the batch can't corrupt rollback data.
        prepared: list[tuple[FileAction, Path, FileSnapshot]] = []
        rollback_map: dict[str, FileSnapshot] = {}
        for action in actions:
            abs_path = resolve_safe_path(self.workspace_root, action.path)
            if abs_path is None:
                ar = ActionResult(action=action, status=ActionStatus.FAILED.value,
                                  error="path refused (sensitive/out-of-root)")
                result.results.append(ar)
                continue
            rel = action.path.replace("\\", "/")
            pre = snapshot_file(
                abs_path, rel,
                max_bytes=self.max_snapshot_bytes,
                keep_content=True,
            )
            if pre.sha256 == "oversize" and action.op in (
                ActionOp.MODIFY.value, ActionOp.DELETE.value,
            ):
                ar = ActionResult(action=action, status=ActionStatus.FAILED.value,
                                  pre=pre.summary(),
                                  error=f"file >{self.max_snapshot_bytes}B; "
                                        "modify/delete blocked for safety")
                result.results.append(ar)
                continue
            # Record the original state for potential rollback. Cache
            # by rel-path so multiple actions on the same path keep
            # the FIRST pre-snapshot.
            if rel not in rollback_map:
                rollback_map[rel] = pre
            prepared.append((action, abs_path, pre))

        # Execute. Stop on first hard exception; verify still runs on
        # whatever landed so the receipt trail is complete.
        for action, abs_path, pre in prepared:
            try:
                self._apply(action, abs_path)
            except Exception as exc:  # noqa: BLE001 — surface to caller
                ar = ActionResult(
                    action=action, status=ActionStatus.FAILED.value,
                    pre=pre.summary(),
                    error=f"apply failed: {type(exc).__name__}: {exc}",
                )
                result.results.append(ar)
                result.errors.append(ar.error)
                continue

            # Snapshot_After + Verify.
            post = snapshot_file(
                abs_path, action.path.replace("\\", "/"),
                max_bytes=self.max_snapshot_bytes,
                keep_content=False,
            )
            ar = self._verify(action, pre, post)
            result.results.append(ar)

        # Commit or Rollback.
        if self.strict and any(not r.succeeded for r in result.results):
            self._rollback(rollback_map, result)

        return result

    # ---- internal --------------------------------------------------------

    def _apply(self, action: FileAction, abs_path: Path) -> None:
        op = action.op
        if op == ActionOp.READ.value:
            # READ is a no-op write. Snapshot work is enough.
            return
        if op == ActionOp.CREATE.value:
            if action.content is None:
                raise ValueError("CREATE requires content")
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(action.content)
            return
        if op == ActionOp.MODIFY.value:
            if action.content is None:
                raise ValueError("MODIFY requires content")
            abs_path.write_bytes(action.content)
            return
        if op == ActionOp.DELETE.value:
            try:
                abs_path.unlink()
            except FileNotFoundError:
                # Already gone; verify will catch via intent check.
                return
            return
        raise ValueError(f"unknown op: {op!r}")

    def _verify(
        self,
        action: FileAction,
        pre: FileSnapshot,
        post: FileSnapshot,
    ) -> ActionResult:
        intent = action.intent
        ar = ActionResult(
            action=action,
            pre=pre.summary(),
            post=post.summary(),
        )
        # Structural checks first.
        if action.op == ActionOp.CREATE.value:
            if pre.exists:
                ar.status = ActionStatus.DRIFT.value
                ar.error = "CREATE target already existed pre-apply"
                return ar
            if not post.exists:
                ar.status = ActionStatus.FAILED.value
                ar.error = "CREATE failed: file not present after apply"
                return ar
            expected_hash = _hash_bytes(action.content)
            if expected_hash and post.sha256 != expected_hash:
                ar.status = ActionStatus.DRIFT.value
                ar.error = (
                    f"CREATE content hash drift "
                    f"(expected {expected_hash[:12]}, got {post.sha256[:12]})"
                )
                return ar
            ar.status = ActionStatus.VERIFIED.value
            return ar
        if action.op == ActionOp.DELETE.value:
            if not pre.exists:
                ar.status = ActionStatus.NOOP.value
                ar.error = "DELETE target did not exist (noop)"
                return ar
            if post.exists:
                ar.status = ActionStatus.FAILED.value
                ar.error = "DELETE failed: file still present after apply"
                return ar
            ar.status = ActionStatus.VERIFIED.value
            return ar
        if action.op == ActionOp.MODIFY.value:
            if not pre.exists:
                ar.status = ActionStatus.FAILED.value
                ar.error = "MODIFY target did not exist pre-apply"
                return ar
            if not post.exists:
                ar.status = ActionStatus.FAILED.value
                ar.error = "MODIFY target disappeared after apply"
                return ar
            # Intent enforcement — the load-bearing check.
            if intent == ActionIntent.MUTATE.value and post.sha256 == pre.sha256:
                ar.status = ActionStatus.FAILED.value
                ar.error = (
                    "intent=MUTATE but file hash unchanged "
                    "(hallucinated edit or no-op patch)"
                )
                return ar
            if intent == ActionIntent.NOOP.value and post.sha256 != pre.sha256:
                ar.status = ActionStatus.DRIFT.value
                ar.error = (
                    "intent=NOOP but file changed "
                    "(concurrent write or wrong file)"
                )
                return ar
            expected_hash = _hash_bytes(action.content)
            if expected_hash and post.sha256 != expected_hash:
                ar.status = ActionStatus.DRIFT.value
                ar.error = (
                    f"MODIFY content hash drift "
                    f"(expected {expected_hash[:12]}, got {post.sha256[:12]})"
                )
                return ar
            ar.status = (
                ActionStatus.NOOP.value
                if post.sha256 == pre.sha256
                else ActionStatus.VERIFIED.value
            )
            return ar
        if action.op == ActionOp.READ.value:
            if pre.sha256 != post.sha256:
                ar.status = ActionStatus.DRIFT.value
                ar.error = "READ target changed during action (concurrent write)"
                return ar
            ar.status = ActionStatus.NOOP.value
            return ar
        ar.error = f"unknown op: {action.op!r}"
        return ar

    def _rollback(
        self,
        rollback_map: dict[str, FileSnapshot],
        result: SWDRunResult,
    ) -> None:
        """Reverse executed writes. Skip paths where the current disk
        state has drifted from what we recorded post-apply — that's a
        concurrent edit and we don't trample it."""
        # Walk in reverse so later writes get undone before earlier ones
        # on the same path collide.
        applied_paths = list(reversed([
            (r.action.path.replace("\\", "/"), r) for r in result.results
            if r.post is not None
        ]))
        for rel, ar in applied_paths:
            original = rollback_map.get(rel)
            if original is None:
                continue
            abs_path = resolve_safe_path(self.workspace_root, rel)
            if abs_path is None:
                result.errors.append(f"rollback refused: {rel}")
                continue
            # Confirm disk still matches what we wrote — if it doesn't,
            # someone else touched it after our verify; skip.
            current = snapshot_file(
                abs_path, rel,
                max_bytes=self.max_snapshot_bytes,
                keep_content=False,
            )
            if ar.post and current.sha256 != ar.post.sha256:
                result.errors.append(
                    f"concurrency drift during rollback: {rel} "
                    f"(expected {ar.post.sha256[:12]}, got {current.sha256[:12]})"
                )
                continue
            try:
                if original.exists and original.content is not None:
                    abs_path.write_bytes(original.content)
                elif not original.exists and current.exists:
                    abs_path.unlink()
                # Preserve the verify verdict on ``ar.status`` so
                # callers can see WHY rollback happened. The run-level
                # ``rolled_back`` flag carries the rollback fact.
            except OSError as exc:
                result.errors.append(f"rollback failed for {rel}: {exc}")
        result.rolled_back = True


# ---------------------------------------------------------------------------
# Text protocol parser — [FILE_ACTION:...] blocks
# ---------------------------------------------------------------------------


import re as _re

_FILE_ACTION_RE = _re.compile(
    r"\[FILE_ACTION:\s*"
    r"op=(?P<op>create|modify|delete|read),\s*"
    r"path=(?P<path>[^,\]]+?)"
    r"(?:,\s*intent=(?P<intent>mutate|noop|unknown))?"
    r"\](?P<body>.*?)\[/FILE_ACTION\]",
    _re.IGNORECASE | _re.DOTALL,
)


def parse_file_actions(output: str) -> list[FileAction]:
    """Extract ``[FILE_ACTION:op=..., path=..., intent=...]...[/FILE_ACTION]``
    blocks from LLM text. Fallback for models with unreliable tool-calls.

    Body of the block (between the open/close tags) is the
    ``content`` for create/modify ops; for delete/read it's a free-text
    reason that lands in the receipt.
    """
    if not output:
        return []
    actions: list[FileAction] = []
    for m in _FILE_ACTION_RE.finditer(output):
        op = m.group("op").lower()
        path = m.group("path").strip()
        intent = (m.group("intent") or ActionIntent.UNKNOWN.value).lower()
        body = m.group("body") or ""
        if op in (ActionOp.CREATE.value, ActionOp.MODIFY.value):
            content = body.lstrip("\n").encode("utf-8")
            reason = ""
        else:
            content = None
            reason = body.strip()
        actions.append(FileAction(
            op=op, path=path, intent=intent,
            content=content, reason=reason,
        ))
    return actions


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def run_actions(
    workspace_root: Path,
    actions: list[FileAction],
    *,
    strict: bool = True,
) -> SWDRunResult:
    """One-shot helper for the common case."""
    engine = SWDEngine(workspace_root=workspace_root, strict=strict)
    return engine.run(actions)


def result_to_log(result: SWDRunResult) -> dict[str, Any]:
    """Render an SWDRunResult into a flat dict suitable for structured
    logging or for the audit trail."""
    return {
        "success": result.success,
        "rolled_back": result.rolled_back,
        "actions": [
            {
                "op": r.action.op,
                "path": r.action.path,
                "intent": r.action.intent,
                "status": r.status,
                "pre_hash": r.pre.sha256 if r.pre else "",
                "post_hash": r.post.sha256 if r.post else "",
                "error": r.error,
                "finding_id": r.action.finding_id,
            }
            for r in result.results
        ],
        "errors": list(result.errors),
        "ts": int(time.time()),
    }
