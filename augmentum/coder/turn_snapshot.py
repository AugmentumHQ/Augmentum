"""Turn snapshot — per-turn pre-write file capture for diff review.

The reviewable-turn flow needs two things the existing coder harness
doesn't provide:

  1. **A diff at turn end.** What did the agent change across this
     whole turn, not just per-tool? The user reviews the *turn* as
     the unit of intent, not individual writes.
  2. **A reject path.** If the user rejects the turn (or a subset of
     files), we need to roll the disk back to pre-turn state.

Two architectures were considered:

* **Staging overlay** — agent writes land in ``.augmentum/staging/``
  instead of real disk; ``shell_exec`` must flush before running.
  Atomic reject, but breaks the "agent's writes are ground truth
  mid-turn" contract that every other tool relies on. npm install
  can't see a staged ``package.json`` without a flush. Complicated.

* **Snapshot-then-observe** (chosen). Writes go directly to disk as
  they do today. Before each first-write to a path P, we capture
  P's pre-turn disk bytes. At turn end we diff snapshots vs current
  disk. Reject = restore from snapshots. Shell-friendly — commands
  always see real state; no flush dance.

Trade-off: the reject path is only as good as the snapshot coverage.
If a path wasn't snapshotted (snapshot call failed, size limit hit,
or the agent created the file outside a mutating tool via shell_exec),
we can't restore it. Those cases log a warning and the review panel
surfaces them as "non-reversible" rather than silently losing data.

Scope — deliberate non-goals for this module:

* **Binary files beyond the size limit.** Snapshot cap is 10 MB total
  across the turn; larger captures are skipped with a warning. Binary
  diffs aren't rendered as unified text anyway; the review panel will
  surface them as "binary changed, N bytes".
* **Shell-created files.** ``shell_exec`` hits disk directly and has
  no hook for snapshotting. If the agent runs ``mkdir src && touch
  src/x.py`` we see the file appear at turn end, diff it as "added
  from empty", but can only reject by deleting (no pre-state to
  restore to — which is correct; there was no pre-state).
* **Concurrent turns.** Each ``CoderState`` holds one active
  snapshot; handlers are cached per (user, session) so contention is
  impossible by construction.
"""
from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# 10 MB total across the turn's snapshots. A single source file rarely
# crosses 1 MB; this ceiling trips only on accidental captures of
# lockfiles, binaries, or minified bundles. Hitting the limit logs a
# warning and skips subsequent snapshots — the review panel will note
# affected paths as non-reversible rather than silently losing data.
_SIZE_LIMIT_BYTES = 10 * 1024 * 1024


@dataclass
class DiffEntry:
    """One file's change within a reviewable turn."""

    path: str
    # "added" — new file that didn't exist pre-turn
    # "modified" — existing file edited
    # "deleted" — file existed pre-turn, gone at turn end
    status: str
    unified_diff: str
    old_size: int
    new_size: int
    # True when a snapshot exists AND the pre-content is recoverable;
    # False when the diff was observed (file appeared/changed) but we
    # have no pre-state to restore to (e.g. shell_exec-created file).
    # The review panel marks ``reversible=False`` entries visibly so
    # the user isn't surprised by a reject that can't fully undo.
    reversible: bool = True


@dataclass
class TurnSnapshot:
    """Pre-write snapshot store for one agent turn.

    Lifecycle:

    * Constructed at turn start by the handler; attached to
      ``CoderState.active_turn_snapshot``.
    * Mutating tools call :meth:`snapshot_before_write` before their
      disk write. Idempotent — repeated calls for the same path are
      no-ops, so e.g. a tool that reads-then-writes (``code_edit``)
      doesn't need to coordinate with a tool that pure-writes
      (``file_write``) that hit the same path earlier.
    * At turn end the handler calls :meth:`collect_diffs` to produce
      the review bundle.
    * The bundle lives in :class:`~augmentum.coder.reviews.ReviewRegistry`
      for the user to accept / reject / partial-accept. Reject paths
      call :meth:`restore` on specific paths.
    """

    turn_id: str
    workspace_id: str
    # Container manager reference. Not in the dataclass field default
    # because typed as Protocol via ContainerManager stubs; passed by
    # the caller.
    container_manager: object = field(repr=False)

    # path → bytes-or-None. None sentinel = "did not exist pre-turn";
    # bytes = captured pre-turn content. Missing key = not snapshotted
    # (never called for this path). The three states matter for the
    # restore path — None restores to "deleted", bytes restores to
    # that content, missing logs as non-reversible.
    _snapshots: dict[str, bytes | None] = field(default_factory=dict, repr=False)

    # Running byte count of captured content. Trips _SIZE_LIMIT_BYTES
    # → subsequent snapshot calls log and bail.
    _total_bytes: int = 0

    # Paths the snapshot was SKIPPED for (size limit, read error).
    # collect_diffs still emits entries for these (status + diff from
    # the NEW file against an empty pre-state, if we can read the new
    # state) but flags ``reversible=False``.
    _skipped: set[str] = field(default_factory=set, repr=False)

    # Timestamp of snapshot construction — used for turn duration in
    # the review bundle metadata.
    started_at: float = field(default_factory=time.time)

    # -----------------------------------------------------------------
    # Write-time capture
    # -----------------------------------------------------------------

    async def snapshot_before_write(self, path: str) -> None:
        """Capture pre-write disk state of ``path`` if not already held.

        Idempotent per path. Failures (file not found → treated as
        "did not exist"; any other exception → marked skipped and
        logged) never block the write; we lose only the restore
        capability for that path.
        """
        if path in self._snapshots or path in self._skipped:
            return

        if self._total_bytes >= _SIZE_LIMIT_BYTES:
            self._skipped.add(path)
            log.warning(
                "turn_snapshot.size_limit_exceeded",
                turn_id=self.turn_id, path=path, total=self._total_bytes,
            )
            return

        try:
            content_str = await self.container_manager.file_read(
                self.workspace_id, path,
            )
            # file_read is cat-based — missing files raise. An empty
            # file is legitimately empty content (""), distinct from
            # "didn't exist". Caller wouldn't hit this path for
            # missing files (read returns via exception).
            content = (content_str or "").encode("utf-8", errors="replace")
        except FileNotFoundError:
            # Pre-turn: path didn't exist. Store sentinel.
            self._snapshots[path] = None
            return
        except Exception as exc:
            # Any other failure — container down, permission, exotic
            # path. Mark skipped so the review bundle flags the file
            # as non-reversible but still shows the diff.
            log.warning(
                "turn_snapshot.capture_failed",
                turn_id=self.turn_id, path=path, error=str(exc),
            )
            self._skipped.add(path)
            return

        self._snapshots[path] = content
        self._total_bytes += len(content)

    def register_created_path(self, path: str) -> None:
        """Register ``path`` as absent at turn start.

        Used for shell-created files discovered via a workspace-tree
        delta at turn end. A ``None`` sentinel is enough for
        ``collect_diffs`` to render the file as "added" and for
        ``restore`` to reject it by deleting the file.
        """
        if path in self._snapshots or path in self._skipped:
            return
        self._snapshots[path] = None

    # -----------------------------------------------------------------
    # Turn-end diff
    # -----------------------------------------------------------------

    async def collect_diffs(self) -> list[DiffEntry]:
        """Enumerate per-file changes since turn start.

        Reads current disk state for every snapshotted or skipped
        path, computes a unified diff against the snapshot (or empty
        if skipped / new), and returns one :class:`DiffEntry` per
        actual change. Unchanged files (snapshot matches current) are
        filtered out — a tool that ``file_write``s identical content
        to disk doesn't clutter the review.
        """
        entries: list[DiffEntry] = []
        # Union: every path either snapshotted or known-skipped.
        # Ordering: sorted for deterministic output (tests + review
        # panel both want stable order).
        touched = sorted(set(self._snapshots) | self._skipped)

        for path in touched:
            pre = self._snapshots.get(path)
            pre_existed = pre is not None
            skipped = path in self._skipped

            try:
                post_str = await self.container_manager.file_read(
                    self.workspace_id, path,
                )
                post: bytes | None = (post_str or "").encode(
                    "utf-8", errors="replace",
                )
            except FileNotFoundError:
                post = None
            except Exception as exc:
                log.warning(
                    "turn_snapshot.post_read_failed",
                    turn_id=self.turn_id, path=path, error=str(exc),
                )
                continue

            # No-op: identical content pre and post. Skip quietly.
            if pre == post:
                continue

            status = _classify_status(pre_existed, post is not None, skipped)
            unified = _unified_diff(path, pre, post)
            entries.append(DiffEntry(
                path=path,
                status=status,
                unified_diff=unified,
                old_size=len(pre) if pre else 0,
                new_size=len(post) if post else 0,
                reversible=not skipped,
            ))
        return entries

    # -----------------------------------------------------------------
    # Restore path (reject)
    # -----------------------------------------------------------------

    async def restore(self, paths: list[str]) -> list[str]:
        """Roll back ``paths`` to their pre-turn disk state.

        Returns the list of paths that could NOT be restored (missing
        snapshot, container error). Callers should surface those to
        the user rather than claim a clean reject.

        Contract:

        * ``snapshots[path] is None`` → path didn't exist pre-turn;
          delete it now.
        * ``snapshots[path] is bytes`` → write bytes back.
        * ``path in skipped`` → we captured nothing; caller must use
          another mechanism (git revert, manual undo).
        * ``path not in snapshots and not skipped`` → we never saw a
          write for this path; nothing to restore. Treated as a
          failure so the caller doesn't silently no-op what the user
          clicked "reject" on.
        """
        failed: list[str] = []

        for path in paths:
            if path in self._skipped:
                failed.append(path)
                continue
            if path not in self._snapshots:
                failed.append(path)
                log.warning(
                    "turn_snapshot.restore_unknown_path",
                    turn_id=self.turn_id, path=path,
                )
                continue

            content = self._snapshots[path]
            try:
                if content is None:
                    # Pre-turn didn't exist → delete the current file.
                    # shell -c rm -f so missing files don't fail.
                    await self.container_manager.run_command(
                        self.workspace_id,
                        ["sh", "-c", f"rm -f {path!r}"],
                    )
                else:
                    # Restore captured content verbatim.
                    await self.container_manager.file_write(
                        self.workspace_id, path,
                        content.decode("utf-8", errors="replace"),
                    )
            except Exception as exc:
                log.warning(
                    "turn_snapshot.restore_failed",
                    turn_id=self.turn_id, path=path, error=str(exc),
                )
                failed.append(path)

        return failed

    # -----------------------------------------------------------------
    # Introspection — for tests + review-panel "what was captured" UI
    # -----------------------------------------------------------------

    @property
    def touched_paths(self) -> list[str]:
        """Union of snapshotted + skipped paths, sorted."""
        return sorted(set(self._snapshots) | self._skipped)

    @property
    def total_bytes_captured(self) -> int:
        return self._total_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_status(pre_existed: bool, post_exists: bool, skipped: bool) -> str:
    """Map (pre, post, skipped) → status string for DiffEntry.

    When ``skipped``, we don't have a reliable pre-state, so we
    heuristically assume the file is new (most common cause of
    skips is a fresh file that triggered an error during the first
    snapshot read — e.g. permissions on a write to a non-existent
    parent dir). The reversible=False flag on the entry already
    communicates the uncertainty.
    """
    if skipped:
        return "added" if post_exists else "deleted"
    if not pre_existed and post_exists:
        return "added"
    if pre_existed and not post_exists:
        return "deleted"
    return "modified"


def _unified_diff(path: str, pre: bytes | None, post: bytes | None) -> str:
    """Render a git-style unified diff between pre and post content.

    Binary-safe-ish: decodes with ``errors='replace'`` so non-UTF-8
    bytes become U+FFFD rather than exploding. Truly-binary files
    will produce a diff full of replacement chars — ugly but
    non-fatal. The review panel's renderer can detect this and
    collapse to a "binary file changed" note.
    """
    pre_text = pre.decode("utf-8", errors="replace") if pre else ""
    post_text = post.decode("utf-8", errors="replace") if post else ""
    pre_lines = pre_text.splitlines(keepends=True)
    post_lines = post_text.splitlines(keepends=True)
    # n=3 matches git's default context. fromfile/tofile get the a/b
    # prefix convention so downstream renderers can parse with any
    # standard diff library without stripping our custom labels.
    return "".join(difflib.unified_diff(
        pre_lines, post_lines,
        fromfile=f"a/{path.lstrip('/')}",
        tofile=f"b/{path.lstrip('/')}",
        n=3,
    ))
