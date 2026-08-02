"""Review registry for the reviewable-turn flow.

Mirrors the shape of ``augmentum/coder/permissions.py`` but with
different semantics:

* **Permission requests** are synchronous — the agent loop blocks on
  an ``asyncio.Future`` until the user clicks Allow/Deny.
* **Reviews** are asynchronous — the agent turn finishes and
  publishes a bundle; the user reviews at their own pace. If the
  user never clicks, the bundle lives in the registry indefinitely
  (until the session is cleared or the server restarts).

The registry stores ``ReviewBundle`` objects keyed by turn_id. Each
bundle contains:

* Metadata (user_id, workspace_id, turn_id, user_message, created_at)
* Per-file diff entries (path, status, unified_diff, sizes, reversible)
* A reference to the :class:`TurnSnapshot` used to produce the diffs
  — needed for the reject path, which calls
  :meth:`TurnSnapshot.restore` on rejected paths.

Route handlers consume bundles via :meth:`accept` / :meth:`reject` /
:meth:`partial` (which remove the bundle from pending and return it
so the caller can do the actual disk work — restore for rejected
paths, git-commit for accepted ones). The registry itself is pure
bookkeeping.

Persistence — Sprint 1 non-goal. Bundles live in-memory only. If the
server restarts mid-review, agent writes are still on disk (they hit
directly; staging was rejected for the ``shell_exec``-compatibility
reason in turn_snapshot.py), so the user can recover via ``git diff``
and checkpoints. Sprint 2 will add a SQLite-backed store for
durability.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.coder.turn_snapshot import DiffEntry, TurnSnapshot

log = get_logger(__name__)


@dataclass
class ReviewBundle:
    """One turn's reviewable changes.

    The ``snapshot`` field holds the live :class:`TurnSnapshot` so
    the reject path can restore rejected files. It's excluded from
    ``to_dict`` because the frontend doesn't need it and serialising
    raw bytes would bloat the API response.
    """

    turn_id: str
    user_id: str
    workspace_id: str
    session_id: str
    user_message: str
    files: list[DiffEntry]
    snapshot: TurnSnapshot = field(repr=False)
    created_at: float = field(default_factory=time.time)
    # One of: pending | accepted | rejected | partial. Used to filter
    # the pending-list endpoint. The registry REMOVES non-pending
    # bundles on resolve, so this is mostly observational — helpful
    # in tests and if we ever add a history endpoint.
    status: str = "pending"

    def to_dict(self) -> dict:
        """Serialisable shape for the /api/coder/reviews/... endpoints."""
        return {
            "turn_id":       self.turn_id,
            "workspace_id":  self.workspace_id,
            "session_id":    self.session_id,
            "user_message":  self.user_message,
            "created_at":    self.created_at,
            "status":        self.status,
            "files": [
                {
                    "path":          f.path,
                    "status":        f.status,
                    "unified_diff":  f.unified_diff,
                    "old_size":      f.old_size,
                    "new_size":      f.new_size,
                    "reversible":    f.reversible,
                }
                for f in self.files
            ],
            "summary": {
                "files_changed":    len(self.files),
                "added":   sum(1 for f in self.files if f.status == "added"),
                "modified": sum(1 for f in self.files if f.status == "modified"),
                "deleted":  sum(1 for f in self.files if f.status == "deleted"),
                "non_reversible": sum(1 for f in self.files if not f.reversible),
            },
        }


class ReviewRegistry:
    """Process-wide registry of pending turn reviews.

    Keyed by turn_id; ``pending_for(user_id)`` filters by owner so
    users only see their own turns (multi-tenant hygiene — same
    pattern as PermissionRegistry).
    """

    def __init__(self) -> None:
        self._pending: dict[str, ReviewBundle] = {}

    def publish(self, bundle: ReviewBundle) -> None:
        """Register a new bundle. Turn-end hook calls this."""
        self._pending[bundle.turn_id] = bundle
        log.info(
            "coder.review_published",
            turn_id=bundle.turn_id,
            user_id=bundle.user_id,
            files=len(bundle.files),
        )

    def pending_for(self, user_id: str) -> list[ReviewBundle]:
        """All unresolved bundles belonging to ``user_id``.

        Empty ``user_id`` returns everything — matches the
        single-tenant-dev convention in PermissionRegistry.
        """
        if not user_id:
            return list(self._pending.values())
        return [b for b in self._pending.values() if b.user_id == user_id]

    def get(self, turn_id: str) -> ReviewBundle | None:
        return self._pending.get(turn_id)

    def resolve(self, turn_id: str, status: str) -> ReviewBundle | None:
        """Remove and return the bundle, marked with the final status.

        Route handlers call this after they've applied the accept /
        reject / partial work. Returns None if the bundle is unknown
        or already resolved — caller should treat that as a 404.
        """
        bundle = self._pending.pop(turn_id, None)
        if bundle is None:
            return None
        bundle.status = status
        log.info(
            "coder.review_resolved", turn_id=turn_id, status=status,
        )
        return bundle

    def size(self) -> int:
        return len(self._pending)

    def clear_for_workspace(self, workspace_id: str) -> int:
        """Drop all pending bundles for a workspace. Returns count dropped.

        Called when a workspace is deleted — prevents orphan bundles
        pointing to a snapshot whose container is gone.
        """
        drop = [
            turn_id for turn_id, b in self._pending.items()
            if b.workspace_id == workspace_id
        ]
        for turn_id in drop:
            self._pending.pop(turn_id, None)
        return len(drop)
