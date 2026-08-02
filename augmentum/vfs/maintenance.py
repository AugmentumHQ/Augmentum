"""Background maintenance for the file subsystem.

Two jobs, both safe to run as a single periodic task:

- `purge_old_trash` — hard-deletes file_index rows in trash older than N
  days, dispatching through registered adapters first so blob refcounts
  get released and physical files actually go away.  The previous loop
  in `server.py` called the bare `purge_all_old_trash` index method,
  which deleted rows but left blobs with refcount > 0 for any remaining
  references — and orphaned blobs on the disk for rows that were the
  only reference.  This wrapper closes that gap.

- `sweep_orphan_blobs` — catches blobs whose refcount has hit zero
  through any path (manual SQL, future bulk delete, race condition).
  The safety net for the case `purge_old_trash` can't see.

Both helpers are pure-async and take their dependencies as arguments so
they can be unit-tested without the full app context.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.vfs.blobs import BlobStore
    from augmentum.vfs.index import FileIndexService

log = get_logger(__name__)


class _AdapterLike(Protocol):
    async def delete(self, source_id: str, *, user_id: str) -> bool: ...


async def purge_old_trash(
    file_index: FileIndexService,
    *,
    days: int,
    adapter_lookup: Callable[[str], _AdapterLike | None],
    batch: int = 1000,
) -> dict[str, int]:
    """Hard-delete trash older than `days` across all users.

    Per-source adapters get first crack so blob refcounts release; rows
    without a registered adapter fall through to the bare index delete.

    `days <= 0` disables the purge entirely (returns zero counts).
    """
    if days <= 0:
        return {"adapter_deleted": 0, "index_deleted": 0, "errors": 0}

    entries = await file_index.list_trashed_older_than(days, limit=batch)
    if not entries:
        return {"adapter_deleted": 0, "index_deleted": 0, "errors": 0}

    adapter_deleted = 0
    errors = 0
    handled_ids: set[str] = set()
    for entry in entries:
        adapter = adapter_lookup(entry.source)
        if adapter is None:
            continue
        try:
            ok = await adapter.delete(entry.source_id, user_id=entry.user_id)
        except Exception as err:
            log.warning(
                "trash_purge_adapter_error",
                source=entry.source, source_id=entry.source_id,
                user_id=entry.user_id, err=str(err),
            )
            errors += 1
            continue
        if ok:
            adapter_deleted += 1
            handled_ids.add(entry.id)

    # Anything left (no adapter registered for that source, or adapter
    # said "no such row") gets the bare index delete so it doesn't sit
    # in trash forever.
    index_deleted = await file_index.purge_all_old_trash(days)

    if adapter_deleted or index_deleted or errors:
        log.info(
            "trash_auto_purged",
            adapter_deleted=adapter_deleted,
            index_deleted=index_deleted,
            errors=errors,
            ttl_days=days,
        )
    return {
        "adapter_deleted": adapter_deleted,
        "index_deleted": index_deleted,
        "errors": errors,
    }


async def sweep_orphan_blobs(blob_store: BlobStore, *, batch: int = 1000) -> int:
    """Delete blobs left at refcount<=0. Returns count purged."""
    return await blob_store.sweep_orphans(limit=batch)


async def run_maintenance(
    *,
    file_index: FileIndexService | None,
    blob_store: BlobStore | None,
    adapter_lookup: Callable[[str], _AdapterLike | None],
    trash_ttl_days: int,
) -> dict:
    """One full maintenance cycle. Safe to call from a background loop;
    individual failures are logged and don't abort the rest of the cycle.
    """
    summary: dict = {"trash": None, "orphans": 0}
    if file_index is not None:
        try:
            summary["trash"] = await purge_old_trash(
                file_index, days=trash_ttl_days, adapter_lookup=adapter_lookup,
            )
        except Exception:
            log.warning("trash_purge_cycle_failed", exc_info=True)
    if blob_store is not None:
        try:
            summary["orphans"] = await sweep_orphan_blobs(blob_store)
        except Exception:
            log.warning("orphan_sweep_cycle_failed", exc_info=True)
    return summary
