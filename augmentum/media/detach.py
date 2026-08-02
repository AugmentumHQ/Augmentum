"""Teardown for revoked media shares: hide the rows, keep the history.

When an admin un-shares (or deletes) a media server, every *borrower's*
``file_index`` rows survive pointing at a server ``get_visible`` now
refuses them. The listing reads ``file_index`` directly, so the items
keep rendering as normal playable cards while every stream/image route
502s. See ``state/migrations/323_file_index_detached.sql`` for the full
diagnosis and why the fix reuses ``is_trashed`` for invisibility with
``detached_at`` carrying the reason.

This module owns both directions of that lifecycle:

* :func:`detach_server_rows` — tombstone a server's rows for everyone
  except its owner. Invisible everywhere (``is_trashed = 1``), exempt
  from every trash *semantic* (``detached_at IS NOT NULL``), progress
  and history preserved.
* :func:`reattach_server_rows` — the exact inverse, for a re-share.

Deliberately tombstones EVERYTHING at this stage. Cascading the rows a
user never touched is a later, separate step gated on measuring the
split on real data first — being wrong here has to stay reversible, and
a DELETE isn't.

Why batched, and why not in the request handler
-----------------------------------------------
``file_index`` has an unqualified ``AFTER UPDATE`` trigger maintaining
``file_index_fts`` (migration 074), so touching a row costs an FTS
delete + insert on top of the write. A shared library is ~63k rows.
One bare ``UPDATE ... WHERE json_extract(...)`` would be a full table
scan plus 63k trigger firings inside a single statement on the shared
aiosqlite connection — which serializes ALL app traffic behind it (see
``vfs/bulk.py`` for the full explanation). So callers run this as a
background job, and it walks in batches with a yield between them.

The batch queries are **self-draining**: each pass selects rows that
still need work, and the update is what removes them from the next
pass's result set. That makes the walk restart-safe with no cursor to
persist — a job killed halfway resumes exactly where it stopped.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

DEFAULT_BATCH_SIZE = 500
DEFAULT_INTER_BATCH_SLEEP_S = 0.05  # yield to ticks / voice / chat

# A walk that never drains would spin forever. Each pass must shrink the
# candidate set; if it somehow doesn't, this stops us rather than pinning
# a worker. 63k rows / 500 = 126 passes, so this is ~80x headroom.
_MAX_PASSES = 10_000


# Rows the user themselves deleted are left strictly alone: they already
# carry `is_trashed = 1, detached_at IS NULL`, which means "trash
# semantics apply". Detaching them would exempt them from the 30-day
# purge and make them unrestorable — silently changing the meaning of a
# deletion the user performed on purpose.
_DETACH_CANDIDATES = """
    SELECT id FROM file_index
     WHERE json_extract(source_metadata, '$.server_id') = ?
       AND user_id != ?
       AND is_trashed = 0
       AND detached_at IS NULL
     LIMIT ?
"""

_REATTACH_CANDIDATES = """
    SELECT id FROM file_index
     WHERE detached_server_id = ?
       AND detached_at IS NOT NULL
       AND user_id != ?
     LIMIT ?
"""


async def _walk(
    db: aiosqlite.Connection,
    *,
    select_sql: str,
    update_sql: str,
    params: tuple,
    batch_size: int,
    sleep_s: float,
    update_params: tuple = (),
) -> int:
    """Drive a self-draining batched UPDATE. Returns rows touched."""
    touched = 0
    for _ in range(_MAX_PASSES):
        cursor = await db.execute(select_sql, (*params, batch_size))
        ids = [row[0] for row in await cursor.fetchall()]
        await cursor.close()
        if not ids:
            return touched

        placeholders = ",".join("?" * len(ids))
        result = await db.execute(
            update_sql.format(placeholders=placeholders),
            (*update_params, *ids),
        )
        await db.commit()
        changed = result.rowcount or 0
        touched += changed
        if changed == 0:
            # The select found rows the update didn't move. Without this
            # the loop would re-select the same ids forever.
            log.warning("media_detach_walk_stalled", pending=len(ids))
            return touched
        if sleep_s:
            await asyncio.sleep(sleep_s)

    log.warning("media_detach_walk_max_passes", touched=touched)
    return touched


async def detach_server_rows(
    db: aiosqlite.Connection,
    server_id: str,
    *,
    owner_user_id: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_s: float = DEFAULT_INTER_BATCH_SLEEP_S,
) -> int:
    """Tombstone every borrower's rows for ``server_id``.

    ``owner_user_id`` is excluded — on an un-share the owner keeps a
    perfectly working private server, and on a delete the owner's rows
    are hard-purged by ``purge_server_data`` instead. Pass ``""`` to
    detach for every user (no real ``user_id`` is empty).

    Idempotent: rows already detached are not candidates, so a re-run
    (or a retried job) touches nothing.
    """
    if not server_id:
        raise ValueError("detach_server_rows requires server_id")

    touched = await _walk(
        db,
        select_sql=_DETACH_CANDIDATES,
        update_sql=(
            "UPDATE file_index SET "
            "is_trashed = 1, "
            "trashed_at = datetime('now'), "
            "detached_at = datetime('now'), "
            "detached_server_id = ?, "
            "updated_at = datetime('now') "
            "WHERE id IN ({placeholders})"
        ),
        update_params=(server_id,),
        params=(server_id, owner_user_id),
        batch_size=batch_size,
        sleep_s=sleep_s,
    )
    if touched:
        log.info(
            "media_server_detached",
            server_id=server_id, owner_user_id=owner_user_id, rows=touched,
        )
    return touched


async def reattach_server_rows(
    db: aiosqlite.Connection,
    server_id: str,
    *,
    owner_user_id: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep_s: float = DEFAULT_INTER_BATCH_SLEEP_S,
) -> int:
    """Restore rows detached by ``server_id`` — the inverse of detach.

    Matches on ``detached_server_id`` so re-sharing one server never
    resurrects rows tombstoned by a different one, and never touches a
    row the user deleted themselves (those have ``detached_at IS NULL``).

    Progress survives the round trip untouched: detach only ever wrote
    the four lifecycle columns.
    """
    if not server_id:
        raise ValueError("reattach_server_rows requires server_id")

    touched = await _walk(
        db,
        select_sql=_REATTACH_CANDIDATES,
        update_sql=(
            "UPDATE file_index SET "
            "is_trashed = 0, "
            "trashed_at = NULL, "
            "detached_at = NULL, "
            "detached_server_id = '', "
            "updated_at = datetime('now') "
            "WHERE id IN ({placeholders})"
        ),
        params=(server_id, owner_user_id),
        batch_size=batch_size,
        sleep_s=sleep_s,
    )
    if touched:
        log.info(
            "media_server_reattached",
            server_id=server_id, owner_user_id=owner_user_id, rows=touched,
        )
    return touched
