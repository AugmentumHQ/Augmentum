"""Teardown of a revoked media share tombstones, not deletes.

Pins the three things that make step 3 safe:

  * borrowers' rows go invisible, owner's are untouched
  * the tombstone is fully reversible and preserves progress
  * rows the USER deleted keep their own trash semantics

Runs against the REAL migration set (SQLiteBackend applies them on
connect) so schema drift can't hide behind a hand-written fixture.
"""

from __future__ import annotations

import json

import pytest

from augmentum.media.detach import detach_server_rows, reattach_server_rows
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.vfs.index import FileIndexService

pytestmark = pytest.mark.asyncio

OWNER = "u_admin"
BORROWER = "u_bench"
OTHER = "u_third"
SRV = "srv1"


async def _backend():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    for uid in (OWNER, BORROWER, OTHER):
        # NOT `INSERT OR IGNORE` with just an id: users.username and
        # .password_hash are NOT NULL with no default, and OR IGNORE
        # swallows that violation silently — leaving no user row and
        # failing every file_index insert on the FK instead.
        await backend.conn.execute(
            "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
            (uid, uid, "x"),
        )
    await backend.conn.commit()
    return backend


async def _item(conn, *, user_id: str, source_id: str, server_id: str = SRV,
                progress: float = 0.0) -> str:
    idx = FileIndexService(conn)
    return await idx.register(
        user_id=user_id, source="emby", source_id=source_id, name=source_id,
        source_metadata={"server_id": server_id, "progress_pct": progress},
    )


async def _state(conn, file_id: str) -> tuple:
    cur = await conn.execute(
        "SELECT is_trashed, detached_at, detached_server_id, source_metadata "
        "FROM file_index WHERE id = ?",
        (file_id,),
    )
    return await cur.fetchone()


# --- the teardown --------------------------------------------------------

async def test_detach_hides_borrower_rows_and_spares_the_owner():
    backend = await _backend()
    try:
        conn = backend.conn
        owner_row = await _item(conn, user_id=OWNER, source_id="a")
        borrowed = await _item(conn, user_id=BORROWER, source_id="a")

        rows = await detach_server_rows(
            conn, SRV, owner_user_id=OWNER, sleep_s=0,
        )

        assert rows == 1
        is_trashed, detached_at, detached_srv, _ = await _state(conn, borrowed)
        assert is_trashed == 1
        assert detached_at is not None
        assert detached_srv == SRV

        is_trashed, detached_at, _, _ = await _state(conn, owner_row)
        assert (is_trashed, detached_at) == (0, None)
    finally:
        await backend.close()


async def test_detach_covers_every_borrower_not_just_one():
    backend = await _backend()
    try:
        conn = backend.conn
        await _item(conn, user_id=BORROWER, source_id="a")
        await _item(conn, user_id=OTHER, source_id="a")

        assert await detach_server_rows(
            conn, SRV, owner_user_id=OWNER, sleep_s=0,
        ) == 2
    finally:
        await backend.close()


async def test_detach_ignores_other_servers():
    backend = await _backend()
    try:
        conn = backend.conn
        keep = await _item(conn, user_id=BORROWER, source_id="b",
                           server_id="srv2")
        await detach_server_rows(conn, SRV, owner_user_id=OWNER, sleep_s=0)
        assert (await _state(conn, keep))[0] == 0
    finally:
        await backend.close()


async def test_detached_rows_vanish_from_the_library_listing():
    """The actual bug: ghosts kept rendering as playable cards."""
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        await _item(conn, user_id=BORROWER, source_id="gone")
        await _item(conn, user_id=BORROWER, source_id="local", server_id="srv2")

        await detach_server_rows(conn, SRV, owner_user_id=OWNER, sleep_s=0)

        listed = await idx.list_recent(user_id=BORROWER)
        assert [e.source_id for e in listed] == ["local"]
        # And not in Trash either — the user didn't delete it.
        assert await idx.list_trash(user_id=BORROWER) == []
    finally:
        await backend.close()


async def test_detach_is_idempotent():
    backend = await _backend()
    try:
        conn = backend.conn
        await _item(conn, user_id=BORROWER, source_id="a")
        first = await detach_server_rows(conn, SRV, owner_user_id=OWNER,
                                         sleep_s=0)
        second = await detach_server_rows(conn, SRV, owner_user_id=OWNER,
                                          sleep_s=0)
        assert (first, second) == (1, 0)
    finally:
        await backend.close()


async def test_detach_leaves_user_deleted_rows_alone():
    """Their trash semantics are deliberate — don't make them un-purgeable."""
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        file_id = await _item(conn, user_id=BORROWER, source_id="a")
        await idx.soft_delete(file_id, user_id=BORROWER)

        assert await detach_server_rows(
            conn, SRV, owner_user_id=OWNER, sleep_s=0,
        ) == 0
        is_trashed, detached_at, _, _ = await _state(conn, file_id)
        assert (is_trashed, detached_at) == (1, None)
        # Still restorable by the user who trashed it.
        assert await idx.restore(file_id, user_id=BORROWER) is True
    finally:
        await backend.close()


async def test_walk_drains_across_multiple_batches():
    """Self-draining select: the update is what shrinks the candidate set."""
    backend = await _backend()
    try:
        conn = backend.conn
        for n in range(7):
            await _item(conn, user_id=BORROWER, source_id=f"i{n}")
        assert await detach_server_rows(
            conn, SRV, owner_user_id=OWNER, batch_size=2, sleep_s=0,
        ) == 7
    finally:
        await backend.close()


# --- the inverse ---------------------------------------------------------

async def test_reattach_restores_rows_with_progress_intact():
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        file_id = await _item(conn, user_id=BORROWER, source_id="a",
                              progress=61.5)

        await detach_server_rows(conn, SRV, owner_user_id=OWNER, sleep_s=0)
        assert await reattach_server_rows(
            conn, SRV, owner_user_id=OWNER, sleep_s=0,
        ) == 1

        is_trashed, detached_at, detached_srv, meta = await _state(conn, file_id)
        assert (is_trashed, detached_at, detached_srv) == (0, None, "")
        assert json.loads(meta)["progress_pct"] == 61.5
        assert [e.source_id for e in await idx.list_recent(user_id=BORROWER)] \
            == ["a"]
    finally:
        await backend.close()


async def test_reattach_only_touches_the_server_that_detached_them():
    backend = await _backend()
    try:
        conn = backend.conn
        a = await _item(conn, user_id=BORROWER, source_id="a")
        b = await _item(conn, user_id=BORROWER, source_id="b", server_id="srv2")
        await detach_server_rows(conn, SRV, owner_user_id=OWNER, sleep_s=0)
        await detach_server_rows(conn, "srv2", owner_user_id=OWNER, sleep_s=0)

        assert await reattach_server_rows(
            conn, SRV, owner_user_id=OWNER, sleep_s=0,
        ) == 1
        assert (await _state(conn, a))[0] == 0
        assert (await _state(conn, b))[0] == 1
    finally:
        await backend.close()


async def test_reattach_never_resurrects_a_user_deleted_row():
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        file_id = await _item(conn, user_id=BORROWER, source_id="a")
        await idx.soft_delete(file_id, user_id=BORROWER)

        assert await reattach_server_rows(
            conn, SRV, owner_user_id=OWNER, sleep_s=0,
        ) == 0
        assert (await _state(conn, file_id))[0] == 1
    finally:
        await backend.close()
