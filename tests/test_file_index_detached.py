"""Detached rows are invisible but exempt from trash semantics.

Migration 323 overloads ``is_trashed`` to mean two things, separated by
``detached_at``:

  detached_at IS NULL     — user deleted it. Trash semantics apply.
  detached_at IS NOT NULL — a media server was un-shared out from under
                            the row. Hidden, but its progress/history is
                            PRESERVED for a possible re-share.

The dangerous failure is silent and delayed: ``purge_all_old_trash``
runs unattended and would hard-delete preserved history 30 days later.
These tests pin every trash-semantics path against that.

They run against the REAL migration set (SQLiteBackend applies them on
connect), so schema drift can't hide behind a hand-written fixture.
"""

from __future__ import annotations

import pytest

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.vfs.index import FileIndexService

pytestmark = pytest.mark.asyncio

USER = "u1"


async def _backend():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    # NOT `INSERT OR IGNORE` with just an id: users.username and
    # .password_hash are NOT NULL with no default, and OR IGNORE swallows
    # that violation silently — leaving no user row and failing every
    # file_index insert on the FK instead.
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
        (USER, "tester", "x"),
    )
    await backend.conn.commit()
    return backend


async def _row(conn, source_id: str, *, trashed: bool, detached: bool,
               trashed_age_days: int = 0) -> str:
    """Insert one file_index row in a chosen lifecycle state."""
    idx = FileIndexService(conn)
    file_id = await idx.register(
        user_id=USER, source="emby", source_id=source_id, name=source_id,
    )
    if trashed:
        await conn.execute(
            "UPDATE file_index SET is_trashed = 1, "
            "trashed_at = datetime('now', ?) WHERE id = ?",
            (f"-{trashed_age_days} days", file_id),
        )
    if detached:
        await conn.execute(
            "UPDATE file_index SET detached_at = datetime('now'), "
            "detached_server_id = 'srv1' WHERE id = ?",
            (file_id,),
        )
    await conn.commit()
    return file_id


# --- the migration itself -------------------------------------------------

async def test_migration_adds_columns_without_touching_existing_rows():
    """Additive only: every pre-existing row must read as NOT detached."""
    backend = await _backend()
    try:
        conn = backend.conn
        await _row(conn, "a", trashed=False, detached=False)
        cur = await conn.execute(
            "SELECT COUNT(*) FROM file_index WHERE detached_at IS NOT NULL",
        )
        assert (await cur.fetchone())[0] == 0
    finally:
        await backend.close()


# --- the delayed, silent killer ------------------------------------------

async def test_unattended_purge_spares_detached_rows():
    """The whole reason detached_at exists as its own column.

    Without the guard, preserved share history is hard-deleted 30 days
    after an un-share, across all users, with no user action.
    """
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        kept = await _row(conn, "detached", trashed=True, detached=True,
                          trashed_age_days=90)
        doomed = await _row(conn, "user_deleted", trashed=True, detached=False,
                            trashed_age_days=90)

        purged = await idx.purge_all_old_trash(older_than_days=30)

        assert purged == 1
        cur = await conn.execute(
            "SELECT id FROM file_index WHERE id IN (?, ?)", (kept, doomed),
        )
        remaining = [r[0] for r in await cur.fetchall()]
        assert remaining == [kept]
    finally:
        await backend.close()


async def test_maintenance_sweep_does_not_list_detached_rows():
    """list_trashed_older_than feeds the same loop — guard it too."""
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        await _row(conn, "detached", trashed=True, detached=True,
                   trashed_age_days=90)
        entries = await idx.list_trashed_older_than(30)
        assert entries == []
    finally:
        await backend.close()


# --- user-facing trash semantics -----------------------------------------

async def test_detached_rows_absent_from_trash_listing_and_count():
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        await _row(conn, "detached", trashed=True, detached=True)
        await _row(conn, "user_deleted", trashed=True, detached=False)

        listed = await idx.list_trash(user_id=USER)
        assert [e.source_id for e in listed] == ["user_deleted"]

        stats = await idx.stats(user_id=USER)
        assert stats["trash"] == 1
    finally:
        await backend.close()


async def test_empty_trash_spares_detached_rows():
    """"Empty Trash" must not delete what it never showed the user."""
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        kept = await _row(conn, "detached", trashed=True, detached=True)
        await _row(conn, "user_deleted", trashed=True, detached=False)

        removed = await idx.purge_trash(user_id=USER)

        assert removed == 1
        cur = await conn.execute("SELECT id FROM file_index")
        assert [r[0] for r in await cur.fetchall()] == [kept]
    finally:
        await backend.close()


async def test_restore_refuses_detached_rows():
    """Takes a raw file_id, so it's reachable even though the listing hides it."""
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        file_id = await _row(conn, "detached", trashed=True, detached=True)

        assert await idx.restore(file_id, user_id=USER) is False

        cur = await conn.execute(
            "SELECT is_trashed FROM file_index WHERE id = ?", (file_id,),
        )
        assert (await cur.fetchone())[0] == 1
    finally:
        await backend.close()


# --- regression guard: normal trash still works --------------------------

async def test_ordinary_trash_lifecycle_unaffected():
    """Nothing above may change behavior for rows the user actually deleted."""
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        file_id = await _row(conn, "normal", trashed=False, detached=False)

        assert await idx.soft_delete(file_id, user_id=USER) is True
        assert [e.id for e in await idx.list_trash(user_id=USER)] == [file_id]
        assert await idx.restore(file_id, user_id=USER) is True
        assert await idx.list_trash(user_id=USER) == []
    finally:
        await backend.close()


async def test_detached_rows_are_invisible_to_normal_listing():
    """Reusing is_trashed buys invisibility across ~30 queries for free.

    This pins that the reuse actually delivers it, via the shared
    `is_trashed = 0` filter every listing path carries.
    """
    backend = await _backend()
    try:
        conn = backend.conn
        idx = FileIndexService(conn)
        await _row(conn, "detached", trashed=True, detached=True)
        await _row(conn, "live", trashed=False, detached=False)

        entries = await idx.list_recent(user_id=USER)
        assert [e.source_id for e in entries] == ["live"]
    finally:
        await backend.close()
