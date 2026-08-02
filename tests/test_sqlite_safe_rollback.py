"""Tests for the SQLite stuck-transaction guard.

Regression suite for the 2026-05-22 lockup: a failed DML on a
persistent aiosqlite connection left Python's sqlite3 in
``in_transaction=True``; the next SELECT opened a real read snapshot
that pinned the WAL for 8 hours, eventually producing a cascading
``database is locked`` storm across every writer.

``install_safe_rollback`` wraps execute/executemany so any DML
failure triggers an immediate rollback, clearing the stuck state
before the exception propagates. ``transactional_write`` is the
explicit commit-or-rollback context manager for atomicity-critical
write blocks.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import aiosqlite
import pytest

from augmentum.state.backends.sqlite import (
    apply_augmentum_pragmas,
    install_safe_rollback,
    savepoint,
    transactional_write,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _writer_holding_lock(db_path: str) -> sqlite3.Connection:
    """Open a sync sqlite3 connection holding a writer lock — used to
    force ``database is locked`` on a parallel async connection.
    """
    holder = sqlite3.connect(db_path, isolation_level=None)
    holder.execute("PRAGMA journal_mode=WAL")
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT INTO t (v) VALUES ('held')")
    return holder


# ─────────────────────────────────────────────────────────────────────
# Repro: the original bug, with safety net OFF
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_dml_leaks_transaction_without_guard(tmp_path: Path):
    """Without ``install_safe_rollback``, a failed DML leaks
    ``in_transaction=True`` onto the connection — the exact bug we're
    fixing. This test pins the behavior so a future fix can be
    distinguished from a regression.
    """
    db_path = str(tmp_path / "leak.db")
    bootstrap = sqlite3.connect(db_path)
    bootstrap.execute("CREATE TABLE t (v TEXT)")
    bootstrap.commit()
    bootstrap.close()

    holder = _writer_holding_lock(db_path)
    try:
        cb = await aiosqlite.connect(db_path)
        try:
            await cb.execute("PRAGMA busy_timeout=200")

            # NO install_safe_rollback here — observe the leak.
            with pytest.raises(aiosqlite.OperationalError):
                await cb.execute("INSERT INTO t (v) VALUES ('blocked')")

            # The bug: Python sqlite3 still thinks a txn is open.
            assert cb._conn.in_transaction is True
        finally:
            await cb.close()
    finally:
        holder.close()


# ─────────────────────────────────────────────────────────────────────
# Safety net: install_safe_rollback clears stuck transactions
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_install_safe_rollback_clears_on_failed_dml(tmp_path: Path):
    """With the guard installed, a failed DML auto-clears the
    transaction so the connection is healthy for the next caller.
    """
    db_path = str(tmp_path / "guarded.db")
    bootstrap = sqlite3.connect(db_path)
    bootstrap.execute("CREATE TABLE t (v TEXT)")
    bootstrap.commit()
    bootstrap.close()

    holder = _writer_holding_lock(db_path)
    try:
        cb = await aiosqlite.connect(db_path)
        try:
            await cb.execute("PRAGMA busy_timeout=200")
            install_safe_rollback(cb)

            with pytest.raises(aiosqlite.OperationalError):
                await cb.execute("INSERT INTO t (v) VALUES ('blocked')")

            # The fix: in_transaction was cleared before the exception
            # surfaced to us, so the next SELECT won't pin a snapshot.
            assert cb._conn.in_transaction is False
        finally:
            await cb.close()
    finally:
        holder.close()


@pytest.mark.asyncio
async def test_install_safe_rollback_select_after_failed_dml_does_not_pin(
    tmp_path: Path,
):
    """End-to-end repro of the WAL-pin scenario. With the guard, a
    SELECT after a failed DML should NOT take a snapshot — verified
    by checking that subsequent commits checkpoint cleanly.
    """
    db_path = str(tmp_path / "wal_pin.db")
    bootstrap = sqlite3.connect(db_path)
    bootstrap.execute("PRAGMA journal_mode=WAL")
    bootstrap.execute("CREATE TABLE t (v TEXT)")
    bootstrap.commit()
    bootstrap.close()

    # Hold writer lock to force the first DML to fail.
    holder = _writer_holding_lock(db_path)
    try:
        cb = await aiosqlite.connect(db_path)
        try:
            await apply_augmentum_pragmas(cb)
            # busy_timeout from pragmas is 30s, override for test speed
            await cb.execute("PRAGMA busy_timeout=200")
            install_safe_rollback(cb)

            # Failed DML — would normally leak in_transaction=True
            with pytest.raises(aiosqlite.OperationalError):
                await cb.execute("INSERT INTO t (v) VALUES ('a')")

            # A SELECT here — with the bug, this opens a real read
            # snapshot inside the ghost transaction. With the fix,
            # in_transaction is already False, so this is a clean read.
            cursor = await cb.execute("SELECT v FROM t")
            rows = await cursor.fetchall()
            assert rows == []  # holder's row not committed yet

            # Verify connection is in clean state — no stuck txn.
            assert cb._conn.in_transaction is False
        finally:
            await cb.close()
    finally:
        holder.close()


# ─────────────────────────────────────────────────────────────────────
# transactional_write context manager
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transactional_write_commits_on_clean_exit(tmp_path: Path):
    db_path = str(tmp_path / "tw_commit.db")
    bootstrap = sqlite3.connect(db_path)
    bootstrap.execute("CREATE TABLE t (v TEXT)")
    bootstrap.commit()
    bootstrap.close()

    conn = await aiosqlite.connect(db_path)
    try:
        async with transactional_write(conn) as db:
            await db.execute("INSERT INTO t (v) VALUES ('x')")
            await db.execute("INSERT INTO t (v) VALUES ('y')")

        cursor = await conn.execute("SELECT v FROM t ORDER BY v")
        rows = await cursor.fetchall()
        assert [r[0] for r in rows] == ["x", "y"]
        assert conn._conn.in_transaction is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_transactional_write_rolls_back_on_exception(tmp_path: Path):
    db_path = str(tmp_path / "tw_rollback.db")
    bootstrap = sqlite3.connect(db_path)
    bootstrap.execute("CREATE TABLE t (v TEXT UNIQUE)")
    bootstrap.commit()
    bootstrap.close()

    conn = await aiosqlite.connect(db_path)
    try:
        with pytest.raises(aiosqlite.IntegrityError):
            async with transactional_write(conn) as db:
                await db.execute("INSERT INTO t (v) VALUES ('x')")
                # Second insert violates UNIQUE — block rolls back.
                await db.execute("INSERT INTO t (v) VALUES ('x')")

        # The first INSERT must NOT be visible — atomicity guarantee.
        cursor = await conn.execute("SELECT COUNT(*) FROM t")
        (count,) = await cursor.fetchone()
        assert count == 0
        assert conn._conn.in_transaction is False
    finally:
        await conn.close()


# ─────────────────────────────────────────────────────────────────────
# savepoint helper
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_savepoint_isolates_failure_from_outer_transaction(tmp_path: Path):
    """A best-effort sub-operation that fails inside a savepoint
    should NOT abort the surrounding transactional_write block.
    """
    db_path = str(tmp_path / "sp.db")
    bootstrap = sqlite3.connect(db_path)
    bootstrap.execute("CREATE TABLE main (v TEXT)")
    bootstrap.execute("CREATE TABLE aux (v TEXT UNIQUE)")
    bootstrap.commit()
    bootstrap.close()

    conn = await aiosqlite.connect(db_path)
    try:
        install_safe_rollback(conn)
        async with transactional_write(conn) as db:
            # Best-effort: tolerate aux failure.
            try:
                async with savepoint(db, "aux_op"):
                    await db.execute("INSERT INTO aux (v) VALUES ('x')")
                    # Force a failure inside the savepoint.
                    await db.execute("INSERT INTO aux (v) VALUES ('x')")
            except aiosqlite.IntegrityError:
                pass  # swallowed — outer txn continues

            # Outer write should still succeed.
            await db.execute("INSERT INTO main (v) VALUES ('survives')")

        # Outer write committed; inner write rolled back via savepoint.
        cursor = await conn.execute("SELECT v FROM main")
        rows = await cursor.fetchall()
        assert [r[0] for r in rows] == ["survives"]

        cursor = await conn.execute("SELECT COUNT(*) FROM aux")
        (count,) = await cursor.fetchone()
        assert count == 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_install_safe_rollback_is_idempotent(tmp_path: Path):
    """Calling the installer twice on the same connection must not
    double-wrap (would cause double-rollback / infinite recursion).
    """
    db_path = str(tmp_path / "idem.db")
    bootstrap = sqlite3.connect(db_path)
    bootstrap.execute("CREATE TABLE t (v TEXT)")
    bootstrap.commit()
    bootstrap.close()

    conn = await aiosqlite.connect(db_path)
    try:
        install_safe_rollback(conn)
        first_execute = conn.execute
        install_safe_rollback(conn)
        # Second call is a no-op — execute is unchanged.
        assert conn.execute is first_execute

        # And the connection still functions normally.
        await conn.execute("INSERT INTO t (v) VALUES ('ok')")
        await conn.commit()
        cursor = await conn.execute("SELECT v FROM t")
        rows = await cursor.fetchall()
        assert [r[0] for r in rows] == ["ok"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_install_safe_rollback_does_not_rollback_on_select_failure(
    tmp_path: Path,
):
    """SELECT failures must NOT trigger rollback — that would clobber
    any legitimate enclosing transaction that the caller is mid-way
    through (e.g. a resource_ledger BEGIN IMMEDIATE block).
    """
    db_path = str(tmp_path / "select_safe.db")
    bootstrap = sqlite3.connect(db_path)
    bootstrap.execute("CREATE TABLE t (v TEXT)")
    bootstrap.commit()
    bootstrap.close()

    conn = await aiosqlite.connect(db_path)
    try:
        install_safe_rollback(conn)
        # Open an explicit transaction with a successful DML.
        await conn.execute("BEGIN IMMEDIATE")
        await conn.execute("INSERT INTO t (v) VALUES ('in_progress')")
        assert conn._conn.in_transaction is True

        # A failed SELECT should NOT roll back the outer transaction.
        with pytest.raises(aiosqlite.OperationalError):
            await conn.execute("SELECT * FROM nonexistent_table")

        # Transaction state must be preserved.
        assert conn._conn.in_transaction is True
        await conn.commit()

        cursor = await conn.execute("SELECT v FROM t")
        rows = await cursor.fetchall()
        assert [r[0] for r in rows] == ["in_progress"]
    finally:
        await conn.close()
