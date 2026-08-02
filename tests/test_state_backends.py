"""Tests for state backends (MemoryBackend, SQLiteBackend, StateManager)."""

from __future__ import annotations

import pytest

from augmentum.state.backends.memory import MemoryBackend
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager


class TestMemoryBackend:
    """In-memory backend CRUD operations."""

    async def test_create_session(self):
        backend = MemoryBackend()
        session = await backend.create_session("s1", mode="analytical")
        assert session["id"] == "s1"
        assert session["mode"] == "analytical"
        assert session["message_count"] == 0

    async def test_get_session(self):
        backend = MemoryBackend()
        await backend.create_session("s1")
        session = await backend.get_session("s1")
        assert session is not None
        assert session["id"] == "s1"

    async def test_get_nonexistent_session(self):
        backend = MemoryBackend()
        assert await backend.get_session("nope") is None

    async def test_update_session_mode(self):
        backend = MemoryBackend()
        await backend.create_session("s1", mode="passthrough")
        await backend.update_session("s1", mode="narrative")
        session = await backend.get_session("s1")
        assert session["mode"] == "narrative"

    async def test_update_session_increment_messages(self):
        backend = MemoryBackend()
        await backend.create_session("s1")
        await backend.update_session("s1", increment_messages=True)
        await backend.update_session("s1", increment_messages=True)
        session = await backend.get_session("s1")
        assert session["message_count"] == 2

    async def test_update_session_metadata(self):
        backend = MemoryBackend()
        await backend.create_session("s1")
        await backend.update_session("s1", metadata='{"key":"val"}')
        session = await backend.get_session("s1")
        assert session["metadata"] == '{"key":"val"}'

    async def test_update_nonexistent_session_no_error(self):
        backend = MemoryBackend()
        # Should not raise
        await backend.update_session("ghost", mode="analytical")


class TestSQLiteBackend:
    """SQLite backend lifecycle and session operations."""

    async def test_connect_and_close(self):
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        assert backend._conn is not None
        await backend.close()
        assert backend._conn is None

    async def test_conn_property_raises_before_connect(self):
        backend = SQLiteBackend(":memory:")
        with pytest.raises(RuntimeError, match="not connected"):
            _ = backend.conn

    async def test_create_and_get_session(self):
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            session = await backend.create_session("s1", mode="narrative")
            assert session["id"] == "s1"
            assert session["mode"] == "narrative"

            fetched = await backend.get_session("s1")
            assert fetched is not None
            assert fetched["id"] == "s1"
        finally:
            await backend.close()

    async def test_get_nonexistent_session(self):
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            assert await backend.get_session("nope") is None
        finally:
            await backend.close()

    async def test_update_session(self):
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            await backend.create_session("s1")
            await backend.update_session("s1", mode="analytical", increment_messages=True)
            session = await backend.get_session("s1")
            assert session["mode"] == "analytical"
            assert session["message_count"] == 1
        finally:
            await backend.close()


class TestSQLiteHealthCheck:
    """Post-startup health-check sweep: detects FTS5 corruption that
    PRAGMA integrity_check misses, repairs it in place, and trips the
    recovery-gate stamp on unrepairable failures."""

    async def test_clean_db_passes_health_check(self):
        backend = SQLiteBackend(":memory:")
        await backend.connect()
        try:
            # Clean :memory: DB should have run quick_check + FTS5
            # integrity-check on every fts table without raising. The
            # connect() returning normally is the assertion.
            fts = await backend._list_fts5_tables()
            assert len(fts) >= 1
            for t in fts:
                assert await backend._fts5_integrity_ok(t)
        finally:
            await backend.close()

    async def test_fts5_corruption_is_auto_repaired(self, tmp_path):
        import sqlite3

        db_path = tmp_path / "fts5_corrupt.db"
        backend = SQLiteBackend(str(db_path))
        await backend.connect()
        try:
            # Seed a row so the FTS index has content to verify against.
            await backend.conn.execute(
                "INSERT INTO memories (id, user_id, content, memory_type, "
                "created_at) VALUES "
                "('m1', 'u1', 'the quick brown fox', 'fact', "
                "datetime('now'))"
            )
            await backend.conn.commit()

            assert await backend._fts5_integrity_ok("memories_fts")
        finally:
            await backend.close()

        # Forcibly corrupt the FTS shadow with the sync-conn (bypasses
        # FTS5's normal write path). Overwriting the 'data' BLOB column
        # with junk produces the same internally-inconsistent state we
        # saw in production after a .recover-based rebuild.
        raw = sqlite3.connect(str(db_path))
        try:
            raw.execute(
                "UPDATE memories_fts_data SET block = X'00' WHERE id = 1"
            )
            raw.commit()
        finally:
            raw.close()

        # Reopening must DETECT the corruption AND repair it. After
        # connect() returns, FTS5 integrity-check should pass and a
        # MATCH query should return the seeded row.
        backend = SQLiteBackend(str(db_path))
        await backend.connect()
        try:
            assert await backend._fts5_integrity_ok("memories_fts")
            cursor = await backend.conn.execute(
                "SELECT count(*) FROM memories_fts "
                "WHERE memories_fts MATCH ?",
                ("fox",),
            )
            row = await cursor.fetchone()
            assert row[0] == 1
        finally:
            await backend.close()

    async def test_recovery_stamp_appends(self, tmp_path):
        db_path = tmp_path / "stamp_test.db"
        backend = SQLiteBackend(str(db_path))
        await backend.connect()
        try:
            backend._touch_recovery_stamp("reason_one")
            backend._touch_recovery_stamp("reason_two")
        finally:
            await backend.close()

        stamp = db_path.parent / ".augmentum_recovery_stamp"
        assert stamp.exists()
        contents = stamp.read_text(encoding="utf-8")
        assert "reason_one" in contents
        assert "reason_two" in contents
        # One line per touch.
        assert contents.count("\n") == 2


class TestStateManager:
    """StateManager delegates to backend."""

    async def test_get_or_create_session_creates(self):
        backend = MemoryBackend()
        mgr = StateManager(backend)
        session = await mgr.get_or_create_session("s1", mode="passthrough")
        assert session["id"] == "s1"

    async def test_get_or_create_session_returns_existing(self):
        backend = MemoryBackend()
        mgr = StateManager(backend)
        await mgr.get_or_create_session("s1", mode="passthrough")
        session = await mgr.get_or_create_session("s1", mode="analytical")
        # Should return existing with original mode
        assert session["mode"] == "passthrough"

    async def test_update_session_delegates(self):
        backend = MemoryBackend()
        mgr = StateManager(backend)
        await mgr.get_or_create_session("s1")
        await mgr.update_session("s1", increment_messages=True)
        session = await backend.get_session("s1")
        assert session["message_count"] == 1

    async def test_backend_property(self):
        backend = MemoryBackend()
        mgr = StateManager(backend)
        assert mgr.backend is backend
