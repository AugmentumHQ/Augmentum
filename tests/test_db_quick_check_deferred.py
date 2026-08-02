"""Tests for the size-gated / deferred boot quick_check (2026-07-02).

``SQLiteBackend.connect()`` runs ``PRAGMA quick_check`` inline only when
the DB file is under ``AUGMENTUM_QUICK_CHECK_INLINE_MAX_MB`` (default 256).
Above the threshold it sets ``deferred_quick_check`` and the lifespan runs
``run_quick_check(deferred=True)`` as a background task — measured 40.5s
on a 1.5GB DB when it was inline, all of it gating first paint.

Locked-in behaviors:
  1. Small DBs keep the historical inline check (deferred_quick_check False).
  2. Above-threshold DBs skip inline and flag deferral.
  3. Deferred mode NEVER raises on corruption — it stamps the recovery
     gate, sets ``quick_check_failed``, and returns False.
  4. Inline mode still raises ``sqlite3.DatabaseError`` on corruption.
"""

from __future__ import annotations

import sqlite3

import pytest

from augmentum.state.backends.sqlite import SQLiteBackend


@pytest.fixture
async def file_backend(tmp_path):
    """A connected file-backed SQLiteBackend (small — inline path)."""
    be = SQLiteBackend(str(tmp_path / "t.db"))
    await be.connect()
    yield be
    await be.close()


class TestSizeGate:
    async def test_small_db_stays_inline(self, file_backend):
        # Under the 256MB default the historical inline path runs.
        assert file_backend.deferred_quick_check is False
        assert file_backend.quick_check_failed is False

    async def test_large_db_defers(self, tmp_path, monkeypatch):
        # Force the threshold to 0MB so any real file trips deferral.
        monkeypatch.setenv("AUGMENTUM_QUICK_CHECK_INLINE_MAX_MB", "0")
        be = SQLiteBackend(str(tmp_path / "big.db"))
        await be.connect()
        try:
            assert be.deferred_quick_check is True
            # The deferred run then passes on a healthy DB.
            assert await be.run_quick_check(deferred=True) is True
            assert be.quick_check_failed is False
        finally:
            await be.close()

    async def test_memory_db_always_inline(self):
        be = SQLiteBackend(":memory:")
        await be.connect()
        try:
            assert be.deferred_quick_check is False
        finally:
            await be.close()


class _CorruptCursor:
    async def fetchall(self):
        return [("row 1 missing from index idx_x",)]


class _CorruptConn:
    """Stub connection whose quick_check reports corruption."""

    async def execute(self, sql, *a, **k):
        assert "quick_check" in sql
        return _CorruptCursor()


class TestCorruptionPaths:
    def _stamped(self, be, calls):
        be._touch_recovery_stamp = lambda reason: calls.append(reason)

    async def test_deferred_corruption_never_raises(self, tmp_path):
        be = SQLiteBackend(str(tmp_path / "c.db"))
        be._conn = _CorruptConn()
        calls: list[str] = []
        self._stamped(be, calls)

        ok = await be.run_quick_check(deferred=True)

        assert ok is False
        assert be.quick_check_failed is True
        assert calls and calls[0].startswith("quick_check_failed:")

    async def test_inline_corruption_still_raises(self, tmp_path):
        be = SQLiteBackend(str(tmp_path / "c2.db"))
        be._conn = _CorruptConn()
        calls: list[str] = []
        self._stamped(be, calls)

        with pytest.raises(sqlite3.DatabaseError):
            await be.run_quick_check()

        assert be.quick_check_failed is True
        assert calls and calls[0].startswith("quick_check_failed:")

    async def test_requires_connection(self):
        be = SQLiteBackend(":memory:")
        with pytest.raises(RuntimeError):
            await be.run_quick_check()
