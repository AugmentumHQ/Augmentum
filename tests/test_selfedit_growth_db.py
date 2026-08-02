"""Isolated growth-DB store tests.

Proves the growth system's bookkeeping lives in its OWN SQLite file (derived onto
the /data volume), opens lazily, is cached, carries the full self-edit schema, and
— the point — is a separate file from the main DB so it can't affect it.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from augmentum.selfedit import growth_db, store


def _app_state(tmp_path):
    db_path = str(tmp_path / "augmentum.db")
    return SimpleNamespace(state_manager=SimpleNamespace(backend=SimpleNamespace(_db_path=db_path)))


def test_growth_path_is_separate_file_on_data_volume(tmp_path):
    st = _app_state(tmp_path)
    p = growth_db.growth_db_path(st)
    assert p == str(tmp_path / "selfedit" / "growth.db")
    assert p != st.state_manager.backend._db_path  # NOT the main DB


async def test_lazy_open_creates_schema_and_caches(tmp_path):
    st = _app_state(tmp_path)
    conn = await growth_db.get_growth_conn(st)
    try:
        assert conn is not None
        # the file exists, separate from main
        assert os.path.exists(str(tmp_path / "selfedit" / "growth.db"))
        # the full self-edit schema is present
        cur = await conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {r[0] for r in await cur.fetchall()}
        assert {"self_edit_attempts", "claude_runs", "claude_run_events"} <= tables
        # cached: a second call returns the same connection object
        assert await growth_db.get_growth_conn(st) is conn
    finally:
        await growth_db.close_growth_conn(st)


async def test_archive_writes_go_to_growth_not_main(tmp_path):
    st = _app_state(tmp_path)
    conn = await growth_db.get_growth_conn(st)
    try:
        await store.create_attempt(conn, attempt_id="g1", user_id="u1",
                                   objective="isolate me", surface="config")
        await store.finalize(conn, attempt_id="g1", user_id="u1", status="promoted",
                             outcome="ok", lesson="isolated + durable")
        got = await store.get_attempt(conn, attempt_id="g1", user_id="u1")
        assert got and got["status"] == "promoted" and got["lesson"] == "isolated + durable"
        # the main DB file was never created by any of this
        assert not os.path.exists(st.state_manager.backend._db_path)
    finally:
        await growth_db.close_growth_conn(st)


async def test_open_failure_is_contained(tmp_path):
    # An unresolvable path → returns None (caller degrades), never raises.
    st = SimpleNamespace(state_manager=SimpleNamespace(
        backend=SimpleNamespace(_db_path="\x00/illegal/augmentum.db")))
    conn = await growth_db.get_growth_conn(st)
    assert conn is None  # contained — the main app is never affected


# ---------------------------------------------------------------------------
# durability — the archive is backed up, not just isolated (Gap 4 fix)
# ---------------------------------------------------------------------------

async def test_open_snapshots_the_archive(tmp_path):
    # The docstring's promise, now enforced: opening the growth DB drops a
    # VACUUM INTO snapshot into selfedit/backups/ — the archive is never the
    # one store with no copy anywhere.
    st = _app_state(tmp_path)
    conn = await growth_db.get_growth_conn(st)
    try:
        await store.create_attempt(conn, attempt_id="g1", user_id="u1",
                                   objective="worth preserving", surface="config")
        backups = tmp_path / "selfedit" / "backups"
        assert backups.exists()
        snaps = list(backups.glob("growth_*.db"))
        assert len(snaps) == 1
        # the snapshot is a real, self-contained SQLite DB carrying the schema
        import aiosqlite
        snap = await aiosqlite.connect(str(snaps[0]))
        try:
            cur = await snap.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in await cur.fetchall()}
            assert "self_edit_attempts" in tables
        finally:
            await snap.close()
    finally:
        await growth_db.close_growth_conn(st)


async def test_backup_is_interval_gated(tmp_path):
    # A second open within the interval window must NOT stack a second snapshot
    # (the same lock-storm guard the main DB uses). Re-open by dropping the cache.
    st = _app_state(tmp_path)
    conn = await growth_db.get_growth_conn(st)
    assert conn is not None
    await growth_db.close_growth_conn(st)   # drops the cached conn
    assert await growth_db.get_growth_conn(st) is not None  # re-opens
    try:
        snaps = list((tmp_path / "selfedit" / "backups").glob("growth_*.db"))
        assert len(snaps) == 1  # still one — the interval gate held
    finally:
        await growth_db.close_growth_conn(st)


async def test_backup_failure_never_blocks_open(tmp_path, monkeypatch):
    # If the backup itself throws, opening the archive still succeeds — a lost
    # backup must never cost us the working store.
    async def _boom(*a, **k):
        raise RuntimeError("disk full mid-VACUUM")
    monkeypatch.setattr("augmentum.state.backup.backup_database", _boom)
    st = _app_state(tmp_path)
    conn = await growth_db.get_growth_conn(st)
    try:
        assert conn is not None  # the open survived the backup failure
    finally:
        await growth_db.close_growth_conn(st)
