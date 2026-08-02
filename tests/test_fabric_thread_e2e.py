"""Per-thread E2E state store (Connect E2E P3)."""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.fabric.thread_e2e import get_e2e, is_e2e, set_e2e


async def _db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    with open("augmentum/state/migrations/295_connect_thread_e2e.sql") as f:
        await conn.executescript(f.read())
    await conn.commit()
    return conn


@pytest.mark.asyncio
async def test_default_absent_is_host_trusted():
    conn = await _db()
    try:
        st = await get_e2e(conn, user_id="u1", thread_id="t1")
        assert st.enabled is False and st.peer_master_did == ""
        assert await is_e2e(conn, user_id="u1", thread_id="t1") is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_enable_disable_round_trip():
    conn = await _db()
    try:
        await set_e2e(conn, user_id="u1", thread_id="t1", enabled=True,
                      peer_master_did="did:key:zMASTER")
        st = await get_e2e(conn, user_id="u1", thread_id="t1")
        assert st.enabled is True and st.peer_master_did == "did:key:zMASTER"
        assert await is_e2e(conn, user_id="u1", thread_id="t1") is True
        # per-user: u2 unaffected
        assert await is_e2e(conn, user_id="u2", thread_id="t1") is False
        # disable
        await set_e2e(conn, user_id="u1", thread_id="t1", enabled=False)
        assert await is_e2e(conn, user_id="u1", thread_id="t1") is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_requires_ids():
    conn = await _db()
    try:
        with pytest.raises(ValueError):
            await set_e2e(conn, user_id="", thread_id="t1", enabled=True)
    finally:
        await conn.close()
