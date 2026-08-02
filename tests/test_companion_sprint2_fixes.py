"""Sprint 2 persistence-atomicity fixes — regression pins (audit 2026-06-17).

The transactional refactors (recovery, healing, skills) are exercised for
*regressions* by their existing suites (test_companion_growth_substrate,
test_companion_skills, test_companion_healing). This file pins the one
genuinely NEW behavioral guarantee that no existing test covered: the
growth act-log append is now atomic under concurrency (was a lost-update
read-modify-write).
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from augmentum.companion.growth import GrowthStore

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "augmentum" / "state" / "migrations"
    / "216_companion_growth_substrate.sql"
)
_SCHEMA_VERSION_TABLE = (
    "CREATE TABLE IF NOT EXISTS schema_version ("
    "version INTEGER PRIMARY KEY, description TEXT NOT NULL DEFAULT '', "
    "applied_at INTEGER NOT NULL DEFAULT (strftime('%s','now')));"
)


async def _mkstore() -> tuple[GrowthStore, aiosqlite.Connection]:
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_VERSION_TABLE)
    await conn.executescript(_MIGRATION_PATH.read_text(encoding="utf-8"))
    await conn.commit()
    return GrowthStore(conn), conn


@pytest.mark.asyncio
async def test_append_act_step_atomic_under_concurrency():
    """20 concurrent appends must all land. The prior read-modify-write
    interleaved SELECT/UPDATE across the shared aiosqlite connection and
    lost steps; the json_insert single-statement append can't."""
    import asyncio
    store, conn = await _mkstore()
    try:
        entry = await store.start_session(user_id="u1")
        await asyncio.gather(*[
            store.append_act_step(entry.id, user_id="u1", step={"i": i})
            for i in range(20)
        ])
        final = await store.get_session(entry.id, user_id="u1")
        assert final is not None
        assert len(final.act_log) == 20
        # All 20 distinct indices present — none lost or duplicated.
        assert sorted(s["i"] for s in final.act_log) == list(range(20))
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_append_act_step_missing_row_warns_not_crashes():
    """A non-existent session id is a no-op + warning, not a crash."""
    store, conn = await _mkstore()
    try:
        # Should not raise; rowcount 0 → logged warning, no row touched.
        await store.append_act_step("nope", user_id="u1", step={"x": 1})
    finally:
        await conn.close()
