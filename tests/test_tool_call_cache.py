"""Unit tests for the agentic tool-call cache (migrations 079 + 087)."""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from augmentum.modes.agentic.task_state import ToolCallCache, hash_tool_call

_MIGRATIONS = Path(__file__).parent.parent / "augmentum" / "state" / "migrations"


def _load_cache_schema() -> str:
    """Load every migration the cache touches.

    079 creates the table; 087 adds user_id + the composite (user_id, task_id)
    index. Both must run against an in-memory DB so these tests exercise the
    same shape the handler sees in production. 087's UPDATE against the
    ``users`` table no-ops when the table is absent, so we guard with a
    fallback schema line that provides a users stub — migration 072 creates
    it in reality but tests don't want the full cascade.
    """
    sql = (_MIGRATIONS / "079_tool_call_cache.sql").read_text(encoding="utf-8")
    # Stub users table so 087's ALTER ... REFERENCES users(id) resolves and
    # the backfill UPDATE has a table to read from.
    sql += "\nCREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, created_at TEXT);\n"
    sql += "\nCREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, description TEXT);\n"
    sql += (_MIGRATIONS / "087_agentic_user_id.sql").read_text(encoding="utf-8")
    # The 087 migration also ALTERs agentic_tasks, so stub the table with
    # the pre-087 shape — 087's own ALTER adds user_id, matching prod order.
    sql = (
        "CREATE TABLE IF NOT EXISTS agentic_tasks (id TEXT PRIMARY KEY, session_id TEXT);\n"
        + sql
    )
    return sql


MIGRATION_SQL = _load_cache_schema()


@pytest.mark.asyncio
async def test_hash_tool_call_is_deterministic():
    a = hash_tool_call("web_search", {"query": "blue light", "limit": 3})
    b = hash_tool_call("web_search", {"limit": 3, "query": "blue light"})
    assert a == b, "hash must be invariant to arg dict ordering"


@pytest.mark.asyncio
async def test_hash_tool_call_distinguishes_args():
    a = hash_tool_call("web_search", {"query": "cats"})
    b = hash_tool_call("web_search", {"query": "dogs"})
    assert a != b


@pytest.mark.asyncio
async def test_cache_roundtrip():
    async with aiosqlite.connect(":memory:") as db:
        await db.executescript(MIGRATION_SQL)
        cache = ToolCallCache(db)

        h = hash_tool_call("calc", {"expr": "2+2"})
        await cache.put(
            "task-1", 3, h,
            tool_name="calc", output="4",
            metadata={"ms": 7}, success=True,
            user_id="alice",
        )

        got = await cache.get("task-1", 3, h, user_id="alice")
        assert got is not None
        assert got.tool_name == "calc"
        assert got.output == "4"
        assert got.metadata == {"ms": 7}
        assert got.success is True


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    async with aiosqlite.connect(":memory:") as db:
        await db.executescript(MIGRATION_SQL)
        cache = ToolCallCache(db)
        assert await cache.get("nope", 0, "deadbeef", user_id="alice") is None


@pytest.mark.asyncio
async def test_cache_is_tenant_scoped():
    """A row written by alice must not be visible to bob.

    This locks in migration 087 — without the user_id filter a second user
    with the same (task_id, step_idx, call_hash) would replay alice's
    result, which is both a privacy leak and a correctness bug because the
    two users can have different arg-reaching context.
    """
    async with aiosqlite.connect(":memory:") as db:
        await db.executescript(MIGRATION_SQL)
        cache = ToolCallCache(db)

        h = hash_tool_call("web_search", {"query": "x"})
        await cache.put(
            "shared-task", 0, h,
            tool_name="web_search", output="alice result",
            user_id="alice",
        )

        # Bob asks for the same (task_id, step_idx, call_hash) — miss.
        assert await cache.get("shared-task", 0, h, user_id="bob") is None
        # Alice still hits.
        alice_hit = await cache.get("shared-task", 0, h, user_id="alice")
        assert alice_hit is not None
        assert alice_hit.output == "alice result"


@pytest.mark.asyncio
async def test_clear_for_task():
    async with aiosqlite.connect(":memory:") as db:
        await db.executescript(MIGRATION_SQL)
        cache = ToolCallCache(db)

        h = hash_tool_call("t", {"x": 1})
        await cache.put("t1", 0, h, tool_name="t", output="ok", user_id="alice")
        await cache.put("t2", 0, h, tool_name="t", output="ok", user_id="alice")
        await cache.clear_for_task("t1", user_id="alice")

        assert await cache.get("t1", 0, h, user_id="alice") is None
        assert await cache.get("t2", 0, h, user_id="alice") is not None


@pytest.mark.asyncio
async def test_clear_for_task_is_tenant_scoped():
    """Clearing alice's task must not touch bob's cached rows."""
    async with aiosqlite.connect(":memory:") as db:
        await db.executescript(MIGRATION_SQL)
        cache = ToolCallCache(db)

        h = hash_tool_call("t", {"x": 1})
        await cache.put("shared", 0, h, tool_name="t", output="a", user_id="alice")
        await cache.put("shared", 0, h, tool_name="t", output="b", user_id="bob")

        await cache.clear_for_task("shared", user_id="alice")

        assert await cache.get("shared", 0, h, user_id="alice") is None
        assert await cache.get("shared", 0, h, user_id="bob") is not None
