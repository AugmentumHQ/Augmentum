# tests/test_memory_events.py
"""Tests for memory event logging."""
from __future__ import annotations

import asyncio
import json
import tempfile
import os

import aiosqlite
import pytest


@pytest.fixture
def event_db():
    """Create a temporary SQLite DB with the events table."""
    path = tempfile.mktemp(suffix=".db")

    async def setup():
        conn = await aiosqlite.connect(path)
        await conn.execute("""
            CREATE TABLE memory_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                event_type TEXT NOT NULL,
                memory_id TEXT,
                detail TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await conn.commit()
        return conn

    conn = asyncio.get_event_loop().run_until_complete(setup())
    yield conn
    asyncio.get_event_loop().run_until_complete(conn.close())
    os.unlink(path)


_UID = "usr_test_owner"


@pytest.mark.asyncio
async def test_log_event_stores_to_db(event_db):
    from augmentum.memory.events import log_event

    await log_event(event_db, "promotion", user_id=_UID, memory_id="m1", detail={"from_tier": "active", "to_tier": "core"})

    cursor = await event_db.execute("SELECT event_type, memory_id, detail, user_id FROM memory_events")
    row = await cursor.fetchone()
    assert row[0] == "promotion"
    assert row[1] == "m1"
    assert json.loads(row[2])["from_tier"] == "active"
    assert row[3] == _UID


@pytest.mark.asyncio
async def test_log_event_without_memory_id(event_db):
    from augmentum.memory.events import log_event

    await log_event(event_db, "extraction", user_id=_UID, detail={"count": 3})

    cursor = await event_db.execute("SELECT memory_id FROM memory_events")
    row = await cursor.fetchone()
    assert row[0] is None


@pytest.mark.asyncio
async def test_get_events_returns_newest_first(event_db):
    from augmentum.memory.events import log_event, get_events

    await log_event(event_db, "extraction", user_id=_UID, detail={"count": 1})
    await log_event(event_db, "promotion", user_id=_UID, memory_id="m2", detail={"to_tier": "core"})

    events = await get_events(event_db, user_id=_UID, limit=10)
    assert len(events) == 2
    assert events[0]["event_type"] == "promotion"  # newest first


@pytest.mark.asyncio
async def test_get_events_filters_by_type(event_db):
    from augmentum.memory.events import log_event, get_events

    await log_event(event_db, "extraction", user_id=_UID, detail={"count": 1})
    await log_event(event_db, "promotion", user_id=_UID, memory_id="m2", detail={})

    events = await get_events(event_db, user_id=_UID, event_type="promotion", limit=10)
    assert len(events) == 1
    assert events[0]["event_type"] == "promotion"


# ----------------------------------------------------------------------
# Regression: user_id MUST be passed (no default-sentinel fallback)
#
# Background: the original implementation defaulted user_id to the
# literal string "default". Five real call sites silently fell back to
# it, stranding 120 events under a non-existent user. The contract is
# now: log_event() / get_events() require user_id as a kwarg. If the
# caller has no user, they pass "" (the project convention for unscoped)
# — never "default".
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_event_requires_user_id(event_db):
    from augmentum.memory.events import log_event

    with pytest.raises(TypeError, match="user_id"):
        await log_event(event_db, "promotion", memory_id="m1")


@pytest.mark.asyncio
async def test_get_events_requires_user_id(event_db):
    from augmentum.memory.events import get_events

    with pytest.raises(TypeError, match="user_id"):
        await get_events(event_db, limit=10)


@pytest.mark.asyncio
async def test_log_event_isolates_users(event_db):
    """A's events are not visible to B and vice-versa."""
    from augmentum.memory.events import log_event, get_events

    await log_event(event_db, "promotion", user_id="usr_a", memory_id="m1")
    await log_event(event_db, "promotion", user_id="usr_b", memory_id="m2")

    a_events = await get_events(event_db, user_id="usr_a", limit=10)
    b_events = await get_events(event_db, user_id="usr_b", limit=10)

    assert len(a_events) == 1 and a_events[0]["memory_id"] == "m1"
    assert len(b_events) == 1 and b_events[0]["memory_id"] == "m2"


@pytest.mark.asyncio
async def test_dream_scheduler_passes_user_id():
    """Static check: scheduler.py routes user_id to log_event as a kwarg,
    not inside the detail dict — that pattern (silent + misleading) is
    what stranded the 32 dream_cycle rows."""
    import inspect
    from augmentum.dream import scheduler
    src = inspect.getsource(scheduler)
    # The call site must pass user_id as a top-level kwarg
    assert "user_id=user_id" in src, "dream/scheduler.py must pass user_id to log_event"


@pytest.mark.asyncio
async def test_memory_store_promotion_passes_user_id():
    """Static check: every log_event call in memory/store.py must pass
    user_id. The 88 stranded promotion/tier_change rows came from these
    callers omitting the kwarg."""
    import inspect
    import re
    from augmentum.memory import store
    src = inspect.getsource(store)
    # Find every log_event(...) call signature, verify each carries user_id
    calls = re.findall(r"log_event\([^)]*\)", src, re.DOTALL)
    assert calls, "no log_event calls found — test scaffold broken"
    for call in calls:
        assert "user_id=" in call, f"log_event call missing user_id: {call!r}"
