# tests/test_memory_notifications_persist.py
"""Tests for persistent memory notifications."""
from __future__ import annotations

import asyncio
import tempfile
import os

import aiosqlite
import pytest


@pytest.fixture
def notif_db():
    """Create a temp DB with notifications table."""
    path = tempfile.mktemp(suffix=".db")

    async def setup():
        conn = await aiosqlite.connect(path)
        await conn.execute("""
            CREATE TABLE memory_notifications (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL DEFAULT 'default',
                content TEXT NOT NULL,
                evidence TEXT,
                tier TEXT NOT NULL DEFAULT 'provisional',
                confidence REAL NOT NULL DEFAULT 0.5,
                memory_type TEXT NOT NULL DEFAULT 'fact',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                resolved_at TEXT
            )
        """)
        await conn.commit()
        return conn

    conn = asyncio.get_event_loop().run_until_complete(setup())
    yield conn
    asyncio.get_event_loop().run_until_complete(conn.close())
    os.unlink(path)


@pytest.mark.asyncio
async def test_queue_persistent_notification(notif_db):
    from augmentum.memory.notifications import queue_notification, get_pending

    await queue_notification(notif_db, "m1", "Likes Python", tier="provisional", confidence=0.6, memory_type="preference")

    pending = await get_pending(notif_db)
    assert len(pending) == 1
    assert pending[0]["id"] == "m1"
    assert pending[0]["content"] == "Likes Python"
    assert pending[0]["status"] == "pending"


@pytest.mark.asyncio
async def test_resolve_notification(notif_db):
    from augmentum.memory.notifications import queue_notification, resolve_notification, get_pending

    await queue_notification(notif_db, "m1", "Likes Python")
    await resolve_notification(notif_db, "m1", "approved")

    pending = await get_pending(notif_db)
    assert len(pending) == 0


@pytest.mark.asyncio
async def test_get_pending_excludes_resolved(notif_db):
    from augmentum.memory.notifications import queue_notification, resolve_notification, get_pending

    await queue_notification(notif_db, "m1", "Fact 1")
    await queue_notification(notif_db, "m2", "Fact 2")
    await resolve_notification(notif_db, "m1", "dismissed")

    pending = await get_pending(notif_db)
    assert len(pending) == 1
    assert pending[0]["id"] == "m2"


@pytest.mark.asyncio
async def test_duplicate_notification_ignored(notif_db):
    from augmentum.memory.notifications import queue_notification, get_pending

    await queue_notification(notif_db, "m1", "Likes Python")
    await queue_notification(notif_db, "m1", "Likes Python")  # duplicate

    pending = await get_pending(notif_db)
    assert len(pending) == 1
