# augmentum/memory/events.py
"""Memory event logging — thin INSERT wrapper for the memory_events table."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, UTC

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def log_event(
    conn: aiosqlite.Connection,
    event_type: str,
    *,
    user_id: str,
    memory_id: str | None = None,
    detail: dict | None = None,
) -> str:
    """Log a memory event. Returns the event ID.

    ``user_id`` is required (no default). The previous ``"default"``
    sentinel default stranded 120 events under a non-existent user,
    silently passing the FK because the column wasn't strict-enforced.
    Callers without a real user_id should pass ``""`` (the convention for
    unscoped) — never the literal ``"default"``.
    """
    event_id = uuid.uuid4().hex[:16]
    detail_json = json.dumps(detail or {})
    now = datetime.now(UTC).isoformat()
    await conn.execute(
        "INSERT INTO memory_events (id, user_id, event_type, memory_id, detail, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (event_id, user_id, event_type, memory_id, detail_json, now),
    )
    await conn.commit()
    log.debug("memory_event_logged", event_type=event_type, memory_id=memory_id)
    return event_id


async def get_events(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    event_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Fetch events newest-first with optional type filter."""
    query = "SELECT id, user_id, event_type, memory_id, detail, created_at FROM memory_events WHERE user_id = ?"
    params: list = [user_id]
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)
    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    cursor = await conn.execute(query, params)
    rows = await cursor.fetchall()
    return [
        {
            "id": r[0],
            "user_id": r[1],
            "event_type": r[2],
            "memory_id": r[3],
            "detail": json.loads(r[4]) if r[4] else {},
            "created_at": r[5],
        }
        for r in rows
    ]
