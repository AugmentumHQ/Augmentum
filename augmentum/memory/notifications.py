# augmentum/memory/notifications.py
"""Persistent memory notifications backed by SQLite."""
from __future__ import annotations

from datetime import UTC, datetime

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def queue_notification(
    conn: aiosqlite.Connection,
    memory_id: str,
    content: str,
    *,
    user_id: str = "default",
    evidence: str | None = None,
    tier: str = "provisional",
    confidence: float = 0.5,
    memory_type: str = "fact",
) -> None:
    """Queue a notification for a newly extracted memory. Ignores duplicates."""
    now = datetime.now(UTC).isoformat()
    status = "pending" if tier == "provisional" else "active"
    await conn.execute(
        """INSERT OR IGNORE INTO memory_notifications
           (id, user_id, content, evidence, tier, confidence, memory_type, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, user_id, content, evidence, tier, confidence, memory_type, status, now),
    )
    await conn.commit()


async def get_undelivered(
    conn: aiosqlite.Connection,
    user_id: str = "default",
    limit: int = 10,
) -> list[dict]:
    """Fetch notifications the chat UI has not delivered yet, oldest-first."""
    cursor = await conn.execute(
        """SELECT id, content, evidence, tier, confidence, memory_type, created_at, status
           FROM memory_notifications
           WHERE user_id = ?
             AND delivered_at IS NULL
             AND status IN ('active', 'pending')
           ORDER BY created_at ASC LIMIT ?""",
        (user_id, limit),
    )
    rows = await cursor.fetchall()
    items = [
        {
            "id": r[0],
            "content": r[1],
            "evidence": r[2],
            "tier": r[3],
            "confidence": r[4],
            "type": r[5],
            "created_at": r[6],
            "status": r[7],
        }
        for r in rows
    ]
    if items:
        now = datetime.now(UTC).isoformat()
        ids = [item["id"] for item in items]
        placeholders = ",".join("?" for _ in ids)
        await conn.execute(
            f"UPDATE memory_notifications SET delivered_at = ? WHERE id IN ({placeholders})",
            (now, *ids),
        )
        await conn.commit()
    return items


async def get_pending(
    conn: aiosqlite.Connection,
    user_id: str = "default",
    limit: int = 20,
) -> list[dict]:
    """Fetch pending notifications newest-first."""
    cursor = await conn.execute(
        """SELECT id, content, evidence, tier, confidence, memory_type, created_at
           FROM memory_notifications
           WHERE user_id = ? AND status = 'pending'
           ORDER BY created_at DESC LIMIT ?""",
        (user_id, limit),
    )
    rows = await cursor.fetchall()
    return [
        {
            "id": r[0],
            "content": r[1],
            "evidence": r[2],
            "tier": r[3],
            "confidence": r[4],
            "type": r[5],
            "created_at": r[6],
            "status": "pending",
        }
        for r in rows
    ]


async def resolve_notification(
    conn: aiosqlite.Connection,
    memory_id: str,
    status: str,
    *,
    user_id: str | None = None,
) -> bool:
    """Mark a notification as approved or dismissed."""
    now = datetime.now(UTC).isoformat()
    if user_id is None:
        cursor = await conn.execute(
            "UPDATE memory_notifications SET status = ?, resolved_at = ? WHERE id = ?",
            (status, now, memory_id),
        )
    else:
        cursor = await conn.execute(
            "UPDATE memory_notifications SET status = ?, resolved_at = ? WHERE id = ? AND user_id = ?",
            (status, now, memory_id, user_id),
        )
    await conn.commit()
    return cursor.rowcount > 0
