"""Connect presence store — durable last-seen / online state (Phase 1).

Thin layer over ``connect_presence`` (migration 283). The ConnectHub writes
here on the first-connect / last-disconnect transitions via its presence-sink
hook, so presence survives restarts and "last seen" becomes possible for
offline peers. One row per user (global presence, not per-viewer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import aiosqlite


async def mark_presence(
    conn: aiosqlite.Connection, *, user_id: str, online: bool,
) -> None:
    """Record an online/offline transition, stamping last_seen_at = now.

    ``last_seen_at`` is bumped on BOTH transitions: when going online it's the
    session start; when going offline it's the moment they left (which is what
    "last seen" should report). Best-effort — a write failure must never break
    the WS attach/detach path, so callers wrap this defensively.
    """
    state = "online" if online else "offline"
    await conn.execute(
        """INSERT INTO connect_presence (user_id, state, last_seen_at, updated_at)
           VALUES (?, ?, datetime('now'), datetime('now'))
           ON CONFLICT(user_id) DO UPDATE SET
               state = excluded.state,
               last_seen_at = datetime('now'),
               updated_at = datetime('now')""",
        (user_id, state),
    )
    await conn.commit()


async def get_presence_for(
    conn: aiosqlite.Connection, user_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Bulk-fetch persisted presence for a set of user_ids.

    Returns ``{user_id: {"state", "last_seen_at"}}`` for users with a row.
    Callers treat a missing entry as never-seen.
    """
    if not user_ids:
        return {}
    placeholders = ", ".join("?" for _ in user_ids)
    cur = await conn.execute(
        "SELECT user_id, state, last_seen_at FROM connect_presence "
        f"WHERE user_id IN ({placeholders})",
        tuple(user_ids),
    )
    rows = await cur.fetchall()
    await cur.close()
    return {r[0]: {"state": r[1], "last_seen_at": r[2]} for r in rows}
