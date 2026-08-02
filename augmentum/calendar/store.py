"""Calendar event cache — aiosqlite-backed read/write primitives.

Events are synced from connected CalDAV servers. The store is
user-scoped: every method accepts ``user_id`` and filters by it.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from augmentum.calendar.caldav_client import CalendarEvent
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


async def upsert_event(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    service_id: str,
    event: CalendarEvent,
) -> None:
    """Insert or update a cached event (deduped on uid + user_id + service_id)."""
    start_str = _dt_to_str(event.start)
    end_str = _dt_to_str(event.end) if event.end else start_str
    now = _now_iso()
    await conn.execute(
        """INSERT INTO calendar_events
           (user_id, service_id, uid, summary, start_dt, end_dt,
            location, description, calendar_name, calendar_path,
            last_seen_at, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (user_id, service_id, uid) DO UPDATE SET
            summary = excluded.summary,
            start_dt = excluded.start_dt,
            end_dt = excluded.end_dt,
            location = excluded.location,
            description = excluded.description,
            calendar_name = excluded.calendar_name,
            calendar_path = excluded.calendar_path,
            last_seen_at = excluded.last_seen_at,
            updated_at = excluded.updated_at""",
        (
            user_id, service_id, event.uid, event.summary, start_str, end_str,
            event.location, event.description,
            event.calendar_name, event.calendar_path,
            now, now, now,
        ),
    )
    await conn.commit()


async def delete_stale_events(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    service_id: str,
    seen_uids: set[str],
) -> int:
    """Remove cached events whose UIDs are no longer on the server.
    Returns count of deleted rows."""
    if not seen_uids:
        cur = await conn.execute(
            "DELETE FROM calendar_events WHERE user_id = ? AND service_id = ?",
            (user_id, service_id),
        )
        await conn.commit()
        deleted = cur.rowcount
        await cur.close()
        return deleted

    placeholders = ",".join("?" for _ in seen_uids)
    cur = await conn.execute(
        f"""DELETE FROM calendar_events
            WHERE user_id = ? AND service_id = ?
            AND uid NOT IN ({placeholders})""",
        (user_id, service_id, *seen_uids),
    )
    await conn.commit()
    deleted = cur.rowcount
    await cur.close()
    return deleted


async def list_events(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    range_start: date | datetime | None = None,
    range_end: date | datetime | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return cached calendar events, optionally scoped to a date range."""
    if range_start is None:
        range_start = date.today()
    if range_end is None:
        range_end = range_start + __import__("datetime").timedelta(days=1)

    start_str = _dt_to_str(range_start)
    end_str = _dt_to_str(range_end)

    cur = await conn.execute(
        """SELECT uid, summary, start_dt, end_dt, location, description,
                  calendar_name, calendar_path, service_id
           FROM calendar_events
           WHERE user_id = ?
             AND start_dt < ?
             AND (end_dt > ? OR end_dt = start_dt)
           ORDER BY start_dt ASC
           LIMIT ?""",
        (user_id, end_str, start_str, limit),
    )
    rows = await cur.fetchall()
    await cur.close()

    events: list[dict[str, Any]] = []
    for r in rows:
        events.append({
            "uid": r[0],
            "summary": r[1],
            "start": r[2],
            "end": r[3],
            "location": r[4],
            "description": r[5],
            "calendar_name": r[6],
            "calendar_path": r[7],
            "service_id": r[8],
        })
    return events


async def count_events_today(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
) -> int:
    """Quick count of events today — used to gate prompt injection.
    Hits the ``idx_calendar_events_range`` index, ~1ms."""
    today = date.today().isoformat()
    tomorrow = (date.today() + __import__("datetime").timedelta(days=1)).isoformat()
    cur = await conn.execute(
        """SELECT COUNT(*) FROM calendar_events
           WHERE user_id = ?
             AND start_dt < ?
             AND (end_dt > ? OR end_dt = start_dt)""",
        (user_id, tomorrow, today),
    )
    row = await cur.fetchone()
    await cur.close()
    return int(row[0]) if row and row[0] is not None else 0


async def purge_old_events(
    conn: aiosqlite.Connection,
    *,
    user_id: str = "",
    older_than_days: int = 90,
) -> int:
    """Delete events whose start date is more than ``older_than_days``
    in the past. Called periodically to keep the cache lean. Returns
    count of deleted rows."""
    cutoff = (date.today() - __import__("datetime").timedelta(days=older_than_days)).isoformat()
    if user_id:
        cur = await conn.execute(
            "DELETE FROM calendar_events WHERE start_dt < ? AND user_id = ?",
            (cutoff, user_id),
        )
    else:
        cur = await conn.execute(
            "DELETE FROM calendar_events WHERE start_dt < ?",
            (cutoff,),
        )
    await conn.commit()
    deleted = cur.rowcount
    await cur.close()
    if deleted:
        log.info("calendar_purged_old_events", deleted=deleted, older_than_days=older_than_days)
    return deleted


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dt_to_str(dt: date | datetime | None) -> str:
    if dt is None:
        return ""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.isoformat()
