"""Calendar sync engine — pull events from a CalDAV server into the cache.

Runs at hook install and on a periodic schedule (cron). The diff is
UID-based: events on the server but not in cache → upsert; events in
cache but not on server → delete. All-or-nothing per calendar.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from augmentum.calendar.caldav_client import CalDAVClient
from augmentum.calendar.store import delete_stale_events, upsert_event
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

# How far to look ahead when syncing (next N days of events).
_SYNC_WINDOW_DAYS = 60


async def sync_calendar_events(
    conn: "aiosqlite.Connection",
    *,
    user_id: str,
    service_id: str,
    base_url: str,
    username: str = "",
    password: str = "",
    calendar_path: str = "",
) -> int:
    """Pull events from a CalDAV server into the cache.

    Returns the number of events synced (inserts + updates). Raises on
    connection failure so the caller can surface the error.
    """
    client = CalDAVClient(base_url, username=username, password=password)

    # Discover calendars if no specific path given.
    calendars: list[dict[str, str]]
    if calendar_path:
        calendars = [{"name": "calendar", "path": calendar_path}]
    else:
        calendars = await client.discover_calendars()

    if not calendars:
        log.warning(
            "caldav_no_calendars_found",
            service_id=service_id, base_url=base_url,
        )
        return 0

    now = datetime.now(timezone.utc)
    range_start = now - timedelta(days=1)   # include yesterday (overlap buffer)
    range_end = now + timedelta(days=_SYNC_WINDOW_DAYS)

    synced = 0
    for cal in calendars:
        cal_path = cal["path"]
        cal_name = cal["name"]

        events = await client.list_events(cal_path, range_start, range_end)
        seen_uids: set[str] = set()
        for ev in events:
            ev.calendar_name = cal_name
            ev.calendar_path = cal_path
            await upsert_event(
                conn, user_id=user_id, service_id=service_id, event=ev,
            )
            seen_uids.add(ev.uid)
            synced += 1

        # Remove events no longer on the server (scoped to this calendar path).
        # We only delete events with this calendar_path so we don't wipe
        # events from other calendar collections under the same service.
        if calendar_path:
            deleted = await delete_stale_events(
                conn, user_id=user_id, service_id=service_id,
                seen_uids=seen_uids,
            )
        else:
            deleted = 0  # multi-calendar: skip stale-delete for now

        log.info(
            "caldav_calendar_synced",
            service_id=service_id, calendar=cal_name,
            upserted=len(events), deleted=deleted,
        )

    return synced


async def create_calendar_event(
    *,
    base_url: str,
    username: str = "",
    password: str = "",
    calendar_path: str = "",
    summary: str,
    start_dt: datetime,
    end_dt: datetime,
    description: str = "",
    location: str = "",
) -> Any | None:
    """Create an event on the CalDAV server. Returns the CalendarEvent or None."""
    client = CalDAVClient(base_url, username=username, password=password)
    return await client.create_event(
        calendar_path, summary, start_dt, end_dt,
        description=description, location=location,
    )
