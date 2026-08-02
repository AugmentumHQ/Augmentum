"""Calendar integration — CalDAV client, event cache, and sync engine."""

from augmentum.calendar.caldav_client import (
    CalDAVClient,
    CalendarEvent,
)
from augmentum.calendar.store import (
    count_events_today,
    list_events,
    purge_old_events,
    upsert_event,
)
from augmentum.calendar.sync import (
    create_calendar_event,
    sync_calendar_events,
)

__all__ = [
    "CalDAVClient",
    "CalendarEvent",
    "count_events_today",
    "create_calendar_event",
    "list_events",
    "purge_old_events",
    "sync_calendar_events",
    "upsert_event",
]
