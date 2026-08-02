"""Native Augmentum calendar events — user-owned, first-class events.

The complement to :mod:`augmentum.calendar.store` (which caches events pulled
FROM connected CalDAV servers). These events are created IN Augmentum and,
when the user opts in, mirrored OUT to a CalDAV server so they reach the
user's phone/laptop through open standards.

User-scoped: every function takes ``user_id`` and filters by it. Times are
stored as ISO-8601 UTC strings (``YYYY-MM-DDTHH:MM:SSZ``); all-day events
store a bare ``YYYY-MM-DD`` date in ``start_dt``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_to_dict(r: Any) -> dict[str, Any]:
    return {
        "id": r[0],
        "title": r[1],
        "start": r[2],
        "end": r[3],
        "all_day": bool(r[4]),
        "location": r[5],
        "description": r[6],
        "color": r[7],
        "rrule": r[8],
        "caldav_service_id": r[9],
        "caldav_uid": r[10],
        "caldav_href": r[11],
    }


_SELECT_COLS = (
    "id, title, start_dt, end_dt, all_day, location, description, "
    "color, rrule, caldav_service_id, caldav_uid, caldav_href"
)


async def create_event(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    title: str,
    start_dt: str,
    end_dt: str = "",
    all_day: bool = False,
    location: str = "",
    description: str = "",
    color: str = "",
    rrule: str = "",
) -> int:
    """Insert a native event. Returns the new row id."""
    now = _now_iso()
    cur = await conn.execute(
        """INSERT INTO calendar_user_events
           (user_id, title, start_dt, end_dt, all_day, location,
            description, color, rrule, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, title, start_dt, end_dt or start_dt, 1 if all_day else 0,
            location, description, color, rrule, now, now,
        ),
    )
    await conn.commit()
    new_id = cur.lastrowid
    await cur.close()
    return int(new_id)


async def get_event(
    conn: aiosqlite.Connection, event_id: int, *, user_id: str,
) -> dict[str, Any] | None:
    cur = await conn.execute(
        f"SELECT {_SELECT_COLS} FROM calendar_user_events "
        "WHERE id = ? AND user_id = ?",
        (event_id, user_id),
    )
    row = await cur.fetchone()
    await cur.close()
    return _row_to_dict(row) if row else None


async def update_event(
    conn: aiosqlite.Connection,
    event_id: int,
    *,
    user_id: str,
    fields: dict[str, Any],
) -> bool:
    """Patch mutable fields on a native event. Returns True if a row changed."""
    allowed = {
        "title", "start_dt", "end_dt", "all_day", "location",
        "description", "color", "rrule",
        "caldav_service_id", "caldav_uid", "caldav_href",
    }
    sets: list[str] = []
    vals: list[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        sets.append(f"{k} = ?")
        vals.append(1 if (k == "all_day" and v) else (0 if k == "all_day" else v))
    if not sets:
        return False
    sets.append("updated_at = ?")
    vals.append(_now_iso())
    vals.extend([event_id, user_id])
    cur = await conn.execute(
        f"UPDATE calendar_user_events SET {', '.join(sets)} "
        "WHERE id = ? AND user_id = ?",
        tuple(vals),
    )
    await conn.commit()
    changed = cur.rowcount > 0
    await cur.close()
    return changed


async def delete_event(
    conn: aiosqlite.Connection, event_id: int, *, user_id: str,
) -> dict[str, Any] | None:
    """Delete an event. Returns the deleted row (so the caller can unlink any
    CalDAV mirror) or None if nothing matched."""
    existing = await get_event(conn, event_id, user_id=user_id)
    if existing is None:
        return None
    cur = await conn.execute(
        "DELETE FROM calendar_user_events WHERE id = ? AND user_id = ?",
        (event_id, user_id),
    )
    await conn.commit()
    await cur.close()
    return existing


async def list_events(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    range_start: datetime,
    range_end: datetime,
) -> list[dict[str, Any]]:
    """Return native events overlapping ``[range_start, range_end)``, with
    recurring events (``rrule``) expanded into concrete occurrences."""
    start_iso = range_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = range_end.strftime("%Y-%m-%dT%H:%M:%SZ")
    # Non-recurring: simple overlap. Recurring: fetch all (rrule set), expand
    # in Python. Recurring rows are rare, so fetching them unfiltered is fine.
    cur = await conn.execute(
        f"""SELECT {_SELECT_COLS} FROM calendar_user_events
            WHERE user_id = ?
              AND ((rrule = '' AND start_dt < ? AND (end_dt >= ? OR end_dt = ''))
                   OR rrule != '')""",
        (user_id, end_iso, start_iso),
    )
    rows = await cur.fetchall()
    await cur.close()

    out: list[dict[str, Any]] = []
    for r in rows:
        base = _row_to_dict(r)
        if not base["rrule"]:
            out.append(base)
            continue
        out.extend(_expand_recurring(base, range_start, range_end))
    return out


# ── RRULE expansion (a pragmatic iCal subset) ───────────────────────────
#
# Supports FREQ=DAILY|WEEKLY|MONTHLY|YEARLY with INTERVAL, COUNT, UNTIL and
# BYDAY (weekly). This covers the repeat presets the UI offers; anything
# richer round-trips the raw rule to CalDAV untouched but shows only the
# first instance in-app rather than guessing wrong.

_RRULE_DAYS = {"MO": 1, "TU": 2, "WE": 3, "TH": 4, "FR": 5, "SA": 6, "SU": 7}


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        norm = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(norm) if "T" in norm else datetime.combine(
            date.fromisoformat(norm[:10]), datetime.min.time(),
        )
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except (ValueError, TypeError):
        return None


def _expand_recurring(
    base: dict[str, Any], range_start: datetime, range_end: datetime,
) -> list[dict[str, Any]]:
    start = _parse_dt(base["start"])
    if start is None:
        return [base]
    end = _parse_dt(base["end"]) or start
    duration = end - start if end > start else timedelta(0)

    parts: dict[str, str] = {}
    for chunk in base["rrule"].replace("RRULE:", "").split(";"):
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            parts[k.strip().upper()] = v.strip().upper()

    freq = parts.get("FREQ", "")
    if freq not in {"DAILY", "WEEKLY", "MONTHLY", "YEARLY"}:
        return [base]  # unknown rule — show the single base instance
    interval = max(1, int(parts["INTERVAL"])) if parts.get("INTERVAL", "").isdigit() else 1
    count = int(parts["COUNT"]) if parts.get("COUNT", "").isdigit() else None
    until = _parse_dt(parts.get("UNTIL", "")) if parts.get("UNTIL") else None
    bydays = {
        _RRULE_DAYS[d] for d in parts.get("BYDAY", "").split(",")
        if d in _RRULE_DAYS
    }

    out: list[dict[str, Any]] = []
    emitted = 0
    cursor = start
    guard = 0
    while guard < 2000:
        guard += 1
        if until and cursor > until:
            break
        if count is not None and emitted >= count:
            break
        # For WEEKLY+BYDAY, emit every matching weekday within the week step.
        candidates = [cursor]
        if freq == "WEEKLY" and bydays:
            week_start = cursor - timedelta(days=cursor.isoweekday() - 1)
            candidates = [
                week_start + timedelta(days=d - 1) for d in sorted(bydays)
            ]
        for occ in candidates:
            if occ < start:
                continue
            if until and occ > until:
                continue
            if count is not None and emitted >= count:
                break
            emitted += 1
            if range_start <= occ < range_end:
                inst = dict(base)
                inst["start"] = occ.strftime("%Y-%m-%dT%H:%M:%SZ")
                inst["end"] = (occ + duration).strftime("%Y-%m-%dT%H:%M:%SZ")
                inst["_recurring"] = True
                out.append(inst)
        # Stop once the stepping cursor has moved past the visible window.
        if cursor >= range_end:
            break
        # Advance the cursor by one interval step.
        if freq == "DAILY":
            cursor = cursor + timedelta(days=interval)
        elif freq == "WEEKLY":
            cursor = cursor + timedelta(weeks=interval)
        elif freq == "MONTHLY":
            month = cursor.month - 1 + interval
            year = cursor.year + month // 12
            cursor = cursor.replace(year=year, month=month % 12 + 1)
        else:  # YEARLY
            cursor = cursor.replace(year=cursor.year + interval)
    return out
