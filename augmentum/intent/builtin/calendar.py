"""Calendar verbs — read today's events, add events.

calendar.today: "what's on my calendar today?" — reads cached events
from the calendar_events table. Tier 1 (regex) + Tier 3 (LLM tool).

calendar.add: "add dentist appointment Friday at 2pm" — creates a
CalDAV event on the connected server. Tier 3 only (needs LLM for
date/time parsing).

Both verbs are gated: they only resolve when ≥1 CalDAV service is
connected. Without a connected calendar, the companion says "I don't
have access to your calendar" rather than a generic error.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


# ── Helpers ────────────────────────────────────────────────────────────


def _resolve_conn(session: SessionContext) -> Any:
    """Best-effort aiosqlite connection from session app_state."""
    sm = getattr(
        getattr(getattr(session, "app_state", None), "state_manager", None),
        "backend", None,
    )
    return getattr(sm, "conn", None) if sm is not None else None


def _fmt_time(dt_str: str) -> str:
    """Format an ISO datetime string as a human-readable time."""
    try:
        if "T" in dt_str:
            dt = datetime.strptime(dt_str[:19], "%Y-%m-%dT%H:%M:%S")
            return dt.strftime("%I:%M %p").lstrip("0").lower()
    except (ValueError, TypeError):
        pass
    return ""


# ── Natural-language time parser for calendar.add ──────────────────────
# The LLM extracts the user's stated time into natural language
# ("tomorrow at 2pm", "Friday at noon"). This parser converts those
# strings into UTC datetimes. Common patterns only — the LLM is
# instructed to use standard phrasings; anything unparseable falls
# back to now + 1 hour.

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_AM_PM_TIMES: dict[str, int] = {
    "noon": 12, "midnight": 0,
}


def _parse_time_str(t: str) -> tuple[int, int]:
    """Parse a time string like '2pm', '2:30pm', '14:00', 'noon' into (hour, minute)."""
    t = t.strip().lower().rstrip(".")
    if t in _AM_PM_TIMES:
        return _AM_PM_TIMES[t], 0
    # 2:30pm, 2.30pm, 2pm, 14:00, 1430
    import re
    m = re.match(r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?", t)
    if not m:
        return -1, -1
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    meridiem = m.group(3)
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    # 24-hour time (14:00 with no am/pm)
    if meridiem is None and hour > 12:
        pass  # already 24h
    elif meridiem is None and hour <= 12:
        pass  # ambiguous — treat as given (2 = 2pm in conversation)
    if hour > 23 or minute > 59:
        return -1, -1
    return hour, minute


def _parse_when(when: str, now: datetime) -> datetime:
    """Parse a natural-language time string into a UTC datetime.

    Handles the phrasings the LLM is taught to emit:
      - "tomorrow at 2pm", "tomorrow at 2:30pm", "tomorrow 2pm"
      - "Friday at 3pm", "next Tuesday at noon"
      - "today at 5pm", "today 5pm"
      - "in 2 hours", "in 30 minutes"
      - "July 25th at 10am", "July 25 10am"
      - "this afternoon" → 2pm today, "tomorrow morning" → 9am tomorrow
      - Bare times: "at 3pm", "2pm" → today at that time (or tomorrow if past)

    Returns now + 1 hour when the string can't be parsed.
    """
    import re
    w = when.strip().lower()
    direction = "today"
    date_hint = None  # date to combine with parsed time
    default_hour = None  # default hour for phrases like "tomorrow morning"

    # ── Relative day anchor ──────────────────────────────────────────
    if "tomorrow" in w:
        direction = "tomorrow"
        w = w.replace("tomorrow", "").strip()
    elif "today" in w:
        direction = "today"
        w = w.replace("today", "").strip()

    # ── Day-of-week anchor ───────────────────────────────────────────
    dow_match = re.match(
        r"(next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        w,
    )
    if dow_match:
        is_next = bool(dow_match.group(1))
        target_dow = _WEEKDAYS[dow_match.group(2)]
        current_dow = now.weekday()
        days_ahead = target_dow - current_dow
        if days_ahead <= 0:
            days_ahead += 7
        if is_next:
            days_ahead += 7
        direction = "dow"
        date_hint = now.date() + timedelta(days=days_ahead)
        w = w[dow_match.end():].strip()

    # ── Month-day anchor ("July 25th") ────────────────────────────────
    md_match = re.match(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?",
        w,
    )
    if md_match:
        month = _MONTHS[md_match.group(1)]
        day = int(md_match.group(2))
        try:
            date_hint = date(now.year, month, day)
            if date_hint < now.date():
                date_hint = date(now.year + 1, month, day)
            direction = "abs"
        except ValueError:
            pass
        w = w[md_match.end():].strip()

    # ── "in N hours/minutes" ─────────────────────────────────────────
    in_match = re.match(r"in\s+(\d+)\s*(hour|min|minute)s?", w)
    if in_match:
        n = int(in_match.group(1))
        unit = in_match.group(2)
        if unit.startswith("hour"):
            return now + timedelta(hours=n)
        else:
            return now + timedelta(minutes=n)

    # ── Defaults for vague phrasings ─────────────────────────────────
    if "morning" in w:
        default_hour = 9
        w = w.replace("morning", "").strip()
    elif "afternoon" in w:
        default_hour = 14
        w = w.replace("afternoon", "").strip()
    elif "evening" in w:
        default_hour = 19
        w = w.replace("evening", "").strip()
    elif "night" in w and "tonight" not in w:
        default_hour = 20
        w = w.replace("night", "").strip()
    elif "tonight" in w:
        default_hour = 20
        w = w.replace("tonight", "").strip()

    # ── Time extraction ──────────────────────────────────────────────
    # Strip leading "at" or punctuations
    w = re.sub(r"^(at|@)\s+", "", w).strip()
    hour, minute = _parse_time_str(w) if w else (-1, -1)
    if hour < 0 and default_hour is not None:
        hour, minute = default_hour, 0

    # ── Combine date + time ──────────────────────────────────────────
    if hour < 0:
        # Couldn't parse time — fall back to now + 1 hour
        return now + timedelta(hours=1)

    # Resolve the date
    if date_hint is not None:
        target_date = date_hint
    elif direction == "tomorrow":
        target_date = now.date() + timedelta(days=1)
    else:
        target_date = now.date()

    target = datetime(target_date.year, target_date.month, target_date.day,
                      hour, minute, 0, tzinfo=timezone.utc)

    # If the resolved time is in the past today, push to tomorrow
    # (unless the user explicitly said "today")
    if target <= now and direction == "today":
        target += timedelta(days=1)

    return target


def _format_today(events: list[dict]) -> str:
    """Format today's events for the companion prompt."""
    if not events:
        return "No events today."

    today_str = date.today().strftime("%A, %B %d")
    lines = [f"Today ({today_str}):"]
    for ev in events:
        start = _fmt_time(ev["start"])
        summary = ev["summary"]
        if start:
            lines.append(f"  {start} — {summary}")
        else:
            lines.append(f"  {summary}")
        if ev.get("location"):
            lines[-1] += f" @ {ev['location']}"
    return "\n".join(lines)


# ── calendar.today ─────────────────────────────────────────────────────


async def _calendar_today(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    """Read today's events from the cached calendar."""
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't reach your calendar for a signed-out session.",
        )

    conn = _resolve_conn(session)
    if conn is None:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't see the calendar right now.",
        )

    today = date.today()
    try:
        from augmentum.calendar.store import list_events
        events = await list_events(
            conn, user_id=session.user_id,
            range_start=today, range_end=today + timedelta(days=1),
        )
    except Exception:
        log.warning("calendar_today_list_failed", exc_info=True)
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I had trouble reading the calendar.",
        )

    if not events:
        return ActionResult(
            short_circuit=True,
            speak="Your calendar is clear today.",
            toast="No events today",
        )

    # Build a spoken summary.
    parts = [f"You have {len(events)} event{'s' if len(events) > 1 else ''} today."]
    for ev in events:
        start = _fmt_time(ev["start"])
        summary = ev["summary"]
        if start:
            parts.append(f"{start}: {summary}")
        else:
            parts.append(summary)
    spoken = ". ".join(parts)

    return ActionResult(
        short_circuit=True,
        speak=spoken,
        toast=f"{len(events)} event{'s' if len(events) > 1 else ''} today",
        digest=f"reported {len(events)} calendar events",
    )


register_action(
    id="calendar.today",
    summary=(
        "Read today's events from the user's connected calendar. Use when "
        "asked about today's schedule, appointments, or 'what's on my "
        "calendar'. Returns the list of events with times and locations."
    ),
    examples=[
        "what's on my calendar today",
        "what do I have today",
        "any appointments today",
        "what's my schedule",
        "do I have anything scheduled",
    ],
    arg_schema={},
    handler=_calendar_today,
    delivery="verbal",
    stakes="personal",
)


# ── calendar.add ────────────────────────────────────────────────────────


async def _calendar_add(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    """Add an event to the user's calendar."""
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't reach your calendar for a signed-out session.",
        )

    summary = str(args.get("summary") or "").strip()
    when = str(args.get("when") or "").strip()  # natural language date/time
    description = str(args.get("description") or "").strip()
    location = str(args.get("location") or "").strip()

    if not summary:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="What should the event be called?",
            clarify={"missing": ["summary"], "args": dict(args)},
        )
    if not when:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="When is that?",
            clarify={"missing": ["when"], "args": dict(args)},
        )

    # Resolve the connected CalDAV service. We take the first one with
    # a calendar integration (most users have exactly one).
    conn = _resolve_conn(session)
    if conn is None:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I can't see the calendar right now.",
        )

    # Find an active CalDAV service.
    try:
        from augmentum.marketplace.store import MarketplaceStore

        # Best-effort: query for connected calendar services via the
        # calendar_events table — if there are events cached, we have
        # at least one connected service.
        cur = await conn.execute(
            "SELECT DISTINCT service_id FROM calendar_events WHERE user_id = ? LIMIT 1",
            (session.user_id,),
        )
        row = await cur.fetchone()
        await cur.close()
        service_id = row[0] if row else ""
    except Exception:
        service_id = ""

    if not service_id:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I don't have a connected calendar to add events to. First install Radicale or another CalDAV server from Discover.",
        )

    # Resolve service details from the managed_services row.
    try:
        cur2 = await conn.execute(
            "SELECT config_json FROM managed_services WHERE id = ? AND enabled = 1",
            (service_id,),
        )
        row2 = await cur2.fetchone()
        await cur2.close()
        if not row2 or not row2[0]:
            return ActionResult(
                short_circuit=True,
                fulfilled=False,
                speak="Your calendar server is offline.",
            )
        import json
        cfg = json.loads(row2[0]) if isinstance(row2[0], str) else (row2[0] or {})
    except Exception:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I had trouble reaching your calendar server.",
        )

    # Resolve credentials and URL.
    base_url = f"http://augmentum-{service_id}:5232"  # Radicale default
    username = cfg.get("auth_user", "")
    password = cfg.get("auth_pass", "")
    calendar_path = cfg.get("calendar_path", f"/{service_id}/")

    # Parse the LLM-extracted natural-language time into a real datetime.
    now = datetime.now(timezone.utc)
    start_dt = _parse_when(when, now)
    end_dt = start_dt + timedelta(hours=1)

    try:
        from augmentum.calendar.sync import create_calendar_event
        created = await create_calendar_event(
            base_url=base_url,
            username=username,
            password=password,
            calendar_path=calendar_path,
            summary=summary,
            start_dt=start_dt,
            end_dt=end_dt,
            description=description,
            location=location,
        )
    except Exception:
        log.warning("calendar_add_create_failed", exc_info=True)
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I wasn't able to add that to your calendar.",
        )

    if created is None:
        return ActionResult(
            short_circuit=True,
            fulfilled=False,
            speak="I couldn't add that event — the calendar server didn't accept it.",
        )

    time_str = start_dt.strftime("%I:%M %p").lstrip("0").lower()
    return ActionResult(
        short_circuit=True,
        speak=f"Added {summary} to your calendar at {time_str}.",
        toast=f"Added: {summary}",
        digest=f"added calendar event: {summary}",
    )


register_action(
    id="calendar.add",
    summary=(
        "Add an event to the user's connected calendar. Use when asked to "
        "schedule, add, or create a calendar event, appointment, or "
        "reminder. The 'when' field is natural language that describes "
        "when the event should occur (e.g. 'tomorrow at 2pm', 'Friday "
        "afternoon', 'next Tuesday 3pm')."
    ),
    examples=[
        "add dentist appointment tomorrow at 2pm",
        "schedule lunch with Sarah Friday noon",
        "put dinner at 7pm on my calendar",
        "remind me to call mom Tuesday morning",
    ],
    arg_schema={
        "summary": {
            "type": "string",
            "description": "The event title — what it is.",
        },
        "when": {
            "type": "string",
            "description": (
                "When the event happens, in natural language — "
                "'tomorrow at 2pm', 'Friday', 'next Tuesday 3pm'. "
                "The server parses this into an exact date/time."
            ),
        },
        "description": {
            "type": "string",
            "description": "Optional notes or details about the event.",
        },
        "location": {
            "type": "string",
            "description": "Optional location or address.",
        },
    },
    required=["summary", "when"],
    fanout=_TIER3_ONLY,
    handler=_calendar_add,
    delivery="verbal",
    stakes="personal",
)
