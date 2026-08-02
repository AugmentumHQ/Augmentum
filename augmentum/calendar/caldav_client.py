"""Async CalDAV client — read and write calendar events.

Minimal CalDAV implementation over httpx. Only what Augmentum needs:

* List events in a date range (REPORT calendar-query with time-range filter)
* Create a single VEVENT (PUT)
* Discover calendar home sets and calendar collections (PROPFIND)

Parses iCalendar (RFC 5545) VEVENTs inline — no external dependency.
XML parsing uses stdlib ``xml.etree.ElementTree``.

The client is protocol-agnostic about the server (Radicale, Nextcloud,
Baikal, iCloud all speak the same CalDAV HTTP dialect). Auth is Basic
(username + password) which is what every self-hosted CalDAV server
uses and what Augmentum provisions for Radicale via managed credentials.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from xml.etree import ElementTree as ET

import httpx

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# CalDAV XML namespaces
_NS = {
    "D": "DAV:",
    "C": "urn:ietf:params:xml:ns:caldav",
    "CS": "http://calendarserver.org/ns/",
}
# iCalendar line folding (RFC 5545 §3.1): lines can continue with a
# leading space or tab on the next line.
_FOLD_RE = re.compile(r"\r?\n[ \t]")
# DTSTART/DTEND value formats we can parse.
_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_DATETIME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?$")


@dataclass
class CalendarEvent:
    """One calendar event, parsed from or ready for iCalendar."""

    uid: str = ""
    summary: str = ""
    start: datetime | date | None = None
    end: datetime | date | None = None
    location: str = ""
    description: str = ""
    calendar_name: str = ""
    calendar_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "summary": self.summary,
            "start": self.start.isoformat() if self.start else "",
            "end": self.end.isoformat() if self.end else "",
            "location": self.location,
            "description": self.description,
            "calendar_name": self.calendar_name,
        }


def _parse_date_value(value: str) -> datetime | date | None:
    """Parse an iCalendar DATE or DATE-TIME value into a Python object."""
    value = value.strip()
    m = _DATETIME_RE.match(value)
    if m:
        return datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)), int(m.group(6)),
            tzinfo=timezone.utc,
        )
    m = _DATE_RE.match(value)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _parse_vevents(ics_text: str) -> list[CalendarEvent]:
    """Extract VEVENT components from raw iCalendar text.

    Simple line-based parser — no recursion needed; we only care about
    VEVENT children of VCALENDAR, and only the properties Augmentum
    surfaces (SUMMARY, DTSTART, DTEND, UID, LOCATION, DESCRIPTION).
    """
    # Unfold folded lines first.
    unfolded = _FOLD_RE.sub("", ics_text)
    events: list[CalendarEvent] = []
    in_vevent = False
    current: dict[str, str] = {}

    for raw_line in unfolded.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        value = value.strip()

        if key == "BEGIN" and value == "VEVENT":
            in_vevent = True
            current = {}
        elif key == "END" and value == "VEVENT" and in_vevent:
            in_vevent = False
            start = _parse_date_value(current.get("DTSTART", ""))
            end = _parse_date_value(current.get("DTEND", ""))
            if start is not None:  # minimum viable event
                events.append(CalendarEvent(
                    uid=current.get("UID", ""),
                    summary=current.get("SUMMARY", ""),
                    start=start,
                    end=end,
                    location=current.get("LOCATION", ""),
                    description=current.get("DESCRIPTION", ""),
                ))
        elif in_vevent and key in (
            "SUMMARY", "DTSTART", "DTEND", "UID", "LOCATION", "DESCRIPTION",
        ):
            current[key] = value

    return events


def _build_vevent(summary: str, start_dt: datetime, end_dt: datetime,
                  uid: str = "", description: str = "",
                  location: str = "") -> str:
    """Build a minimal VEVENT iCalendar string for PUT."""
    if not uid:
        uid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Augmentum//Calendar//EN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{now}",
        f"DTSTART:{start_dt.strftime('%Y%m%dT%H%M%SZ')}",
        f"DTEND:{end_dt.strftime('%Y%m%dT%H%M%SZ')}",
        f"SUMMARY:{summary}",
    ]
    if description:
        lines.append(f"DESCRIPTION:{description}")
    if location:
        lines.append(f"LOCATION:{location}")
    lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


class CalDAVClient:
    """Async CalDAV client over a single base URL + Basic auth."""

    def __init__(self, base_url: str, username: str = "",
                 password: str = "", timeout: float = 30.0) -> None:
        self._base = base_url.rstrip("/")
        self._auth = (username, password) if username else None
        self._timeout = timeout

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self._base}{path}" if path else self._base
        async with httpx.AsyncClient(
            auth=httpx.BasicAuth(*self._auth) if self._auth else None,
            timeout=httpx.Timeout(self._timeout),
        ) as client:
            return await client.request(method, url, **kwargs)

    async def discover_calendars(self) -> list[dict[str, str]]:
        """PROPFIND the root to find calendar home sets, then list calendar
        collections. Returns ``[{name, path}]`` for each writable calendar."""
        calendars: list[dict[str, str]] = []

        # PROPFIND the root URL to find calendar-home-set or direct
        # calendar resources. Many servers (Radicale) list calendars
        # at the root directly.
        try:
            resp = await self._request(
                "PROPFIND", "",
                headers={"Depth": "1", "Content-Type": "application/xml"},
                content=(
                    '<?xml version="1.0" encoding="utf-8"?>'
                    '<D:propfind xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">'
                    "<D:prop><D:resourcetype/><D:displayname/>"
                    "<C:supported-calendar-component-set/>"
                    "</D:prop></D:propfind>"
                ),
            )
            if resp.status_code in (207, 200):
                calendars = _parse_calendar_resources(resp.text)
        except Exception:
            log.warning("caldav_discover_failed", base_url=self._base, exc_info=True)

        return calendars

    async def list_events(self, calendar_path: str,
                          range_start: datetime,
                          range_end: datetime) -> list[CalendarEvent]:
        """REPORT calendar-query for events in a date range.

        ``calendar_path`` is the href of a calendar collection (e.g.
        ``/radicale/user/calendar.ics/``). Returns parsed VEVENTs.
        """
        start_iso = range_start.strftime("%Y%m%dT%H%M%SZ")
        end_iso = range_end.strftime("%Y%m%dT%H%M%SZ")
        body = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">'
            "<D:prop><D:getetag/>"
            "<C:calendar-data/>"
            "</D:prop>"
            "<C:filter>"
            "<C:comp-filter name=\"VCALENDAR\">"
            "<C:comp-filter name=\"VEVENT\">"
            f"<C:time-range start=\"{start_iso}\" end=\"{end_iso}\"/>"
            "</C:comp-filter>"
            "</C:comp-filter>"
            "</C:filter>"
            "</C:calendar-query>"
        )
        try:
            resp = await self._request(
                "REPORT", calendar_path,
                headers={"Depth": "1", "Content-Type": "application/xml"},
                content=body,
            )
            if resp.status_code == 207:
                return _parse_calendar_data(resp.text)
        except Exception:
            log.warning("caldav_list_failed", calendar_path=calendar_path, exc_info=True)
        return []

    async def create_event(self, calendar_path: str, summary: str,
                           start_dt: datetime, end_dt: datetime,
                           uid: str = "", description: str = "",
                           location: str = "") -> CalendarEvent | None:
        """PUT a new VEVENT into a calendar collection. Returns the
        created event on success, None on failure."""
        ics = _build_vevent(summary, start_dt, end_dt,
                            uid=uid, description=description, location=location)
        event_uid = uid or str(uuid.uuid4())
        event_path = f"{calendar_path.rstrip('/')}/{event_uid}.ics"
        try:
            resp = await self._request(
                "PUT", event_path,
                headers={
                    "Content-Type": "text/calendar; charset=utf-8",
                    "If-None-Match": "*",
                },
                content=ics,
            )
            if resp.status_code in (201, 204):
                return CalendarEvent(
                    uid=event_uid, summary=summary,
                    start=start_dt, end=end_dt,
                    description=description, location=location,
                    calendar_path=calendar_path,
                )
            log.warning(
                "caldav_create_failed",
                status=resp.status_code, calendar_path=calendar_path,
            )
        except Exception:
            log.warning("caldav_create_failed", calendar_path=calendar_path, exc_info=True)
        return None

    async def ping(self) -> bool:
        """Quick reachability check — OPTIONS or GET the root."""
        try:
            resp = await self._request("OPTIONS", "")
            return resp.status_code < 500
        except Exception:
            return False


def _parse_calendar_resources(xml_text: str) -> list[dict[str, str]]:
    """Extract calendar collections from a PROPFIND multi-status response."""
    calendars: list[dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return calendars

    for response in root.findall(".//D:response", _NS):
        href_el = response.find("D:href", _NS)
        href = href_el.text.strip() if href_el is not None and href_el.text else ""
        if not href:
            continue

        # Check if this resource is a calendar collection.
        is_calendar = False
        res_type = response.find(".//D:resourcetype", _NS)
        if res_type is not None and res_type.find("C:calendar", _NS) is not None:
            is_calendar = True

        if not is_calendar:
            continue

        name = ""
        display_name = response.find(".//D:displayname", _NS)
        if display_name is not None and display_name.text:
            name = display_name.text.strip()
        if not name:
            # Fall back to the last path segment.
            name = href.rstrip("/").rsplit("/", 1)[-1]

        calendars.append({"name": name, "path": href})

    return calendars


def _parse_calendar_data(xml_text: str) -> list[CalendarEvent]:
    """Extract calendar-data (iCalendar) from a calendar-query multi-status
    response, one VEVENT per calendar-data element."""
    events: list[CalendarEvent] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return events

    for response in root.findall(".//D:response", _NS):
        href_el = response.find("D:href", _NS)
        href = href_el.text.strip() if href_el is not None and href_el.text else ""

        cal_data = response.find(".//C:calendar-data", _NS)
        if cal_data is None or not cal_data.text:
            continue

        vevents = _parse_vevents(cal_data.text)
        for ev in vevents:
            ev.calendar_path = href
            if not ev.calendar_name:
                # Use the parent path segment as calendar name.
                parts = href.rstrip("/").rsplit("/", 2)
                ev.calendar_name = parts[1] if len(parts) >= 3 else parts[-1]

        events.extend(vevents)

    return events
