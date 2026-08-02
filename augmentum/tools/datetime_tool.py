"""DateTime tool — date/time operations, formatting, and calendar queries."""

from __future__ import annotations

import calendar
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from augmentum.tools.base import Tool, ToolCategory, ToolResult


def _parse_date(date_str: str) -> datetime:
    """Parse a date string in common formats."""
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y%m%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str}")


class DateTimeTool(Tool):
    """Date/time utility: current time, date math, formatting, calendar queries."""

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return (
            "Date and time operations. Actions: 'now' (current time), "
            "'parse' (parse a date string), 'diff' (difference between two dates), "
            "'add' (add days/hours to a date), 'format' (format a date), "
            "'calendar' (month calendar), 'day_of_week' (what day is a date), "
            "'timezone' (convert between timezones)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["now", "parse", "diff", "add", "format", "calendar", "day_of_week", "timezone"],
                    "description": "The datetime operation to perform",
                },
                "date": {"type": "string", "description": "Date string (for parse/diff/add/format/day_of_week)"},
                "date2": {"type": "string", "description": "Second date string (for diff)"},
                "days": {"type": "number", "description": "Number of days to add (for add)"},
                "hours": {"type": "number", "description": "Number of hours to add (for add)"},
                "format": {"type": "string", "description": "Output format string (for format)"},
                "year": {"type": "integer", "description": "Year (for calendar)"},
                "month": {"type": "integer", "description": "Month 1-12 (for calendar)"},
                "timezone": {"type": "string", "description": "Timezone name (e.g. 'US/Eastern', 'Europe/London')"},
                "to_timezone": {"type": "string", "description": "Target timezone (for timezone conversion)"},
            },
            "required": ["action"],
        }

    @property
    def timeout(self) -> float:
        return 2.0

    @property
    def cacheable(self) -> bool:
        return False

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "now")

        try:
            if action == "now":
                return self._now(kwargs.get("timezone"))
            if action == "parse":
                return self._parse(kwargs.get("date", ""))
            if action == "diff":
                return self._diff(kwargs.get("date", ""), kwargs.get("date2", ""))
            if action == "add":
                return self._add(
                    kwargs.get("date", ""),
                    kwargs.get("days", 0),
                    kwargs.get("hours", 0),
                )
            if action == "format":
                return self._format(
                    kwargs.get("date", ""),
                    kwargs.get("format", "%Y-%m-%d %H:%M:%S"),
                )
            if action == "calendar":
                return self._calendar(
                    kwargs.get("year"), kwargs.get("month"),
                )
            if action == "day_of_week":
                return self._day_of_week(kwargs.get("date", ""))
            if action == "timezone":
                return self._timezone_convert(
                    kwargs.get("date", ""),
                    kwargs.get("timezone", "UTC"),
                    kwargs.get("to_timezone", "UTC"),
                )
            return ToolResult(success=False, error=f"Unknown action: {action}")
        except (ValueError, KeyError, OverflowError) as e:
            return ToolResult(success=False, error=str(e))

    def _now(self, tz_name: str | None = None) -> ToolResult:
        if tz_name:
            try:
                tz = ZoneInfo(tz_name)
            except Exception:
                return ToolResult(success=False, error=f"Unknown timezone: {tz_name}")
            now = datetime.now(tz)
        else:
            now = datetime.now(UTC)
        return ToolResult(
            success=True,
            output=now.isoformat(),
            metadata={"timestamp": now.timestamp(), "timezone": tz_name or "UTC"},
        )

    def _parse(self, date_str: str) -> ToolResult:
        if not date_str:
            return ToolResult(success=False, error="No date string provided")
        dt = _parse_date(date_str)
        return ToolResult(
            success=True,
            output=dt.isoformat(),
            metadata={
                "year": dt.year, "month": dt.month, "day": dt.day,
                "hour": dt.hour, "minute": dt.minute, "second": dt.second,
                "day_of_week": dt.strftime("%A"),
                "day_of_year": dt.timetuple().tm_yday,
            },
        )

    def _diff(self, date1: str, date2: str) -> ToolResult:
        if not date1 or not date2:
            return ToolResult(success=False, error="Two dates required for diff")
        dt1 = _parse_date(date1)
        dt2 = _parse_date(date2)
        delta = dt2 - dt1
        return ToolResult(
            success=True,
            output=f"{delta.days} days, {delta.seconds} seconds",
            metadata={
                "total_days": delta.days,
                "total_seconds": int(delta.total_seconds()),
                "weeks": delta.days // 7,
                "remaining_days": delta.days % 7,
            },
        )

    def _add(self, date_str: str, days: float, hours: float) -> ToolResult:
        dt = datetime.now(UTC) if not date_str else _parse_date(date_str)
        result = dt + timedelta(days=float(days), hours=float(hours))
        return ToolResult(
            success=True,
            output=result.isoformat(),
            metadata={"original": dt.isoformat(), "added_days": days, "added_hours": hours},
        )

    def _format(self, date_str: str, fmt: str) -> ToolResult:
        if not date_str:
            return ToolResult(success=False, error="No date string provided")
        dt = _parse_date(date_str)
        return ToolResult(success=True, output=dt.strftime(fmt))

    def _calendar(self, year: int | None, month: int | None) -> ToolResult:
        now = datetime.now(UTC)
        y = int(year) if year else now.year
        m = int(month) if month else now.month
        if not (1 <= m <= 12):
            return ToolResult(success=False, error=f"Invalid month: {m}")
        cal_text = calendar.month(y, m)
        return ToolResult(
            success=True,
            output=cal_text.strip(),
            metadata={"year": y, "month": m, "days_in_month": calendar.monthrange(y, m)[1]},
        )

    def _day_of_week(self, date_str: str) -> ToolResult:
        if not date_str:
            return ToolResult(success=False, error="No date string provided")
        dt = _parse_date(date_str)
        return ToolResult(
            success=True,
            output=dt.strftime("%A"),
            metadata={"day_number": dt.weekday(), "iso_day": dt.isoweekday()},
        )

    def _timezone_convert(self, date_str: str, from_tz: str, to_tz: str) -> ToolResult:
        if not date_str:
            return ToolResult(success=False, error="No date string provided")
        dt = _parse_date(date_str)
        source_tz = ZoneInfo(from_tz)
        target_tz = ZoneInfo(to_tz)
        dt_with_tz = dt.replace(tzinfo=source_tz)
        converted = dt_with_tz.astimezone(target_tz)
        return ToolResult(
            success=True,
            output=converted.isoformat(),
            metadata={"from_timezone": from_tz, "to_timezone": to_tz},
        )
