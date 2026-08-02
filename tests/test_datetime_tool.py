"""Tests for DateTimeTool — date/time operations, formatting, timezone conversion."""

from __future__ import annotations

from datetime import datetime

import pytest

from augmentum.tools.datetime_tool import DateTimeTool, _parse_date


class TestDateTimeParsing:
    """Date string parsing in various formats."""

    def test_parse_iso_format(self):
        dt = _parse_date("2025-06-15")
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.day == 15

    def test_parse_us_format(self):
        dt = _parse_date("06/15/2025")
        assert dt.month == 6
        assert dt.day == 15

    def test_parse_long_format(self):
        dt = _parse_date("June 15, 2025")
        assert dt.year == 2025

    def test_parse_compact_format(self):
        dt = _parse_date("20250615")
        assert dt.year == 2025

    def test_parse_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_date("not a date")


class TestDateTimeNow:
    """Current time action."""

    async def test_now_returns_iso_string(self):
        tool = DateTimeTool()
        result = await tool.execute(action="now")
        assert result.success is True
        # Output is an ISO datetime string
        assert "T" in result.output
        assert result.metadata["timezone"] == "UTC"

    async def test_now_with_timezone(self):
        tool = DateTimeTool()
        result = await tool.execute(action="now", timezone="US/Eastern")
        assert result.success is True
        assert result.metadata["timezone"] == "US/Eastern"

    async def test_now_invalid_timezone(self):
        tool = DateTimeTool()
        result = await tool.execute(action="now", timezone="Not/Real")
        assert result.success is False
        assert "timezone" in result.error.lower()


class TestDateTimeDiff:
    """Date difference calculations."""

    async def test_diff_same_dates(self):
        tool = DateTimeTool()
        result = await tool.execute(action="diff", date="2025-01-01", date2="2025-01-01")
        assert result.success is True
        assert result.metadata["total_days"] == 0

    async def test_diff_one_week(self):
        tool = DateTimeTool()
        result = await tool.execute(action="diff", date="2025-01-01", date2="2025-01-08")
        assert result.success is True
        assert result.metadata["total_days"] == 7
        assert result.metadata["weeks"] == 1

    async def test_diff_missing_date_returns_error(self):
        tool = DateTimeTool()
        result = await tool.execute(action="diff", date="2025-01-01")
        assert result.success is False


class TestDateTimeAdd:
    """Date arithmetic."""

    async def test_add_days(self):
        tool = DateTimeTool()
        result = await tool.execute(action="add", date="2025-01-01", days=10)
        assert result.success is True
        assert "2025-01-11" in result.output

    async def test_add_hours(self):
        tool = DateTimeTool()
        result = await tool.execute(action="add", date="2025-01-01T00:00:00", hours=24)
        assert result.success is True
        assert "2025-01-02" in result.output


class TestDateTimeTimezone:
    """Timezone conversion."""

    async def test_timezone_utc_to_eastern(self):
        tool = DateTimeTool()
        result = await tool.execute(
            action="timezone",
            date="2025-06-15T12:00:00",
            timezone="UTC",
            to_timezone="US/Eastern",
        )
        assert result.success is True
        assert result.metadata["from_timezone"] == "UTC"
        assert result.metadata["to_timezone"] == "US/Eastern"

    async def test_timezone_missing_date(self):
        tool = DateTimeTool()
        result = await tool.execute(action="timezone")
        assert result.success is False


class TestDateTimeOther:
    """Other actions: calendar, day_of_week, format, parse."""

    async def test_day_of_week(self):
        tool = DateTimeTool()
        # 2025-01-01 is a Wednesday
        result = await tool.execute(action="day_of_week", date="2025-01-01")
        assert result.success is True
        assert result.output == "Wednesday"

    async def test_calendar_output(self):
        tool = DateTimeTool()
        result = await tool.execute(action="calendar", year=2025, month=1)
        assert result.success is True
        assert "January" in result.output
        assert result.metadata["days_in_month"] == 31

    async def test_format_date(self):
        tool = DateTimeTool()
        result = await tool.execute(action="format", date="2025-06-15", format="%B %d, %Y")
        assert result.success is True
        assert result.output == "June 15, 2025"

    async def test_parse_action(self):
        tool = DateTimeTool()
        result = await tool.execute(action="parse", date="2025-06-15")
        assert result.success is True
        assert result.metadata["year"] == 2025
        assert result.metadata["day_of_week"] == "Sunday"

    async def test_unknown_action(self):
        tool = DateTimeTool()
        result = await tool.execute(action="invalid")
        assert result.success is False

    async def test_tool_properties(self):
        tool = DateTimeTool()
        assert tool.name == "datetime"
        assert tool.cacheable is False
