"""Tests for augmentum/utils/datetime_context.py — system datetime for LLM prompts."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import patch

from augmentum.utils.datetime_context import _get_local_tz, get_datetime_context


class TestGetDatetimeContext:
    """Verify datetime context string generation."""

    def test_returns_string(self):
        result = get_datetime_context()
        assert isinstance(result, str)

    def test_contains_current_date_tag(self):
        result = get_datetime_context()
        assert "<current_time>" in result
        assert "</current_time>" in result

    def test_contains_date_label(self):
        result = get_datetime_context()
        assert "Current date:" in result

    def test_contains_time_label(self):
        result = get_datetime_context()
        assert "Current time:" in result

    def test_contains_utc_offset(self):
        result = get_datetime_context()
        assert "UTC" in result

    def test_contains_day_of_week(self):
        result = get_datetime_context()
        # Should contain a day name like Monday, Tuesday, etc.
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(d in result for d in days)

    def test_does_not_mention_future(self):
        result = get_datetime_context()
        assert "Never mention this timestamp" in result


class TestGetLocalTz:
    """Verify timezone resolution priority."""

    def test_returns_timezone_object(self):
        tz = _get_local_tz()
        assert tz is not None

    def test_tz_env_var_override(self):
        with patch.dict(os.environ, {"TZ": "UTC"}):
            # Force re-evaluation by clearing settings timezone
            with patch("augmentum.config.settings") as mock_settings:
                mock_settings.timezone = ""
                tz = _get_local_tz()
                # Should resolve to UTC or equivalent
                assert tz is not None

    def test_can_create_datetime_with_result(self):
        tz = _get_local_tz()
        now = datetime.now(tz)
        assert now.tzinfo is not None

    def test_utc_conversion(self):
        tz = _get_local_tz()
        now = datetime.now(tz)
        utc_now = now.astimezone(timezone.utc)
        assert utc_now.tzinfo == timezone.utc
