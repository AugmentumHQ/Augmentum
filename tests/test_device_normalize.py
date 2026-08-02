"""Tests for the device.* natural-language normalization layer.

This is the cross-model robustness guarantee for the phone action
verbs: the verbs accept the user's own words and these deterministic
parsers turn them into phone primitives, so the result quality does not
depend on how good the active model is at emitting clean integers. If
these pass, "set an alarm for quarter past seven" works the same whether
the turn ran on Gemma-E2B or Claude.
"""
from __future__ import annotations

import pytest

from augmentum.intent.device_normalize import (
    ClockTime,
    parse_clock_time,
    parse_duration,
)

# ── parse_duration ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text,expected",
    [
        ("10 minutes", 600),
        ("10 min", 600),
        ("10m", 600),
        ("90 seconds", 90),
        ("45s", 45),
        ("1 hour", 3600),
        ("2 hours", 7200),
        ("1h30m", 5400),
        ("an hour and a half", 5400),
        ("a minute", 60),
        ("two hours", 7200),
        ("half an hour", 1800),
        ("a couple of minutes", 120),
        ("a few minutes", 180),
        ("forty five minutes", 2700),
    ],
)
def test_parse_duration_ok(text: str, expected: int) -> None:
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "10", "soon", "later", "a bunch"])
def test_parse_duration_none(text: str) -> None:
    # Bare numbers and vague words are too ambiguous to guess — the
    # handler degrades to asking rather than setting a wrong timer.
    assert parse_duration(text) is None


# ── parse_clock_time: absolute ──────────────────────────────────────

@pytest.mark.parametrize(
    "text,hour,minute",
    [
        ("7am", 7, 0),
        ("7 am", 7, 0),
        ("7pm", 19, 0),
        ("12pm", 12, 0),
        ("12am", 0, 0),
        ("7:30am", 7, 30),
        ("7:30 pm", 19, 30),
        ("19:00", 19, 0),
        ("07:05", 7, 5),
        ("noon", 12, 0),
        ("midnight", 0, 0),
        ("quarter past seven", 7, 15),
        ("half past six", 18, 30),  # bare-hour guess → pm for 6
        ("quarter to eight", 7, 45),
        ("half past 7", 7, 30),
    ],
)
def test_parse_clock_absolute(text: str, hour: int, minute: int) -> None:
    ct = parse_clock_time(text)
    assert ct is not None
    assert not ct.is_relative
    assert (ct.hour, ct.minute) == (hour, minute)


# ── parse_clock_time: relative ──────────────────────────────────────

@pytest.mark.parametrize(
    "text,seconds",
    [
        ("in 20 minutes", 1200),
        ("in 2 hours", 7200),
        ("in an hour and a half", 5400),
        ("in 90 seconds", 90),
    ],
)
def test_parse_clock_relative(text: str, seconds: int) -> None:
    ct = parse_clock_time(text)
    assert ct is not None
    assert ct.is_relative
    assert ct.in_seconds == seconds


@pytest.mark.parametrize("text", ["", "sometime", "whenever", "in a bit"])
def test_parse_clock_none(text: str) -> None:
    assert parse_clock_time(text) is None


# ── payload shape (what gets sent to the phone) ─────────────────────

def test_to_payload_absolute() -> None:
    assert ClockTime(hour=7, minute=30).to_payload() == {"hour": 7, "minute": 30}


def test_to_payload_relative() -> None:
    assert ClockTime(in_seconds=1200).to_payload() == {"in_seconds": 1200}


def test_bare_hour_waking_heuristic() -> None:
    # No am/pm marker: 1-6 read as pm, 7-11 as am — the documented
    # phone-assistant guess so a bare "set alarm for 5" doesn't fire at
    # 5 in the morning.
    assert parse_clock_time("5").hour == 17
    assert parse_clock_time("9").hour == 9
