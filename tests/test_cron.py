"""Unit tests for augmentum/utils/cron.py — the hand-rolled 5-field
cron parser behind the scheduling substrate's ``params.cron`` rung."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from augmentum.utils.cron import describe, next_after, parse, validate

NY = ZoneInfo("America/New_York")


def _dt(y, mo, d, h=0, m=0, tz=NY):
    return datetime(y, mo, d, h, m, tzinfo=tz)


# ─── parse ──────────────────────────────────────────────────────────────


def test_parse_basic_fields():
    spec = parse("30 9 * * 1-5")
    assert spec.minutes == frozenset({30})
    assert spec.hours == frozenset({9})
    assert spec.dom_star and not spec.dow_star
    assert spec.weekdays == frozenset({1, 2, 3, 4, 5})


def test_parse_names_and_seven_as_sunday():
    spec = parse("0 12 * jan,jul sun")
    assert spec.months == frozenset({1, 7})
    assert spec.weekdays == frozenset({0})
    assert parse("0 12 * * 7").weekdays == frozenset({0})


def test_parse_steps_ranges_lists():
    spec = parse("*/15 8-17/3 1,15 * *")
    assert spec.minutes == frozenset({0, 15, 30, 45})
    assert spec.hours == frozenset({8, 11, 14, 17})
    assert spec.days == frozenset({1, 15})


def test_parse_open_step():  # "a/n" = a..max step n
    assert parse("5/20 * * * *").minutes == frozenset({5, 25, 45})


def test_parse_aliases():
    assert parse("@daily").hours == frozenset({0})
    assert parse("@weekly").weekdays == frozenset({0})
    assert parse("@monthly").days == frozenset({1})


@pytest.mark.parametrize("bad", [
    "", "* * * *", "* * * * * *",       # wrong field count
    "60 * * * *", "* 24 * * *",         # out of range
    "* * 0 * *", "* * * 13 *",
    "*/0 * * * *",                      # zero step
    "5-1 * * * *",                      # reversed range
    "a * * * *",                        # junk
    "0 0 L * *",                        # quartz L unsupported
])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse(bad)


def test_validate_returns_message_not_raise():
    assert validate("not a cron") is not None
    assert validate("0 9 * * mon-fri") is None
    # Satisfiable-but-never date (Feb 31, both day fields restricted
    # would OR — use dom-only so it must match Feb 31).
    assert validate("0 0 31 2 *") is not None
    # Feb 29 IS satisfiable (leap years) within the 4-year probe.
    assert validate("0 0 29 2 *") is None


# ─── next_after ─────────────────────────────────────────────────────────


def test_next_after_same_day():
    # Wed 2026-07-01 08:00 → daily 09:00 fires same day.
    got = next_after("0 9 * * *", _dt(2026, 7, 1, 8, 0))
    assert got == _dt(2026, 7, 1, 9, 0)


def test_next_after_rolls_to_next_day():
    got = next_after("0 9 * * *", _dt(2026, 7, 1, 9, 0))  # exactly 9:00
    assert got == _dt(2026, 7, 2, 9, 0)  # strictly-after


def test_next_after_weekday_restriction():
    # Fri 2026-07-03 10:00 → next weekday-9am is Mon 07-06.
    got = next_after("0 9 * * mon-fri", _dt(2026, 7, 3, 10, 0))
    assert got == _dt(2026, 7, 6, 9, 0)


def test_next_after_hourly_step():
    got = next_after("0 */2 * * *", _dt(2026, 7, 1, 13, 5))
    assert got == _dt(2026, 7, 1, 14, 0)


def test_next_after_monthly():
    got = next_after("0 9 1 * *", _dt(2026, 7, 2, 0, 0))
    assert got == _dt(2026, 8, 1, 9, 0)


def test_next_after_dom_dow_or_semantics():
    # crontab(5): both restricted → OR. From Wed 2026-07-01,
    # "0 0 15 * fri" fires Fri 07-03 (dow) before 07-15 (dom).
    got = next_after("0 0 15 * fri", _dt(2026, 7, 1, 0, 0))
    assert got == _dt(2026, 7, 3, 0, 0)


def test_next_after_feb29():
    got = next_after("0 0 29 2 *", _dt(2026, 3, 1, 0, 0))
    assert got == _dt(2028, 2, 29, 0, 0)


def test_next_after_unsatisfiable_returns_none():
    assert next_after("0 0 31 2 *", _dt(2026, 1, 1)) is None


def test_next_after_minute_granularity_within_hour():
    got = next_after("15,45 10 * * *", _dt(2026, 7, 1, 10, 20))
    assert got == _dt(2026, 7, 1, 10, 45)


def test_next_after_timezone_preserved():
    got = next_after("0 9 * * *", _dt(2026, 7, 1, 8, 0, tz=NY))
    assert got.tzinfo is not None
    assert got.utcoffset() == _dt(2026, 7, 1, 9, 0, tz=NY).utcoffset()


def test_next_after_across_dst_spring_forward():
    # US spring-forward 2027-03-14: 02:30 doesn't exist that day.
    # The walk must not crash or skip the schedule permanently.
    got = next_after("30 2 * * *", _dt(2027, 3, 13, 3, 0, tz=NY))
    assert got is not None
    assert (got.month, got.day) == (3, 14)


# ─── describe ───────────────────────────────────────────────────────────


def test_describe_common_shapes():
    assert describe("0 9 * * *") == "daily at 09:00"
    assert describe("30 7 * * mon-fri") == "Mon,Tue,Wed,Thu,Fri at 07:30"
    assert describe("0 * * * *") == "hourly"
    # Non-trivial → raw passthrough, never raises.
    assert describe("*/7 3-5 2 * *") == "*/7 3-5 2 * *"
    assert describe("garbage") == "garbage"
