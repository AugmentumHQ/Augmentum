"""Minimal 5-field cron expression support for the scheduling substrate.

Hand-rolled instead of croniter: the substrate must not grow a dependency
for ~150 lines of well-specified date math (croniter isn't in the image,
and a new wheel means a rebuild for every install). Scope is classic
POSIX cron plus the common niceties LLMs and users actually emit:

  * five fields — minute hour day-of-month month day-of-week
  * ``*``, numbers, ranges ``a-b``, steps ``*/n`` / ``a-b/n`` / ``a/n``,
    comma lists, month names (``jan``-``dec``), weekday names
    (``sun``-``sat``), and ``7`` as an alias for Sunday
  * ``@hourly`` / ``@daily`` / ``@weekly`` / ``@monthly`` / ``@yearly``
  * standard DOM/DOW semantics: when BOTH day fields are restricted the
    match is an OR (crontab(5)); otherwise the restricted one must match

Not supported (rejected with a clear error, not silently misread):
six-field seconds syntax, ``L``/``W``/``#`` Quartz extensions.

Evaluation is timezone-aware: ``next_after`` walks calendar days in the
zone of the ``after`` datetime it's given, so "0 9 * * mon" means 9am in
the user's zone when callers pass a user-local "now" — matching how the
anchored ``local_time`` mode in standing_tasks resolves schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

_MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# Cron day-of-week: 0 = Sunday .. 6 = Saturday (7 also accepted as Sunday).
_DOW_NAMES = {
    "sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6,
}

_ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * 0",
    "@monthly": "0 0 1 * *",
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
}

# Field metadata: (label, lo, hi, name-map)
_FIELDS: tuple[tuple[str, int, int, dict[str, int] | None], ...] = (
    ("minute", 0, 59, None),
    ("hour", 0, 23, None),
    ("day-of-month", 1, 31, None),
    ("month", 1, 12, _MONTH_NAMES),
    ("day-of-week", 0, 7, _DOW_NAMES),
)

# Upper bound on the day walk in next_after. Four years covers any
# satisfiable spec including "Feb 29" (0 0 29 2 *); anything still
# unmatched after that is unsatisfiable (e.g. "0 0 31 2 *").
_MAX_SEARCH_DAYS = 366 * 4 + 2


@dataclass(frozen=True, slots=True)
class CronSpec:
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]  # 0=Sunday .. 6=Saturday
    dom_star: bool
    dow_star: bool
    raw: str


def _atom_value(atom: str, label: str, names: dict[str, int] | None) -> int:
    a = atom.strip().lower()
    if names and a in names:
        return names[a]
    try:
        return int(a)
    except ValueError:
        raise ValueError(f"{label}: '{atom}' is not a number"
                         + (" or name" if names else "")) from None


def _parse_field(
    field: str, label: str, lo: int, hi: int,
    names: dict[str, int] | None,
) -> tuple[set[int], bool]:
    """Return (values, is_star). is_star = the field was a bare ``*``."""
    field = field.strip()
    if not field:
        raise ValueError(f"{label}: empty field")
    if field == "*":
        return set(range(lo, hi + 1)), True

    out: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"{label}: empty list item")
        step = 1
        if "/" in part:
            base, _, step_s = part.partition("/")
            try:
                step = int(step_s)
            except ValueError:
                raise ValueError(f"{label}: bad step '{step_s}'") from None
            if step < 1:
                raise ValueError(f"{label}: step must be >= 1")
        else:
            base = part
        base = base.strip()
        if base == "*":
            start, end = lo, hi
        elif "-" in base and not base.lstrip("-").isdigit():
            # a-b range (guard: a bare negative number is not a range)
            a_s, _, b_s = base.partition("-")
            start = _atom_value(a_s, label, names)
            end = _atom_value(b_s, label, names)
            if start > end:
                # crontab(5) wrap-around ranges (e.g. fri-mon) are rare and
                # ambiguous across implementations — reject explicitly.
                raise ValueError(f"{label}: range '{base}' is reversed")
        else:
            start = _atom_value(base, label, names)
            # "a/n" (no range) means "a through max, step n" per cron.
            end = hi if "/" in part else start
        if start < lo or end > hi:
            raise ValueError(
                f"{label}: {base} out of range {lo}-{hi}",
            )
        out.update(range(start, end + 1, step))

    if label == "day-of-week" and 7 in out:
        # 7 is Sunday too.
        out.discard(7)
        out.add(0)
    return out, False


def parse(expr: str) -> CronSpec:
    """Parse ``expr`` into a :class:`CronSpec`. Raises ValueError with a
    user-showable message on any problem."""
    raw = (expr or "").strip()
    raw_expanded = _ALIASES.get(raw.lower(), raw)
    fields = raw_expanded.split()
    if len(fields) != 5:
        raise ValueError(
            f"cron needs 5 fields (minute hour day month weekday), got "
            f"{len(fields)}: '{raw[:64]}'",
        )
    parsed: list[tuple[set[int], bool]] = []
    for value, (label, lo, hi, names) in zip(fields, _FIELDS, strict=True):
        parsed.append(_parse_field(value, label, lo, hi, names))
    (minutes, _), (hours, _), (days, dom_star), (months, _), \
        (weekdays, dow_star) = parsed
    # dow field: after 7→0 normalization the range check above allowed 7;
    # the working set is 0-6.
    weekdays = {d for d in weekdays if 0 <= d <= 6}
    return CronSpec(
        minutes=frozenset(minutes), hours=frozenset(hours),
        days=frozenset(days), months=frozenset(months),
        weekdays=frozenset(weekdays),
        dom_star=dom_star, dow_star=dow_star, raw=raw,
    )


def validate(expr: str) -> str | None:
    """Return an error message, or None when ``expr`` is a usable cron
    expression that fires at least once in the next 4 years."""
    try:
        spec = parse(expr)
    except ValueError as exc:
        return str(exc)
    probe = next_after(spec, datetime.now().astimezone())
    if probe is None:
        return f"'{expr}' never matches a real date (e.g. Feb 31)"
    return None


def _day_matches(spec: CronSpec, d: datetime) -> bool:
    if d.month not in spec.months:
        return False
    dom_ok = d.day in spec.days
    dow_ok = (d.isoweekday() % 7) in spec.weekdays  # Sunday → 0
    if spec.dom_star and spec.dow_star:
        return True
    if spec.dom_star:
        return dow_ok
    if spec.dow_star:
        return dom_ok
    return dom_ok or dow_ok  # both restricted → OR, per crontab(5)


def next_after(spec: CronSpec | str, after: datetime) -> datetime | None:
    """Smallest minute-aligned datetime strictly after ``after`` that
    matches ``spec``, in ``after``'s timezone. None if nothing matches
    within ~4 years (unsatisfiable spec)."""
    if isinstance(spec, str):
        spec = parse(spec)
    start = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    hours_sorted = sorted(spec.hours)
    minutes_sorted = sorted(spec.minutes)
    day = start
    for offset in range(_MAX_SEARCH_DAYS):
        if offset:
            # Walk by calendar day at a fixed probe hour so a DST jump
            # can't skip a day; the real time-of-day is applied below.
            day = (start + timedelta(days=offset)).replace(hour=12, minute=0)
        if not _day_matches(spec, day):
            continue
        first_day = offset == 0
        for h in hours_sorted:
            if first_day and h < start.hour:
                continue
            for m in minutes_sorted:
                if first_day and h == start.hour and m < start.minute:
                    continue
                candidate = day.replace(hour=h, minute=m,
                                        second=0, microsecond=0)
                if candidate >= start:
                    return candidate
    return None


def describe(expr: str) -> str:
    """Short human-readable gloss, best-effort. Falls back to the raw
    expression for anything non-trivial — never raises."""
    try:
        spec = parse(expr)
    except ValueError:
        return expr
    try:
        dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        time_part = ""
        if len(spec.minutes) == 1 and len(spec.hours) == 1:
            time_part = (
                f"{next(iter(spec.hours)):02d}:{next(iter(spec.minutes)):02d}"
            )
        every_day = spec.dom_star and spec.dow_star and len(spec.months) == 12
        if time_part and every_day:
            return f"daily at {time_part}"
        if time_part and spec.dom_star and len(spec.months) == 12 \
                and 0 < len(spec.weekdays) < 7:
            days = ",".join(dow_names[d] for d in sorted(spec.weekdays))
            return f"{days} at {time_part}"
        if len(spec.minutes) == 1 and len(spec.hours) == 24 and every_day:
            m = next(iter(spec.minutes))
            return "hourly" if m == 0 else f"hourly at :{m:02d}"
    except Exception:  # noqa: BLE001 — display-only helper
        pass
    return expr


__all__ = ["CronSpec", "parse", "validate", "next_after", "describe"]
