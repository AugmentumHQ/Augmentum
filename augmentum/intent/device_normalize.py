"""Deterministic natural-language → device-primitive normalization.

This is the cross-model robustness layer for the ``device.*`` action
verbs. The verbs expose *natural-language* slots ("7am", "quarter past
six", "an hour and a half") rather than structured integers, because
that is the one shape every model in our zoo can fill reliably: a weak
local model (Gemma-E2B, SmolLM) only has to echo the user's own words,
and a strong cloud model can too. The Siri/Google-grade parsing then
happens HERE, deterministically and unit-tested — so the quality of the
result does not ride on how good the active model is at emitting clean
hour/minute integers.

Two parsers:

* :func:`parse_clock_time` → an absolute wall-clock ``(hour, minute)``
  for "7am" / "19:30" / "quarter past seven", OR a relative
  ``in_seconds`` for "in 20 minutes" (the *phone* turns a relative
  alarm into a wall-clock time against its own clock, because the
  server's timezone is not the user's).
* :func:`parse_duration` → ``seconds`` for "10 minutes" / "1h30m" /
  "an hour and a half".

Both return ``None`` when they can't confidently parse — the caller
degrades to asking the user rather than guessing wrong (an alarm set
for the wrong time is worse than no alarm).

Nothing here touches the model or the network; it is pure text → number
so it can be exhaustively tested. See
tests/test_device_normalize.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Number words 0-59 are enough for clock minutes and small hour counts.
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


def _words_to_int(text: str) -> int | None:
    """Tiny spelled-out-number reader for 0-59 ("forty five" → 45)."""
    text = text.strip().lower().replace("-", " ")
    if not text:
        return None
    if text in _ONES:
        return _ONES[text]
    if text in _TENS:
        return _TENS[text]
    parts = text.split()
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES:
        val = _TENS[parts[0]] + _ONES[parts[1]]
        return val if val < 60 else None
    return None


@dataclass(frozen=True)
class ClockTime:
    """Either an absolute wall-clock time or a relative offset.

    Exactly one of (``hour``/``minute``) or ``in_seconds`` is set.
    Relative offsets are resolved on the *device* against its own clock.
    """

    hour: int | None = None
    minute: int | None = None
    in_seconds: int | None = None

    @property
    def is_relative(self) -> bool:
        return self.in_seconds is not None

    def to_payload(self) -> dict[str, int]:
        if self.is_relative:
            return {"in_seconds": int(self.in_seconds or 0)}
        return {"hour": int(self.hour or 0), "minute": int(self.minute or 0)}


# ── duration ────────────────────────────────────────────────────────

_DUR_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1, "s": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60, "m": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600, "h": 3600,
}

# "1h30m", "90m", "45s" — compact unit-suffixed form.
_COMPACT_DUR = re.compile(r"(\d+)\s*(h|hr|hrs|m|min|mins|s|sec|secs)", re.I)
# "10 minutes", "two hours", "an hour and a half"
_PHRASE_DUR = re.compile(
    r"(?P<num>\d+|an?|a couple of|a few|half|"
    r"(?:[a-z]+(?:[ -][a-z]+)?))\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
    re.I,
)


def parse_duration(text: str) -> int | None:
    """Natural-language duration → whole seconds (or ``None``).

    Handles "10 minutes", "90 seconds", "1h30m", "two hours",
    "an hour and a half", "a minute". Returns ``None`` when no time
    unit is present at all (a bare "10" is too ambiguous to guess).
    """
    if not text:
        return None
    s = text.strip().lower()

    # "half an hour" / "half a minute" — idiom that isn't a number+unit
    # shape, so handle it before the general parser. ("an hour and a
    # half" uses "a half" and is summed by the phrasal parser below.)
    half = re.fullmatch(r"\s*half\s+an?\s+(hour|hr|minute|min)s?\s*", s)
    if half:
        return 1800 if half.group(1).startswith("h") else 30

    # Compact form first ("1h30m", possibly several segments).
    compact = _COMPACT_DUR.findall(s)
    if compact and re.fullmatch(r"[\dhmrsicen\s]+", s):
        total = 0
        for num, unit in compact:
            total += int(num) * _DUR_UNIT_SECONDS[unit.lower()]
        if total > 0:
            return total

    # Phrasal form, summing each "<num> <unit>" chunk so
    # "an hour and a half" = 3600 + 1800.
    total = 0
    matched = False
    for m in _PHRASE_DUR.finditer(s):
        unit = m.group("unit").lower()
        unit_secs = _DUR_UNIT_SECONDS.get(unit)
        if unit_secs is None:
            continue
        num_raw = m.group("num").strip()
        if num_raw in ("a", "an"):
            qty: float = 1.0
        elif num_raw in ("half",):
            qty = 0.5
        elif num_raw in ("a couple of", "a couple"):
            qty = 2.0
        elif num_raw in ("a few",):
            qty = 3.0
        elif num_raw.isdigit():
            qty = float(num_raw)
        else:
            w = _words_to_int(num_raw)
            if w is None:
                continue
            qty = float(w)
        total += int(round(qty * unit_secs))
        matched = True

    # "... and a half" tail attached to the last unit (hours/minutes).
    if matched and re.search(r"and a half\b", s):
        last = _PHRASE_DUR.findall(s)
        if last:
            last_unit = last[-1][1].lower()
            half = _DUR_UNIT_SECONDS.get(last_unit, 0) // 2
            # Only add if not already captured by a "half <unit>" chunk.
            if not re.search(r"half\s+(?:hour|minute|hr|min)", s):
                total += half

    return total if matched and total > 0 else None


# ── clock time ──────────────────────────────────────────────────────

_REL_PREFIX = re.compile(r"\bin\b\s+(.+)$", re.I)
_HHMM = re.compile(
    r"\b(?P<h>\d{1,2})[:.](?P<m>\d{2})\s*(?P<ap>am|pm|a\.m\.|p\.m\.)?\b",
    re.I,
)
_HONLY = re.compile(r"\b(?P<h>\d{1,2})\s*(?P<ap>am|pm|a\.m\.|p\.m\.)\b", re.I)
_PAST_TO = re.compile(
    r"\b(?P<frac>quarter|half|ten|twenty|five|twenty[- ]five)\s+"
    r"(?P<dir>past|to|after|till|til|before)\s+"
    r"(?P<hr>\w+(?:[ -]\w+)?)\b",
    re.I,
)


def _apply_ampm(hour: int, ap: str | None) -> int:
    """Fold a 12-hour reading + am/pm marker into 24-hour."""
    ap = (ap or "").replace(".", "").lower()
    if ap == "pm":
        return hour if hour == 12 else hour + 12
    if ap == "am":
        return 0 if hour == 12 else hour
    return hour


def _bare_hour_guess(hour: int) -> int:
    """Best-effort am/pm for a bare hour with no marker.

    We don't know the phone's clock, so we use the same waking-hours
    heuristic a phone assistant uses: 1-6 → afternoon/evening (pm),
    7-11 → morning (am), 12 → noon, and 13-23 read as already-24h.
    Documented + tested so the guess is intentional, not accidental.
    """
    if hour >= 13:
        return hour
    if hour == 12:
        return 12
    if 1 <= hour <= 6:
        return hour + 12
    return hour


def parse_clock_time(text: str) -> ClockTime | None:
    """Natural-language time → :class:`ClockTime` (or ``None``).

    Absolute: "7am", "7:30 pm", "19:00", "noon", "midnight",
    "quarter past seven", "half past six", "quarter to eight".
    Relative: "in 20 minutes", "in an hour and a half" → ``in_seconds``.
    """
    if not text:
        return None
    s = text.strip().lower()

    # Named times.
    if re.search(r"\bnoon\b|\bmidday\b", s):
        return ClockTime(hour=12, minute=0)
    if re.search(r"\bmidnight\b", s):
        return ClockTime(hour=0, minute=0)

    # Relative ("in 20 minutes") — delegate to the duration parser.
    rel = _REL_PREFIX.search(s)
    if rel:
        secs = parse_duration(rel.group(1))
        if secs:
            return ClockTime(in_seconds=secs)
        return None

    # "quarter past seven" / "half past six" / "quarter to eight".
    m = _PAST_TO.search(s)
    if m:
        frac_word = m.group("frac").lower().replace("-", " ")
        frac = {
            "quarter": 15, "half": 30, "ten": 10,
            "twenty": 20, "five": 5, "twenty five": 25,
        }.get(frac_word)
        base_hr = _words_to_int(m.group("hr")) or _hr_from_digits(m.group("hr"))
        if frac is not None and base_hr is not None:
            direction = m.group("dir").lower()
            if direction in ("to", "till", "til", "before"):
                hour = (base_hr - 1) % 24
                minute = 60 - frac
            else:  # past / after
                hour = base_hr % 24
                minute = frac
            return ClockTime(hour=_bare_hour_guess(hour), minute=minute)

    # "7:30 pm" / "07:30".
    m = _HHMM.search(s)
    if m:
        hour = int(m.group("h"))
        minute = int(m.group("m"))
        if minute > 59:
            return None
        ap = m.group("ap")
        hour = _apply_ampm(hour, ap) if ap else (
            hour if hour <= 23 else None
        )
        if hour is None or hour > 23:
            return None
        return ClockTime(hour=hour, minute=minute)

    # "7am" / "7 pm".
    m = _HONLY.search(s)
    if m:
        hour = _apply_ampm(int(m.group("h")), m.group("ap"))
        if hour > 23:
            return None
        return ClockTime(hour=hour, minute=0)

    # Bare hour as a last resort ("set an alarm for 7").
    m = re.search(r"\b(\d{1,2})\b", s)
    if m:
        hour = int(m.group(1))
        if hour > 23:
            return None
        return ClockTime(hour=_bare_hour_guess(hour), minute=0)

    return None


def _hr_from_digits(text: str) -> int | None:
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        v = int(m.group(1))
        return v if v <= 23 else None
    return None
