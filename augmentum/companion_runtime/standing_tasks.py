"""Standing tasks — recurring jobs Becca runs for the user on a cadence.

The relational complement to the curator (curator.py) and tracked topics
(migration 238). Where curator says "I noticed this in the world today,"
a standing task says "I went and checked X for you because you asked me
to keep an eye on it."

Built-in kinds (extensible — add new ones by registering in _TASK_KINDS):

  * ``feed_digest``       — curator-style digest of a topic. Pulls fresh
                            feed items, ranks by relevance, surfaces the
                            top N as a single rolled-up note.
  * ``github_releases``   — poll a GitHub repo, surface new releases via
                            their public Atom feed (no auth needed).
  * ``url_watch``         — fetch a URL, hash the body, surface a diff
                            note when the hash changes.
  * ``recurring_search``  — SearXNG query, surface novel results vs. the
                            last_result url-set baseline.

Surfacing:
  * Always writes a journal entry (so the result appears in the notes
    drawer alongside curator output, with URL refs for dedup).
  * When the result is "noteworthy" (kind-specific definition) also
    publishes via the notifications hub so a Web Push reaches the user
    when they're offline.

Scheduling:
  * Each row carries its own ``interval_seconds`` and ``next_run_at``
    (modes: deadline / cron / anchored local_time / interval — see
    :func:`_compute_next_run_at`).
  * ``step`` picks ONE due task per call per user (cheapest gate —
    oldest next_run_at first). Tasks are independent — no global rate
    limit.

Dispatchers (two lanes, no overlap):
  * The companion's ``tick_scheduler`` verb — the OWNER's lane when the
    companion runtime is on (presence-gated, verb-ledger-cited).
  * :class:`augmentum.scheduling.SchedulerService` — the app-level lane
    covering every other user, and ALL users when the companion is off
    (via a headless context). Scheduling is a platform substrate; the
    companion is one entrypoint into it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import aiosqlite

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────


# Built-in task kinds. Each maps to a runner coroutine of shape
# ``async def run(runtime, *, user_id, params) -> dict``. The dict shape:
#   {
#     "summary": str,         # short user-facing line for notes
#     "noteworthy": bool,     # True → also publish via notifications hub
#     "refs": [{"kind": "url", ...}],   # URL refs for dedup
#     "details": dict,        # arbitrary kind-specific payload
#   }
_TASK_KINDS: dict[str, Any] = {}


def _register_kind(name: str):
    def deco(fn):
        _TASK_KINDS[name] = fn
        return fn
    return deco


# Sensible default: a task that fails 5 times in a row gets auto-paused
# so a broken feed doesn't burn cycles forever. User can re-enable.
# Shared with the management-verb dispatcher (event_bus.py) via
# dispatch_policy.DEFAULT_MAX_CONSECUTIVE_ERRORS — both substrates must
# use the same threshold so "Becca pauses things that fail 5x" stays a
# single user-facing rule. Re-exported as a module local for callers
# that already import _MAX_CONSECUTIVE_ERRORS from this module.
from augmentum.companion_runtime.dispatch_policy import (
    DEFAULT_MAX_CONSECUTIVE_ERRORS as _MAX_CONSECUTIVE_ERRORS,
)

# A merely-slow task (its verb's wallclock budget cancels it) gets a much
# looser pause threshold than a broken one, so transient slowness recovers
# but a permanently-stuck task still eventually pauses (audit 2026-06-17,
# the 2026-06-08 7-hour-pause incident root).
_MAX_CONSECUTIVE_BUDGET_TIMEOUTS = 20

# ─── Schedule resolution ────────────────────────────────────────────────


# Jitter windows — added to anchored fire times to prevent thundering
# herds when many users share an anchor like "7am morning briefing."
# Without jitter, 1,000 users all hit SearXNG at 07:00:00 + tick window.
# With ±10min recurring jitter, that load smears across a 20-minute window.
# One-shot reminders use ±90s — too short to matter to the user, long
# enough to break the synchronized-fire pattern. Borrowed from Claude
# Code's deterministic-per-task-id design.
_JITTER_RECURRING_S: int = 600   # ±10min on recurring tasks
_JITTER_ONE_SHOT_S: int = 90     # ±90s on one-shot reminders


def _deterministic_jitter_seconds(seed: str, window_s: int) -> int:
    """Return a stable signed offset in ``[-window_s, +window_s)`` derived
    from ``seed``. Same seed → same offset, every call. SHA-256 because
    Python's hash() is salted per-process and would drift across restarts."""
    if not seed or window_s <= 0:
        return 0
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    # First 4 bytes → 32-bit unsigned int → modulo (2*window) − window
    val = int.from_bytes(digest[:4], "big")
    return (val % (2 * window_s)) - window_s


def _compute_next_run_at(
    *,
    params: dict[str, Any] | None,
    interval_seconds: int,
    user_timezone: str = "",
    jitter_seed: str = "",
) -> str:
    """Return next_run_at as a SQLite-storable absolute string (UTC).

    Scheduling modes (precedence: deadline > cron > anchored > interval):
      * Cron — ``params.cron`` (5-field expression, @daily-style aliases)
        via :mod:`augmentum.utils.cron`, evaluated in the user's zone.
        The expressive rung: "every 2 hours", "1st of the month at 9".
      * Anchored — ``params.local_time`` (HH:MM, 24h) sets a wall-clock
        target. Optional ``params.weekdays`` (list of 1-7 where 1=Mon..
        7=Sun, ISO) restricts which days qualify. Empty or missing
        weekdays = every day. The next occurrence at or after "now+1
        second" wins.
      * Interval — fallback when neither is set or both are malformed.
        next_run_at = now + interval_seconds.

    ``user_timezone`` (IANA name like "America/New_York") interprets
    ``local_time`` in the user's zone. Empty string = server local time
    (legacy). Malformed zone names fall back to server local time rather
    than raising — a bad TZ shouldn't take a scheduled task down.

    ``jitter_seed`` (any stable string per task — task_id, title, etc.)
    enables deterministic jitter: ±10min on recurring, ±90s on one-shots.
    Empty disables jitter (test paths + back-compat). The offset is
    derived from sha256(seed) so the same task always lands in the same
    slot within its window.

    Returned format is ``YYYY-MM-DD HH:MM:SS`` UTC, matching SQLite's
    ``datetime('now')`` output so reads/writes round-trip cleanly.

    This is the broad substrate move — every task kind, existing or
    future, gets time-of-day + per-user-TZ scheduling for free by
    including ``local_time`` (+ optional ``weekdays``) in its params.
    """
    from datetime import UTC, datetime, timedelta

    p = params or {}

    # Deadline mode: countdown toward ``params.target_date``, firing at each
    # lead-time offset (``offsets_days`` e.g. [30,14,7,1,0]). The next fire is
    # the soonest offset-boundary strictly after now; once none remain the
    # day-of moment is returned (the deadline runner completes the row on the
    # day-of fire, so this can't loop on a past target). Takes precedence over
    # the regular anchored / interval modes below.
    # Only the deadline kind sets target_date, so its presence alone selects
    # deadline mode (offsets default below if missing/empty).
    raw_target = p.get("target_date")
    if isinstance(raw_target, str) and raw_target.strip():
        tz = _resolve_zoneinfo(user_timezone)
        now_local = datetime.now(tz) if tz else datetime.now().astimezone()
        fh, fm = 9, 0  # default fire time-of-day 09:00 local
        lt = p.get("local_time")
        if isinstance(lt, str) and ":" in lt:
            try:
                _h = int(lt.split(":")[0])
                _m = int(lt.split(":", 1)[1][:2])
                if 0 <= _h <= 23 and 0 <= _m <= 59:
                    fh, fm = _h, _m
            except (ValueError, TypeError):
                pass
        try:
            _y, _mo, _dd = (int(x) for x in raw_target.strip().split("-", 2))
            target = now_local.replace(
                year=_y, month=_mo, day=_dd,
                hour=fh, minute=fm, second=0, microsecond=0,
            )
        except (ValueError, TypeError):
            log.warning("standing_tasks_bad_target_date", target=str(raw_target)[:16])
            target = None
        if target is not None:
            offsets: set[int] = set()
            for o in (p.get("offsets_days") or []):
                try:
                    oi = int(o)
                except (ValueError, TypeError):
                    continue
                if oi >= 0:
                    offsets.add(oi)
            if not offsets:
                offsets = {30, 14, 7, 1, 0}
            boundaries = sorted(target - timedelta(days=o) for o in offsets)
            chosen = next((b for b in boundaries if b > now_local), None)
            if chosen is None:
                chosen = target  # exhausted → day-of; runner completes the row
            if jitter_seed:
                chosen = chosen + timedelta(
                    seconds=_deterministic_jitter_seconds(jitter_seed, _JITTER_ONE_SHOT_S),
                )
            return chosen.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Cron mode: ``params.cron`` (5-field expression or @daily-style alias)
    # evaluated in the user's timezone. The expressiveness rung above
    # anchored local_time — "every 2 hours", "9am on the 1st", "hourly
    # during weekdays" — that neither local_time nor a drifting interval
    # can say. Every task kind inherits it, same as local_time. Malformed
    # expressions log and fall through to the anchored/interval modes so
    # a bad edit can't kill an existing schedule.
    raw_cron = p.get("cron")
    if isinstance(raw_cron, str) and raw_cron.strip():
        from augmentum.utils import cron as _cron

        tz = _resolve_zoneinfo(user_timezone)
        now_local = datetime.now(tz) if tz else datetime.now().astimezone()
        try:
            spec = _cron.parse(raw_cron)
        except ValueError as exc:
            log.warning(
                "standing_tasks_bad_cron",
                cron=str(raw_cron)[:64], error=str(exc)[:120],
            )
            spec = None
        if spec is not None:
            candidate = _cron.next_after(spec, now_local)
            if candidate is None:
                log.warning(
                    "standing_tasks_cron_unsatisfiable",
                    cron=str(raw_cron)[:64],
                )
            else:
                if jitter_seed:
                    window = _JITTER_ONE_SHOT_S if bool(p.get("one_shot")) \
                        else _JITTER_RECURRING_S
                    offset = _deterministic_jitter_seconds(jitter_seed, window)
                    candidate = candidate + timedelta(seconds=offset)
                    # Same strict-future re-assert as the anchored mode:
                    # negative jitter can pull the fire back across "now",
                    # leaving the row perpetually due. Advance through
                    # occurrences (same fixed offset each time) until the
                    # jittered moment is genuinely in the future.
                    guard = 0
                    while candidate <= now_local and guard < 400:
                        base = _cron.next_after(
                            spec, candidate - timedelta(seconds=offset),
                        )
                        if base is None:
                            break
                        candidate = base + timedelta(seconds=offset)
                        guard += 1
                if candidate is not None and candidate > now_local:
                    return candidate.astimezone(UTC).strftime(
                        "%Y-%m-%d %H:%M:%S",
                    )

    local_time = p.get("local_time")
    if isinstance(local_time, str) and ":" in local_time:
        try:
            hh_s, mm_s = local_time.split(":", 1)
            target_h = int(hh_s)
            target_m = int(mm_s.split(":")[0])
            if not (0 <= target_h <= 23 and 0 <= target_m <= 59):
                raise ValueError
        except (ValueError, TypeError):
            target_h = target_m = None
        if target_h is not None:
            raw_weekdays = p.get("weekdays") or []
            if not isinstance(raw_weekdays, list):
                raw_weekdays = []
            valid_weekdays = {
                int(w) for w in raw_weekdays
                if isinstance(w, int | float) and 1 <= int(w) <= 7
            }
            tz = _resolve_zoneinfo(user_timezone)
            now_local = datetime.now(tz) if tz else datetime.now().astimezone()
            # Dated anchor: ``params.date`` (YYYY-MM-DD) pins the fire to
            # a specific calendar day — "tomorrow morning", "on the 20th".
            # A stale or malformed date falls through to the regular
            # next-occurrence rules, so recomputing after a fire can't
            # loop on a dead date. Date wins over weekdays (more specific).
            raw_date = p.get("date")
            if isinstance(raw_date, str) and raw_date.strip():
                dated = None
                try:
                    y, mo, dd = (int(x) for x in raw_date.strip().split("-", 2))
                    dated = now_local.replace(
                        year=y, month=mo, day=dd,
                        hour=target_h, minute=target_m,
                        second=0, microsecond=0,
                    )
                except (ValueError, TypeError):
                    log.warning(
                        "standing_tasks_bad_date", date=str(raw_date)[:16],
                    )
                if dated is not None and dated > now_local:
                    if jitter_seed:
                        offset = _deterministic_jitter_seconds(
                            jitter_seed, _JITTER_ONE_SHOT_S,
                        )
                        dated = dated + timedelta(seconds=offset)
                    return dated.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")
            target_today = now_local.replace(
                hour=target_h, minute=target_m, second=0, microsecond=0,
            )
            # Strictly later than "now" so a just-fired task doesn't
            # immediately re-fire on the same minute boundary.
            candidate = target_today if target_today > now_local \
                else target_today + timedelta(days=1)
            if valid_weekdays:
                # At most 7 hops to land on a valid weekday.
                for _ in range(8):
                    if candidate.isoweekday() in valid_weekdays:
                        break
                    candidate = candidate + timedelta(days=1)
            # Apply deterministic jitter AFTER weekday hop so the window
            # doesn't push us off a valid day. One-shots get a tighter
            # window since their fire time is the whole point.
            if jitter_seed:
                window = _JITTER_ONE_SHOT_S if bool(p.get("one_shot")) \
                    else _JITTER_RECURRING_S
                offset = _deterministic_jitter_seconds(jitter_seed, window)
                candidate = candidate + timedelta(seconds=offset)
            # Re-assert the strict-future invariant AFTER jitter. The branch
            # above only proves candidate > now for the UN-jittered anchor; a
            # negative jitter offset can pull the result back across "now"
            # (e.g. anchor 20:30, offset -3min -> 20:27 today, already past).
            # When that happens the row stays "due now" and the dispatcher
            # re-picks + fully re-runs it every tick until the wall clock
            # crawls past the raw anchor — burning a full gather+synthesis
            # per tick. Bump whole days (re-hopping weekdays) until the
            # jittered candidate is genuinely in the future.
            guard = 0
            while candidate <= now_local and guard < 400:
                candidate = candidate + timedelta(days=1)
                guard += 1
                if valid_weekdays:
                    for _ in range(8):
                        if candidate.isoweekday() in valid_weekdays:
                            break
                        candidate = candidate + timedelta(days=1)
            return candidate.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")

    # Interval mode. No jitter needed — "now + interval" already disperses
    # naturally since "now" varies per task.
    next_run = datetime.now(UTC) + timedelta(seconds=int(interval_seconds))
    return next_run.strftime("%Y-%m-%d %H:%M:%S")


def iter_occurrences(
    *,
    params: dict[str, Any] | None,
    interval_seconds: int,
    user_timezone: str = "",
    range_start,
    range_end,
    next_run_at: str | None = None,
    max_count: int = 400,
) -> list:
    """Return the UTC fire moments of a task within ``[range_start, range_end)``.

    Display-only expansion for the calendar grid — the AUTHORITATIVE next
    fire is still :func:`_compute_next_run_at`. This reuses the very same
    primitives (cron ``next_after``, the anchored local_time + weekday +
    date rules, the deadline offset boundaries) so the grid can never
    disagree with the engine about WHEN a task runs; it just enumerates a
    window instead of the single next moment.

    ``range_start`` / ``range_end`` are timezone-aware UTC datetimes.
    Returns a sorted list of aware UTC ``datetime`` objects (jitter-free —
    the grid shows the scheduled slot, not the ±10min dispersion).
    """
    from datetime import UTC, datetime, timedelta

    p = params or {}
    tz = _resolve_zoneinfo(user_timezone)

    def _to_utc(dt: datetime) -> datetime:
        return dt.astimezone(UTC)

    out: list[datetime] = []

    # ── Deadline mode: countdown offset boundaries toward target_date. ──
    raw_target = p.get("target_date")
    if isinstance(raw_target, str) and raw_target.strip():
        fh, fm = 9, 0
        lt = p.get("local_time")
        if isinstance(lt, str) and ":" in lt:
            try:
                _h = int(lt.split(":")[0]); _m = int(lt.split(":", 1)[1][:2])
                if 0 <= _h <= 23 and 0 <= _m <= 59:
                    fh, fm = _h, _m
            except (ValueError, TypeError):
                pass
        try:
            _y, _mo, _dd = (int(x) for x in raw_target.strip().split("-", 2))
            anchor = datetime(_y, _mo, _dd, fh, fm, tzinfo=tz or UTC)
        except (ValueError, TypeError):
            anchor = None
        if anchor is not None:
            offsets: set[int] = set()
            for o in (p.get("offsets_days") or []):
                try:
                    oi = int(o)
                except (ValueError, TypeError):
                    continue
                if oi >= 0:
                    offsets.add(oi)
            if not offsets:
                offsets = {30, 14, 7, 1, 0}
            for o in offsets:
                moment = _to_utc(anchor - timedelta(days=o))
                if range_start <= moment < range_end:
                    out.append(moment)
        return sorted(out)

    # ── Cron mode: enumerate via the engine's next_after in the user zone. ──
    raw_cron = p.get("cron")
    if isinstance(raw_cron, str) and raw_cron.strip():
        from augmentum.utils import cron as _cron

        try:
            spec = _cron.parse(raw_cron)
        except ValueError:
            spec = None
        if spec is not None:
            cursor = (range_start.astimezone(tz) if tz else range_start) - timedelta(minutes=1)
            end_local = range_end.astimezone(tz) if tz else range_end
            for _ in range(max_count):
                nxt = _cron.next_after(spec, cursor)
                if nxt is None or nxt >= end_local:
                    break
                out.append(_to_utc(nxt))
                cursor = nxt
        return sorted(out)

    # ── Anchored mode: local_time (+ optional weekdays / one-shot date). ──
    local_time = p.get("local_time")
    if isinstance(local_time, str) and ":" in local_time:
        try:
            th = int(local_time.split(":")[0]); tm = int(local_time.split(":", 1)[1][:2])
            if not (0 <= th <= 23 and 0 <= tm <= 59):
                raise ValueError
        except (ValueError, TypeError):
            th = tm = None
        if th is not None:
            # One-shot on a specific date.
            raw_date = p.get("date")
            if isinstance(raw_date, str) and raw_date.strip():
                try:
                    y, mo, dd = (int(x) for x in raw_date.strip().split("-", 2))
                    moment = _to_utc(datetime(y, mo, dd, th, tm, tzinfo=tz or UTC))
                    if range_start <= moment < range_end:
                        out.append(moment)
                except (ValueError, TypeError):
                    pass
                return sorted(out)
            raw_weekdays = p.get("weekdays") or []
            valid = {
                int(w) for w in raw_weekdays
                if isinstance(w, int | float) and 1 <= int(w) <= 7
            } if isinstance(raw_weekdays, list) else set()
            start_local = range_start.astimezone(tz) if tz else range_start
            end_local = range_end.astimezone(tz) if tz else range_end
            day = start_local.replace(hour=th, minute=tm, second=0, microsecond=0)
            if day < start_local:
                day = day + timedelta(days=1)
            guard = 0
            while day < end_local and guard < max_count:
                guard += 1
                if not valid or day.isoweekday() in valid:
                    out.append(_to_utc(day))
                day = day + timedelta(days=1)
                day = day.replace(hour=th, minute=tm, second=0, microsecond=0)
        return sorted(out)

    # ── Interval mode: project from next_run_at, stepping by interval. ──
    step = int(interval_seconds) if interval_seconds and interval_seconds > 0 else 86400
    step_td = timedelta(seconds=step)
    anchor_utc = None
    if next_run_at:
        try:
            norm = next_run_at.replace(" ", "T")
            anchor_utc = datetime.fromisoformat(norm)
            if anchor_utc.tzinfo is None:
                anchor_utc = anchor_utc.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            anchor_utc = None
    if anchor_utc is None:
        anchor_utc = range_start
    # Walk back to at-or-before range_start, then forward through the window.
    cur = anchor_utc
    guard = 0
    while cur > range_start and guard < max_count:
        cur = cur - step_td
        guard += 1
    guard = 0
    while cur < range_end and guard < max_count:
        if cur >= range_start:
            out.append(cur)
        cur = cur + step_td
        guard += 1
    return sorted(out)


def _resolve_zoneinfo(name: str):
    """Return a ZoneInfo for ``name`` or None.

    Bad TZ names fall back to server-local rather than raising — a
    scheduled task should not die because a user typed 'EDT' instead of
    'America/New_York'.
    """
    if not name:
        return None
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    except ImportError:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        log.warning("standing_tasks_bad_timezone", tz=name[:32])
        return None


async def _resolve_user_timezone(app_state: Any, user_id: str) -> str:
    """Look up the user's IANA timezone via settings_store, with fallback
    to the install-wide ``timezone`` setting. Empty string when unset."""
    if not user_id or app_state is None:
        return ""
    store = getattr(app_state, "settings_store", None)
    if store is None:
        return ""
    try:
        tz = await store.get_user_or_global(user_id, "timezone")
    except Exception:
        return ""
    return (tz or "").strip()


# ─── Row dataclass ──────────────────────────────────────────────────────


@dataclass(slots=True)
class StandingTask:
    id: int
    user_id: str
    companion_id: str
    title: str
    kind: str
    params: dict[str, Any]
    interval_seconds: int
    last_run_at: str | None
    next_run_at: str | None
    last_result_summary: str | None
    last_error: str | None
    enabled: bool
    consecutive_error_count: int
    # Slow-but-healthy runs (verb wallclock-budget cancels) count here,
    # NOT toward consecutive_error_count, so a transiently-slow task isn't
    # auto-paused like a broken one (audit 2026-06-17). Default keeps any
    # other constructor working pre-migration.
    consecutive_budget_timeout_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "params": self.params,
            "interval_seconds": self.interval_seconds,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "last_result_summary": self.last_result_summary,
            "last_error": self.last_error,
            "enabled": self.enabled,
            "consecutive_error_count": self.consecutive_error_count,
            "consecutive_budget_timeout_count": self.consecutive_budget_timeout_count,
        }


# Kind → the params key that IS the task's target. An exact match on
# this (same kind, same target) is the strongest duplicate signal —
# two watches on the same URL are the same watch regardless of title.
_DUP_TARGET_KEYS: dict[str, str] = {
    "url_watch": "url",
    "feed_watch": "feed_url",
    "github_releases": "repo",
    "recurring_search": "query",
    "feed_digest": "topic",
    "prompt_fire": "prompt",
    "verb_fire": "verb",
}


def _dup_norm(value: Any) -> str:
    """Normalize a target for duplicate comparison: lowercase, collapsed
    whitespace, capped — 'Watch  BTC' == 'watch btc'."""
    return " ".join(str(value or "").lower().split())[:200]


async def find_similar_tasks(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    companion_id: str,
    kind: str,
    title: str,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Existing live tasks the proposed one likely duplicates.

    The substrate-level read-before-create rule, shared by every
    creation surface (the same discipline schedule_briefing pioneered
    with _find_similar_briefings — enforced in the tool, deterministic,
    never left to the model remembering to list first). Signals:

      * exact target match, same kind (+4) — same URL/feed/repo/query/
        topic/prompt/verb; the strongest signal
      * title similarity, any kind (+2) — case-insensitive containment
        either way (min 4 chars, so 'AI' can't match everything)
      * same kind + same schedule anchor (+1) — local_time or cron

    Score >= 2 qualifies. Delivered one-shots are excluded (a finished
    reminder competes with nothing). Returns up to 3 matches as plain
    dicts (id, title, kind, schedule fields) for a dup-review reply.
    """
    p = params or {}
    existing = await list_tasks(
        conn, user_id=user_id, companion_id=companion_id,
    )
    title_l = _dup_norm(title)
    target_key = _DUP_TARGET_KEYS.get(kind)
    my_target = _dup_norm(p.get(target_key)) if target_key else ""
    my_anchor = _dup_norm(p.get("cron") or p.get("local_time"))

    scored: list[tuple[int, dict[str, Any]]] = []
    for t in existing:
        tp = t.params or {}
        if tp.get("delivered_at"):
            continue
        score = 0
        if t.kind == kind:
            if my_target and target_key \
                    and _dup_norm(tp.get(target_key)) == my_target:
                score += 4
            if kind == "deadline" and p.get("target_date") \
                    and str(p.get("target_date")) == str(tp.get("target_date")):
                score += 3
            their_anchor = _dup_norm(tp.get("cron") or tp.get("local_time"))
            if my_anchor and my_anchor == their_anchor:
                score += 1
        their_title = _dup_norm(t.title)
        if len(title_l) >= 4 and len(their_title) >= 4 and (
            title_l in their_title or their_title in title_l
        ):
            score += 2
        if score >= 2:
            scored.append((score, {
                "id": t.id,
                "title": t.title,
                "kind": t.kind,
                "local_time": tp.get("local_time", ""),
                "cron": tp.get("cron", ""),
                "target": tp.get(_DUP_TARGET_KEYS.get(t.kind, ""), ""),
                "enabled": t.enabled,
            }))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for _, row in scored[:3]]


def _validate_cron_param(params: dict[str, Any] | None) -> None:
    """Reject an unusable ``params.cron`` at write time with ValueError.

    The single validation home for every creation surface (chat tools,
    HTTP routes, future callers) — the engine's own read path stays
    tolerant (malformed cron on an existing row falls through to the
    anchored/interval modes rather than killing the task), but nothing
    unusable should be ACCEPTED into a row in the first place.
    """
    raw = (params or {}).get("cron")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return
    if not isinstance(raw, str):
        raise ValueError("cron must be a string expression")
    from augmentum.utils.cron import validate as _cron_validate
    err = _cron_validate(raw)
    if err:
        raise ValueError(f"cron: {err}")


# ─── CRUD ───────────────────────────────────────────────────────────────


async def add_task(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    companion_id: str,
    title: str,
    kind: str,
    params: dict[str, Any] | None = None,
    interval_seconds: int = 86400,
    user_timezone: str = "",
) -> StandingTask | None:
    if not user_id or not companion_id or not title or not kind:
        raise ValueError("add_task requires user_id, companion_id, title, kind")
    if kind not in _TASK_KINDS:
        raise ValueError(f"unknown task kind: {kind}")
    _validate_cron_param(params)
    interval_seconds = max(300, int(interval_seconds))  # 5min floor
    params_json = json.dumps(params or {})

    # Anchored schedules (local_time or cron set) wait until the next
    # wall-clock occurrence; interval-only tasks run immediately at
    # creation. The user expects "wake me at 9am" to fire at 9am
    # tomorrow, not now.
    #
    # Jitter seed at creation time: row hasn't been INSERTed yet, so no
    # task_id. Use a stable fingerprint of the user-task identity so the
    # initial anchored fire lands in the same jitter slot as every future
    # fire (we'll switch _persist_run to use task_id post-insert).
    has_anchor = bool(
        (params or {}).get("local_time") or (params or {}).get("cron"),
    )
    creation_jitter_seed = (
        f"{user_id}:{companion_id}:{title}:{kind}" if has_anchor else ""
    )
    initial_next = _compute_next_run_at(
        params=params or {}, interval_seconds=interval_seconds,
        user_timezone=user_timezone,
        jitter_seed=creation_jitter_seed,
    ) if has_anchor else None

    try:
        if initial_next:
            cur = await conn.execute(
                """INSERT INTO companion_standing_tasks
                   (user_id, companion_id, title, kind, params,
                    interval_seconds, next_run_at, enabled)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                (user_id, companion_id, title, kind, params_json,
                 interval_seconds, initial_next),
            )
        else:
            cur = await conn.execute(
                """INSERT INTO companion_standing_tasks
                   (user_id, companion_id, title, kind, params,
                    interval_seconds, next_run_at, enabled)
                   VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 1)""",
                (user_id, companion_id, title, kind, params_json, interval_seconds),
            )
        await conn.commit()
        row_id = cur.lastrowid
        await cur.close()
    except aiosqlite.IntegrityError:
        return None

    return await get_task(
        conn, task_id=int(row_id or 0), user_id=user_id, companion_id=companion_id,
    )


async def list_tasks(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    companion_id: str,
) -> list[StandingTask]:
    cur = await conn.execute(
        """SELECT id, user_id, companion_id, title, kind, params,
                  interval_seconds, last_run_at, next_run_at,
                  last_result_summary, last_error,
                  enabled, consecutive_error_count,
                  consecutive_budget_timeout_count
           FROM companion_standing_tasks
           WHERE user_id = ? AND companion_id = ?
           ORDER BY created_at DESC""",
        (user_id, companion_id),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [_row_to_task(r) for r in rows]


async def get_task(
    conn: aiosqlite.Connection,
    *,
    task_id: int,
    user_id: str,
    companion_id: str,
) -> StandingTask | None:
    cur = await conn.execute(
        """SELECT id, user_id, companion_id, title, kind, params,
                  interval_seconds, last_run_at, next_run_at,
                  last_result_summary, last_error,
                  enabled, consecutive_error_count,
                  consecutive_budget_timeout_count
           FROM companion_standing_tasks
           WHERE id = ? AND user_id = ? AND companion_id = ?""",
        (int(task_id), user_id, companion_id),
    )
    row = await cur.fetchone()
    await cur.close()
    return _row_to_task(row) if row else None


async def remove_task(
    conn: aiosqlite.Connection,
    *,
    task_id: int,
    user_id: str,
    companion_id: str,
) -> bool:
    cur = await conn.execute(
        """DELETE FROM companion_standing_tasks
           WHERE id = ? AND user_id = ? AND companion_id = ?""",
        (int(task_id), user_id, companion_id),
    )
    affected = cur.rowcount or 0
    await cur.close()
    await conn.commit()
    return affected > 0


async def set_enabled(
    conn: aiosqlite.Connection,
    *,
    task_id: int,
    user_id: str,
    companion_id: str,
    enabled: bool,
) -> bool:
    cur = await conn.execute(
        """UPDATE companion_standing_tasks
           SET enabled = ?, consecutive_error_count = 0
           WHERE id = ? AND user_id = ? AND companion_id = ?""",
        (1 if enabled else 0, int(task_id), user_id, companion_id),
    )
    affected = cur.rowcount or 0
    await cur.close()
    await conn.commit()
    return affected > 0


# Param keys the engine itself writes/mutates as a task runs (cursors,
# hashes, dedup sets, extraction state). An edit from the UI carries only
# user-facing fields, so these are preserved from the existing params
# rather than wiped — editing a url_watch's schedule must not reset its
# diff baseline.
_INTERNAL_PARAM_KEYS: frozenset[str] = frozenset({
    "last_hash", "last_seen_id", "seen", "metric_state", "extract_method",
    "_lat", "_lon", "delivered_at", "delivery_summary",
})


async def update_task(
    conn: aiosqlite.Connection,
    *,
    task_id: int,
    user_id: str,
    companion_id: str,
    title: str | None = None,
    params: dict[str, Any] | None = None,
    interval_seconds: int | None = None,
    user_timezone: str = "",
) -> StandingTask | None:
    """Edit a task's title / params / interval in place.

    Kind is immutable (changing it would invalidate the params contract —
    delete + re-add for that). When params or interval change, next_run_at
    is recomputed so a schedule edit takes effect immediately. Engine-owned
    bookkeeping keys (``_INTERNAL_PARAM_KEYS``) are carried over from the
    existing params so an edit doesn't reset a watch's diff baseline or a
    digest's dedup set. Returns the updated task, or None if not found.
    """

    _validate_cron_param(params)
    existing = await get_task(
        conn, task_id=task_id, user_id=user_id, companion_id=companion_id,
    )
    if existing is None:
        return None

    new_title = existing.title if title is None else title
    if interval_seconds is None:
        new_interval = existing.interval_seconds
    else:
        new_interval = max(300, int(interval_seconds))

    if params is None:
        new_params = dict(existing.params)
        recompute = interval_seconds is not None
    else:
        preserved = {
            k: v for k, v in (existing.params or {}).items()
            if k in _INTERNAL_PARAM_KEYS
        }
        new_params = {**preserved, **params}
        recompute = True

    sets = ["title = ?", "params = ?", "interval_seconds = ?"]
    vals: list[Any] = [new_title, json.dumps(new_params), new_interval]

    if recompute:
        next_run = _compute_next_run_at(
            params=new_params, interval_seconds=new_interval,
            user_timezone=user_timezone,
            jitter_seed=f"task:{task_id}",
        )
        sets.append("next_run_at = ?")
        vals.append(next_run)

    vals.extend([int(task_id), user_id, companion_id])
    cur = await conn.execute(
        f"""UPDATE companion_standing_tasks
            SET {', '.join(sets)}
            WHERE id = ? AND user_id = ? AND companion_id = ?""",
        vals,
    )
    affected = cur.rowcount or 0
    await cur.close()
    await conn.commit()
    if not affected:
        return None
    return await get_task(
        conn, task_id=task_id, user_id=user_id, companion_id=companion_id,
    )


def _row_to_task(row: tuple) -> StandingTask:
    try:
        params = json.loads(row[5] or "{}")
    except (json.JSONDecodeError, TypeError):
        params = {}
    return StandingTask(
        id=int(row[0]), user_id=row[1], companion_id=row[2],
        title=row[3] or "", kind=row[4] or "", params=params,
        interval_seconds=int(row[6] or 86400),
        last_run_at=row[7], next_run_at=row[8],
        last_result_summary=row[9],
        last_error=row[10],
        enabled=bool(row[11]),
        consecutive_error_count=int(row[12] or 0),
        consecutive_budget_timeout_count=int(
            (row[13] if len(row) > 13 else 0) or 0
        ),
    )


# ─── Persistence after a run ────────────────────────────────────────────


# Newest N run rows kept per task. The history is a trust surface, not
# an archive — enough to render "checked 2h ago, nothing new" timelines
# without unbounded growth.
_RUN_HISTORY_KEEP: int = 20


async def _record_run(
    conn: aiosqlite.Connection,
    *,
    task_id: int,
    user_id: str,
    status: str,
    summary: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Append one row to companion_standing_task_runs and trim history.

    Best-effort: a history-write failure must never break the run path
    (the run itself already happened), but it logs at warning because a
    silently empty timeline reads as "never ran" to the user.
    """
    try:
        await conn.execute(
            """INSERT INTO companion_standing_task_runs
               (task_id, user_id, status, summary, details)
               VALUES (?, ?, ?, ?, ?)""",
            (
                int(task_id), user_id, status,
                (summary or "")[:500],
                json.dumps(details or {}),
            ),
        )
        await conn.execute(
            """DELETE FROM companion_standing_task_runs
               WHERE task_id = ?
                 AND id NOT IN (
                     SELECT id FROM companion_standing_task_runs
                     WHERE task_id = ?
                     ORDER BY id DESC LIMIT ?
                 )""",
            (int(task_id), int(task_id), _RUN_HISTORY_KEEP),
        )
        await conn.commit()
    except Exception:
        log.warning("standing_task_run_record_failed", task_id=task_id, exc_info=True)


async def list_runs(
    conn: aiosqlite.Connection,
    *,
    task_id: int,
    user_id: str,
    limit: int = _RUN_HISTORY_KEEP,
) -> list[dict[str, Any]]:
    """Run history for one task, newest first. User-scoped."""
    cur = await conn.execute(
        """SELECT id, ran_at, status, summary, details
           FROM companion_standing_task_runs
           WHERE task_id = ? AND user_id = ?
           ORDER BY id DESC LIMIT ?""",
        (int(task_id), user_id, max(1, min(int(limit), 100))),
    )
    rows = await cur.fetchall()
    await cur.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            details = json.loads(r[4] or "{}")
        except (json.JSONDecodeError, TypeError):
            details = {}
        out.append({
            "id": int(r[0]), "ran_at": r[1], "status": r[2] or "",
            "summary": r[3] or "", "details": details,
        })
    return out


async def _record_observation(
    conn: aiosqlite.Connection,
    *,
    task_id: int,
    user_id: str,
    observation: dict[str, Any],
) -> None:
    """Append one numeric reading to companion_metric_observations.
    Best-effort like _record_run — the series is an audit trail, not a
    gate on the run itself."""
    try:
        await conn.execute(
            """INSERT INTO companion_metric_observations
               (task_id, user_id, series, value, scale, unit, method,
                status, evidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(task_id), user_id,
                str(observation.get("series") or "value"),
                observation.get("value"),       # already scaled int or None
                int(observation.get("scale") or 100),
                str(observation.get("unit") or "")[:16],
                str(observation.get("method") or "")[:16],
                str(observation.get("status") or "ok")[:16],
                (str(observation.get("evidence") or "")[:300] or None),
            ),
        )
        await conn.commit()
    except Exception:
        log.warning(
            "standing_task_observation_record_failed",
            task_id=task_id, exc_info=True,
        )


async def _persist_run(
    conn: aiosqlite.Connection,
    *,
    task_id: int,
    interval_seconds: int,
    result: dict | None,
    error: str | None,
    consecutive_error_count: int,
    auto_pause: bool = False,
    params: dict[str, Any] | None = None,
    user_timezone: str = "",
    consecutive_budget_timeout_count: int | None = None,
    persist_params: dict[str, Any] | None = None,
) -> None:
    """Persist run metadata + advance next_run_at.

    next_run_at uses :func:`_compute_next_run_at`, which honors anchored
    ``local_time`` scheduling when the task's params declare it and falls
    back to ``interval_seconds`` otherwise. Pass the task's params here so
    anchored tasks land on their next wall-clock occurrence rather than
    drifting on the interval clock.

    ``consecutive_budget_timeout_count`` (None = leave unchanged) and
    ``persist_params`` (None = leave unchanged) fold the budget-counter
    bookkeeping and the recurring param-cursor write into THIS single
    UPDATE, so the cursor advance can't be lost if a separate write
    failed between them (audit 2026-06-17).
    """
    result_json = json.dumps(result) if result else None
    summary = (result or {}).get("summary") if result else None
    # After the first fire we have a stable task_id — use it as the
    # jitter seed so every future fire lands in the same slot within
    # its anchor's jitter window.
    next_run_at = _compute_next_run_at(
        params=params or {}, interval_seconds=interval_seconds,
        user_timezone=user_timezone,
        jitter_seed=str(task_id),
    )
    await conn.execute(
        """UPDATE companion_standing_tasks
           SET last_run_at = datetime('now'),
               next_run_at = ?,
               last_result = ?,
               last_result_summary = COALESCE(?, last_result_summary),
               last_error = ?,
               consecutive_error_count = ?,
               consecutive_budget_timeout_count =
                   COALESCE(?, consecutive_budget_timeout_count),
               params = COALESCE(?, params),
               enabled = CASE WHEN ? THEN 0 ELSE enabled END
           WHERE id = ?""",
        (
            next_run_at,
            result_json,
            summary,
            (error[:200] if error else None),
            int(consecutive_error_count),
            (int(consecutive_budget_timeout_count)
             if consecutive_budget_timeout_count is not None else None),
            (json.dumps(persist_params) if persist_params is not None else None),
            1 if auto_pause else 0,
            int(task_id),
        ),
    )
    await conn.commit()


# ─── Task kinds ─────────────────────────────────────────────────────────


@_register_kind("feed_digest")
async def _kind_feed_digest(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Curator on-demand. params = {"topic": str, "max_items": int=3}.
    Reuses curator's poll + scoring for a single topic, rolls top N into
    one digest note."""
    from augmentum.companion_runtime import curator

    topic = (params.get("topic") or "").strip()
    if not topic:
        raise ValueError("feed_digest requires params.topic")
    max_items = max(1, min(int(params.get("max_items", 3)), 10))

    http_client = curator._resolve_http_client(runtime)
    if http_client is None:
        return {"summary": "no http_client; deferred", "noteworthy": False, "refs": []}

    items = await curator.poll_for_topic(
        runtime, topic=topic, http_client=http_client,
    )
    scored = sorted(
        ((curator.score_relevance(it, topic), it) for it in items),
        key=lambda p: p[0], reverse=True,
    )

    picked: list[dict] = []
    seen_urls: set[str] = set()
    for score, it in scored:
        if score < curator._MIN_RELEVANCE_SCORE:
            break
        url = (it.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        if await curator._seen_url_recently(
            runtime.backend.conn,
            user_id=user_id, companion_id=runtime.companion_id, url=url,
        ):
            continue
        seen_urls.add(url)
        picked.append(it)
        if len(picked) >= max_items:
            break

    if not picked:
        return {
            "summary": f"{topic}: nothing new",
            "noteworthy": False, "refs": [],
        }

    lines = [f"{topic} digest"]
    refs: list[dict] = []
    for it in picked:
        title = (it.get("title") or "").strip()
        snippet = (it.get("snippet") or "")[:120].rstrip()
        lines.append(f"• {title}" + (f" — {snippet}" if snippet else ""))
        url = (it.get("url") or "").strip()
        if url:
            refs.append({
                "kind": "url", "url": url,
                "id": curator._url_hash(url),
            })

    return {
        "summary": f"{topic}: {len(picked)} new",
        "noteworthy": True,
        "refs": refs,
        "details": {"content": "\n".join(lines)},
    }


@_register_kind("github_releases")
async def _kind_github_releases(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Poll a repo's releases.atom feed (no auth required).
    params = {"repo": "owner/name"}. Surfaces only releases newer than
    the last_seen_tag stored in params (mutated in-place on hit)."""
    from augmentum.companion_runtime import curator

    repo = (params.get("repo") or "").strip().strip("/")
    if not repo or "/" not in repo:
        raise ValueError("github_releases requires params.repo = owner/name")

    http_client = curator._resolve_http_client(runtime)
    if http_client is None:
        return {"summary": "no http_client; deferred", "noteworthy": False, "refs": []}

    url = f"https://github.com/{repo}/releases.atom"
    try:
        resp = await http_client.get(
            url, timeout=15.0,
            headers={"User-Agent": "Augmentum/1.0 (standing-tasks)"},
        )
        resp.raise_for_status()
        body = resp.text
    except Exception as exc:
        raise RuntimeError(f"github fetch failed: {str(exc)[:120]}") from exc

    # Parse minimal atom — pull <entry><title>...<id>tag:...,...:Repository/123/v...
    entries = re.findall(
        r"<entry>.*?<title>([^<]+)</title>.*?<id>([^<]+)</id>.*?<link[^>]+href=\"([^\"]+)\"",
        body, re.DOTALL,
    )
    if not entries:
        return {"summary": f"{repo}: no releases parsed", "noteworthy": False, "refs": []}

    last_seen = (params.get("last_seen_id") or "").strip()
    if not last_seen:
        # First poll — every release in the feed is "new" only in the
        # trivial sense. Capture the baseline silently instead of
        # notifying with the repo's entire release history.
        params["last_seen_id"] = entries[0][1]
        latest_title = entries[0][0].strip()
        return {
            "summary": f"{repo}: watching (latest release: {latest_title})",
            "noteworthy": False,
            "refs": [],
            "details": {"params_update": dict(params)},
        }
    new_entries: list[tuple[str, str, str]] = []
    for title, entry_id, link in entries:
        if entry_id == last_seen:
            break
        new_entries.append((title.strip(), entry_id, link.strip()))

    if not new_entries:
        return {"summary": f"{repo}: no new releases", "noteworthy": False, "refs": []}

    # Mutate params for next-run baseline.
    params["last_seen_id"] = entries[0][1]

    lines = [f"{repo} — {len(new_entries)} new release{'s' if len(new_entries) != 1 else ''}"]
    refs = []
    for title, _, link in new_entries[:5]:
        lines.append(f"• {title}")
        if link:
            refs.append({
                "kind": "url", "url": link,
                "id": curator._url_hash(link),
            })

    return {
        "summary": f"{repo}: {len(new_entries)} new release{'s' if len(new_entries) != 1 else ''}",
        "noteworthy": True,
        "refs": refs,
        "details": {"content": "\n".join(lines), "params_update": dict(params)},
    }


@_register_kind("feed_watch")
async def _kind_feed_watch(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Poll ANY RSS/Atom feed and surface new entries — the generic
    "follow a creator" kind: YouTube channels (keyless per-channel Atom),
    podcasts, blogs/Substack, subreddits, arXiv, alert feeds.

    params = {"feed_url": str, "source_label": str="", "max_items": 3}.
    Baseline semantics mirror github_releases: the first poll captures
    ``last_seen_id`` silently (following someone is not news), later
    polls surface entries newer than it. ``feed_resolve.parse_feed``
    handles both formats; creation surfaces (UI resolve endpoint /
    watch_for) already validated the feed once.
    """
    from augmentum.companion_runtime import curator
    from augmentum.companion_runtime.feed_resolve import parse_feed

    feed_url = (params.get("feed_url") or "").strip()
    if not feed_url.startswith(("http://", "https://")):
        raise ValueError("feed_watch requires params.feed_url (http/https)")
    label = (params.get("source_label") or "").strip() or _short_host(feed_url)
    max_items = max(1, min(int(params.get("max_items", 3)), 10))

    http_client = curator._resolve_http_client(runtime)
    if http_client is None:
        return {"summary": "no http_client; deferred", "noteworthy": False, "refs": []}

    try:
        resp = await http_client.get(
            feed_url, timeout=15.0,
            headers={"User-Agent": "Augmentum/1.0 (feed-watch)"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        body = resp.text
    except Exception as exc:
        raise RuntimeError(f"feed fetch failed: {str(exc)[:120]}") from exc

    entries = parse_feed(body)
    if not entries:
        return {
            "summary": f"{label}: feed returned no entries",
            "noteworthy": False, "refs": [],
        }

    last_seen = (params.get("last_seen_id") or "").strip()
    if not last_seen:
        # First poll — capture the baseline silently instead of dumping
        # the channel's entire back catalog on the user.
        params["last_seen_id"] = entries[0]["id"]
        latest = entries[0].get("title") or "(untitled)"
        return {
            "summary": f"{label}: following (latest: {latest[:80]})",
            "noteworthy": False,
            "refs": [],
            "details": {"params_update": dict(params)},
        }

    new_entries: list[dict[str, str]] = []
    for e in entries:
        if e["id"] == last_seen:
            break
        new_entries.append(e)

    if not new_entries:
        return {"summary": f"{label}: nothing new", "noteworthy": False, "refs": []}

    params["last_seen_id"] = entries[0]["id"]

    shown = new_entries[:max_items]
    n = len(new_entries)
    lines = [f"{label} — {n} new post{'s' if n != 1 else ''}"]
    refs: list[dict] = []
    for e in shown:
        title = (e.get("title") or "(untitled)").strip()
        lines.append(f"• {title}")
        url = (e.get("url") or "").strip()
        if url:
            lines.append(f"  {url}")
            refs.append({
                "kind": "url", "url": url, "id": curator._url_hash(url),
            })
    if n > len(shown):
        lines.append(f"…and {n - len(shown)} more")

    return {
        "summary": f"{label}: {n} new post{'s' if n != 1 else ''}"
                   + (f" — {shown[0].get('title', '')[:60]}" if shown else ""),
        "noteworthy": True,
        "refs": refs,
        "details": {"content": "\n".join(lines), "params_update": dict(params)},
    }


@_register_kind("url_watch")
async def _kind_url_watch(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Fetch a URL, hash the body, surface when changed. Mutates
    params['last_hash'] in place."""
    from augmentum.companion_runtime import curator

    url = (params.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("url_watch requires params.url with http(s):// scheme")

    http_client = curator._resolve_http_client(runtime)
    if http_client is None:
        return {"summary": "no http_client; deferred", "noteworthy": False, "refs": []}

    try:
        resp = await http_client.get(
            url, timeout=20.0,
            headers={"User-Agent": "Augmentum/1.0 (standing-tasks)"},
        )
        resp.raise_for_status()
        body = resp.text
    except Exception as exc:
        raise RuntimeError(f"fetch failed: {str(exc)[:120]}") from exc

    # Condition-carrying watch ("below $500"): the unit of change is the
    # NUMBER, not the bytes. Extract via the ladder, classify through the
    # quarantine/confirm state machine, fire only on a confirmed
    # condition crossing. Hash-diff semantics below never run for these —
    # a retail page's rotating chrome would otherwise drown the signal.
    condition = params.get("condition")
    if isinstance(condition, dict) and condition:
        from augmentum.companion_runtime import metrics as _metrics
        from augmentum.config import settings

        ext = _metrics.extract_value(
            body,
            hint=str(params.get("extract_hint") or ""),
            pinned_method=str(params.get("extract_method") or ""),
        )
        if ext.ambiguous:
            # Multiple distinct candidates — an error state, not a guess
            # (changedetection.io's MoreThanOnePriceFound rule).
            verdict = _metrics.Verdict(
                status="missing",
                note=(
                    f"{len(ext.candidates)} different values on the page "
                    "— couldn't tell which one to track"
                ),
                state=dict(params.get("metric_state") or {}),
            )
        else:
            verdict = _metrics.classify_reading(
                ext.value,
                state=params.get("metric_state"),
                condition=condition,
                quarantine_pct=float(getattr(
                    settings, "companion_metric_quarantine_pct", 60.0,
                ) or 60.0),
                confirm_readings=int(getattr(
                    settings, "companion_metric_confirm_readings", 2,
                ) or 2),
            )
        params["metric_state"] = verdict.state
        if ext.method and not params.get("extract_method"):
            params["extract_method"] = ext.method  # pin the working rung
        unit = str(condition.get("unit") or "")
        observation = {
            "series": "value",
            "value": (
                _metrics.to_scaled_int(ext.value)
                if ext.value is not None else None
            ),
            "scale": 100,
            "unit": unit,
            "method": ext.method,
            "status": verdict.status,
            "evidence": ext.evidence,
        }
        fired_line = ""
        if verdict.fire:
            fired_line = (
                f"{_short_host(url)}: {ext.value:g}{(' ' + unit) if unit else ''} "
                f"— condition {condition.get('op')} {condition.get('value'):g} met"
            )
        return {
            "summary": fired_line or f"{_short_host(url)}: {verdict.note}",
            "noteworthy": bool(verdict.fire),
            "refs": [{"kind": "url", "url": url, "id": curator._url_hash(url)}],
            "details": {
                "content": fired_line or "",
                "observation": observation,
                "params_update": dict(params),
            },
        }

    # Hash body; strip whitespace differences and obvious volatile
    # bits (timestamps, csrf tokens, build hashes) before hashing —
    # otherwise every fetch looks "changed."
    normalized = re.sub(r"\s+", " ", body).strip()
    normalized = re.sub(
        r"(?i)(csrf|nonce|sessionid|build[_-]?id)[^\"'\s]*[\"'][^\"']+[\"']",
        "", normalized,
    )
    body_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    prior = (params.get("last_hash") or "").strip()
    if body_hash == prior:
        return {"summary": f"{_short_host(url)}: unchanged", "noteworthy": False, "refs": []}

    params["last_hash"] = body_hash
    if not prior:
        # First fetch — nothing to compare against. Capture the baseline
        # silently; "we just started watching" is not a change.
        return {
            "summary": f"{_short_host(url)}: baseline captured",
            "noteworthy": False,
            "refs": [{"kind": "url", "url": url, "id": curator._url_hash(url)}],
            "details": {"params_update": dict(params)},
        }
    return {
        "summary": f"{_short_host(url)}: changed",
        "noteworthy": True,
        "refs": [{"kind": "url", "url": url, "id": curator._url_hash(url)}],
        "details": {
            "content": f"{params.get('title') or _short_host(url)} changed — go check.",
            "params_update": dict(params),
        },
    }


@_register_kind("recurring_search")
async def _kind_recurring_search(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """SearXNG query on a cadence, surface novel results.
    params = {"query": str, "max_results": int=3}. Tracks seen url hashes
    in params['seen'] (capped to last 100)."""
    from augmentum.companion_runtime import curator
    from augmentum.config import settings

    query = (params.get("query") or "").strip()
    if not query:
        raise ValueError("recurring_search requires params.query")
    max_results = max(1, min(int(params.get("max_results", 3)), 10))

    http_client = curator._resolve_http_client(runtime)
    if http_client is None:
        return {"summary": "no http_client; deferred", "noteworthy": False, "refs": []}

    searxng_base = getattr(settings, "searxng_base_url", "")
    if not searxng_base:
        return {"summary": "no searxng configured", "noteworthy": False, "refs": []}

    try:
        resp = await http_client.get(
            f"{searxng_base.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            timeout=15.0,
            headers={"User-Agent": "Augmentum/1.0 (standing-tasks)"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"searxng query failed: {str(exc)[:120]}") from exc

    results = data.get("results") or []
    if not results:
        return {"summary": f"\"{query}\": no results", "noteworthy": False, "refs": []}

    seen: list[str] = list(params.get("seen") or [])
    seen_set = set(seen)

    novel: list[dict] = []
    for r in results:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        h = curator._url_hash(url)
        if h in seen_set:
            continue
        novel.append({
            "title": (r.get("title") or "").strip(),
            "snippet": (r.get("content") or "").strip(),
            "url": url,
            "_h": h,
        })
        if len(novel) >= max_results:
            break

    if not novel:
        return {"summary": f"\"{query}\": nothing novel", "noteworthy": False, "refs": []}

    for n in novel:
        seen.append(n["_h"])
    # Keep last 100 to stop params from growing forever.
    params["seen"] = seen[-100:]

    lines = [f"\"{query}\" — {len(novel)} new"]
    refs: list[dict] = []
    for n in novel:
        lines.append(f"• {n['title']}" + (f" — {n['snippet'][:120]}" if n['snippet'] else ""))
        refs.append({"kind": "url", "url": n["url"], "id": n["_h"]})

    return {
        "summary": f"\"{query}\": {len(novel)} new",
        "noteworthy": True,
        "refs": refs,
        "details": {"content": "\n".join(lines), "params_update": dict(params)},
    }


@_register_kind("deadline")
async def _kind_deadline(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Countdown toward a target date — fires at each lead-time offset with
    days-remaining + an optional checklist of what's still outstanding.

    No external data: pure date math + the standard delivery path. Completes
    itself on the day-of (or any past) fire by handing ``step()`` the
    ``one_shot`` flag, so the row retires once the date arrives. The schedule
    of fires is computed by :func:`_compute_next_run_at`'s deadline mode.
    """
    from datetime import date, datetime

    raw_target = str(params.get("target_date") or "").strip()
    if not raw_target:
        raise ValueError("deadline requires params.target_date (YYYY-MM-DD)")
    try:
        ty, tmo, tdd = (int(x) for x in raw_target.split("-", 2))
        target = date(ty, tmo, tdd)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"deadline bad target_date: {raw_target[:16]}") from exc

    title = str(params.get("title") or "Deadline").strip() or "Deadline"
    app_state = getattr(runtime, "_app_state", None)
    tzname = await _resolve_user_timezone(app_state, user_id)
    tz = _resolve_zoneinfo(tzname)
    today = (datetime.now(tz) if tz else datetime.now().astimezone()).date()
    days = (target - today).days

    if days > 1:
        lead = f"{days} days left"
    elif days == 1:
        lead = "tomorrow"
    elif days == 0:
        lead = "today"
    else:
        lead = f"{abs(days)} day{'s' if abs(days) != 1 else ''} ago"

    lines = [f"{title} — {lead} ({target.isoformat()})"]
    note = str(params.get("note") or "").strip()
    if note:
        lines.append(note)
    checklist = params.get("checklist")
    if isinstance(checklist, list) and checklist:
        lines.append("Still to do:")
        for c in checklist[:12]:
            item = str(c).strip()
            if item:
                lines.append(f"• {item}")
    content = "\n".join(lines)

    details: dict[str, Any] = {"content": content}
    # Complete on the day-of (or any past) fire: hand step() the one_shot
    # flag so the existing one-shot path disables + marks the row delivered.
    if days <= 0:
        completed = dict(params)
        completed["one_shot"] = True
        details["params_update"] = completed

    return {
        "summary": f"{title} — {lead}",
        "noteworthy": True,
        "refs": [],
        "details": details,
    }


async def _gather_image_candidates_for_briefing(
    runtime: CompanionRuntime, query: str, count: int = 3,
    user_id: str = "",
) -> list[dict[str, str]]:
    """Return up to ``count`` image candidates as ``[{url, alt, source}]``.

    Goes through the registered ``image_search`` tool (which already
    handles SearXNG image categories, quality filtering, download +
    artifact storage). Returns empty list on any failure — the briefing
    falls back to text-only synthesis.
    """
    app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        return []
    registry = getattr(app_state, "tool_registry", None)
    if registry is None:
        return []
    tool = registry.resolve("image_search")
    if tool is None:
        return []
    try:
        # _user_id: the artifact store refuses anonymous saves, and this
        # gather runs headless — no chain/orchestrator context to inject it.
        result = await tool.execute(
            query=query, count=count, _user_id=user_id,
        )
        if not result.success:
            log.info(
                "briefing_image_gather_empty",
                query=query[:80], error=(result.error or "")[:120],
            )
            return []
        items = (result.metadata or {}).get("images") or []
        if not items:
            log.info(
                "briefing_image_gather_no_items", query=query[:80],
            )
    except Exception as exc:
        log.warning(
            "briefing_image_gather_failed", error=str(exc)[:200],
        )
        return []
    out: list[dict[str, str]] = []
    for i in items[:count]:
        url = i.get("embed_url") or i.get("url") or ""
        if not url:
            continue
        out.append({
            "url": url,
            "alt": i.get("title") or "",
            "source": i.get("source") or "",
        })
    return out


async def _gather_video_candidate_for_briefing(
    runtime: CompanionRuntime, query: str,
) -> dict[str, str] | None:
    """Search YouTube via the registered ``youtube`` tool, then fetch
    transcript for the top result. Returns ``{url, title, channel,
    thumbnail, transcript_summary}`` or None on any failure.

    Two-call flow because YouTubeTool's search mode returns metadata
    candidates and direct mode (URL or video_id) returns transcript.
    """
    app_state = getattr(runtime, "_app_state", None)
    if app_state is None:
        return None
    registry = getattr(app_state, "tool_registry", None)
    if registry is None:
        return None
    tool = registry.resolve("youtube")
    if tool is None:
        return None
    try:
        search_result = await tool.execute(query=query)
        if not search_result.success:
            return None
        md = search_result.metadata or {}
        # Either we hit search mode (list of candidates) or got direct
        # (one transcript) — handle both shapes.
        if md.get("youtube_mode") == "search":
            results = md.get("results") or []
            if not results:
                return None
            top = results[0]
            video_url = top.get("url", "")
            base_title = top.get("title", "")
            base_channel = top.get("channel", "")
            base_thumbnail = top.get("thumbnail", "")
        else:
            video_url = md.get("url") or ""
            base_title = md.get("title", "")
            base_channel = md.get("channel", "")
            base_thumbnail = md.get("thumbnail", "")
        if not video_url:
            return None

        transcript_summary = ""
        if md.get("youtube_mode") == "search":
            # Second pass to fetch transcript for the top result.
            try:
                transcript_result = await tool.execute(query=video_url)
                if transcript_result.success:
                    tmd = transcript_result.metadata or {}
                    paragraphs = tmd.get("paragraphs") or []
                    chunks: list[str] = []
                    total = 0
                    for p in paragraphs:
                        t = (p.get("text") or "").strip()
                        if not t:
                            continue
                        if total + len(t) > 800:
                            break
                        chunks.append(t)
                        total += len(t)
                    transcript_summary = " ".join(chunks)[:800]
            except Exception:
                log.warning(
                    "briefing_video_transcript_fetch_failed", exc_info=True,
                )
        else:
            paragraphs = md.get("paragraphs") or []
            chunks = []
            total = 0
            for p in paragraphs:
                t = (p.get("text") or "").strip()
                if not t:
                    continue
                if total + len(t) > 800:
                    break
                chunks.append(t)
                total += len(t)
            transcript_summary = " ".join(chunks)[:800]
    except Exception as exc:
        log.warning(
            "briefing_video_gather_failed", error=str(exc)[:200],
        )
        return None

    return {
        "url": video_url,
        "title": base_title,
        "channel": base_channel,
        "thumbnail": base_thumbnail,
        "transcript_summary": transcript_summary,
    }


# Topic tokens that mark a briefing as ephemeral / time-sensitive: the
# answer's value lives in the LIVE data, not in any prior briefing. For
# these topics the prior-briefing recall layer is structurally disabled
# (Layer B of the anti-anchor fix) so the model literally cannot see a
# prior price/score/temperature/etc. to parrot. The Layer A system-prompt
# rules are the safety net for mixed-topic briefings where some topics
# qualify and others don't — there we still send recall but tell the
# model strictly how to use it.
_EPHEMERAL_TOPIC_TOKENS: frozenset[str] = frozenset({
    # Markets / finance
    "price", "prices", "pricing", "rate", "rates", "stock", "stocks",
    "ticker", "quote", "quotes", "yield", "yields", "btc", "eth",
    "crypto", "currency", "currencies", "forex", "fx",
    # Weather
    "weather", "forecast", "temperature", "temp", "rain", "snow",
    "humidity", "wind", "uv", "aqi", "air",
    # Sports / live events
    "score", "scores", "scoreboard", "result", "results", "standings",
    "live", "kickoff", "tipoff",
    # System / status / network
    "status", "uptime", "outage", "outages", "online", "offline",
    "incident", "incidents",
    # Generic time-anchored
    "current", "now", "today",
})


def _is_ephemeral_topic(topic: str) -> bool:
    """True when ``topic`` mentions a live-data signal whose value lives
    only in the current results. Token match on lowercase words so
    'Bitcoin price', 'BTC/USD rate', 'sf weather forecast' all qualify
    while 'AI industry news' / 'kubernetes 1.30 release notes' don't."""
    if not topic:
        return False
    tl = topic.lower()
    # Whole-word match avoids 'temperature' matching 'tempo' etc., but
    # token-level is fast and correct enough — the set is whole tokens.
    for tok in tl.replace("/", " ").replace("-", " ").split():
        tok = tok.strip(".,;:?!()[]'\"")
        if tok in _EPHEMERAL_TOPIC_TOKENS:
            return True
    return False


def _should_skip_recall(topics: list[str]) -> bool:
    """Skip prior-briefing recall entirely when ALL topics are
    ephemeral. Mixed briefings still get recall (with Layer A's strict
    prompt rules as the guard). No topics → skip is moot, return False."""
    if not topics:
        return False
    return all(_is_ephemeral_topic(t) for t in topics)


async def _recall_prior_briefing_context(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    companion_id: str,
    topics: list[str],
    limit: int = 3,
    days_back: int = 30,
) -> list[dict[str, str]]:
    """Scoped recall for the briefing synthesis pass.

    Topic-overlap query across TWO sources:
      1. ``companion_standing_tasks`` rows where ``params.delivered_at``
         is set (delivered one-shots). Pull ``params.delivery_summary``.
      2. ``companion_journal`` entries with ``source='standing_task'``
         in the lookback window. These are recurring-fire outputs.

    Filtering: a row is "relevant" if any current topic string appears
    (case-insensitive substring) in either the stored topics list or
    the entry text. Cheaper than FTS5/embeddings and sufficient for
    MVP. Escalation path: full-text or vec match when this proves too
    narrow on user feedback.

    Returns up to ``limit`` most-recent {when, summary, source} dicts
    sorted desc by timestamp. Designed to be safe to inject — never
    pulls relational/affect/persona/companion-journal entries (those
    have different ``source`` values), so the "introduce friendly
    facts about the user's cat" failure mode is structurally
    prevented.
    """
    if not topics:
        return []
    candidates: list[dict[str, str]] = []
    topic_norm = [t.strip().lower() for t in topics if t.strip()]

    # 1. Delivered one-shots — read params blob, decode, topic-match.
    try:
        cur = await conn.execute(
            """SELECT params, last_run_at
               FROM companion_standing_tasks
               WHERE user_id = ? AND companion_id = ?
                 AND kind = 'briefing'
                 AND json_extract(params, '$.delivered_at') IS NOT NULL
                 AND last_run_at IS NOT NULL
               ORDER BY last_run_at DESC
               LIMIT 50""",
            (user_id, companion_id),
        )
        rows = await cur.fetchall()
        await cur.close()
        for row in rows:
            try:
                p = json.loads(row[0] or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            their_topics = [
                str(t).strip().lower() for t in (p.get("topics") or [])
            ]
            if not any(
                tn in their or their in tn
                for tn in topic_norm for their in their_topics
            ):
                continue
            candidates.append({
                "when": str(row[1]),
                "summary": (p.get("delivery_summary") or "")[:300],
                "source": "delivered_one_shot",
            })
    except Exception:
        log.warning("recall_delivered_query_failed", exc_info=True)

    # 2. Journal entries from prior recurring fires — content-string match.
    # Match both the full topic phrase ("bitcoin price") AND its
    # individual significant words (>=4 chars, skip stopwords). Phrase
    # matching alone is too narrow — a journal entry saying "Bitcoin at
    # $63k" wouldn't match "bitcoin price" without word-level fallback.
    try:
        _STOPWORDS = {
            "the", "and", "for", "with", "from", "this", "that",
            "your", "into", "what", "where", "when", "today",
            "tomorrow", "tonight", "morning", "evening",
        }
        words: set[str] = set()
        for tn in topic_norm:
            for w in tn.split():
                w = w.strip(".,;:?!").lower()
                if len(w) >= 4 and w not in _STOPWORDS:
                    words.add(w)
        like_clauses = []
        like_params: list[Any] = []
        # Phrase matches first (higher signal), then word fallbacks.
        for tn in topic_norm:
            like_clauses.append("LOWER(content) LIKE ?")
            like_params.append(f"%{tn}%")
        for w in words:
            like_clauses.append("LOWER(content) LIKE ?")
            like_params.append(f"%{w}%")
        if like_clauses:
            sql = (
                "SELECT content, created_at FROM companion_journal "
                "WHERE user_id = ? AND companion_id = ? "
                "AND source = 'standing_task' "
                "AND created_at > datetime('now', ?) "
                "AND (" + " OR ".join(like_clauses) + ") "
                "ORDER BY created_at DESC LIMIT 20"
            )
            args = [user_id, companion_id, f"-{int(days_back)} days"] + like_params
            cur = await conn.execute(sql, args)
            rows = await cur.fetchall()
            await cur.close()
            for row in rows:
                content = (row[0] or "").strip()
                if not content:
                    continue
                # First line of the journal entry is typically the
                # title/headline — perfect summary length.
                first_line = content.splitlines()[0][:300] if content else ""
                candidates.append({
                    "when": str(row[1]),
                    "summary": first_line,
                    "source": "journal",
                })
    except Exception:
        log.warning("recall_journal_query_failed", exc_info=True)

    # Merge: sort by timestamp desc, dedupe by summary first ~60 chars
    # (avoids near-identical entries from delivered + journal paths),
    # cap at limit.
    candidates.sort(key=lambda c: c.get("when") or "", reverse=True)
    seen_fingerprints: set[str] = set()
    out: list[dict[str, str]] = []
    for c in candidates:
        fp = (c.get("summary") or "")[:60].strip().lower()
        if fp and fp in seen_fingerprints:
            continue
        if fp:
            seen_fingerprints.add(fp)
        out.append(c)
        if len(out) >= limit:
            break
    return out


_BRIEFING_SYNTHESIS_SYSTEM_PROMPT = (
    "You synthesize search-result briefings into genuinely useful "
    "answers. Given the user's briefing topics and the raw search "
    "results gathered for each one, produce an answer focused on what "
    "the briefing was actually asking. Lead with the most useful thing. "
    "Be specific and concrete — names, numbers, what changed, what to "
    "do — not vague summaries. Do not regurgitate snippet text "
    "verbatim; read across the hits and write the answer in your own "
    "words. EVERY topic the user configured gets its own section, in the order given — when a topic's results are thin or useless, say so in one short honest line ('nothing solid surfaced for X today') rather than omitting the section; a configured section silently vanishing reads as a malfunction.\n\n"
    "PRIOR BRIEFINGS — STRICT RULES:\n"
    "  * Prior briefings show what was reported THEN, not what is "
    "TRUE NOW. Treat them as narrative context for continuity and "
    "tone only.\n"
    "  * Specific numerical claims — prices, scores, temperatures, "
    "percentages, counts, status values — MUST come from the CURRENT "
    "search results. Never copy a number from a prior briefing into "
    "this one.\n"
    "  * NEVER claim a value 'is unchanged from' or 'matches' a "
    "prior briefing unless the CURRENT search results contain that "
    "exact value. Asserted continuity without a current source is "
    "fabrication.\n"
    "  * If the current results lack a fresh value for a time-"
    "sensitive topic, say the data wasn't available — DO NOT "
    "substitute the prior value as a fallback.\n"
    "  * Surface a delta ('up $3k since Tuesday') only when both the "
    "current AND prior values are present and the comparison is "
    "meaningful. Otherwise omit the comparison entirely.\n\n"
    "LOCATION CONTEXT: only reference the user's location when the "
    "topic is genuinely location-bound (weather, traffic, local "
    "events, local news, local services). For globally-uniform "
    "topics (crypto, programming, science, world news, tech "
    "industry, sports leagues) DO NOT mention the user's location "
    "— it makes the answer feel padded and reveals nothing.\n\n"
    "When image candidates are provided, choose ONE for "
    "hero_image_url if any genuinely illustrate the briefing's "
    "subject — otherwise set it to null. Don't pick an image just "
    "because one is available. When a video candidate is provided, "
    "include video_url + a 1-2 sentence video_transcript_summary "
    "that captures what the video actually adds beyond the text "
    "results. Skip both if they don't add information.\n\n"
    "OUTPUT FORMAT — STRICT JSON, no markdown fences, no preamble:\n"
    "{\n"
    '  "headline": "<= 140 chars, the single most useful thing the '
    'user should know. THIS IS THE PUSH NOTIFICATION BODY — write the '
    "answer, not metadata about how many topics were covered.\",\n"
    '  "body": "The useful answer (~80-350 words). One tight paragraph '
    "per topic that had useful results, each leading with its single "
    "most useful fact. Specific and scannable — no filler, no restating "
    'the question, no meta-commentary about the search.",\n'
    '  "hero_image_url": "<url from candidates>" or null,\n'
    '  "video_url": "<url from candidate>" or null,\n'
    '  "video_transcript_summary": "1-2 sentences distilling the '
    'video\'s key contribution" or null,\n'
    '  "citations": [{"title": "...", "url": "..."}]  (REQUIRED whenever '
    "any source was useful: the 3-5 best links for the user to read "
    "next, each with its real page title — this is the user's fast path "
    "into the underlying research)\n"
    "}\n\n"
    "If the results contain nothing useful, set headline to "
    "'<title> — no useful results', body to a one-line explanation, "
    "and all other fields to null."
)


# Read-through budget: top results per topic to fetch in full, and the hard
# cap across the whole briefing so a many-topic brief can't fan out into
# dozens of fetches. Reuses web_fetch (Wikipedia API / Reddit Atom / PDF /
# trafilatura), the same extractor the research tool reads pages with.
_BRIEFING_READ_TOP = 2
_BRIEFING_READ_MAX = 6
_BRIEFING_READ_CHARS = 2400
_BRIEFING_READ_CONCURRENCY = 4


async def _attach_read_through(
    app_state: Any, gathered: list[dict[str, Any]],
) -> int:
    """Deep-read the top result(s) of each gathered topic via web_fetch and
    attach the extracted text as ``item['excerpt']``, so synthesis works
    from real page content instead of search snippets.

    Best-effort + bounded: skipped entirely when research is disabled or
    web_fetch isn't registered; a per-page failure just leaves that item
    snippet-only. Fetches run concurrently (bounded) so the whole pass is
    one short wave, not N sequential reads. Returns pages read (for logging).
    """
    from augmentum.config import settings
    if not bool(getattr(settings, "companion_research_enabled", True)):
        return 0
    registry = getattr(app_state, "tool_registry", None) if app_state else None
    if registry is None:
        return 0
    fetch_tool = registry.resolve("web_fetch")
    if fetch_tool is None:
        return 0

    # Top N items per topic, capped overall, in gathered order.
    targets: list[dict[str, Any]] = []
    for g in gathered:
        picked = 0
        for item in g.get("items") or []:
            if picked >= _BRIEFING_READ_TOP or len(targets) >= _BRIEFING_READ_MAX:
                break
            url = (item.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            targets.append(item)
            picked += 1
        if len(targets) >= _BRIEFING_READ_MAX:
            break
    if not targets:
        return 0

    sem = asyncio.Semaphore(_BRIEFING_READ_CONCURRENCY)

    async def _read(item: dict[str, Any]) -> bool:
        async with sem:
            try:
                fr = await fetch_tool.execute(
                    url=item["url"], max_chars=_BRIEFING_READ_CHARS,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort depth
                log.debug(
                    "briefing_read_through_failed",
                    url=str(item.get("url"))[:80], error=str(exc)[:120],
                )
                return False
            if getattr(fr, "success", False) and getattr(fr, "output", ""):
                item["excerpt"] = fr.output.strip()[:_BRIEFING_READ_CHARS]
                return True
            return False

    results = await asyncio.gather(*[_read(t) for t in targets])
    return sum(1 for r in results if r)


def _build_synthesis_prompt(
    *,
    title: str,
    location: str,
    gathered: list[dict[str, Any]],
    prior_history: list[dict[str, str]] | None = None,
    image_candidates: list[dict[str, str]] | None = None,
    video_candidate: dict[str, str] | None = None,
) -> str:
    lines = [f"Briefing: {title}"]
    if location:
        lines.append(f"User location context: {location}")
    lines.append("")
    if prior_history:
        # Inject BEFORE the search results so the model sees the
        # continuity arc and can frame the new content as "what
        # changed" rather than restating context. The system prompt
        # already tells the model to only reference history when
        # materially informative.
        lines.append("Prior briefings on this topic (newest first):")
        for h in prior_history:
            when = (h.get("when") or "")[:16]  # YYYY-MM-DD HH:MM
            summary = h.get("summary") or ""
            lines.append(f"  [{when}] {summary}")
        lines.append("")
    if image_candidates:
        lines.append(
            "Image candidates (pick ONE for hero_image_url if any "
            "genuinely illustrate the subject; otherwise null):"
        )
        for img in image_candidates:
            url = img.get("url") or ""
            alt = (img.get("alt") or "").strip()
            src = img.get("source") or ""
            line = f"  - {url}"
            if alt:
                line += f"  ({alt[:120]})"
            if src:
                line += f" [{src}]"
            lines.append(line)
        lines.append("")
    if video_candidate:
        v = video_candidate
        lines.append("Video candidate available:")
        lines.append(f"  URL: {v.get('url') or ''}")
        if v.get("title"):
            lines.append(f"  Title: {v['title']}")
        if v.get("channel"):
            lines.append(f"  Channel: {v['channel']}")
        ts = (v.get("transcript_summary") or "").strip()
        if ts:
            lines.append(f"  Transcript excerpt: {ts}")
        lines.append("")
    lines.append("Topics covered (in order):")
    for g in gathered:
        topic = g.get("topic") or ""
        q = g.get("search_query") or ""
        if q and q != topic:
            lines.append(f"  • {topic} (queried as: {q})")
        else:
            lines.append(f"  • {topic}")
    lines.append("")
    lines.append("Search results:")
    for g in gathered:
        lines.append("")
        lines.append(f"# {g.get('topic') or ''}")
        items = g.get("items") or []
        if not items:
            lines.append("  (no results)")
            continue
        for item in items:
            t = (item.get("title") or "").strip()
            s = (item.get("snippet") or "").strip()
            u = (item.get("url") or "").strip()
            excerpt = (item.get("excerpt") or "").strip()
            line = f"- {t}" if t else "-"
            if s:
                # Trim snippets aggressively — the model doesn't need
                # the full 200-char block for synthesis, and short
                # snippets keep the context window manageable across
                # multi-topic briefings.
                line += f": {s[:140]}"
            if u:
                line += f" [{u}]"
            lines.append(line)
            # When we deep-read this page, give synthesis the real content
            # (indented under the result) instead of just the snippet — this
            # is what lifts a briefing from headline-skimming to actually
            # reading the sources. Bounded per item to keep the prompt sane.
            if excerpt:
                lines.append(f"    {excerpt[:1400]}")
    lines.append("")
    lines.append(
        "Produce the JSON now. Remember: the headline IS the push "
        "notification body — write the answer."
    )
    return "\n".join(lines)


def _parse_synthesis_response(text: str) -> dict[str, Any] | None:
    """Tolerate code fences and stray prose around the JSON."""
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    # Strip ```json fences (and bare ``` fences).
    if s.startswith("```"):
        # Drop the first fence line + any trailing fence.
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    # Try direct parse first.
    parsed: Any = None
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        # Fall back to finding the outermost { ... } block.
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None
    if not isinstance(parsed, dict):
        return None
    headline = str(parsed.get("headline") or "").strip()
    body = str(parsed.get("body") or "").strip()
    if not headline or not body:
        return None
    out: dict[str, Any] = {
        "headline": headline[:140],  # Web Push body convention
        "body": body,
    }
    # Optional multimodal fields — only include when the model wrote
    # something usable (non-empty string). Null / empty / missing all
    # collapse to "no media for this briefing."
    hero = parsed.get("hero_image_url")
    if isinstance(hero, str) and hero.strip().startswith(("http://", "https://")):
        out["hero_image_url"] = hero.strip()
    vurl = parsed.get("video_url")
    if isinstance(vurl, str) and vurl.strip().startswith(("http://", "https://")):
        out["video_url"] = vurl.strip()
    vts = parsed.get("video_transcript_summary")
    if isinstance(vts, str) and vts.strip():
        out["video_transcript_summary"] = vts.strip()[:600]
    cites = parsed.get("citations")
    if isinstance(cites, list):
        clean_cites: list[dict[str, str]] = []
        for c in cites[:8]:  # cap at 8
            if not isinstance(c, dict):
                continue
            ct = str(c.get("title") or "").strip()
            cu = str(c.get("url") or "").strip()
            if cu.startswith(("http://", "https://")):
                clean_cites.append({"title": ct[:200], "url": cu})
        if clean_cites:
            out["citations"] = clean_cites
    return out


def _resolve_synthesis_candidates() -> list[tuple[str, str]]:
    """Return (model_name, source_label) tuples in fallback order.

    Order:
      1. primary_chat_model — user's actual chat model. The companion
         "voice" the user is used to.
      2. utility_model — set when the operator wants a cheaper/faster
         model for internal tasks.
      3. classifier_model — last resort, usually smallest of the three.

    Empty model names are dropped. Duplicates (same model name across
    settings) are dropped so we don't retry the same call.
    """
    from augmentum.config import settings as _settings
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source_label, attr in (
        ("primary", "primary_chat_model"),
        ("utility", "utility_model"),
        ("classifier", "classifier_model"),
    ):
        name = (getattr(_settings, attr, "") or "").strip()
        if name and name not in seen:
            seen.add(name)
            candidates.append((name, source_label))
    return candidates


async def _call_briefing_synthesis(
    backend: Any, model: str, *, system: str, user: str,
) -> dict[str, Any] | None:
    """One synthesis attempt. Returns the parsed structured response
    (headline + body + optional multimedia fields) or None."""
    from augmentum.models.base import InternalChatRequest, Message
    req = InternalChatRequest(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=user),
        ],
        model=model,
        # Low-temperature factual synthesis. We're rewriting source text
        # into a focused answer; creativity is a regression vector here.
        temperature=0.3,
        max_tokens=700,
        # Synthesis rewrites source text into strict JSON — no chain-of-
        # thought needed. A reasoning model (Qwen 3.x / GLM-4.x) would
        # otherwise spend the 700-token budget thinking and time out before
        # emitting the JSON, silently dropping the briefing to the URL-only
        # fallback. No-op on non-reasoning models / backends that ignore it.
        chat_template_kwargs={"enable_thinking": False},
    )
    try:
        resp = await asyncio.wait_for(backend.chat(req), timeout=60.0)
    except TimeoutError:
        log.warning("briefing_synthesis_timeout", model=model)
        return None
    except Exception as exc:
        log.warning(
            "briefing_synthesis_call_failed",
            model=model, error=str(exc)[:200],
        )
        return None
    if resp is None or resp.message is None:
        return None
    raw = resp.message.content or ""
    parsed = _parse_synthesis_response(raw)
    if parsed is None:
        log.warning(
            "briefing_synthesis_parse_failed",
            model=model, raw_preview=raw[:200],
        )
    return parsed


async def _try_synthesize_briefing(
    runtime: CompanionRuntime,
    *,
    user_id: str,
    title: str,
    location: str,
    gathered: list[dict[str, Any]],
    topics: list[str],
    image_candidates: list[dict[str, str]] | None = None,
    video_candidate: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Walk the fallback chain. Returns (result, attempted_models).

    attempted_models is included so the URL-only fallback path can
    honestly tell the user which models failed.

    Pulls scoped prior-briefing context (delivered one-shots + recent
    journal fires matching current topics) into the synthesis prompt so
    the model can frame the new content as deltas/continuity rather
    than restating context. Safe by construction: only references the
    user's own prior BRIEFINGS, never random journal/affect/relational
    entries — so the "introduce friendly cat facts" failure mode can't
    happen.
    """
    app_state = getattr(runtime, "_app_state", None)
    registry = getattr(app_state, "provider_registry", None) if app_state else None
    if registry is None:
        log.warning("briefing_synthesis_no_provider_registry")
        return None, []

    candidates = _resolve_synthesis_candidates()
    if not candidates:
        log.info("briefing_synthesis_no_models_configured")
        return None, []

    # Recall is best-effort — query failures degrade to no-recall, not
    # synthesis failure. The current run is far more important than the
    # historical context.
    #
    # Layer B of the anti-anchor fix: when every topic is ephemeral
    # (price/score/weather/etc.) we structurally skip recall so the
    # model literally can't see a stale value to parrot. Mixed-topic
    # briefings still get recall — Layer A's strict system-prompt
    # rules cover the model's behavior there.
    prior_history: list[dict[str, str]] = []
    if _should_skip_recall(topics):
        log.info(
            "briefing_recall_skipped_ephemeral",
            topic_count=len(topics),
        )
    else:
        try:
            prior_history = await _recall_prior_briefing_context(
                runtime.backend.conn,
                user_id=user_id, companion_id=runtime.companion_id,
                topics=topics,
            )
        except Exception:
            log.warning("briefing_recall_failed", exc_info=True)

    system = _BRIEFING_SYNTHESIS_SYSTEM_PROMPT
    user = _build_synthesis_prompt(
        title=title, location=location, gathered=gathered,
        prior_history=prior_history,
        image_candidates=image_candidates,
        video_candidate=video_candidate,
    )
    if prior_history:
        log.info(
            "briefing_synthesis_with_recall",
            recall_entries=len(prior_history),
        )

    attempted: list[str] = []
    for model_name, source in candidates:
        attempted.append(f"{model_name} ({source})")
        try:
            backend, resolved_model = await registry.resolve_backend_with_fabric(
                model_name,
            )
        except Exception as exc:
            log.warning(
                "briefing_synthesis_resolve_failed",
                model=model_name, source=source, error=str(exc)[:200],
            )
            continue
        if backend is None or not resolved_model:
            continue
        result = await _call_briefing_synthesis(
            backend, resolved_model, system=system, user=user,
        )
        if result is not None:
            log.info(
                "briefing_synthesis_ok",
                model=resolved_model, source=source,
                headline_chars=len(result["headline"]),
                body_chars=len(result["body"]),
            )
            return result, attempted

    log.warning(
        "briefing_synthesis_all_models_failed",
        attempted=attempted,
    )
    return None, attempted


async def _utc_now_iso(conn: aiosqlite.Connection) -> str:
    """Return current UTC time as an ISO string using SQLite's clock.
    Using the DB's clock matches ``last_run_at = datetime('now')``
    semantics for delivered/last-run consistency on the same row."""
    cur = await conn.execute("SELECT datetime('now')")
    row = await cur.fetchone()
    await cur.close()
    return str(row[0]) if row and row[0] else ""


async def _refine_topic_queries(
    runtime: CompanionRuntime, *, title: str, topics: list[str],
    location: str,
) -> dict[str, str]:
    """LLM-write search queries for topics that came without one.

    Instruction-shaped topics ("surprise me", "teach me something") and
    bare category words ("news") search terribly verbatim - the
    Merriam-Webster-definition failure class. ``search_queries`` is the
    canonical fix when the ScheduleBriefingTool's LLM wrote the params,
    but UI-created briefings have no LLM in the loop, so this runs one
    cheap call at gather time for the topics still missing a query.
    Best-effort: any failure returns {} and the caller falls back to
    :func:`_clean_topic_for_query` exactly as before.
    """
    app_state = getattr(runtime, "_app_state", None)
    registry = getattr(app_state, "provider_registry", None) if app_state else None
    if registry is None or not topics:
        return {}
    from augmentum.models.base import InternalChatRequest, Message
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(topics))
    loc_line = f"User location: {location}\n" if location else ""
    user_prompt = (
        f"Briefing title: {title}\n{loc_line}"
        "For each numbered briefing topic below, write ONE short web search "
        "query that would surface fresh, substantive results for it today. "
        'Topics may be instructions ("surprise me") - infer what the '
        "person wants and write a real query (e.g. an interesting fact or "
        "story worth sharing today). Respond with ONLY a JSON object "
        "mapping the topic numbers to query strings, like "
        '{"1": "query", "2": "query"}.\n\n'
        f"{numbered}"
    )
    for model_name, _source in _resolve_synthesis_candidates():
        try:
            backend, resolved = await registry.resolve_backend_with_fabric(model_name)
        except Exception:
            continue
        if backend is None or not resolved:
            continue
        req = InternalChatRequest(
            messages=[Message(role="user", content=user_prompt)],
            model=resolved, temperature=0.2, max_tokens=250,
            chat_template_kwargs={"enable_thinking": False},
        )
        try:
            resp = await asyncio.wait_for(backend.chat(req), timeout=25.0)
        except Exception as exc:
            log.warning(
                "briefing_query_refine_failed",
                model=resolved, error=str(exc)[:150],
            )
            continue
        raw = (resp.message.content or "") if resp and resp.message else ""
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            parsed = json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            continue
        out: dict[str, str] = {}
        for i, topic in enumerate(topics):
            q = str(parsed.get(str(i + 1), "") or "").strip()
            if q:
                out[topic] = q[:200]
        if out:
            log.info("briefing_queries_refined", count=len(out))
            return out
    return {}


def _clean_topic_for_query(topic: str) -> str:
    """Strip imperative verbs / instruction-shaped phrasing from a topic
    so it works as a search query.

    The :class:`ScheduleBriefingTool` accepts free-form NL topics ("Check
    current Bitcoin price", "Tell me about post-quantum crypto"). Without
    cleanup these get submitted to SearXNG verbatim and the most
    distinctive word — "check", "tell" — dominates the results
    (Merriam-Webster definition wins for "check"). This helper strips
    the leading instruction so the actual subject is what's searched.

    Conservative: only strips well-known imperative openers, leaves
    everything else alone. A topic that was already a clean noun
    phrase passes through unchanged.
    """
    if not topic:
        return ""
    # Match leading verb (case-insensitive) followed by whitespace.
    # Captures the common "tell me about" / "let me know about" forms.
    import re
    # NOTE on alternation order: regex picks the FIRST matching branch,
    # not the longest. Longer phrases must come before their prefixes
    # ("find out about" before bare "find"; "look up" / "look into"
    # before bare "look"). Without this ordering, "find out about X"
    # strips only "find" and leaves "out about X" as the query.
    pattern = re.compile(
        r"^\s*(?:"
        r"find\s+out\s+about|find|"
        r"check(?:\s+on)?|"
        r"tell\s+me\s+about|tell\s+me|"
        r"look\s+up|look\s+into|"
        r"show\s+me|give\s+me|bring\s+me|"
        r"keep\s+me\s+(?:updated|posted)\s+(?:on|about)|"
        r"let\s+me\s+know\s+about|let\s+me\s+know|"
        r"alert\s+me\s+(?:about|to)|"
        r"update\s+me\s+(?:on|about)|update\s+me|"
        r"remind\s+me\s+about|"
        r"monitor|watch|track|follow|"
        r"get|fetch|grab|pull|"
        r"summari[sz]e"
        r")\s+",
        re.IGNORECASE,
    )
    cleaned = pattern.sub("", topic).strip()
    # Strip trailing punctuation that turns a query into a question.
    cleaned = cleaned.rstrip("?!.,;:")
    return cleaned or topic


def _short_host(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").removeprefix("www.")
        return host or url
    except Exception:
        return url


@_register_kind("verb_fire")
async def _kind_verb_fire(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a registry verb at a scheduled time. params:
      * ``verb`` (str, required) — canonical tool id (media.pause,
        weather.today, note.append, …)
      * ``verb_args`` (dict, optional) — arguments for the verb

    The unification of alarms, timers-with-actions, and "check X at
    5pm and tell me": anchored local_time / weekdays / one_shot all
    come free from the scheduler, the fire is restart-survivable
    (unlike asyncio timers), and the result's content becomes the
    notification body via the standing-task surfacing contract.

    Safety mirrors ``app.act`` and timer then-actions: only
    ``trivial_reversible`` verbs may fire unattended — checked at
    EVERY fire, not just at creation, so a stakes change in the
    registry takes effect immediately. surface_emit effects reach the
    screen through the headless bus-forwarding path (empty session_id).
    """
    verb = str(params.get("verb") or "").strip().lower()
    if not verb:
        raise ValueError("verb_fire requires params.verb")
    raw_args = params.get("verb_args") or {}
    verb_args = (
        {str(k): str(v) for k, v in raw_args.items()}
        if isinstance(raw_args, dict) else {}
    )

    from augmentum.intent.registry import DEFERRED_ACTION_STAKES, REGISTRY
    action = REGISTRY.get(verb)
    if action is None:
        return {
            "summary": f"Scheduled action failed — I don't have a tool called {verb}.",
            "noteworthy": True, "refs": [],
            "details": {"verb": verb, "ok": False, "reason": "unknown_verb"},
        }
    if getattr(action, "stakes", "") not in DEFERRED_ACTION_STAKES:
        return {
            "summary": (
                f"Scheduled action {verb} refused — it needs you "
                "present to run."
            ),
            "noteworthy": True, "refs": [],
            "details": {"verb": verb, "ok": False, "reason": "stakes"},
        }

    from augmentum.companion_runtime import tools as tool_bridge
    from augmentum.companion_runtime.tool_protocol import ToolCall
    call = ToolCall(
        kind="tool", name=verb, args=verb_args, raw="verb_fire", span=(0, 0),
    )
    result = await tool_bridge.execute_tool(call, runtime, user_id=user_id)

    content = ""
    if isinstance(result.payload, dict):
        content = str(result.payload.get("content") or "").strip()
    if result.ok:
        return {
            "summary": content or f"Done — ran {verb}.",
            "noteworthy": True, "refs": [],
            "details": {"verb": verb, "ok": True},
        }
    reason = result.error.message if result.error else "unknown error"
    return {
        "summary": f"Scheduled action {verb} failed: {reason[:160]}",
        "noteworthy": True, "refs": [],
        "details": {"verb": verb, "ok": False, "reason": reason[:200]},
    }


async def _gather_weather_for_briefing(
    runtime: CompanionRuntime, *, user_id: str, location: str = "",
) -> dict[str, str] | None:
    """Direct-sources weather block for a briefing section.

    Replaces the SearXNG round-trip for weather topics with typed
    Open-Meteo data: resolves the briefing's ``location`` param when
    given, otherwise the user's saved home (the blob weather.today
    learns). Appends any active Severe/Extreme NWS alert for US
    locations. Returns {"title", "snippet"} or None — best-effort,
    a miss just means the briefing stays text-only for that section.
    """
    try:
        from augmentum.intent.builtin.weather import _load_home
        from augmentum.sources import nws, open_meteo

        place: dict[str, Any] | None = None
        if location:
            place = await open_meteo.geocode(location)
        if place is None:
            app_state = getattr(runtime, "_app_state", None)
            store = getattr(app_state, "settings_store", None) if app_state else None
            if store is not None:
                place = await _load_home(store, user_id)
        if place is None:
            return None

        imperial = str(place.get("country_code") or "").upper() == "US"
        fc = await open_meteo.forecast(
            float(place.get("latitude") or 0.0),
            float(place.get("longitude") or 0.0),
            imperial=imperial,
        )
        if fc is None:
            return None
        summary = open_meteo.summarize(place, fc, imperial=imperial)

        alert_line = ""
        if imperial:
            alerts = await nws.active_alerts(
                float(place.get("latitude") or 0.0),
                float(place.get("longitude") or 0.0),
            )
            severe = [
                a for a in alerts if a["severity"] in nws.NOTIFY_SEVERITIES
            ]
            if severe:
                alert_line = f" ⚠ {severe[0]['headline'] or severe[0]['event']}"

        return {
            "title": summary["spoken"],
            "snippet": (
                f"Tomorrow: {summary['tomorrow'].get('high')} / "
                f"{summary['tomorrow'].get('low')}, "
                f"{summary['tomorrow'].get('condition')}.{alert_line}"
            ),
        }
    except Exception:  # noqa: BLE001 — gather is best-effort by contract
        log.warning("briefing_weather_gather_failed", exc_info=True)
        return None


@_register_kind("briefing")
async def _kind_briefing(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Generic multi-source digest. params:
      * ``topics`` (list[str], required) — each topic becomes one section
      * ``location`` (str, optional) — appended to every search query
      * ``max_per_topic`` (int, default 1) — top N results to surface per
        topic; useful when you want each section to have more body

    Compose-time logic deliberately avoids per-topic special-casing
    (no "if topic startswith 'weather' fetch open-meteo"). The substrate
    is general — anything searchable is briefable. Use it for a daily
    morning briefing, a Monday "team standup" digest, an evening "what
    happened in markets today" — all are the same shape: a list of
    topics + a time + a place.

    Briefings are always marked noteworthy when at least one topic
    returned a result, so the notifications hub publishes them as a
    push. Per the standing-tasks contract, surfacing also writes a
    journal entry that lands in the notes drawer.
    """
    from augmentum.companion_runtime import curator
    from augmentum.config import settings

    topics_raw = params.get("topics") or []
    if not isinstance(topics_raw, list) or not topics_raw:
        raise ValueError("briefing requires params.topics as a non-empty list")
    topics = [str(t).strip() for t in topics_raw if str(t).strip()]
    if not topics:
        raise ValueError("briefing requires at least one non-empty topic")
    location = (params.get("location") or "").strip()
    max_per_topic = max(1, min(int(params.get("max_per_topic", 1)), 5))

    # Per-topic search queries when the LLM provided them. Display label
    # still comes from ``topics``; only the SearXNG query is overridden.
    # Length must match topics; missing/extra entries fall back to the
    # cleaned topic. This is the canonical fix for the
    # "Check current Bitcoin price" → Merriam-Webster definition case:
    # the LLM, which understands intent, can write
    # "Bitcoin price USD last 24 hours" while the section header stays
    # human-readable.
    raw_queries = params.get("search_queries") or []
    if not isinstance(raw_queries, list):
        raw_queries = []
    search_queries = [str(q).strip() for q in raw_queries]
    while len(search_queries) < len(topics):
        search_queries.append("")

    # Topics still missing a query get an LLM-written one (best-effort;
    # see _refine_topic_queries). Fixes the "surprise me" -> dictionary
    # definition class for UI-created briefings.
    _missing = [topics[i] for i in range(len(topics)) if not search_queries[i]]
    if _missing:
        _refined = await _refine_topic_queries(
            runtime, title=str(params.get("title") or "Briefing"),
            topics=_missing, location=location,
        )
        for i, t in enumerate(topics):
            if not search_queries[i] and _refined.get(t):
                search_queries[i] = _refined[t]

    http_client = curator._resolve_http_client(runtime)
    if http_client is None:
        return {"summary": "no http_client; deferred", "noteworthy": False, "refs": []}

    searxng_base = getattr(settings, "searxng_base_url", "")
    if not searxng_base:
        return {"summary": "no searxng configured", "noteworthy": False, "refs": []}

    # Phase 1: gather raw structured results per topic (no formatting yet).
    refs: list[dict] = []
    gathered: list[dict[str, Any]] = []
    title_line = params.get("title") or "Briefing"

    for idx, topic in enumerate(topics):
        # Query precedence:
        # 1. Explicit search_queries[idx] when LLM provided it — used
        #    verbatim. Location NOT auto-appended (LLM already decided
        #    per-query whether locality matters).
        # 2. Otherwise: clean the topic of imperative verbs, then
        #    append location if any.
        explicit_query = search_queries[idx] if idx < len(search_queries) else ""
        if explicit_query:
            query = explicit_query
        else:
            cleaned = _clean_topic_for_query(topic)
            query = f"{cleaned} {location}".strip() if location else cleaned

        items: list[dict] = []
        # News-shaped topics get the news category + day freshness -
        # general search for "top news today" returns outlet HOMEPAGES
        # (the CNN-front-page case), not articles. Search-quality
        # tuning, not topic special-casing: the gather stays generic.
        _newsish = (
            "news" in topic.lower()
            or "news" in query.lower()
            or "headline" in query.lower()
        )
        search_params: dict[str, str] = {"q": query, "format": "json"}
        if _newsish:
            search_params["categories"] = "news"
            search_params["time_range"] = "day"
        try:
            resp = await http_client.get(
                f"{searxng_base.rstrip('/')}/search",
                params=search_params,
                timeout=12.0,
                headers={"User-Agent": "Augmentum/1.0 (standing-tasks)"},
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("results") or []
        except Exception as exc:
            log.warning(
                "briefing_topic_query_failed",
                topic=topic[:60], error=str(exc)[:200],
            )

        picked: list[dict[str, str]] = []
        for r in items:
            if len(picked) >= max_per_topic:
                break
            url = (r.get("url") or "").strip()
            t = (r.get("title") or "").strip()
            snippet = (r.get("content") or "").strip()[:200]
            if not t and not snippet:
                continue
            picked.append({"title": t, "snippet": snippet, "url": url})
            if url:
                refs.append({
                    "kind": "url", "url": url,
                    "id": curator._url_hash(url),
                })

        gathered.append({
            "topic": topic, "search_query": query, "items": picked,
        })

    # Direct-sources weather gather (declared like image_search /
    # youtube — a gather TOOL, not a topic special-case). Typed
    # Open-Meteo data + active alerts lead the briefing instead of
    # whatever SearXNG scraped about "weather". Inserted FIRST so the
    # synthesis sees conditions before the news.
    _gt_early = params.get("gather_tools") or []
    if isinstance(_gt_early, list) and "weather" in {
        str(t).strip().lower() for t in _gt_early
    }:
        wx = await _gather_weather_for_briefing(
            runtime, user_id=user_id, location=location,
        )
        if wx is not None:
            gathered.insert(0, {
                "topic": "Weather", "search_query": "",
                "items": [{
                    "title": wx["title"], "snippet": wx["snippet"], "url": "",
                }],
            })

    sections_with_content = sum(1 for g in gathered if g["items"])
    if sections_with_content == 0:
        return {
            "summary": f"{title_line}: no results across {len(topics)} topic(s)",
            "noteworthy": False, "refs": [],
        }

    # Phase 1b: optional multimodal gather based on per-preset
    # ``gather_tools`` declaration. Presets like the recipe one declare
    # ``['searxng', 'image_search', 'youtube']``; text-only presets stay
    # ``['searxng']`` (or omit the field entirely, same result). Each
    # tool gather is best-effort — a SearXNG image timeout doesn't fail
    # the briefing, just falls back to text-only synthesis.
    gather_tools_raw = params.get("gather_tools") or []
    if not isinstance(gather_tools_raw, list):
        gather_tools_raw = []
    gather_tools = {str(t).strip().lower() for t in gather_tools_raw}

    image_candidates: list[dict[str, str]] = []
    video_candidate: dict[str, str] | None = None
    # Image/video query: a real gathered headline is a far better image
    # subject than the briefing's own title ("Evening briefing Rochester"
    # finds nothing; the top story's title finds the story's imagery).
    _top_headline = next(
        (g["items"][0]["title"] for g in gathered
         if g["items"] and g["items"][0].get("title")),
        "",
    )
    media_query = _top_headline or (
        f"{title_line} {location}".strip() if location else title_line
    )
    if "image_search" in gather_tools:
        image_candidates = await _gather_image_candidates_for_briefing(
            runtime, query=media_query, count=3, user_id=user_id,
        )
    if "youtube" in gather_tools:
        video_candidate = await _gather_video_candidate_for_briefing(
            runtime, query=media_query,
        )

    # Phase 1c: deep-read the top results so synthesis works from real page
    # content, not 140-char search snippets. Reuses web_fetch (platform-aware
    # extraction); bounded + best-effort; gated by companion_research_enabled.
    read_pages = await _attach_read_through(
        getattr(runtime, "_app_state", None), gathered,
    )
    if read_pages:
        log.info(
            "briefing_read_through", pages=read_pages, topics=len(gathered),
        )

    # Phase 2: synthesize via LLM. The substrate's job is no longer to
    # mechanically paste search hits — it's to read across them and
    # produce an actual answer to what the briefing was asking. The
    # synthesized headline becomes the OS push notification body; the
    # synthesized response becomes the journal entry.
    synthesis, attempted = await _try_synthesize_briefing(
        runtime,
        user_id=user_id,
        title=title_line, location=location,
        gathered=gathered, topics=topics,
        image_candidates=image_candidates or None,
        video_candidate=video_candidate,
    )

    if synthesis is not None:
        # Multimedia fields land in details so _surface_result and
        # notification payload can carry them downstream. The summary
        # field stays headline-only (OS push body convention).
        details: dict[str, Any] = {"content": synthesis["body"]}
        if synthesis.get("hero_image_url"):
            details["hero_image_url"] = synthesis["hero_image_url"]
        if synthesis.get("video_url"):
            details["video_url"] = synthesis["video_url"]
        if synthesis.get("video_transcript_summary"):
            details["video_transcript_summary"] = (
                synthesis["video_transcript_summary"]
            )
        if synthesis.get("citations"):
            details["citations"] = synthesis["citations"]
        return {
            "summary": synthesis["headline"],
            "noteworthy": True,
            "refs": refs,
            "details": details,
        }

    # Synthesis fallback: honestly tell the user the synthesis didn't
    # work, then list the gathered sources as title + URL pairs so they
    # can investigate. No snippet regex truncation — just the cleanest
    # possible "here's what was found, you decide."
    fallback_lines = [
        f"{title_line}: synthesis unavailable — sources below.",
    ]
    if attempted:
        fallback_lines.append(
            f"(Tried: {', '.join(attempted)}. None produced a usable "
            "answer.)"
        )
    fallback_lines.append("")
    for g in gathered:
        fallback_lines.append(f"— {g['topic']}")
        if not g["items"]:
            fallback_lines.append("  (no results)")
            continue
        for item in g["items"]:
            t = item.get("title") or "(no title)"
            u = item.get("url") or ""
            if u:
                fallback_lines.append(f"  • {t}\n    {u}")
            else:
                fallback_lines.append(f"  • {t}")
    total_sources = sum(len(g["items"]) for g in gathered)
    fallback_summary = (
        f"{title_line}: synthesis unavailable, {total_sources} "
        f"source{'s' if total_sources != 1 else ''} ready"
    )

    # Preserve gathered media on the fallback path so the modal still
    # has something rich to show even when no model could synthesize.
    # The image is whatever the first candidate was; same for video.
    fallback_details: dict[str, Any] = {"content": "\n".join(fallback_lines)}
    if image_candidates:
        fallback_details["hero_image_url"] = image_candidates[0]["url"]
    if video_candidate and video_candidate.get("url"):
        fallback_details["video_url"] = video_candidate["url"]
        if video_candidate.get("transcript_summary"):
            fallback_details["video_transcript_summary"] = (
                video_candidate["transcript_summary"]
            )

    return {
        "summary": fallback_summary,
        "noteworthy": True,
        "refs": refs,
        "details": fallback_details,
    }


@_register_kind("metric_watch")
async def _kind_metric_watch(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Track a number from a structured provider — no scraping, no
    extraction ladder, no judge. params = {"metric": {"provider":
    "open_meteo", "location": str}, "condition": {op, value, unit}?}.

    v1 provider: open_meteo (shipped). The reading runs through the same
    quarantine/confirm state machine as page-extracted numbers, though a
    typed provider value mostly exists to catch provider glitches.
    """
    from augmentum.companion_runtime import metrics as _metrics
    from augmentum.config import settings

    metric = params.get("metric") or {}
    if isinstance(metric, str):
        # The watch tool stores the raw target string for kind=metric;
        # treat it as an open_meteo location ("temperature in Portland").
        metric = {"provider": "open_meteo", "location": metric}
    provider = str(metric.get("provider") or "open_meteo").strip().lower()
    if provider != "open_meteo":
        raise ValueError(f"metric_watch: unknown provider {provider}")

    location = str(metric.get("location") or "").strip()
    if not location:
        raise ValueError("metric_watch requires metric.location")

    from augmentum.sources import open_meteo

    # Geocode once, cache lat/lon in params — the WATCH's place
    # shouldn't drift with the user; they named it at creation.
    lat = metric.get("_lat")
    lon = metric.get("_lon")
    if lat is None or lon is None:
        place = await open_meteo.geocode(location)
        if not place:
            raise ValueError(f"couldn't find a place called {location!r}")
        lat, lon = place["latitude"], place["longitude"]
        metric["_lat"], metric["_lon"] = lat, lon
    params["metric"] = metric

    imperial = bool(metric.get("imperial", True))
    fc = await open_meteo.forecast(lat, lon, imperial=imperial)
    current = (fc or {}).get("current") or {}
    raw = current.get("temperature_2m")
    value = float(raw) if isinstance(raw, int | float) else None
    unit = "F" if imperial else "C"

    condition = params.get("condition")
    verdict = _metrics.classify_reading(
        value,
        state=params.get("metric_state"),
        condition=condition if isinstance(condition, dict) else None,
        quarantine_pct=float(getattr(
            settings, "companion_metric_quarantine_pct", 60.0) or 60.0),
        confirm_readings=int(getattr(
            settings, "companion_metric_confirm_readings", 2) or 2),
    )
    params["metric_state"] = verdict.state

    observation = {
        "series": "temperature",
        "value": _metrics.to_scaled_int(value) if value is not None else None,
        "scale": 100,
        "unit": unit,
        "method": "provider",
        "status": verdict.status,
        "evidence": f"open_meteo current.temperature_2m for {location}",
    }
    fired_line = ""
    if verdict.fire and value is not None and isinstance(condition, dict):
        fired_line = (
            f"{location}: {value:g}°{unit} — condition "
            f"{condition.get('op')} {condition.get('value'):g} met"
        )
    return {
        "summary": fired_line or f"{location}: {verdict.note}",
        "noteworthy": bool(verdict.fire),
        "refs": [],
        "details": {
            "content": fired_line or "",
            "observation": observation,
            "params_update": dict(params),
        },
    }


@_register_kind("prompt_fire")
async def _kind_prompt_fire(
    runtime: CompanionRuntime, *, user_id: str, params: dict[str, Any],
) -> dict[str, Any]:
    """Run a saved request through the shared FC loop at fire time.

    The stakes carve-out (spec §6.2): only user-created rows exist
    (schedule_request refuses without a user), the INFERENCE is the only
    thing exempted from DEFERRED_ACTION_STAKES — actions inside the run
    stay gated by the loop's own dispatch — and surface events are left
    parked (drain_surface_events=False, never delivered): a deferred
    request gathers and reports; it never touches the user's screen.

    Groundedness (§6.1b): a run with zero data-gathering tool calls is
    answered from priors, not the world — retried once with an explicit
    gather-first instruction, then delivered visibly labeled.
    """
    from augmentum.config import settings

    prompt = str(params.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt_fire requires params.prompt")

    app_state = getattr(runtime, "_app_state", None)
    registry = getattr(app_state, "tool_registry", None) if app_state else None
    if registry is None:
        return {"summary": "no tool registry; deferred",
                "noteworthy": False, "refs": []}

    max_calls = int(getattr(
        settings, "companion_prompt_fire_max_tool_calls", 6) or 6)
    max_seconds = float(getattr(
        settings, "companion_prompt_fire_max_seconds", 120.0) or 120.0)

    async def _one_pass(user_text: str) -> tuple[str, int, list[dict]]:
        """Drive the loop once. Returns (final_text, tool_calls, trace)."""
        from augmentum.companion_runtime import tiers
        from augmentum.companion_runtime.native_loop import native_loop_events
        from augmentum.companion_runtime.runtime import Intent
        from augmentum.models.base import InternalChatRequest, Message

        backend, model_name = await tiers.primary(runtime)
        system_text = (
            "You are completing a request the user scheduled earlier — "
            "they are not present and cannot clarify. Gather what you "
            "need with your tools, then answer in plain words. Be "
            "complete but brief; your final text is delivered as a "
            "notification.\n\n"
            "Gathering rules:\n"
            "- Prefer the `research` tool for any factual lookup — it "
            "tries multiple queries and reads sources for you. For an "
            "intricate request, pass a few `alt_queries` (different angles "
            "or the sub-parts of the request).\n"
            "- If a search or tool comes back empty or off-target, do NOT "
            "give up or answer from memory — try a different phrasing, a "
            "broader or narrower query, or a different tool/resource.\n"
            "- If after a genuine effort the information simply isn't "
            "available, say so plainly (what you looked for and couldn't "
            "find). An honest 'I couldn't confirm X' beats a confident "
            "guess.\n"
            "- When you cite a source, end your answer with a short "
            "'Sources:' list giving the actual URLs — don't leave bare "
            "[1]/[2] markers with no links."
        )
        req = InternalChatRequest(
            model=model_name,
            messages=[
                Message(role="system", content=system_text),
                Message(role="user", content=user_text),
            ],
            stream=False,
            # Deferred answers are delivered whole as a notification/note;
            # without an explicit ceiling the backend's small default cut
            # multi-section answers off mid-sentence ("**Key context:**" …).
            max_tokens=1024,
        )
        intent = Intent(text=user_text, user_id=user_id, source="tick")
        cancel = asyncio.Event()
        final_parts: list[str] = []
        trace: list[dict] = []
        source_urls: list[str] = []
        calls = 0

        async def _consume() -> None:
            nonlocal calls
            async for kind, payload in native_loop_events(
                req,
                backend=backend,
                runtime=runtime,
                intent=intent,
                registry=registry,
                user_id=user_id,
                session_id=f"prompt_fire:{user_id}",
                app_state=app_state,
                cancel=cancel,
                drain_surface_events=False,
            ):
                if kind == "tool_call":
                    calls += 1
                    trace.append({
                        "tool": str(payload.get("name") or "")[:60],
                    })
                    if calls >= max_calls:
                        cancel.set()  # budget: loop stops between hops
                elif kind == "tool_result":
                    if trace:
                        trace[-1]["ok"] = bool(payload.get("ok", True))
                    for u in (payload.get("urls") or []):
                        if u and u not in source_urls:
                            source_urls.append(u)
                elif kind == "text":
                    final_parts.append(str(payload.get("text") or ""))

        await asyncio.wait_for(_consume(), timeout=max_seconds)
        return "".join(final_parts).strip(), calls, trace, list(source_urls)

    try:
        text, calls, trace, urls = await _one_pass(prompt)
    except TimeoutError:
        return {
            "summary": "ran out of time before finishing — try a "
                       "narrower request",
            "noteworthy": True, "refs": [],
            "details": {"tool_trace": [], "timed_out": True},
        }

    ungrounded = False
    if calls == 0 and text:
        # Zero-gather: the answer came from model priors. One retry with
        # an explicit instruction; if it still won't gather, deliver
        # honestly labeled rather than passing priors off as a check.
        try:
            retry_text, retry_calls, retry_trace, retry_urls = await _one_pass(
                "You must gather current information with your tools "
                "BEFORE answering — do not answer from memory.\n\n" + prompt,
            )
            if retry_calls > 0 and retry_text:
                text, calls, trace, urls = (
                    retry_text, retry_calls, retry_trace, retry_urls,
                )
            else:
                ungrounded = True
        except TimeoutError:
            ungrounded = True
    if not text:
        return {
            "summary": "couldn't produce an answer this time",
            "noteworthy": False, "refs": [],
            "details": {"tool_trace": trace},
        }
    if ungrounded:
        text = "(from memory, not checked:) " + text

    # Attach the source URLs the run actually used as content refs so the
    # delivered note renders clickable sources instead of dangling [1]/[3].
    refs = [
        {"kind": "url", "url": u,
         "id": hashlib.sha256(u.encode("utf-8")).hexdigest()[:16]}
        for u in urls[:6]
    ]

    return {
        "summary": text[:300],
        "noteworthy": True,  # the user explicitly asked for this output
        "refs": refs,
        "details": {
            "content": text,
            "tool_trace": trace[:12],
            "ungrounded": ungrounded,
        },
    }


# ─── Step ────────────────────────────────────────────────────────────────


async def step(
    runtime: CompanionRuntime,
    *,
    user_id: str | None = None,
    respect_presence_gate: bool = True,
) -> int | None:
    """Pick ONE due task for ``user_id`` and run it. Returns the run
    task id, or None when nothing was due.

    ``user_id`` defaults to the runtime's bound owner (the companion
    tick-verb path). The app-level SchedulerService passes each user
    explicitly — scheduling is a platform substrate, not an owner-only
    companion behavior.

    ``respect_presence_gate``: the companion's presence/silent mode
    gates her AUTONOMY — initiative, drift, curator chatter. An
    explicitly user-created schedule ("wake me at 9") is the user's
    own ask, not initiative, so the generalized dispatcher passes
    False; the companion tick path keeps the historical gate.
    """
    from augmentum.config import settings

    if not getattr(settings, "companion_standing_tasks_enabled", True):
        return None

    user_id = user_id or getattr(runtime, "owner_user_id", "") or ""
    if not user_id:
        return None

    if respect_presence_gate:
        from augmentum.companion_runtime import presence_mode as _pm
        if not _pm.autonomy_allowed():
            return None

    backend = runtime.backend
    conn = backend.conn

    # Pick the next due task. next_run_at NULL → run immediately (just
    # created). enabled=1 only.
    try:
        cur = await conn.execute(
            """SELECT id, user_id, companion_id, title, kind, params,
                      interval_seconds, last_run_at, next_run_at,
                      last_result_summary, last_error,
                  enabled, consecutive_error_count,
                  consecutive_budget_timeout_count
               FROM companion_standing_tasks
               WHERE user_id = ? AND companion_id = ?
                 AND enabled = 1
                 AND (next_run_at IS NULL OR next_run_at <= datetime('now'))
               ORDER BY (next_run_at IS NULL) DESC, next_run_at ASC
               LIMIT 1""",
            (user_id, runtime.companion_id),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("standing_tasks_query_failed", exc_info=True)
        return None

    if row is None:
        return None

    task = _row_to_task(row)
    user_tz = await _resolve_user_timezone(
        getattr(runtime, "_app_state", None), user_id,
    )
    runner = _TASK_KINDS.get(task.kind)
    if runner is None:
        await _persist_run(
            conn, task_id=task.id,
            interval_seconds=task.interval_seconds,
            result=None, error=f"unknown kind: {task.kind}",
            consecutive_error_count=task.consecutive_error_count + 1,
            auto_pause=(task.consecutive_error_count + 1) >= _MAX_CONSECUTIVE_ERRORS,
            params=task.params, user_timezone=user_tz,
        )
        await _record_run(
            conn, task_id=task.id, user_id=user_id,
            status="error", summary=f"unknown kind: {task.kind}",
        )
        return task.id

    started = time.monotonic()
    try:
        result = await runner(runtime, user_id=user_id, params=task.params)
    except asyncio.CancelledError:
        # The verb's asyncio.wait_for cut us off. Without this handler
        # the persist+next_run_at update is skipped, the task remains
        # "due now", and the next tick re-picks it — five repicks trip
        # the verb's auto-pause (2026-06-08 incident: a briefing task
        # over the 5s wallclock paused tick_scheduler for ~7 hours).
        # Shield the persist so it survives the outer cancellation,
        # then re-raise so the verb still records BUDGET_EXCEEDED.
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.warning(
            "standing_task_runner_cancelled",
            task_id=task.id, kind=task.kind, elapsed_ms=elapsed_ms,
        )
        # Budget timeout ≠ error: increment the SEPARATE budget counter
        # (looser threshold), leave consecutive_error_count untouched, so
        # a merely-slow task isn't auto-paused like a broken one (audit
        # 2026-06-17 — the 7-hour-pause incident root).
        next_budget = task.consecutive_budget_timeout_count + 1
        try:
            await asyncio.shield(_persist_run(
                conn, task_id=task.id,
                interval_seconds=task.interval_seconds,
                result=None, error="cancelled (wallclock exceeded)",
                consecutive_error_count=task.consecutive_error_count,
                consecutive_budget_timeout_count=next_budget,
                auto_pause=next_budget >= _MAX_CONSECUTIVE_BUDGET_TIMEOUTS,
                params=task.params, user_timezone=user_tz,
            ))
        except asyncio.CancelledError:
            # Outer cancellation propagated through the shield wrapper;
            # the shielded persist still completes on the loop.
            pass
        try:
            await asyncio.shield(_record_run(
                conn, task_id=task.id, user_id=user_id,
                status="error", summary="cancelled (wallclock exceeded)",
                details={"elapsed_ms": elapsed_ms},
            ))
        except asyncio.CancelledError:
            pass
        raise
    except Exception as exc:
        log.warning(
            "standing_task_runner_failed",
            task_id=task.id, kind=task.kind, error=str(exc)[:200],
        )
        next_errs = task.consecutive_error_count + 1
        await _persist_run(
            conn, task_id=task.id,
            interval_seconds=task.interval_seconds,
            result=None, error=str(exc)[:200],
            consecutive_error_count=next_errs,
            auto_pause=next_errs >= _MAX_CONSECUTIVE_ERRORS,
            params=task.params, user_timezone=user_tz,
        )
        await _record_run(
            conn, task_id=task.id, user_id=user_id,
            status="error", summary=str(exc)[:200],
        )
        return task.id

    elapsed = int((time.monotonic() - started) * 1000)
    log.info(
        "standing_task_ran",
        task_id=task.id, kind=task.kind, ms=elapsed,
        noteworthy=bool(result.get("noteworthy")),
    )

    # If the runner mutated params (last_seen baselines), persist them.
    params_update = (result.get("details") or {}).get("params_update")
    effective_params = params_update if params_update is not None else task.params

    # Numeric watches: append the reading to the observation series —
    # including quarantined/missing readings; the audit trail is the
    # point. Baselines persist regardless of what the judge says below.
    observation = (result.get("details") or {}).get("observation")
    if isinstance(observation, dict):
        await _record_observation(
            conn, task_id=task.id, user_id=user_id, observation=observation,
        )

    # Importance judge (intent-carrying watches only; prompt_fire output
    # is exempt — the user explicitly asked for it). Downgrade-only and
    # fail-open: a broken judge delivers, a verified "not important"
    # suppresses-but-logs. Baseline params already advanced above, so a
    # suppressed change can't re-trigger every fire.
    verdict_dict: dict[str, Any] | None = None
    suppressed = False
    intent_text = str((task.params or {}).get("intent") or "").strip()
    if result.get("noteworthy") and intent_text and task.kind != "prompt_fire":
        from augmentum.companion_runtime import watch_judge
        diff_content = (
            (result.get("details") or {}).get("content")
            or result.get("summary") or ""
        )
        verdict = await watch_judge.judge_change(
            runtime, intent=intent_text, diff_content=diff_content,
        )
        verdict_dict = verdict.as_dict()
        suppressed = not verdict.important

    # Surface FIRST so the journal entry + push notification go out
    # regardless of whether the row is about to be deleted (one-shot)
    # or updated (recurring). The user always gets the fire output.
    if result.get("noteworthy") and not suppressed:
        await _surface_result(runtime, task=task, result=result)

    run_details: dict[str, Any] = {"elapsed_ms": elapsed}
    if verdict_dict is not None:
        run_details["judge"] = verdict_dict
    # The delivered body rides the run row so History shows the actual
    # briefing, not just the one-line headline (2026-07-07 - the runs
    # table always had a details column; no caller ever filled it).
    for body_key in ("content", "hero_image_url", "video_url", "citations"):
        val = (result.get("details") or {}).get(body_key)
        if val:
            run_details[body_key] = val
    for extra_key in ("tool_trace", "ungrounded", "timed_out"):
        extra = (result.get("details") or {}).get(extra_key)
        if extra:
            run_details[extra_key] = extra
    await _record_run(
        conn, task_id=task.id, user_id=user_id,
        status=(
            "suppressed" if suppressed
            else "fired" if result.get("noteworthy") else "silent"
        ),
        summary=result.get("summary") or "",
        details=run_details,
    )

    # One-shot: mark as delivered rather than delete. The row sticks
    # around as a historical artifact the model can recall when the
    # user wants to dig into the findings, re-instate the briefing
    # for a different time, or convert it to recurring. The dup-check
    # filters out delivered rows so a future one-shot on the same
    # topic won't get blocked.
    if bool(effective_params.get("one_shot")):
        delivered_params = dict(effective_params)
        delivered_params["delivered_at"] = (
            await _utc_now_iso(conn)
        )
        delivered_params["delivery_summary"] = (
            (result.get("summary") or "")[:300]
        )
        try:
            await conn.execute(
                """UPDATE companion_standing_tasks
                   SET enabled = 0,
                       last_run_at = datetime('now'),
                       last_result_summary = COALESCE(?, last_result_summary),
                       params = ?
                   WHERE id = ?""",
                (
                    result.get("summary"),
                    json.dumps(delivered_params),
                    task.id,
                ),
            )
            await conn.commit()
        except Exception:
            log.warning("standing_task_one_shot_persist_failed", exc_info=True)
        log.info("standing_task_one_shot_delivered", task_id=task.id)
        return task.id

    # Recurring path: advance the schedule AND persist any param-cursor
    # mutation in ONE atomic UPDATE (via persist_params) so a failure
    # can't lose the cursor and re-fire the same diff (audit 2026-06-17).
    # Success resets both the error and budget-timeout counters.
    await _persist_run(
        conn, task_id=task.id,
        interval_seconds=task.interval_seconds,
        result=result, error=None,
        consecutive_error_count=0,
        consecutive_budget_timeout_count=0,
        params=effective_params, user_timezone=user_tz,
        persist_params=params_update,
    )

    return task.id


# Kinds whose whole point is an actively-scheduled heads-up: the user
# asked to be told at a time, so they must reach every device (the hub's
# always-push path fires at IMPORTANCE_HIGH, even with a tab open).
# Passive watches (feed polls, release/url/metric diffs) stay at DEFAULT:
# they chime in-app when a tab is open and only Web Push to a device when
# none is — so a feed poll doesn't buzz the phone on your desk.
_ACTIVE_DELIVERY_KINDS = frozenset({
    "briefing", "prompt_fire", "verb_fire", "reminder", "deadline",
})


def _surface_importance(task: StandingTask) -> int:
    """Resolve the notification importance for a fired standing task.

    The user's explicit per-task choice wins: ``params.delivery`` is
    ``"alert"`` (push every device, even with a tab open) or ``"quiet"``
    (in-app chime; Web Push only when away). Absent a choice, the kind
    default applies — active-delivery kinds (briefings, reminders,
    deferred prompts/verbs) are HIGH, passive watches DEFAULT. Never
    auto-decide against a choice the user made.
    """

    from augmentum.notifications.catalog import (
        IMPORTANCE_DEFAULT,
        IMPORTANCE_HIGH,
    )

    pref = str((task.params or {}).get("delivery") or "").strip().lower()
    if pref == "alert":
        return IMPORTANCE_HIGH
    if pref == "quiet":
        return IMPORTANCE_DEFAULT

    kind = (getattr(task, "kind", "") or "").strip().lower()
    if kind in _ACTIVE_DELIVERY_KINDS:
        return IMPORTANCE_HIGH
    return IMPORTANCE_DEFAULT


def _media_refs_from_details(details: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build image/video content-refs from a runner's ``details`` payload.

    Briefings gather a hero image + a YouTube video (with transcript
    summary) and the synthesis LLM picks among them — but historically
    only the url refs reached the journal note, so every image and video
    was gathered, reasoned over, then dropped before the user ever saw
    it. These ride through the SAME ``content_refs`` channel as url refs:
    the journal validator tolerates unknown kinds by design
    (``validators.refs_exist_for_user`` — "future ref types add their
    kind"), and the notes drawer renders them. Kinds: ``image`` (hero,
    may be a local artifact path) and ``video`` (external link + summary).
    """
    d = details or {}
    out: list[dict[str, Any]] = []
    hero = str(d.get("hero_image_url") or "").strip()
    # Local artifacts arrive as "/..." embed paths; remote as http(s).
    if hero.startswith(("http://", "https://", "/")):
        out.append({
            "kind": "image",
            "url": hero,
            "id": hashlib.sha256(hero.encode("utf-8")).hexdigest()[:16],
        })
    video = str(d.get("video_url") or "").strip()
    if video.startswith(("http://", "https://")):
        out.append({
            "kind": "video",
            "url": video,
            "summary": str(d.get("video_transcript_summary") or "")[:300],
            "id": hashlib.sha256(video.encode("utf-8")).hexdigest()[:16],
        })
    return out


def _citation_refs_from_details(details: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Build titled ``citation`` content-refs from a runner's ``details``.

    The briefing synthesis returns ``citations`` — the model-curated top
    3-5 sources the answer actually rests on, each with a real page title.
    These are the user's fast path into the underlying research, but
    historically only the raw gathered url refs (domain-only) reached the
    note while the titled citations were dropped. The drawer prefers these
    over bare url refs. Deduped by url; unknown kind is tolerated by the
    journal validator (same channel as image/video refs).
    """
    d = details or {}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in d.get("citations") or []:
        if not isinstance(c, dict):
            continue
        url = str(c.get("url") or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        out.append({
            "kind": "citation",
            "url": url,
            "title": str(c.get("title") or "").strip()[:200],
            "id": hashlib.sha256(url.encode("utf-8")).hexdigest()[:16],
        })
    return out


# Hosts whose links are best opened as in-app *playback* (the client routes
# these to the media player) rather than a generic new-tab link.
_VIDEO_HOSTS: frozenset[str] = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
    "vimeo.com", "www.vimeo.com",
})


def _open_target_from_refs(
    refs: list[dict[str, Any]] | None,
) -> dict[str, str] | None:
    """Pick the primary 'open me' target for a briefing notification.

    A media briefing ("here's a video to wind down to") should let the user
    tap the notification and land ON the thing — not in the notes drawer to
    *manage* the briefing. We surface the synthesis's primary link as
    ``open_url`` on the notification payload; the client deep-links it
    (in-app player for video hosts, new tab otherwise).

    Order mirrors how ``_surface_result`` assembles refs: the runner's own
    ``refs`` (the recommendation the body is about) first, then the gathered
    ``video``, then titled ``citation`` links. First http(s) url wins.
    Returns ``None`` for text-only briefings, leaving Open → drawer intact.
    """
    from urllib.parse import urlparse

    for r in refs or []:
        if not isinstance(r, dict):
            continue
        url = str(r.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        try:
            host = (urlparse(url).hostname or "").lower()
        except ValueError:
            continue
        kind = "video" if host in _VIDEO_HOSTS else "link"
        out: dict[str, str] = {"url": url, "kind": kind}
        title = str(r.get("title") or "").strip()
        if title:
            out["title"] = title[:200]
        return out
    return None


def _task_note_origin(
    kind: str, title: str, params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the journal-note ``origin`` for a fired task.

    Carries the per-briefing ``read_aloud`` toggle (params.read_aloud) out
    to the drawer so the note can offer/auto-start spoken playback. Origin
    is the right channel — it's metadata about the note's provenance, not a
    content ref, and it already round-trips through ``origin_json``.
    """
    origin: dict[str, Any] = {
        "source": "task",
        "detail": f"{kind or 'task'}: {(title or '')[:80]}",
    }
    if (params or {}).get("read_aloud"):
        origin["read_aloud"] = True
    return origin


# Cap the spoken text carried on the notification frame — a runaway body
# shouldn't bloat the WS push, and ~6k chars is several minutes of speech.
_SPEAK_TEXT_MAX = 6000


def _speak_payload(
    params: dict[str, Any] | None, content: str, note_id: int | None,
) -> dict[str, Any] | None:
    """Build the notification payload that drives spoken delivery.

    Only for read_aloud-toggled tasks. The client plays the chime, waits a
    beat, then narrates ``speak_text`` through the user's default voice
    (server-side synthesis via that user's configured provider). ``note_id``
    lets the live notification path and the drawer-open path dedupe so a
    briefing is never narrated twice. None when read-aloud is off or there's
    nothing to say.
    """
    if not (params or {}).get("read_aloud"):
        return None
    text = (content or "").strip()
    if not text:
        return None
    payload: dict[str, Any] = {"read_aloud": True, "speak_text": text[:_SPEAK_TEXT_MAX]}
    if note_id is not None:
        payload["note_id"] = int(note_id)
    return payload


async def _surface_result(
    runtime: CompanionRuntime, *, task: StandingTask, result: dict,
) -> None:
    """Write a journal note for the result, and publish via the
    notifications hub when the user is offline."""
    content = (result.get("details") or {}).get("content") \
        or result.get("summary") or task.title
    if not content:
        return

    # Carry url refs AND any gathered media (hero image / video) AND the
    # model-curated titled citations into the note so the drawer can render
    # them — see _media_refs_from_details / _citation_refs_from_details.
    refs = list(result.get("refs") or [])
    refs.extend(_media_refs_from_details(result.get("details")))
    refs.extend(_citation_refs_from_details(result.get("details")))

    journal_id: int | None = None
    try:
        journal_id = await runtime.memory.safe_journal(
            content,
            source="standing_task",
            user_id=task.user_id,
            entry_type="standing_task",
            affect_tag="curious",
            content_refs=refs,
            confidence_numeric=0.7,
            origin=_task_note_origin(
                getattr(task, "kind", "") or "",
                task.title or "",
                task.params,
            ),
        )
    except Exception:
        log.warning("standing_task_journal_failed", task_id=task.id, exc_info=True)

    # Notifications hub publish. The canonical attribute is
    # ``app.state.notification_hub`` (singular). The hub is lazily
    # created by HTTP routes (notifications_routes._get_hub,
    # connect_routes._get_notification_hub); if a briefing fires before
    # those routes have ever been called, the attribute doesn't exist
    # yet — so we instantiate one here and stash it. Without this
    # the standing-task notification is silently dropped (no journal
    # for the failure, no banner, no Web Push), which is exactly the
    # "I scheduled it and never heard from it" symptom.
    try:
        app_state = getattr(runtime, "_app_state", None)
        hub = getattr(app_state, "notification_hub", None) if app_state else None
        if hub is None:
            from augmentum.notifications.hub import NotificationHub
            hub = NotificationHub()
            if app_state is not None:
                app_state.notification_hub = hub
        # Notification payload: spoken-delivery hints (read_aloud) PLUS a
        # deep-link target so tapping "Open" on a media briefing lands on the
        # video/page instead of the notes drawer. open_url is absent for
        # text-only briefings, so those keep the drawer-open behavior.
        payload = dict(_speak_payload(task.params, content, journal_id) or {})
        open_target = _open_target_from_refs(refs)
        if open_target:
            payload["open_url"] = open_target["url"]
            payload["open_kind"] = open_target["kind"]
            if open_target.get("title"):
                payload["open_title"] = open_target["title"]
        from augmentum.notifications.hub import publish_and_dispatch
        await publish_and_dispatch(
            runtime.backend.conn,
            hub=hub,
            user_id=task.user_id,
            channel_id="companion.tasks",
            source="companion.standing_task",
            title=task.title,
            body=result.get("summary") or "",
            importance=_surface_importance(task),
            dedupe_key=f"task:{task.id}:{int(time.time() // 3600)}",
            payload=payload or None,
        )
    except Exception:
        log.warning("standing_task_notify_failed", task_id=task.id, exc_info=True)


# ─── Manual fire (for the run-now route) ────────────────────────────────


async def run_now(
    runtime: CompanionRuntime, *, task_id: int, user_id: str,
    surface: bool = True,
) -> dict | None:
    """Run a task immediately, bypassing next_run_at. Same result-handling
    as the tick path. Returns the result dict on success, None on
    missing/disabled/unknown-kind.

    ``surface=False`` skips the journal+notification publish — used by
    the watch creation probe, whose result reaches the user through the
    tool reply instead (a probe must never double-notify)."""
    task = await get_task(
        runtime.backend.conn, task_id=task_id,
        user_id=user_id, companion_id=runtime.companion_id,
    )
    if task is None:
        return None
    runner = _TASK_KINDS.get(task.kind)
    if runner is None:
        return None
    user_tz = await _resolve_user_timezone(
        getattr(runtime, "_app_state", None), user_id,
    )
    try:
        result = await runner(runtime, user_id=user_id, params=task.params)
    except Exception as exc:
        log.warning(
            "standing_task_run_now_failed",
            task_id=task_id, error=str(exc)[:200],
        )
        await _record_run(
            runtime.backend.conn, task_id=task_id, user_id=user_id,
            status="error", summary=str(exc)[:200],
        )
        return {"summary": f"error: {str(exc)[:120]}", "noteworthy": False, "refs": []}

    # Persist + surface as if from tick. Param-cursor mutation folds into
    # the same atomic _persist_run UPDATE (persist_params) so it can't be
    # lost between two writes (audit 2026-06-17). Success resets both
    # the error and budget-timeout counters.
    params_update = (result.get("details") or {}).get("params_update")
    effective_params = params_update if params_update is not None else task.params
    await _persist_run(
        runtime.backend.conn,
        task_id=task_id,
        interval_seconds=task.interval_seconds,
        result=result, error=None,
        consecutive_error_count=0,
        consecutive_budget_timeout_count=0,
        params=effective_params, user_timezone=user_tz,
        persist_params=params_update,
    )
    observation = (result.get("details") or {}).get("observation")
    if isinstance(observation, dict):
        await _record_observation(
            runtime.backend.conn, task_id=task_id, user_id=user_id,
            observation=observation,
        )
    if surface and result.get("noteworthy"):
        await _surface_result(runtime, task=task, result=result)
    run_details: dict[str, Any] = {}
    for body_key in ("content", "hero_image_url", "video_url", "citations"):
        val = (result.get("details") or {}).get(body_key)
        if val:
            run_details[body_key] = val
    await _record_run(
        runtime.backend.conn, task_id=task_id, user_id=user_id,
        status="fired" if result.get("noteworthy") else "silent",
        summary=result.get("summary") or "",
        details=run_details or None,
    )
    return result


def known_kinds() -> list[str]:
    """For the UI's kind picker."""
    return sorted(_TASK_KINDS.keys())


__all__ = [
    "StandingTask",
    "add_task", "list_tasks", "get_task", "remove_task", "set_enabled",
    "step", "run_now", "known_kinds", "list_runs",
]
