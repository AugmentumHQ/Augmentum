"""Shared substrate gate for standing-task-backed tools.

``_gate()`` was copy-pasted identically across watch_for,
schedule_reminder, manage_briefings, and recommend_now (and inlined in
schedule_action / schedule_briefing). One home; the kill-switch story
("companion runtime off → every scheduling tool refuses the same way")
stays a single rule.
"""

from __future__ import annotations

from typing import Any

from augmentum.config import settings
from augmentum.tools.base import ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def standing_gate(app_state: Any) -> tuple[bool, ToolResult | None, Any]:
    """Dispatcher + flag checks shared by every standing-task tool.

    Returns ``(ok, error_result, runtime)`` — when ``ok`` is False the
    caller returns ``error_result`` directly.

    Scheduling is an app-level substrate: the returned context is the
    companion runtime when it's up, else the SchedulerService's headless
    context (same duck-typed surface — backend/companion_id/memory/
    _app_state). A schedule tool refuses only when NEITHER dispatcher is
    available or the standing-tasks kill-switch is off.
    """
    if not bool(getattr(settings, "companion_standing_tasks_enabled", True)):
        return False, ToolResult(
            success=False, error="standing_tasks_disabled",
            metadata={"ok": False, "reason": "standing_tasks_disabled"},
        ), None
    runtime = getattr(app_state, "companion_runtime", None)
    if runtime is None:
        service = getattr(app_state, "scheduler_service", None)
        if service is not None:
            runtime = service.ctx
    if runtime is None:
        return False, ToolResult(
            success=False, error="scheduling_disabled",
            metadata={"ok": False, "reason": "scheduling_disabled"},
        ), None
    return True, None, runtime


def parse_cron_param(value: Any) -> tuple[str | None, str | None]:
    """Validate an optional ``cron`` tool argument.

    Returns ``(normalized_expr, error)``. Both are None when the argument
    is absent/empty; exactly one is non-None otherwise. Shared by every
    schedule tool so "what counts as a valid cron" stays a single rule
    (augmentum.utils.cron — 5-field POSIX + @daily-style aliases).
    """
    raw = str(value or "").strip()
    if not raw:
        return None, None
    from augmentum.utils.cron import validate
    err = validate(raw)
    if err:
        return None, (
            f"cron: {err}. Use 5 fields (minute hour day month weekday), "
            "e.g. '0 */2 * * *' for every 2 hours, '0 9 1 * *' for the "
            "1st at 9am."
        )
    return raw, None


# Shared input_schema property for the ``cron`` argument — one wording
# for every schedule tool so the model sees a consistent vocabulary.
CRON_SCHEMA_PROPERTY: dict[str, Any] = {
    "type": "string",
    "description": (
        "Optional 5-field cron expression (minute hour day month "
        "weekday) for schedules local_time can't express: 'every 2 "
        "hours' → '0 */2 * * *', 'hourly 9-5 on weekdays' → "
        "'0 9-17 * * mon-fri', 'the 1st at 9am' → '0 9 1 * *'. "
        "@hourly/@daily/@weekly/@monthly aliases work. Evaluated in "
        "the user's timezone. When set, local_time/weekdays are "
        "ignored. Prefer local_time+weekdays for simple daily/weekly "
        "times — use cron only when they can't say it."
    ),
}


def parse_delivery_param(value: Any) -> tuple[str | None, str | None]:
    """Validate an optional ``delivery`` tool argument.

    Returns ``(normalized, error)`` — both None when absent. Accepted:
    ``alert`` (notify every device with sound, even with a tab open) and
    ``quiet`` (in-app chime; push only when away). Tolerates the obvious
    synonyms models emit. One home so every schedule tool speaks the
    same delivery vocabulary as the Schedule UI.
    """
    raw = str(value or "").strip().lower()
    if not raw:
        return None, None
    aliases = {
        "alert": "alert", "loud": "alert", "notify": "alert",
        "push": "alert", "all_devices": "alert", "high": "alert",
        "quiet": "quiet", "digest": "quiet", "silent": "quiet",
        "low": "quiet", "default": "quiet",
    }
    norm = aliases.get(raw)
    if norm is None:
        return None, (
            f"delivery must be 'alert' or 'quiet', got: {raw[:24]}"
        )
    return norm, None


# Shared input_schema property for the ``delivery`` argument.
DELIVERY_SCHEMA_PROPERTY: dict[str, Any] = {
    "type": "string",
    "enum": ["alert", "quiet"],
    "description": (
        "Optional delivery preference. 'alert' = notify every device "
        "with sound when it fires, even with a tab open ('wake me', "
        "'make sure I see it'). 'quiet' = chime in an open tab, push "
        "only when away ('just add it to my digest', 'no need to ping "
        "me'). Omit to use the sensible default for the task type."
    ),
}


# Shared input_schema property for the ``confirm_replace`` argument.
CONFIRM_REPLACE_SCHEMA_PROPERTY: dict[str, Any] = {
    "type": "boolean",
    "description": (
        "Set true ONLY after the user has explicitly confirmed they "
        "want this created even though a similar task exists (you'll "
        "get a duplicate-review error naming the matches first). "
        "Default false."
    ),
}


async def duplicate_review(
    runtime: Any,
    *,
    user_id: str,
    kind: str,
    title: str,
    params: dict[str, Any] | None,
    confirm_replace: bool = False,
    one_shot: bool = False,
) -> ToolResult | None:
    """The read-before-create rule, enforced in the tool.

    Checks the user's existing schedule for likely duplicates BEFORE
    creating — deterministically, so it never depends on the model
    remembering to call list_briefings first. Returns a dup-review
    ToolResult naming the matches (the model relays them and asks the
    user), or None when creation should proceed.

    Bypasses, mirroring schedule_briefing's precedent:
      * ``confirm_replace`` — the user already said "yes, make another".
      * ``one_shot`` — a one-time fire doesn't compete with a recurring
        schedule ("check it once now" vs "check it every Thursday").
    """
    if confirm_replace or one_shot:
        return None
    from augmentum.companion_runtime import standing_tasks
    try:
        similar = await standing_tasks.find_similar_tasks(
            runtime.backend.conn,
            user_id=user_id, companion_id=runtime.companion_id,
            kind=kind, title=title, params=params,
        )
    except Exception:  # noqa: BLE001 — the check must never block creation
        log.warning("duplicate_review_failed", exc_info=True)
        return None
    if not similar:
        return None
    lines = []
    for m in similar:
        when = m.get("cron") or m.get("local_time") or ""
        bits = [f"#{m['id']} '{m['title']}' ({m['kind']}"]
        if when:
            bits.append(f", {when}")
        if m.get("target"):
            bits.append(f", {str(m['target'])[:60]}")
        bits.append(")" if m.get("enabled") else ", paused)")
        lines.append("".join(bits))
    return ToolResult(
        success=False,
        error=(
            "possible duplicate — the user already has: "
            + "; ".join(lines)
            + ". Ask whether to keep the existing one, replace it "
            "(cancel_briefing then re-create), or create this anyway "
            "(re-call with confirm_replace=true)."
        ),
        metadata={
            "ok": False, "reason": "duplicate_review",
            "matches": similar,
        },
    )


def schedule_moment_error() -> str:
    """The one refusal wording for 'no usable schedule given'."""
    return (
        "need a schedule: local_time (e.g. '5pm', '17:30') or a cron "
        "expression (e.g. '0 */2 * * *')"
    )


def validate_future_date(
    date_str: Any, local_time: str, user_timezone: str = "",
) -> tuple[str | None, str | None]:
    """Validate a YYYY-MM-DD ``date`` for a dated one-shot.

    Returns ``(normalized_date, error)`` — exactly one is non-None.
    The date+local_time moment must be in the future in the user's
    timezone; the model resolves "tomorrow"/"next Tuesday" to ISO, this
    only checks the result. (The engine independently treats stale dates
    as fall-through, so this is the conversational-refusal layer, not
    the safety layer.)
    """
    from datetime import datetime

    from augmentum.companion_runtime.standing_tasks import _resolve_zoneinfo

    raw = str(date_str or "").strip()
    if not raw:
        return None, "date is required (YYYY-MM-DD)"
    try:
        y, mo, dd = (int(x) for x in raw.split("-", 2))
        hh, mm = (int(x) for x in (local_time or "00:00").split(":", 1))
        tz = _resolve_zoneinfo(user_timezone)
        now_local = datetime.now(tz) if tz else datetime.now().astimezone()
        target = now_local.replace(
            year=y, month=mo, day=dd, hour=hh, minute=mm,
            second=0, microsecond=0,
        )
    except (ValueError, TypeError):
        return None, f"date must be a real calendar day (YYYY-MM-DD), got: {raw[:16]}"
    if target <= now_local:
        return None, f"{raw} {local_time} is already past — pick a future moment"
    return f"{y:04d}-{mo:02d}-{dd:02d}", None
