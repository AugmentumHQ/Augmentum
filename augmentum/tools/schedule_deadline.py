"""schedule_deadline — countdown reminders toward a target date.

The conversational front door to the ``deadline`` standing-task kind: "my
PMP renews March 15, nudge me at 30/14/7 days", "remind me taxes are due
April 15", "the grant closes June 1, count me down". The model resolves the
date + lead-times and this persists it; the standing-tasks engine fires once
per lead-time offset (days-remaining + an optional checklist of what's still
outstanding) and retires the row on the day-of fire.

Distinct from ``schedule_reminder`` (a single one-shot at a time) — a deadline
fires MULTIPLE times on an escalating countdown toward one target date. No
external data: pure date math + the standard notification/note delivery.
"""
from __future__ import annotations

from typing import Any

from augmentum.tools._standing_common import (
    CONFIRM_REPLACE_SCHEMA_PROPERTY,
    DELIVERY_SCHEMA_PROPERTY,
    duplicate_review,
    parse_delivery_param,
    standing_gate,
    validate_future_date,
)
from augmentum.tools.base import (
    CoreVerbAutonomyClass,
    CoreVerbMetadata,
    CoreVerbSafetyClass,
    CostEnvelope,
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Sensible escalating cadence when the user doesn't specify lead-times.
_DEFAULT_OFFSETS = [30, 14, 7, 1]
# Day-of (0) always fires regardless of the offsets — you want to know when
# the day actually arrives — so it's added by the engine's exhausted path.


def _normalize_offsets(raw: Any) -> list[int]:
    """Coerce the model's lead-times into a sorted, deduped, non-negative
    int list. Tolerates ints, numeric strings, and "day-of"/"0". Empty or
    junk falls back to the default cadence."""
    if not isinstance(raw, list):
        return list(_DEFAULT_OFFSETS)
    out: set[int] = set()
    for o in raw:
        if isinstance(o, str) and o.strip().lower() in ("day-of", "day of", "today", "0"):
            out.add(0)
            continue
        try:
            oi = int(o)
        except (ValueError, TypeError):
            continue
        if 0 <= oi <= 3650:
            out.add(oi)
    return sorted(out, reverse=True) if out else list(_DEFAULT_OFFSETS)


class ScheduleDeadlineTool(Tool):
    """Create a deadline countdown (deadline standing task)."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "schedule_deadline"

    @property
    def description(self) -> str:
        return (
            "Count down to a deadline with reminders at lead times — 'my "
            "PMP renews March 15, nudge me at 30/14/7 days', 'taxes are due "
            "April 15', 'the grant closes June 1'. Fires once per lead-time "
            "with days-remaining (and an optional checklist), then retires "
            "itself on the day. For a single one-time reminder use "
            "schedule_reminder instead; for a recurring digest use "
            "schedule_briefing."
        )

    @property
    def model_hint(self) -> str:
        return (
            "Resolve the deadline to an absolute target_date (YYYY-MM-DD) "
            "yourself — turn 'March 15' / 'next Friday' / 'in 3 weeks' into "
            "the actual future date. offsets_days = how many days before the "
            "date to remind (default 30/14/7/1); the day itself always "
            "fires. checklist = the items still to do, if the user listed "
            "any. local_time accepts '9am' / '09:00' (default 09:00)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True, coder=False, companion=True, flow=False,
            voice="disruptive",
            voice_capability_line="track a deadline with lead-time nudges (schedule_deadline)",
        )

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        return CoreVerbMetadata(
            safety_class=CoreVerbSafetyClass.WRITE_SELF,
            autonomy_class=CoreVerbAutonomyClass.EXPLICIT,
            cost_envelope=CostEnvelope(max_wallclock_ms=3_000, max_db_ops=6),
            cite_self_required=True,
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def produces(self) -> list[str]:
        return ["text", "structured_data"]

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "target_date": (
                "Resolve the deadline to a real future YYYY-MM-DD date and "
                "retry — don't pass a relative phrase."
            ),
            "already past": (
                "That date is in the past. Confirm the year, or tell the "
                "user the deadline has already passed."
            ),
            "scheduling_disabled": (
                "The scheduling dispatcher is off in this install. Tell "
                "the user to enable scheduling_enabled (or the companion "
                "runtime) in settings."
            ),
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Short name for the deadline (e.g. 'PMP renewal', "
                        "'Taxes due', 'Grant application')."
                    ),
                },
                "target_date": {
                    "type": "string",
                    "description": (
                        "The deadline date as YYYY-MM-DD. Resolve relative "
                        "phrases ('March 15', 'in 2 weeks') to the actual "
                        "future date yourself. Required."
                    ),
                },
                "offsets_days": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Days-before-the-date to send reminders. Default "
                        "[30, 14, 7, 1]. The day itself always fires too."
                    ),
                },
                "local_time": {
                    "type": "string",
                    "description": (
                        "Time of day to send each reminder. '9am', '17:00', "
                        "'noon'. Default 09:00."
                    ),
                },
                "checklist": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of things still to do before the "
                        "deadline — surfaced with each reminder."
                    ),
                },
                "delivery": DELIVERY_SCHEMA_PROPERTY,
                "confirm_replace": CONFIRM_REPLACE_SCHEMA_PROPERTY,
                "note": {
                    "type": "string",
                    "description": "Optional one-line note shown with each reminder.",
                },
            },
            "required": ["title", "target_date"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        ok, err_result, runtime = standing_gate(self._app_state)
        if not ok:
            return err_result

        title = str(kwargs.get("title") or "").strip()
        if not title:
            return ToolResult(
                success=False, error="title is required", validation_error=True,
            )

        # Reuse the briefing time parser for '9am'/'noon'/'17:00'.
        from augmentum.tools.schedule_briefing import _parse_local_time
        local_time = _parse_local_time(kwargs.get("local_time")) or "09:00"

        offsets = _normalize_offsets(kwargs.get("offsets_days"))

        raw_checklist = kwargs.get("checklist")
        checklist = (
            [str(c).strip() for c in raw_checklist if str(c).strip()][:20]
            if isinstance(raw_checklist, list) else []
        )
        note = str(kwargs.get("note") or "").strip()

        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False,
                error="user_id missing — deadlines are per-user",
                metadata={"ok": False, "reason": "missing_user"},
            )

        from augmentum.companion_runtime import standing_tasks
        user_tz = await standing_tasks._resolve_user_timezone(
            self._app_state, user_id,
        )
        raw_date = str(kwargs.get("target_date") or "").strip()
        date_norm, date_err = validate_future_date(raw_date, local_time, user_tz)
        if date_err:
            return ToolResult(
                success=False, error=date_err, validation_error=True,
            )

        params: dict[str, Any] = {
            "target_date": date_norm,
            "offsets_days": offsets,
            "local_time": local_time,
            "title": title,
        }
        if checklist:
            params["checklist"] = checklist
        if note:
            params["note"] = note
        delivery, delivery_err = parse_delivery_param(kwargs.get("delivery"))
        if delivery_err:
            return ToolResult(
                success=False, error=delivery_err, validation_error=True,
            )
        if delivery:
            params["delivery"] = delivery

        # Read-before-create duplicate review -- same target_date +
        # similar title is the same countdown.
        dup = await duplicate_review(
            runtime, user_id=user_id, kind="deadline", title=title,
            params=params,
            confirm_replace=bool(kwargs.get("confirm_replace")),
        )
        if dup is not None:
            return dup

        try:
            task = await standing_tasks.add_task(
                runtime.backend.conn,
                user_id=user_id, companion_id=runtime.companion_id,
                title=title, kind="deadline", params=params,
                interval_seconds=86400,
                user_timezone=user_tz,
            )
        except ValueError as exc:
            return ToolResult(
                success=False, error=str(exc), validation_error=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the turn
            log.warning("schedule_deadline_failed", error=str(exc)[:200])
            return ToolResult(
                success=False,
                error="failed to create deadline",
                metadata={"ok": False, "reason": "internal"},
            )
        if task is None:
            return ToolResult(
                success=False,
                error=f"a deadline titled '{title}' already exists",
                metadata={"ok": False, "reason": "duplicate"},
            )

        offset_words = "/".join(str(o) for o in offsets)
        summary = (
            f"Counting down to {title} on {date_norm} — reminders at "
            f"{offset_words} days out (and on the day). Next: {task.next_run_at} UTC."
        )
        log.info(
            "schedule_deadline_created",
            task_id=task.id, target_date=date_norm, user_id=user_id,
            offsets=offsets,
        )
        return ToolResult(
            success=True,
            output=summary,
            metadata={
                "ok": True, "task_id": task.id,
                "target_date": date_norm, "next_run_at": task.next_run_at,
            },
        )
