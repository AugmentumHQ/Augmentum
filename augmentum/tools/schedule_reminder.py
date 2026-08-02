"""``schedule_reminder`` — companion verbs architecture, Phase 4.

One-shot reminder for a specific local time. Distinct from
``schedule_briefing`` which is recurring + multi-topic; a reminder
is a single note Becca will surface to the user at the requested
moment (e.g. "remind me to call mom at 5pm").

Implementation note: standing tasks have a per-row ``interval_seconds``
floor of 5 min. We don't try to make a truly one-shot row — instead
the reminder is a ``briefing``-kind row anchored to a specific
``local_time``, with a "one-shot" marker in params so the runner
disables the row after the first fire. That keeps reminder semantics
on the same engine as briefings without a new standing-task kind.
"""

from __future__ import annotations

from typing import Any

from augmentum.tools._standing_common import (
    DELIVERY_SCHEMA_PROPERTY,
    parse_delivery_param,
    standing_gate,
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


class ScheduleReminderTool(Tool):
    """One-shot reminder fire at a specific local time."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "schedule_reminder"

    @property
    def description(self) -> str:
        return (
            "Set a one-shot reminder for a specific local time. Use "
            "for 'remind me to X at 5pm' or 'wake me at 6am' — single "
            "fire, then the reminder is done. NOT for recurring "
            "digests (that's schedule_briefing) or for watching a URL "
            "(that's watch_for)."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=True, coder=False, companion=True, flow=False)

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "What to remind the user about.",
                },
                "local_time": {
                    "type": "string",
                    "description": "Local time in HH:MM format (24h).",
                },
                "date_offset_days": {
                    "type": "integer",
                    "description": (
                        "0 = today (if local_time is later than now), "
                        "1 = tomorrow, etc. Default 0."
                    ),
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Specific calendar day (YYYY-MM-DD) — preferred "
                        "over date_offset_days for 'next Tuesday' / 'on "
                        "the 20th' asks. Must be in the future."
                    ),
                },
                "delivery": DELIVERY_SCHEMA_PROPERTY,
            },
            "required": ["title", "local_time"],
        }

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        return CoreVerbMetadata(
            safety_class=CoreVerbSafetyClass.WRITE_SELF,
            autonomy_class=CoreVerbAutonomyClass.EXPLICIT,
            cost_envelope=CostEnvelope(max_wallclock_ms=2_000, max_db_ops=4),
            cite_self_required=True,
        )

    async def execute(self, **kwargs) -> ToolResult:
        ok, err, runtime = standing_gate(self._app_state)
        if not ok:
            return err
        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False, error="user_id missing",
                metadata={"ok": False, "reason": "missing_user"},
            )

        title = str(kwargs.get("title") or "").strip()
        local_time = str(kwargs.get("local_time") or "").strip()
        date_offset = int(kwargs.get("date_offset_days") or 0)
        if not title or not local_time:
            return ToolResult(
                success=False, error="title and local_time required",
                metadata={"ok": False, "reason": "missing_args"},
            )

        from augmentum.companion_runtime import standing_tasks
        params: dict[str, Any] = {
            "local_time": local_time,
            "one_shot": True,
            "topics": [],
        }

        user_tz = await standing_tasks._resolve_user_timezone(
            self._app_state, user_id,
        )
        # Resolve the target day to a concrete params.date — the engine
        # anchors dated one-shots to it. (date_offset_days used to be
        # stored but never read by the scheduler, so "tomorrow at 9"
        # could fire today; translating to a date fixes that.)
        raw_date = str(kwargs.get("date") or "").strip()
        if not raw_date and date_offset > 0:
            from datetime import datetime, timedelta

            from augmentum.companion_runtime.standing_tasks import (
                _resolve_zoneinfo,
            )
            tz = _resolve_zoneinfo(user_tz)
            now_local = datetime.now(tz) if tz else datetime.now().astimezone()
            raw_date = (now_local + timedelta(days=date_offset)).strftime(
                "%Y-%m-%d",
            )
        if raw_date:
            from augmentum.tools._standing_common import validate_future_date
            date_norm, date_err = validate_future_date(
                raw_date, local_time, user_tz,
            )
            if date_err:
                return ToolResult(
                    success=False, error=date_err,
                    metadata={"ok": False, "reason": "bad_date"},
                )
            params["date"] = date_norm
        delivery, delivery_err = parse_delivery_param(kwargs.get("delivery"))
        if delivery_err:
            return ToolResult(
                success=False, error=delivery_err, validation_error=True,
            )
        if delivery:
            params["delivery"] = delivery

        try:
            task = await standing_tasks.add_task(
                runtime.backend.conn,
                user_id=user_id,
                companion_id=runtime.companion_id,
                title=title,
                kind="briefing",
                params=params,
                interval_seconds=86400,  # Floor; one_shot disables after fire.
                user_timezone=user_tz,
            )
        except ValueError as e:
            return ToolResult(
                success=False, error=str(e),
                metadata={"ok": False, "reason": "validation"},
            )

        if task is None:
            return ToolResult(
                success=False, error="schedule failed",
                metadata={"ok": False, "reason": "persist_failed"},
            )

        return ToolResult(
            success=True,
            output=(
                f"Reminder set: '{title}' at {local_time}"
                f"{' tomorrow' if date_offset == 1 else ''}."
            ),
            metadata={
                "ok": True,
                "task_id": task.id,
                "next_run_at": task.next_run_at,
            },
        )
