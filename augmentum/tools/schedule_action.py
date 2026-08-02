"""schedule_action — wire up a scheduled verb fire the user asked for.

The conversational front door to the ``verb_fire`` standing-task kind:
"at 5pm check the weather and tell me if I need a jacket", "every
weekday at 9 open my notes", "pause the music at midnight". The model
parses the ask into (verb, args, when) and this tool persists it; the
standing-tasks engine fires it restart-survivably, in the user's
timezone, and the verb's result lands as a notification + drawer note.

Sibling of ScheduleBriefingTool — same gates, same scheduling
vocabulary (local_time / weekdays / one_shot), same management story
(list_briefings / cancel_briefing operate on ALL standing tasks).
SHORT relative asks ("in 10 minutes pause the music") belong to
``time.set_timer`` with then_verb — second-accurate and cheaper; this
tool is for wall-clock anchored asks.

Safety: the verb must exist in the intent registry AND carry a stakes
tier in DEFERRED_ACTION_STAKES (trivial_reversible / disruptive) —
checked here at creation for a conversational refusal, and re-checked
at every fire by the verb_fire runner.
"""
from __future__ import annotations

from typing import Any

from augmentum.tools._standing_common import (
    CONFIRM_REPLACE_SCHEMA_PROPERTY,
    CRON_SCHEMA_PROPERTY,
    DELIVERY_SCHEMA_PROPERTY,
    duplicate_review,
    parse_cron_param,
    parse_delivery_param,
    schedule_moment_error,
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
from augmentum.tools.schedule_briefing import (
    _normalize_weekdays,
    _parse_local_time,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_WEEKDAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu",
                  5: "Fri", 6: "Sat", 7: "Sun"}


class ScheduleActionTool(Tool):
    """Create a scheduled action (verb_fire standing task)."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "schedule_action"

    @property
    def description(self) -> str:
        return (
            "Schedule one of your tools to run at a time of day — 'at "
            "5pm check the weather and tell me', 'every weekday at 9 "
            "open my notes', 'pause the music at midnight'. The result "
            "reaches the user as a notification."
        )

    @property
    def model_hint(self) -> str:
        return (
            "verb = a tool id you can call (weather.today, media.pause, "
            "note.append, …). Only small/reversible or playback-level "
            "tools are allowed — costly or irreversible ones are "
            "refused. one_shot=true for 'at 5pm do X' (fires once); "
            "false for 'every morning' / 'every weekday' patterns. For "
            "SHORT relative asks ('in 10 minutes pause the music') use "
            "time.set_timer with then_verb instead — it is "
            "second-accurate. local_time accepts '17:00', '5pm', 'noon'."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True, coder=False, companion=True, flow=False,
            voice="disruptive",
            voice_capability_line="fire an app action at a wall-clock time (schedule_action)",
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
            "unknown verb": (
                "The verb isn't a tool id. Check your tool list and "
                "retry with a canonical id like weather.today or "
                "media.pause."
            ),
            "needs you present": (
                "That tool's stakes are too high to fire unattended. "
                "Tell the user it can't be scheduled and offer to run "
                "it now instead."
            ),
            "local_time must be": (
                "local_time must be a time of day (e.g. '5pm', '17:30'). "
                "Re-parse the user's request and retry."
            ),
            "scheduling_disabled": (
                "The scheduling dispatcher is off in this install. "
                "Tell the user to enable scheduling_enabled (or the "
                "companion runtime) in settings."
            ),
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "verb": {
                    "type": "string",
                    "description": (
                        "Canonical tool id to run at the scheduled time "
                        "(e.g. weather.today, media.pause, note.append)."
                    ),
                },
                "verb_args": {
                    "type": "object",
                    "description": (
                        "Arguments for the verb, e.g. {\"location\": "
                        "\"portland\"} for weather.today. Omit for "
                        "arg-less verbs."
                    ),
                },
                "local_time": {
                    "type": "string",
                    "description": (
                        "Time of day to fire. Accepts '17:00', '5pm', "
                        "'noon', 'midnight'. Required unless cron is set."
                    ),
                },
                "cron": CRON_SCHEMA_PROPERTY,
                "delivery": DELIVERY_SCHEMA_PROPERTY,
                "confirm_replace": CONFIRM_REPLACE_SCHEMA_PROPERTY,
                "date": {
                    "type": "string",
                    "description": (
                        "Optional calendar day (YYYY-MM-DD) for one-shots "
                        "not today — resolve 'tomorrow' / 'next Tuesday' "
                        "to the actual date yourself. Must be in the "
                        "future. Only meaningful with one_shot=true."
                    ),
                },
                "weekdays": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional weekday restriction ('mon'..'sun' or "
                        "ISO ints). Empty = every day."
                    ),
                },
                "one_shot": {
                    "type": "boolean",
                    "description": (
                        "true = fire once at the next occurrence then "
                        "delete itself ('at 5pm today do X'). false = "
                        "recurring ('every morning'). Default true."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Optional short display title. Defaults to a "
                        "readable form of the verb."
                    ),
                },
            },
            # local_time OR cron carries the schedule — validated in
            # execute() so cron-only asks don't force a fake time.
            "required": ["verb"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        ok, gate_err, runtime = standing_gate(self._app_state)
        if not ok:
            return gate_err

        verb = str(kwargs.get("verb") or "").strip().lower()
        if not verb:
            return ToolResult(
                success=False, error="verb is required",
                validation_error=True,
            )
        from augmentum.intent.registry import DEFERRED_ACTION_STAKES, REGISTRY
        action = REGISTRY.get(verb)
        if action is None:
            return ToolResult(
                success=False,
                error=f"unknown verb: {verb}",
                validation_error=True,
            )
        if getattr(action, "stakes", "") not in DEFERRED_ACTION_STAKES:
            return ToolResult(
                success=False,
                error=f"{verb} needs you present — it can't fire unattended",
                metadata={"ok": False, "reason": "stakes"},
            )

        raw_args = kwargs.get("verb_args") or {}
        verb_args = (
            {str(k): str(v) for k, v in raw_args.items()}
            if isinstance(raw_args, dict) else {}
        )

        cron_expr, cron_err = parse_cron_param(kwargs.get("cron"))
        if cron_err:
            return ToolResult(
                success=False, error=cron_err, validation_error=True,
            )
        delivery, delivery_err = parse_delivery_param(kwargs.get("delivery"))
        if delivery_err:
            return ToolResult(
                success=False, error=delivery_err, validation_error=True,
            )
        local_time = _parse_local_time(kwargs.get("local_time"))
        if local_time is None and cron_expr is None:
            return ToolResult(
                success=False,
                error=schedule_moment_error(),
                validation_error=True,
            )
        weekdays = _normalize_weekdays(kwargs.get("weekdays"))
        one_shot = kwargs.get("one_shot")
        # A cron cadence is inherently recurring — the one-shot default
        # only applies to time-of-day asks ("at 5pm do X").
        default_one_shot = cron_expr is None
        one_shot = default_one_shot if one_shot is None else bool(one_shot)

        title = str(kwargs.get("title") or "").strip()
        if not title:
            title = f"Scheduled: {verb.replace('.', ' ')}"

        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False,
                error="user_id missing — scheduled actions are per-user",
                metadata={"ok": False, "reason": "missing_user"},
            )

        params: dict[str, Any] = {
            "verb": verb,
            "title": title,
        }
        if local_time is not None:
            params["local_time"] = local_time
        if cron_expr:
            params["cron"] = cron_expr
        if delivery:
            params["delivery"] = delivery
        if verb_args:
            params["verb_args"] = verb_args
        if weekdays:
            params["weekdays"] = weekdays
        if one_shot:
            params["one_shot"] = True

        from augmentum.companion_runtime import standing_tasks
        user_tz = await standing_tasks._resolve_user_timezone(
            self._app_state, user_id,
        )
        raw_date = str(kwargs.get("date") or "").strip()
        if raw_date:
            from augmentum.tools._standing_common import validate_future_date
            date_norm, date_err = validate_future_date(
                raw_date, local_time, user_tz,
            )
            if date_err:
                return ToolResult(
                    success=False, error=date_err, validation_error=True,
                )
            params["date"] = date_norm
        # Read-before-create duplicate review (one-shots bypass).
        dup = await duplicate_review(
            runtime, user_id=user_id, kind="verb_fire", title=title,
            params=params,
            confirm_replace=bool(kwargs.get("confirm_replace")),
            one_shot=one_shot,
        )
        if dup is not None:
            return dup

        try:
            task = await standing_tasks.add_task(
                runtime.backend.conn,
                user_id=user_id, companion_id=runtime.companion_id,
                title=title, kind="verb_fire", params=params,
                interval_seconds=86400,
                user_timezone=user_tz,
            )
        except ValueError as exc:
            return ToolResult(
                success=False, error=str(exc), validation_error=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the turn
            log.warning("schedule_action_failed", error=str(exc)[:200])
            return ToolResult(
                success=False,
                error="failed to create scheduled action",
                metadata={"ok": False, "reason": "internal"},
            )
        if task is None:
            return ToolResult(
                success=False,
                error=f"a scheduled action titled '{title}' already exists",
                metadata={"ok": False, "reason": "duplicate"},
            )

        weekday_words = ""
        if weekdays:
            weekday_words = " on " + "/".join(
                _WEEKDAY_NAMES[w] for w in weekdays
            )
        if cron_expr:
            from augmentum.utils.cron import describe
            when_words = f"on schedule '{describe(cron_expr)}'"
            weekday_words = ""  # cron owns the day pattern
            cadence = "fires once" if one_shot else "recurring"
        else:
            when_words = f"at {local_time}"
            cadence = "fires once" if one_shot else "runs daily"
        summary = (
            f"Scheduled {verb} {when_words}{weekday_words} — "
            f"{cadence}. Next fire: {task.next_run_at} UTC."
        )
        if one_shot:
            summary += " (Deletes itself after firing.)"
        log.info(
            "schedule_action_created",
            task_id=task.id, verb=verb, user_id=user_id,
            one_shot=one_shot, local_time=local_time,
        )
        return ToolResult(
            success=True,
            output=summary,
            metadata={
                "ok": True, "task_id": task.id, "verb": verb,
                "next_run_at": task.next_run_at,
            },
        )
