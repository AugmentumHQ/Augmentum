"""``schedule_request`` — defer a whole request, not just a verb.

"Tomorrow morning, summarize what happened with the llama.cpp release"
isn't a verb fire — it's "run this prompt through me at fire time."
Wraps the ``prompt_fire`` standing-task kind: at the scheduled moment
the companion's shared FC loop runs the saved prompt headlessly
(gathering with tools, budget-capped) and the answer reaches the user
as a notification + drawer note.

Sibling of ScheduleActionTool — same scheduling vocabulary
(local_time / date / weekdays / one_shot), same management story.
The stakes carve-out is §6.2 of the scheduled-requests spec: only a
user's explicit ask creates one of these (initiative paths cannot),
the inference is the only thing exempted from DEFERRED_ACTION_STAKES —
every action *inside* the run stays gated, and the output is words,
never autonomous screen action.
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
from augmentum.tools.schedule_briefing import (
    _normalize_weekdays,
    _parse_local_time,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class ScheduleRequestTool(Tool):
    """Create a deferred request (prompt_fire standing task)."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "schedule_request"

    @property
    def description(self) -> str:
        return (
            "Schedule a request to run later — 'tomorrow morning, "
            "summarize what happened with X', 'at 6pm find me dinner "
            "ideas'. At fire time the request runs with tool access and "
            "the answer reaches the user as a notification."
        )

    @property
    def model_hint(self) -> str:
        return (
            "prompt = the request, SELF-CONTAINED: there is no "
            "conversation context at fire time, so resolve pronouns and "
            "use relative windows ('in the last 24 hours', not 'since "
            "we talked'). Use schedule_action instead when the ask maps "
            "to one tool call; use time.set_timer for short relative "
            "asks ('in 10 minutes'). date = YYYY-MM-DD for 'tomorrow' / "
            "'next Tuesday' (resolve it yourself). one_shot=true unless "
            "the user said 'every'."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True, coder=False, companion=True, flow=False,
            voice="disruptive",
            voice_capability_line="run any request once at a future time (schedule_request)",
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
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The request to run at fire time. Must stand "
                        "alone without this conversation."
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
                        "Optional calendar day (YYYY-MM-DD) for "
                        "one-shots not today. Must be in the future."
                    ),
                },
                "weekdays": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional weekday restriction ('mon'..'sun' or "
                        "ISO ints) for recurring requests."
                    ),
                },
                "one_shot": {
                    "type": "boolean",
                    "description": (
                        "true = fire once (default). false = recurring "
                        "('every morning...')."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Optional short display title.",
                },
            },
            # local_time OR cron carries the schedule — validated in
            # execute() so cron-only asks don't force a fake time.
            "required": ["prompt"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        ok, err, runtime = standing_gate(self._app_state)
        if not ok:
            return err
        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False,
                error="user_id missing — scheduled requests are per-user",
                metadata={"ok": False, "reason": "missing_user"},
            )

        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return ToolResult(
                success=False, error="prompt is required",
                validation_error=True,
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
        # Cron cadences are inherently recurring; the one-shot default
        # applies only to time-of-day asks.
        default_one_shot = cron_expr is None
        one_shot = default_one_shot if one_shot is None else bool(one_shot)

        title = str(kwargs.get("title") or "").strip()
        if not title:
            title = prompt[:60] + ("…" if len(prompt) > 60 else "")

        from augmentum.companion_runtime import standing_tasks
        user_tz = await standing_tasks._resolve_user_timezone(
            self._app_state, user_id,
        )

        params: dict[str, Any] = {
            "prompt": prompt[:2000],
        }
        if local_time is not None:
            params["local_time"] = local_time
        if cron_expr:
            params["cron"] = cron_expr
        if delivery:
            params["delivery"] = delivery
        if weekdays:
            params["weekdays"] = weekdays
        if one_shot:
            params["one_shot"] = True
        raw_date = str(kwargs.get("date") or "").strip()
        if raw_date:
            date_norm, date_err = validate_future_date(
                raw_date, local_time, user_tz,
            )
            if date_err:
                return ToolResult(
                    success=False, error=date_err, validation_error=True,
                )
            params["date"] = date_norm

        # Read-before-create duplicate review (one-shots bypass — a
        # one-time ask doesn't compete with a recurring schedule).
        dup = await duplicate_review(
            runtime, user_id=user_id, kind="prompt_fire", title=title,
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
                title=title, kind="prompt_fire", params=params,
                interval_seconds=86400,
                user_timezone=user_tz,
            )
        except ValueError as exc:
            return ToolResult(
                success=False, error=str(exc), validation_error=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the turn
            log.warning("schedule_request_failed", error=str(exc)[:200])
            return ToolResult(
                success=False,
                error="failed to create scheduled request",
                metadata={"ok": False, "reason": "internal"},
            )
        if task is None:
            return ToolResult(
                success=False,
                error=f"a scheduled request titled '{title}' already exists",
                metadata={"ok": False, "reason": "duplicate"},
            )

        if cron_expr:
            from augmentum.utils.cron import describe
            when = f"on schedule '{describe(cron_expr)}'"
        else:
            when = f"at {local_time}"
        if raw_date:
            when += f" on {params['date']}"
        cadence = "fires once" if one_shot else "runs on schedule"
        summary = (
            f"Scheduled: \"{title}\" {when} — {cadence}. "
            f"Next fire: {task.next_run_at} UTC."
        )
        log.info(
            "schedule_request_created",
            task_id=task.id, user_id=user_id,
            one_shot=one_shot, local_time=local_time,
        )
        return ToolResult(
            success=True,
            output=summary,
            metadata={
                "ok": True, "task_id": task.id,
                "next_run_at": task.next_run_at,
            },
        )
