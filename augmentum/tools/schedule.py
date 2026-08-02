"""Unified ``schedule`` tool — one entrypoint (and one tools-panel button) for
all scheduling.

The scheduling substrate ships several specialized tools — ``schedule_briefing``,
``schedule_reminder``, ``schedule_deadline``, ``watch_for`` — plus list/cancel.
Exposing each as its own chat tool meant no single control in the tools panel
(so it was only really reachable via voice/companion) and a confusing many-way
choice for the model. This facade collapses them into ONE tool: pick an
``action`` (create / list / cancel) and, for create, a ``type`` (briefing /
reminder / deadline / watch); it routes to the specialized tool, which keeps all
its rich lenient parsing. One button, one tool, full capability — nothing is
reimplemented here.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from augmentum.tools.base import (
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
)
from augmentum.tools.manage_briefings import CancelBriefingTool, ListBriefingsTool
from augmentum.tools.schedule_briefing import ScheduleBriefingTool
from augmentum.tools.schedule_deadline import ScheduleDeadlineTool
from augmentum.tools.schedule_reminder import ScheduleReminderTool
from augmentum.tools.watch_for import WatchForTool
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# type aliases the model may emit → canonical create kind.
_TYPE_ALIASES: dict[str, str] = {
    "brief": "briefing", "briefing": "briefing", "digest": "briefing",
    "remind": "reminder", "reminder": "reminder", "alarm": "reminder",
    "deadline": "deadline", "countdown": "deadline", "due": "deadline",
    "watch": "watch", "monitor": "watch", "track": "watch",
}
_LIST_ACTIONS = {"list", "show", "view", "get"}
_CANCEL_ACTIONS = {"cancel", "delete", "remove", "stop", "unschedule"}

# Appended to every list result so the action doubles as an orientation +
# planning step: the model sees what's already set up (or that none are), then
# has the menu of what it can do next and which fields each needs — so it can
# plan and fire the right create/cancel in the same turn without guessing.
_PLAN_GUIDE = (
    "\n\nTo change anything, call `schedule` again:\n"
    "- action=create, type=briefing — recurring digest at a time of day "
    "(fields: topics, local_time; e.g. daily news at 09:00).\n"
    "- action=create, type=reminder — one-off nudge at a time "
    "(fields: title, local_time).\n"
    "- action=create, type=deadline — count down to a date with lead-time "
    "nudges (fields: title, target_date).\n"
    "- action=create, type=watch — notify when a page/price/search changes "
    "(fields: title, target/url).\n"
    "- action=cancel, task_id=<id from above> — remove one.\n"
    "Pick the next step from what's set up above and call it directly."
)


class ScheduleTool(Tool):
    """Single scheduling entrypoint that routes to the specialized tools."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state
        # Own lightweight instances of each delegate (they just hold app_state
        # and do the real parsing/persistence). Independent of whatever the
        # registry holds — no ordering dependency.
        self._create: dict[str, Tool] = {
            "briefing": ScheduleBriefingTool(app_state),
            "reminder": ScheduleReminderTool(app_state),
            "deadline": ScheduleDeadlineTool(app_state),
            "watch": WatchForTool(app_state),
        }
        self._list = ListBriefingsTool(app_state)
        self._cancel = CancelBriefingTool(app_state)

    @property
    def name(self) -> str:
        return "schedule"

    @property
    def description(self) -> str:
        return (
            "Create, view, and cancel scheduled things: recurring briefings "
            "('wake me at 9 with news'), one-off reminders ('remind me at 5pm'), "
            "deadline countdowns (nudges before a date), and watches (tell me "
            "when a page/price/search changes). One tool for all scheduling."
        )

    @property
    def model_hint(self) -> str:
        return (
            "action=create (default) with type: 'briefing' (recurring digest), "
            "'reminder' (one-off at a time), 'deadline' (count down to a date "
            "with lead-time nudges), or 'watch' (notify when a page/price/search "
            "changes). Fill the fields for the chosen type — briefing/reminder "
            "use local_time, deadline uses target_date, watch uses target/url. "
            "For a clear fresh request, just create it. When the ask is about "
            "EXISTING schedules — 'what do I have', 'change/move/cancel my …', "
            "or anything ambiguous — call action=list FIRST: it returns what's "
            "set up (or that none are) plus the menu of next steps, so you can "
            "plan and then call action=create or action=cancel (with the task_id "
            "from the list) in the same turn."
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
    def produces(self) -> list[str]:
        return ["text", "structured_data"]

    @property
    def error_hints(self) -> dict[str, str]:
        # Delegate error hints still apply (their ToolResults flow straight
        # back); these cover the routing layer itself.
        return {
            "unknown schedule type": (
                "Set type to one of: briefing, reminder, deadline, watch."
            ),
            "scheduling_disabled": (
                "The scheduling dispatcher is off in this install. Tell the "
                "user to enable scheduling (companion runtime or SchedulerService)."
            ),
        }

    @property
    def input_schema(self) -> dict:
        # action + type first (tiny models bias toward earlier entries), then
        # the union of every delegate's fields so the model can fill kind-
        # specific params with nothing lost. First declaration wins on a name
        # collision, so shared fields (title, local_time, delivery, task_id)
        # stay single. Auto-syncs when a delegate's schema changes.
        props: dict[str, Any] = {
            "action": {
                "type": "string",
                "enum": ["create", "list", "cancel"],
                "description": (
                    "create a schedule (default), list existing ones, or "
                    "cancel one."
                ),
            },
            "type": {
                "type": "string",
                "enum": ["briefing", "reminder", "deadline", "watch"],
                "description": (
                    "For action=create: what to schedule. briefing=recurring "
                    "digest; reminder=one-off at a time; deadline=countdown to a "
                    "date; watch=notify on a page/price/search change. Default "
                    "briefing."
                ),
            },
        }
        for delegate in (*self._create.values(), self._cancel):
            for pname, pdef in (delegate.input_schema.get("properties") or {}).items():
                if pname not in props:
                    props[pname] = pdef
        return {"type": "object", "properties": props, "required": []}

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.pop("action", "") or "").strip().lower()
        sched_type = str(kwargs.pop("type", "") or "").strip().lower()

        # Infer action when omitted: a bare task_id/title with no create fields
        # reads as cancel; otherwise default to create.
        if not action:
            has_target = bool(kwargs.get("task_id"))
            create_fields = ("topics", "local_time", "target_date", "target",
                             "url", "cron", "date", "checklist")
            has_create = any(kwargs.get(f) for f in create_fields)
            action = "cancel" if (has_target and not has_create) else "create"

        if action in _LIST_ACTIONS:
            res = await self._list.execute(**kwargs)
            # Turn a bare listing into a plan-your-next-step result: current
            # state + the use-case menu, so the model can act in the same turn.
            if getattr(res, "success", False):
                body = res.output or ""
                if not body.strip():
                    body = "No schedules are set up yet."
                return dataclasses.replace(res, output=body + _PLAN_GUIDE)
            return res
        if action in _CANCEL_ACTIONS:
            return await self._cancel.execute(**kwargs)

        # create
        sched_type = _TYPE_ALIASES.get(sched_type, sched_type or "briefing")
        delegate = self._create.get(sched_type)
        if delegate is None:
            return ToolResult(
                success=False,
                error=(
                    f"unknown schedule type '{sched_type}' — use briefing, "
                    "reminder, deadline, or watch"
                ),
                validation_error=True,
            )
        return await delegate.execute(**kwargs)
