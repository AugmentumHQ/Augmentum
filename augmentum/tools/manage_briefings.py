"""``list_briefings`` + ``cancel_briefing`` — lifecycle chat tools.

The :class:`ScheduleBriefingTool` covers create. Without these, a user
can ask Becca to set up a briefing in chat but has to open the topics
modal to see what's scheduled or to cancel one. Two small tools close
the loop:

* ``list_briefings`` — zero-arg enumeration of the user's briefings,
  including next-run, last-error, and pause state. Lets the LLM answer
  "what briefings do I have?" without round-tripping through the UI.
* ``cancel_briefing`` — delete a briefing by title (fuzzy match) or by
  task_id (from a prior ``list_briefings`` call). Title match is
  case-insensitive substring so "cancel my morning briefing" works
  whether the title is "Morning briefing" or "Morning Briefing — News".

Pause/resume is intentionally left to the topics modal — voice/chat
"pause my briefing for a week" is a temporal request the briefing
substrate doesn't currently model. Better to be explicit (cancel +
re-create) than have a half-feature.

See ``augmentum/tools/schedule_briefing.py`` for the create tool and
``augmentum/companion_runtime/standing_tasks.py`` for the engine.
"""

from __future__ import annotations

from typing import Any

from augmentum.tools._standing_common import standing_gate
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


class ListBriefingsTool(Tool):
    """Enumerate the user's scheduled briefings."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "list_briefings"

    @property
    def description(self) -> str:
        return (
            "Show existing briefings (active + delivered one-times). "
            "Only for 'what briefings do I have?' / 'when is my X "
            "briefing?' or before referencing a past briefing's "
            "content. Delivered one-shots are kept as historical "
            "records — the user can ask to revisit, re-instate, or "
            "convert them to recurring."
        )

    @property
    def model_hint(self) -> str:
        return (
            "Read-only. Returns active AND delivered briefings. "
            "Delivered = already fired one-shots, kept for history. "
            "Not for setup requests."
        )

    @property
    def category(self) -> ToolCategory:
        # FETCH = retrieval/read. The enum has no dedicated READ; FETCH
        # is the closest fit for "enumerate user-scoped DB rows."
        return ToolCategory.FETCH

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True, coder=False, companion=True, flow=False,
            voice="disruptive",
            voice_capability_line="list your scheduled briefings and watches (list_briefings)",
        )

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        # Phase 4 — read-only briefing enumeration.
        return CoreVerbMetadata(
            safety_class=CoreVerbSafetyClass.READ,
            autonomy_class=CoreVerbAutonomyClass.EXPLICIT,
            cost_envelope=CostEnvelope(max_wallclock_ms=2_000, max_db_ops=4),
            cite_self_required=False,
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def produces(self) -> list[str]:
        return ["structured_data"]

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "user_id missing": (
                "user_id wasn't routed through — this is an internal "
                "wiring issue, not something the user can fix. Apologize "
                "and suggest they try again."
            ),
            "runtime_not_ready": (
                "Companion runtime is starting up. Tell the user to "
                "wait a moment and retry."
            ),
        }

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

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

        from augmentum.companion_runtime import standing_tasks
        all_tasks = await standing_tasks.list_tasks(
            runtime.backend.conn,
            user_id=user_id, companion_id=runtime.companion_id,
        )
        # verb_fire (scheduled actions) shares the conversational
        # lifecycle — one list, one cancel, regardless of kind.
        briefings = [t for t in all_tasks if t.kind in ("briefing", "verb_fire")]

        if not briefings:
            return ToolResult(
                success=True,
                output="No briefings scheduled.",
                metadata={"ok": True, "count": 0, "briefings": []},
            )

        rendered: list[dict[str, Any]] = []
        lines: list[str] = []
        for t in briefings:
            p = t.params or {}
            is_one_shot = bool(p.get("one_shot"))
            delivered_at = p.get("delivered_at") or ""
            is_delivered = bool(delivered_at)
            row = {
                "id": t.id,
                "title": t.title,
                "local_time": p.get("local_time", ""),
                "cron": p.get("cron", ""),
                "weekdays": p.get("weekdays") or [],
                "topics": p.get("topics") or [],
                "location": p.get("location", ""),
                "next_run_at": t.next_run_at,
                "enabled": t.enabled,
                "one_shot": is_one_shot,
                "delivered_at": delivered_at,
                "delivery_summary": p.get("delivery_summary") or "",
                "last_error": getattr(t, "last_error", None),
                "consecutive_error_count": t.consecutive_error_count,
            }
            rendered.append(row)
            line = f"• {t.title}"
            if p.get("cron"):
                from augmentum.utils.cron import describe
                line += f" — {describe(str(p['cron']))}"
            elif p.get("local_time"):
                line += f" — {p['local_time']}"
            if p.get("weekdays"):
                names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu",
                         5: "Fri", 6: "Sat", 7: "Sun"}
                line += " " + "/".join(
                    names.get(int(w), str(w)) for w in p["weekdays"]
                )
            if p.get("location"):
                line += f" · {p['location']}"
            if is_delivered:
                line += f" (delivered {delivered_at})"
            elif is_one_shot:
                line += " (one-time)"
            if not t.enabled and not is_delivered:
                line += " (paused)"
            if t.consecutive_error_count >= 3:
                line += " ⚠"
            lines.append(line)

        summary = (
            f"{len(briefings)} briefing{'s' if len(briefings) != 1 else ''} "
            f"scheduled:\n" + "\n".join(lines)
        )
        return ToolResult(
            success=True, output=summary,
            metadata={
                "ok": True, "count": len(briefings),
                "briefings": rendered,
            },
        )


class CancelBriefingTool(Tool):
    """Cancel (delete) a scheduled briefing."""

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "cancel_briefing"

    @property
    def description(self) -> str:
        return (
            "Delete a scheduled briefing. Pass task_id or a title "
            "fragment (e.g. 'morning')."
        )

    @property
    def model_hint(self) -> str:
        return (
            "Only for explicit cancel/delete/stop requests. Title is "
            "fine for one obvious match; ambiguous titles get refused."
        )

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "no briefing matching": (
                "Title didn't match any briefing. Call list_briefings "
                "to see what exists, then retry with the correct title "
                "or task_id."
            ),
            "ambiguous": (
                "Multiple briefings match that title. The metadata "
                "lists them — pick the right one with the user, then "
                "retry with task_id."
            ),
            "provide either": (
                "You called cancel_briefing without selecting one. "
                "Pass task_id (from list_briefings) or a title fragment."
            ),
        }

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True, coder=False, companion=True, flow=False,
            voice="disruptive",
            voice_capability_line="cancel a scheduled briefing or watch (cancel_briefing)",
        )

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        # Phase 4 — destructive but reversible (cancel can be re-scheduled).
        return CoreVerbMetadata(
            safety_class=CoreVerbSafetyClass.WRITE_SELF,
            autonomy_class=CoreVerbAutonomyClass.EXPLICIT,
            cost_envelope=CostEnvelope(max_wallclock_ms=2_000, max_db_ops=4),
            cite_self_required=True,
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "integer",
                    "description": (
                        "ID of the briefing to cancel. Get from "
                        "list_briefings. When provided, title is "
                        "ignored."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": (
                        "Case-insensitive substring of the briefing "
                        "title (e.g. 'morning' matches 'Morning "
                        "briefing'). Used when task_id is omitted."
                    ),
                },
            },
        }

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

        raw_id = kwargs.get("task_id")
        title_query = (kwargs.get("title") or "").strip().lower()

        if raw_id is None and not title_query:
            return ToolResult(
                success=False,
                error="provide either task_id or title",
                validation_error=True,
            )

        from augmentum.companion_runtime import standing_tasks
        all_tasks = await standing_tasks.list_tasks(
            runtime.backend.conn,
            user_id=user_id, companion_id=runtime.companion_id,
        )
        # verb_fire (scheduled actions) shares the conversational
        # lifecycle — one list, one cancel, regardless of kind.
        briefings = [t for t in all_tasks if t.kind in ("briefing", "verb_fire")]

        # Prefer explicit id when given.
        if raw_id is not None:
            try:
                tid = int(raw_id)
            except (TypeError, ValueError):
                return ToolResult(
                    success=False, error="task_id must be an integer",
                    validation_error=True,
                )
            target = next((t for t in briefings if t.id == tid), None)
            if target is None:
                return ToolResult(
                    success=False,
                    error=f"no briefing with id {tid}",
                    metadata={"ok": False, "reason": "not_found"},
                )
            matches = [target]
        else:
            matches = [
                t for t in briefings if title_query in t.title.lower()
            ]

        if not matches:
            return ToolResult(
                success=False,
                error=f"no briefing matching '{kwargs.get('title')}'",
                metadata={"ok": False, "reason": "not_found"},
            )
        if len(matches) > 1:
            choices = ", ".join(f"#{t.id} '{t.title}'" for t in matches)
            return ToolResult(
                success=False,
                error=(
                    f"{len(matches)} briefings match — call "
                    f"list_briefings and pass task_id. Matches: "
                    f"{choices}"
                ),
                metadata={
                    "ok": False, "reason": "ambiguous",
                    "matches": [{"id": t.id, "title": t.title} for t in matches],
                },
            )

        target = matches[0]
        removed = await standing_tasks.remove_task(
            runtime.backend.conn, task_id=target.id,
            user_id=user_id, companion_id=runtime.companion_id,
        )
        if not removed:
            return ToolResult(
                success=False,
                error=f"failed to cancel '{target.title}'",
                metadata={"ok": False, "reason": "delete_failed"},
            )
        return ToolResult(
            success=True,
            output=f"Cancelled '{target.title}'.",
            metadata={
                "ok": True, "task_id": target.id, "title": target.title,
            },
        )
