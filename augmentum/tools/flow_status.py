"""ATP ``task_status`` — poll a background flow task for its result.

Named WITHOUT the ``flow_`` prefix on purpose: the flow re-sync in
handler_factory.py unregisters every ``flow_*`` tool when custom flows
change, which would silently nuke this one.

Flow tools (e.g. ``flow_deep_research``) launch a multi-step chain on the
BackgroundChainManager and return a ``task_id`` immediately. Chat surfaces
get the result injected into the conversation automatically; an external
harness has no such channel, so it polls with this tool instead.
Ownership is enforced by ``get_task(user_id=...)``.
"""

from __future__ import annotations

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class FlowStatusTool(Tool):
    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "task_status"

    @property
    def description(self) -> str:
        return (
            "Check a background flow task (started by a flow_* tool such as "
            "flow_deep_research). Returns status and, when completed, the "
            "full result. Poll every few seconds until status is "
            "'completed' or 'failed'."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=False, coder=False, flow=False)

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
            },
            "required": ["task_id"],
        }

    def health_check(self) -> bool:
        return getattr(self._app_state, "background_chain_manager", None) is not None

    async def execute(self, **kwargs) -> ToolResult:
        manager = getattr(self._app_state, "background_chain_manager", None)
        if manager is None:
            return ToolResult(success=False, error="background chain manager unavailable")
        task_id = str(kwargs.get("task_id") or "").strip()
        if not task_id:
            return ToolResult(success=False, validation_error=True,
                              error="'task_id' is required")
        user_id = self.extract_user_id(kwargs)
        task = manager.get_task(task_id, user_id=user_id)
        if task is None:
            return ToolResult(
                success=False,
                error=(
                    f"task {task_id!r} not found (or not yours) — results "
                    "expire about an hour after completion"
                ),
            )
        meta = {
            "task_id": task.task_id,
            "flow_name": task.flow_name,
            "status": task.status,
        }
        if task.status == "running":
            return ToolResult(
                success=True,
                output=f"Task {task_id} ({task.flow_name}) is still running — poll again shortly.",
                metadata=meta,
            )
        if task.status == "failed":
            return ToolResult(
                success=False,
                error=f"Task {task_id} ({task.flow_name}) failed: {task.error or 'unknown error'}",
                metadata=meta,
            )
        return ToolResult(
            success=True,
            output=task.result_summary or "(task completed with no result text)",
            metadata=meta,
        )
