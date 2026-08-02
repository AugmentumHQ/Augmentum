"""FlowTool — wraps a custom flow as a callable Tool.

When the LLM calls a flow tool, execution is delegated to a chain launcher
(typically the BackgroundChainManager) which runs the flow's steps
asynchronously and returns immediately with a task ID.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Coroutine
from typing import Any

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Sanitize flow names → tool names: "Deep Research" → "flow_deep_research"
_SANITIZE_RE = re.compile(r"\W+")


def flow_name_to_tool_name(flow_name: str) -> str:
    """Convert a human-readable flow name to a valid tool name."""
    return "flow_" + _SANITIZE_RE.sub("_", flow_name).lower().strip("_")


class FlowTool(Tool):
    """Virtual tool that launches a custom flow chain.

    The ``chain_launcher`` callable is injected at construction time so the
    tool is decoupled from the execution infrastructure.  It receives
    ``(flow_dict, query, session_id)`` and returns a ``task_id`` string.
    """

    def __init__(
        self,
        flow: dict,
        chain_launcher: Callable[..., Coroutine[Any, Any, str]],
        session_id: str = "",
    ) -> None:
        self._flow = flow
        self._launch = chain_launcher
        self._session_id = session_id
        self._name = flow_name_to_tool_name(flow["name"])

    # -- Tool interface --

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        desc = self._flow.get("description", "")
        return desc or f"Run the {self._flow['name']} multi-step flow"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query or input for this flow",
                },
            },
            "required": ["query"],
        }

    @property
    def timeout(self) -> float:
        return 10.0  # Just launches — doesn't wait for completion

    @property
    def cacheable(self) -> bool:
        return False  # Each invocation is unique

    @property
    def flow_id(self) -> str:
        return self._flow.get("id", "")

    async def execute(self, **kwargs: object) -> ToolResult:
        query = str(kwargs.get("query", ""))
        request_context = kwargs.pop("_request_context", None)
        # Canonical extraction — handles both the chain path (_user_id) and
        # the ATP/orchestrator path (_context={"user_id": ...}).
        user_id = self.extract_user_id(kwargs)
        try:
            task_id = await self._launch(
                self._flow, query, self._session_id,
                user_id=user_id,
                request_context=request_context,
            )
            return ToolResult(
                success=True,
                output=(
                    f"Flow '{self._flow['name']}' started (task {task_id}). "
                    "Results will be available shortly — continue responding "
                    "to the user naturally."
                ),
                metadata={
                    "task_id": task_id,
                    "flow_name": self._flow["name"],
                    "flow_id": self.flow_id,
                    "background": True,
                },
            )
        except Exception as exc:
            log.warning("flow_tool_launch_failed", flow=self._flow["name"], error=str(exc))
            return ToolResult(
                success=False,
                error=str(exc),
                output=f"Error: Failed to start flow '{self._flow['name']}': {exc}",
            )
