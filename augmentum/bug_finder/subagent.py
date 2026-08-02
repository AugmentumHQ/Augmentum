"""Re-export shim: bounded subagent loop primitive.

Substrate moved to ``augmentum.agents`` in 2026-05-31 (split across
``loop.py`` and ``tools.py``) so the coder mode's task_dispatch tool
can share the same primitives. Existing bug_finder import sites
continue to work unchanged via this re-export.
"""

from __future__ import annotations

from augmentum.agents.loop import (
    SubagentResult,
    SubagentSpec,
    ToolCallLog,
    run_subagent,
)
from augmentum.agents.tools import (
    COMPREHENDER_TOOL_NAMES,
    DETECTOR_TOOL_NAMES,
    FIXER_TOOL_NAMES,
    INVESTIGATOR_TOOL_NAMES,
    LEAD_TOOL_NAMES,
    PEN_TESTER_TOOL_NAMES,
    PLANNER_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    VERIFIER_TOOL_NAMES,
    filter_tools,
    tool_names_for_role,
)

__all__ = [
    "COMPREHENDER_TOOL_NAMES",
    "DETECTOR_TOOL_NAMES",
    "FIXER_TOOL_NAMES",
    "INVESTIGATOR_TOOL_NAMES",
    "LEAD_TOOL_NAMES",
    "PEN_TESTER_TOOL_NAMES",
    "PLANNER_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "SubagentResult",
    "SubagentSpec",
    "ToolCallLog",
    "VERIFIER_TOOL_NAMES",
    "filter_tools",
    "run_subagent",
    "tool_names_for_role",
]
