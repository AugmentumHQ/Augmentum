"""Subagent dispatch substrate — shared between bug_finder and coder.

Public surface: ``SubagentSpec``, ``SubagentResult``, ``SubagentBudget``,
``run_subagent``, ``StuckDetector``, ``ToolGuard``, ``AgentRole``,
``AgentRegistry``, ``SubagentDispatcher``, ``resolve_subagent_model``.

Extracted from ``augmentum/bug_finder/`` in 2026-05-31. The bug_finder
package re-exports the same names from here so existing call-sites
keep working without changes.
"""

from __future__ import annotations

from augmentum.agents.budget import BudgetTracker, SubagentBudget
from augmentum.agents.guards import (
    ToolGuard,
    detector_guard,
    fixer_guard,
    planner_guard,
    role_guard,
    verifier_guard,
)
from augmentum.agents.loop import (
    SubagentResult,
    SubagentSpec,
    ToolCallLog,
    run_subagent,
)
from augmentum.agents.stuck import StuckDetector, StuckPattern, StuckResult, Turn
from augmentum.agents.tools import (
    DETECTOR_TOOL_NAMES,
    FIXER_TOOL_NAMES,
    PLANNER_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    VERIFIER_TOOL_NAMES,
    filter_tools,
    tool_names_for_role,
)

__all__ = [
    "BudgetTracker",
    "DETECTOR_TOOL_NAMES",
    "FIXER_TOOL_NAMES",
    "PLANNER_TOOL_NAMES",
    "READ_ONLY_TOOL_NAMES",
    "StuckDetector",
    "StuckPattern",
    "StuckResult",
    "SubagentBudget",
    "SubagentResult",
    "SubagentSpec",
    "ToolCallLog",
    "ToolGuard",
    "Turn",
    "VERIFIER_TOOL_NAMES",
    "detector_guard",
    "filter_tools",
    "fixer_guard",
    "planner_guard",
    "role_guard",
    "run_subagent",
    "tool_names_for_role",
    "verifier_guard",
]
