"""Re-export shim: per-subagent budget accounting.

Substrate moved to ``augmentum.agents.budget`` in 2026-05-31 so the
coder mode's task_dispatch tool can share the same primitives. Existing
bug_finder import sites continue to work unchanged via this re-export.
"""

from __future__ import annotations

from augmentum.agents.budget import BudgetTracker, SubagentBudget

__all__ = ["BudgetTracker", "SubagentBudget"]
