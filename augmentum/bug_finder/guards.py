"""Re-export shim: tool-call guards (reward-hacking defenses).

Substrate moved to ``augmentum.agents.guards`` in 2026-05-31 so the
coder mode's task_dispatch tool can share the same defenses. Existing
bug_finder import sites continue to work unchanged via this re-export.
"""

from __future__ import annotations

from augmentum.agents.guards import (
    ToolGuard,
    detector_guard,
    fixer_guard,
    planner_guard,
    role_guard,
    verifier_guard,
)

__all__ = [
    "ToolGuard",
    "detector_guard",
    "fixer_guard",
    "planner_guard",
    "role_guard",
    "verifier_guard",
]
