"""Augmentum loops — shared agent-loop substrate.

Phase 2 of the Integrated Coding Nervous System spec extracts the
coder's loop machinery into a reusable LoopRunner. Three intensity
tiers — Light, Medium, Heavy — control how aggressively the runner
plans, observes, breakers-out, and verifies before terminating.

PR-2.1 (this file set) is additive substrate only: it defines the
shapes that PR-2.2 ... PR-2.6 will fill in. Coder + agentic
behavior is unchanged until PR-2.4 (coder thin-caller) and PR-2.6
(agentic retarget) land.

See docs/superpowers/specs/2026-05-29-integrated-coding-nervous-system.md.
"""

from __future__ import annotations

from augmentum.loops.ledger import (
    FAILURE_LEDGER_TTL_SECONDS,
    TRACKED_TOOLS_BY_COMMAND,
    TRACKED_TOOLS_BY_PATH,
    TRACKED_TOOLS_BY_QUERY,
    ObservationLedger,
)
from augmentum.loops.protocols import (
    ChunkEmitter,
    FileIO,
    PermissionGate,
    QuestionAsker,
    ToolExecutor,
)
from augmentum.loops.runner import (
    LoopRunner,
    RunResult,
    RunStatus,
)
from augmentum.loops.tier import (
    HEAVY,
    LIGHT,
    MEDIUM,
    BreakerSet,
    Intensity,
)

__all__ = [
    "BreakerSet",
    "ChunkEmitter",
    "FAILURE_LEDGER_TTL_SECONDS",
    "FileIO",
    "HEAVY",
    "Intensity",
    "LIGHT",
    "LoopRunner",
    "MEDIUM",
    "ObservationLedger",
    "PermissionGate",
    "QuestionAsker",
    "RunResult",
    "RunStatus",
    "TRACKED_TOOLS_BY_COMMAND",
    "TRACKED_TOOLS_BY_PATH",
    "TRACKED_TOOLS_BY_QUERY",
    "ToolExecutor",
]
