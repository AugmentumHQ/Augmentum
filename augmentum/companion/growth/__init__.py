"""Companion growth loop — self-improvement cycle for the companion.

Spec: ``docs/superpowers/specs/2026-05-31-companion-growth-loop-design.md``
Action catalog: ``docs/superpowers/specs/2026-05-31-companion-action-catalog.md``

The growth loop is the companion's coder-loop-shaped agent cycle for
working on herself — refining skills, calibrating affect-reading,
consolidating memory, anticipating user needs. Distinct from chat
dispatch per ``project_companion_tool_tree_separate`` memory.

Phase 1 (this package): substrate + one anchor action (Recall) end-to-end.
Subsequent phases per the spec's phasing table.
"""

from __future__ import annotations

from augmentum.companion.growth.economy import Economy
from augmentum.companion.growth.session import CompanionGrowthSession
from augmentum.companion.growth.store import (
    BacklogItem,
    EconomyAccount,
    EconomyTx,
    GrowthLogEntry,
    GrowthStore,
)

__all__ = [
    "BacklogItem",
    "CompanionGrowthSession",
    "Economy",
    "EconomyAccount",
    "EconomyTx",
    "GrowthLogEntry",
    "GrowthStore",
]
