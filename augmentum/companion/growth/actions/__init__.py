"""Growth-loop action registry.

Each action category from the catalog
(``docs/superpowers/specs/2026-05-31-companion-action-catalog.md``) ships
as a handler class registered in :data:`ACTIONS` here. Phase 1 ships
``recall.surface_connection`` only; subsequent phases add the rest.

A handler implements:

  * Class attribute ``action_type`` — the registry key (e.g.
    ``recall_connect``). Backlog rows carry this in ``item_type``.
  * Class attribute ``mana_cost`` — default mana debit per dispatch.
    Session may override per request later.
  * Class attribute ``tier`` — risk gating tier (0..3). The session
    enforces the tier-gated workflow (Phase 1 is Tier 0 only).
  * Async method ``run(ctx) -> ActionResult``.

Registration is explicit (not import-side-effect): the action's module
appends to :data:`ACTIONS` at import. We import the modules from this
``__init__`` so a single ``from augmentum.companion.growth.actions
import ACTIONS`` populates the dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# ── Action-handler protocol ──────────────────────────────────────────


@dataclass(slots=True)
class ActionRequest:
    """Ad-hoc action request (no backlog row).

    Used when the user / system explicitly fires a growth session with a
    target rather than pulling from the backlog.
    """

    action_type: str
    target_ref: str = ""
    rationale: str = ""


@dataclass(slots=True)
class ActionContext:
    """Per-dispatch context passed to handlers.

    ``memory_store`` and ``growth_store`` are both optional — handlers
    that need either should check and return ``ok=False`` with a clear
    reason when absent. Keeping them optional means tests can drive
    handlers in isolation without spinning up the full app state.
    """

    user_id: str
    agent_id: str
    growth_log_id: str
    target_ref: str = ""
    rationale: str = ""
    memory_store: Any = None  # augmentum.memory.store.MemoryStore | None
    # GrowthStore handle for actions that reflect on prior growth-loop
    # work (e.g. ``narrate_growth`` reads ``list_sessions``). Optional —
    # populated by ``CompanionGrowthSession.act`` from ``session.store``.
    growth_store: Any = None  # augmentum.companion.growth.store.GrowthStore | None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionResult:
    """What a handler returns to the session loop."""

    ok: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    surface_event: dict[str, Any] | None = None
    ledger_delta: dict[str, Any] = field(default_factory=dict)
    continue_loop: bool = False
    error: str = ""


class ActionHandler(Protocol):
    """Structural type — handlers don't need to subclass this."""

    action_type: str
    mana_cost: float
    tier: int

    async def run(self, ctx: ActionContext) -> ActionResult: ...


# ── Registry ─────────────────────────────────────────────────────────


ACTIONS: dict[str, ActionHandler] = {}


def register(handler: ActionHandler) -> ActionHandler:
    """Add ``handler`` to :data:`ACTIONS` keyed by ``action_type``."""
    ACTIONS[handler.action_type] = handler
    return handler


# Import builtins so their ``register(...)`` call runs. New handlers must
# be added here too — explicit beats import-side-effect discovery when the
# package is consumed by multiple processes (CLAUDE.md / spine convention).
from augmentum.companion.growth.actions import recall as _recall  # noqa: E402, F401
from augmentum.companion.growth.actions import narrate as _narrate  # noqa: E402, F401
from augmentum.companion.growth.actions import discovery as _discovery  # noqa: E402, F401
from augmentum.companion.growth.actions import care as _care  # noqa: E402, F401
from augmentum.companion.growth.actions import offer as _offer  # noqa: E402, F401
from augmentum.companion.growth.actions import companionship as _companionship  # noqa: E402, F401


__all__ = [
    "ACTIONS",
    "ActionContext",
    "ActionHandler",
    "ActionRequest",
    "ActionResult",
    "register",
]
