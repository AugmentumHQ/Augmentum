"""Agent role spec — user-defined or built-in.

A role bundles model selection (preferred + fallback chain), tool subset,
budget, guards, context-bridge mode, visibility, and permissions into a
named contract that the lead model invokes via ``task_dispatch``.

Roles arrive from two sources:

* **Built-in roles** declared in ``presets.py`` (explore, plan, review,
  research). Always available, no filesystem dependency.
* **User-defined roles** discovered via ``registry.py`` from
  ``.augmentum/agents/*.md`` (workspace-local) or
  ``~/.augmentum/agents/*.md`` (user-global). Workspace-local wins on
  name collision.

Loose-typed dict fields (``model``, ``budget``) accept whatever a role
file passes; ``AgentRegistry`` validates + materializes on load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from augmentum.agents.budget import SubagentBudget


# Per-role context-bridge modes. See ``context_bridge.py`` for what each
# inherits from the parent turn.
CONTEXT_MODES = ("slim", "workspace", "hot")


@dataclass(frozen=True)
class AgentRole:
    """A named subagent contract loaded from a role file or built-in.

    Frozen so dispatcher caches can use roles as dict keys without worry.
    Lists become tuples in ``__post_init__`` for the same reason.
    """

    name: str
    description: str = ""
    system_prompt: str = ""

    preferred_model: str = ""
    """Spec string — see ``resolve.parse_model_spec``."""

    fallback_models: tuple[str, ...] = ()

    tools: tuple[str, ...] = ()
    """Tool name allow-list AFTER resolving any ``read_only`` preset."""

    budget: SubagentBudget = field(default_factory=SubagentBudget)

    tool_guard: str = "detector"
    """Name of a guard from ``guards.py`` — detector, fixer, verifier,
    planner. Use ``"none"`` to disable guarding entirely (not recommended
    for user-supplied roles)."""

    context_mode: str = "workspace"
    """One of ``CONTEXT_MODES``. See ``context_bridge.py``."""

    stream_to_parent: bool = True
    """If True, dispatcher emits progress events for the UI; if False,
    only the final result surfaces."""

    visible_in_ui: bool = True
    log_persistence: bool = True

    can_spawn_subagents: bool = False
    """If False, ``task_dispatch`` is excluded from this role's tool
    set even when listed — prevents accidental recursion."""

    max_concurrent: int = 4
    """Cap on how many concurrent instances of this role can run under
    one parent turn. Larger spawns queue."""

    source: str = "builtin"
    """``"builtin"`` | ``"workspace"`` | ``"user"``. Audit hint surfaced
    in the UI so users can tell where a role came from."""

    file_path: str = ""
    """Absolute path to the source markdown file (empty for built-ins)."""

    mtime: float = 0.0
    """Source mtime for hot-reload (0.0 for built-ins)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Pass-through bucket for forward-compat role-file fields the
    runtime doesn't yet interpret. Surfaced verbatim in the role API
    response so future UI panels can render them."""

    def to_api_dict(self) -> dict[str, Any]:
        """Serializable form for the role list endpoint."""
        return {
            "name": self.name,
            "description": self.description,
            "preferred_model": self.preferred_model,
            "fallback_models": list(self.fallback_models),
            "tools": list(self.tools),
            "tool_guard": self.tool_guard,
            "context_mode": self.context_mode,
            "stream_to_parent": self.stream_to_parent,
            "visible_in_ui": self.visible_in_ui,
            "log_persistence": self.log_persistence,
            "can_spawn_subagents": self.can_spawn_subagents,
            "max_concurrent": self.max_concurrent,
            "budget": {
                "max_iterations": self.budget.max_iterations,
                "max_wallclock_seconds": self.budget.max_wallclock_seconds,
                "max_tokens": self.budget.max_tokens,
            },
            "source": self.source,
            "file_path": self.file_path,
            "extra": dict(self.extra),
        }
