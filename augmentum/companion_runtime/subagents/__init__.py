"""Subagent adapters — each wraps an existing mode orchestrator.

Sprint 2 lives here. One file per mode, all extending ``SubagentBase``
and registering via ``SubagentRegistry.register(...)`` at import time.
The registry is gated by ``companion_subagent_registry_active``: when
the flag is False, ``available()`` returns an empty tuple even though
the registrations exist.
"""

# Import every adapter so each class registers at package-import time.
# Imports are kept here (not eagerly in registry.py) so test code can
# selectively import individual adapters in isolation if needed.
from augmentum.companion_runtime.subagents import (  # noqa: F401
    agentic,
    analytical,
    becca_direct,
    bug_finder,
    build,
    coder,
    narrative,
    passthrough,
)
from augmentum.companion_runtime.subagents.base import (
    SubagentBase,
    SubagentContext,
    SubagentResult,
)
from augmentum.companion_runtime.subagents.registry import SubagentRegistry

__all__ = [
    "SubagentBase",
    "SubagentContext",
    "SubagentResult",
    "SubagentRegistry",
]
