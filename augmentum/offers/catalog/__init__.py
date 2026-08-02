"""Per-kind offer catalogs.

Each kind (``mcp_server``, ``power``, ``mode_switch``, …) is a Python
module that registers its entries with ``register_kind`` from ``base``.
The module is imported once at package init time so its entries land
in the global registry; nothing further is needed at runtime to make
them available to the dispatcher.

v1 ships only the MCP-server kind with a single Gmail reference entry
so the substrate is exercisable end-to-end. Phases 2+ fill in the
other kinds.
"""

from __future__ import annotations

# Importing the per-kind modules registers their entries as a
# side-effect. The list grows as catalog kinds land in phases 2-4.
from . import (  # noqa: F401  -- register on import
    gated_tool,
    knowledge_packs,
    mcp_clients,
    mcp_servers,
    memory_save,
    mode_switch,
    model_swap,
    powers,
    setting_tweak,
    subagent_roles,
    workspace_profiles,
)

__all__ = [
    "gated_tool",
    "knowledge_packs",
    "mcp_clients",
    "mcp_servers",
    "memory_save",
    "mode_switch",
    "model_swap",
    "powers",
    "setting_tweak",
    "subagent_roles",
    "workspace_profiles",
]
