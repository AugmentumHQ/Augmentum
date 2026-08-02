"""Controller framework for the Augmentum Experience Framework.

Per-system canonical button layouts (NES, SNES, GBA, N64, PSX,
Genesis, Master System, GB/GBC, Atari 2600, PCE, Arcade, ...) +
per-user remap overrides resolved at runtime. The canonical layouts
ship as code (``defaults.py``); the user-specific overrides live in
the ``controller_remaps`` table.

Engine adapters (EmulatorJS browser runtime, future server-streamed
RetroArch) consume the resolved layout via the launch handle's
metadata, so the same remap drives every engine the user might pick.
"""

from __future__ import annotations

from augmentum.controllers.defaults import (
    SYSTEM_PROFILES,
    SystemProfile,
    get_system_profile,
    list_systems,
)
from augmentum.controllers.service import (
    ControllerService,
    ResolvedLayout,
)
from augmentum.controllers.store import ControllerRemap, ControllerStore

__all__ = [
    "ControllerRemap",
    "ControllerService",
    "ControllerStore",
    "ResolvedLayout",
    "SYSTEM_PROFILES",
    "SystemProfile",
    "get_system_profile",
    "list_systems",
]
