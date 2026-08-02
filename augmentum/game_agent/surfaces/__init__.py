"""Game-agent surface adapters.

Each module in this package implements the
:class:`augmentum.game_agent.surfaces.base.SurfaceAdapter` Protocol
for one class of game-delivery surface.

The mock adapter is fully wired and is what the unit tests drive. The
four real adapters (js13k, luanti, emulator, curated) ship as
scaffolds: their :meth:`caps` and :attr:`resolver` are populated, but
the wire transport (WebSocket to browser shim, Lua mod bridge,
Selkies bridge, xdotool subprocess) is a TODO that the integration
layer wires per deployment.
"""

from __future__ import annotations

from augmentum.game_agent.surfaces.base import EmitEventFn, SurfaceAdapter
from augmentum.game_agent.surfaces.bridged import BridgedAdapter
from augmentum.game_agent.surfaces.curated import CuratedAdapter
from augmentum.game_agent.surfaces.emulator import EmulatorAdapter
from augmentum.game_agent.surfaces.js13k import Js13kAdapter
from augmentum.game_agent.surfaces.luanti import LuantiAdapter
from augmentum.game_agent.surfaces.mock import MockAdapter, ScriptedEvent

__all__ = [
    "BridgedAdapter",
    "CuratedAdapter",
    "EmitEventFn",
    "EmulatorAdapter",
    "Js13kAdapter",
    "LuantiAdapter",
    "MockAdapter",
    "ScriptedEvent",
    "SurfaceAdapter",
]
