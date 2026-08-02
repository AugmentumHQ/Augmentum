"""Universal control schema for game agents.

Three-layer composition lets the slow-path agent speak ONE vocabulary
across every surface we control (browser games, emulators, scripted
servers, captured desktops):

* :mod:`actions` (Layer 1) — what the agent THINKS. Canonical
  surface-agnostic verbs: ``confirm``, ``cancel``, ``nav_up``,
  ``interact``, etc. The same name means the same intent everywhere.
* :mod:`controllers` (Layer 2) — what the device CAN DO. SDL-style
  positional inputs: ``face_south``, ``dpad_up``, ``shoulder_l``.
  Stable across console families because they describe physical
  geometry, not letters (the A button is ``face_east`` on a GBA,
  ``face_south`` on a SNES; ``face_east`` is Circle on PlayStation).
* :mod:`profile` (Layer 3 wire format) — what the bridge actually
  sends. libretro joypad button id, KeyboardEvent.code, Lua RPC
  payload, etc. The :class:`ComposedProfile` holds both mappings and
  resolves Layer 1 → Layer 3 in one call.

Architecture summary::

                       agent emits "confirm"
                                │
                                ▼
                  game_profile (pokemon_rs.json)
                                │  Layer 1 → Layer 2
                                ▼  ("confirm" → "face_east")
                                │
                                ▼
                controller_profile (gba.json)
                                │  Layer 2 → Layer 3
                                ▼  ("face_east" → libretro id 8)
                                │
                                ▼
                  BridgedAdapter sends {"action": "confirm",
                                        "wire_code": 8, ...}

Why three layers and not two
----------------------------
A naive two-layer design (action → wire) loses portability: PSX
"confirm" is button id 0 (Cross), GBA "confirm" is id 8 (A). Without
an abstract controller layer in between, every game ships a
controller-specific table, and a new console means re-editing every
game profile.

By splitting the abstraction:
* Per-controller knowledge lives in ONE place per device.
* Per-game knowledge lives in ONE place per game.
* Adding a new console = one controller JSON.
* Adding a new game = one game JSON.
* Cross-console games (Pokémon Diamond on DS, FRLG on GBA) share
  most of the game profile -- only the controller binding differs.

Industry sources for this pattern: SDL2 ``SDL_GameController``,
Steam Input action sets, W3C Gamepad API standard mapping. We are
not inventing; we are choosing a well-validated structure.

Public surface
--------------
* :data:`UNIVERSAL_ACTIONS` — canonical Layer-1 action names.
* :data:`CONTROLLER_INPUTS` — canonical Layer-2 input names.
* :class:`ControllerProfile`, :class:`GameProfile`,
  :class:`ComposedProfile` — typed data wrappers.
* :class:`ProfileRegistry` — loads JSON profile files from
  ``control/profiles/`` and composes them on demand.
* :data:`default_registry` — module-level registry pre-loaded with the
  shipped profiles. Use for production code; tests instantiate their
  own ``ProfileRegistry()`` to avoid coupling.
"""

from __future__ import annotations

from augmentum.game_agent.control.actions import (
    ACTION_DESCRIPTIONS,
    UNIVERSAL_ACTIONS,
)
from augmentum.game_agent.control.controllers import CONTROLLER_INPUTS
from augmentum.game_agent.control.profile import (
    ComposedProfile,
    ControllerProfile,
    GameProfile,
)
from augmentum.game_agent.control.registry import ProfileRegistry, default_registry

__all__ = [
    "ACTION_DESCRIPTIONS",
    "CONTROLLER_INPUTS",
    "ComposedProfile",
    "ControllerProfile",
    "GameProfile",
    "ProfileRegistry",
    "UNIVERSAL_ACTIONS",
    "default_registry",
]
