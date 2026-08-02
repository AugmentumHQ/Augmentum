"""Augmentum Game Streaming Platform (AGSP).

Universal browser-delivery substrate for native games. The container
runs the game + a WebRTC streaming stack; the browser is a thin
client; the same Augmentum auth and user_id-scoping that protects the
rest of the app applies here too.

Distinct from ``augmentum/games/`` (which handles web-game *discovery*
via js13k and similar public catalogs) -- this subsystem is about
*running* a game container and piping pixels back to a browser tab.

Public surface (re-exported here for stability):

* ``GameStreamLifecycle`` -- state-machine helpers
* ``PortPool`` -- TCP/UDP port allocator for streaming + game ports
* ``GameProfile`` / ``profile_registry`` -- per-game container metadata
* ``GameStreamRuntime`` -- high-level start/stop/connect API

The container layer (Dockerfile, entrypoints) and the client-side stage
UI live outside this package -- this module is the orchestration brain.
"""

from __future__ import annotations

from augmentum.game_stream.lifecycle import (
    GameStreamLifecycle,
    LifecycleTransitionError,
    SessionStatus,
)
from augmentum.game_stream.port_pool import PortPool, PortPoolExhausted
from augmentum.game_stream.profiles import (
    GameProfile,
    ProfileRegistry,
    profile_registry,
)
from augmentum.game_stream.runtime import (
    ConcurrentStreamLimitError,
    GameStreamRuntime,
    RuntimeError as GameStreamRuntimeError,
    StubContainerAdapter,
)

__all__ = [
    "ConcurrentStreamLimitError",
    "GameProfile",
    "GameStreamLifecycle",
    "GameStreamRuntime",
    "GameStreamRuntimeError",
    "LifecycleTransitionError",
    "PortPool",
    "PortPoolExhausted",
    "ProfileRegistry",
    "SessionStatus",
    "StubContainerAdapter",
    "profile_registry",
]
