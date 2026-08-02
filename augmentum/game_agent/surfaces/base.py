"""SurfaceAdapter Protocol -- the contract every adapter implements.

A surface adapter is the only game-specific code in the game-agent
package. It owns:

* declaring its semantic-input vocabulary + log schema
  (:meth:`SurfaceAdapter.caps`),
* binding semantic ids to wire-format input emitters
  (:attr:`SurfaceAdapter.resolver`),
* producing observations into the orchestrator's event sink
  (:meth:`SurfaceAdapter.start`),
* yielding a fresh frame on demand when vision is budgeted
  (:meth:`SurfaceAdapter.snapshot_frame`).

The Protocol is the *only* type the orchestrator references; adapters
do not subclass anything. Wire your adapter as a regular class with
matching method signatures.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeAlias

from augmentum.game_agent.schema import EventPayload, SurfaceCapsPayload
from augmentum.game_agent.semantic import SemanticInputResolver

EmitEventFn: TypeAlias = Callable[[EventPayload], Awaitable[None]]
"""Adapter -> orchestrator observation channel.

The orchestrator passes one of these into :meth:`SurfaceAdapter.start`;
the adapter awaits it whenever an observation arrives from the
underlying surface. The orchestrator handles log persistence, rule
evaluation, and slow-path novelty signaling -- the adapter does not
need to know any of that.
"""


class SurfaceAdapter(Protocol):
    """Every concrete adapter implements this."""

    @property
    def resolver(self) -> SemanticInputResolver:
        """The binding registry the orchestrator uses to apply actions.

        Adapters expose this so the orchestrator can call
        :meth:`SemanticInputResolver.apply` for plan + rule actions.
        """

        ...

    def caps(self) -> SurfaceCapsPayload:
        """Declare what this adapter accepts and produces.

        Called once at session start. The orchestrator writes a
        :class:`SurfaceCapsEntry` so the slow-path agent (and replay
        tooling) can read the active vocabulary.
        """

        ...

    async def start(self, emit: EmitEventFn) -> None:
        """Begin producing observations.

        Spawn whatever background work is needed (WebSocket connection,
        DOM event hookup, screen-capture loop, ...) and return promptly.
        Continue producing observations until :meth:`stop` is called.

        ``emit`` may be awaited from any task the adapter creates.
        """

        ...

    async def stop(self) -> None:
        """Tear down. Must be idempotent."""

        ...

    async def snapshot_frame(self) -> bytes | None:
        """Return the latest frame as PNG bytes, or ``None`` if vision is
        not wired for this adapter.

        The orchestrator only calls this when ``frame`` is in the
        ``observation_modalities`` declared by :meth:`caps`. Adapters
        without vision should return ``None`` here; calling it on such
        an adapter must not raise.
        """

        ...
