"""Deprecated emulator surface adapter.

NOTICE:
    The original design intent here was an in-process adapter that
    would write to ``/tmp/selkies_js{0..3}.sock`` and read frames from
    the Xvfb display the WebRTC encoder uses. That design does not
    work: ``selkies-gamepad-bridge.py`` is the *reader* on those
    sockets and Selkies-gstreamer is the *writer*, so there is no
    injection endpoint -- and the augmentum host process has no
    access to the container's Xvfb display anyway.
    Replaced by an in-container ``agent-bridge.py`` daemon that dials
    augmentum's bridge WebSocket as a client and owns its own UInput
    gamepad + ``gst-launch ximagesrc`` frame capture. The route layer
    accepts ``surface: "emulator"`` and constructs a regular
    :class:`BridgedAdapter` over the dialled-in WS, identical to
    ``emulatorjs``. See:
      * ``services/game-stream/scripts/agent-bridge.py``
      * ``services/game-stream/scripts/entrypoint-base.sh`` (launcher)
      * ``augmentum/proxy/game_agent_routes.py`` (route)
    Removal condition: any code path that constructs ``EmulatorAdapter``
    is removed. Today nothing inside augmentum does; the class only
    exists so the historical export from :mod:`augmentum.game_agent`
    keeps resolving.
"""

from __future__ import annotations

from augmentum.game_agent.schema import SurfaceCapsPayload
from augmentum.game_agent.semantic import SemanticInputResolver
from augmentum.game_agent.surfaces.base import EmitEventFn


class EmulatorAdapter:
    """Removed scaffold; do not instantiate.

    Kept as a typed placeholder so the re-export from
    :mod:`augmentum.game_agent.surfaces` and :mod:`augmentum.game_agent`
    keeps resolving while downstream code (none in-tree today) gets a
    deprecation cycle. The streamed-emulator surface now flows through
    :class:`BridgedAdapter` -- see the module docstring for the path.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "EmulatorAdapter has been replaced. Streamed emulator sessions "
            "now route through BridgedAdapter; the in-container "
            "agent-bridge.py daemon dials the bridge WS, owns /dev/uinput, "
            "and captures Xvfb frames. See augmentum/game_agent/surfaces/"
            "emulator.py module docstring for the migration path."
        )

    @property
    def resolver(self) -> SemanticInputResolver:  # pragma: no cover -- never reached
        raise NotImplementedError

    def caps(self) -> SurfaceCapsPayload:  # pragma: no cover -- never reached
        raise NotImplementedError

    async def start(self, emit: EmitEventFn) -> None:  # pragma: no cover
        raise NotImplementedError

    async def stop(self) -> None:  # pragma: no cover
        raise NotImplementedError

    async def snapshot_frame(self) -> bytes | None:  # pragma: no cover
        raise NotImplementedError
