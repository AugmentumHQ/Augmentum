"""Luanti surface adapter (scaffold).

Targets a Luanti server with the Augmentum agent mod installed. The
mod exposes a Lua-side WebSocket that publishes structured events
(player position, inventory deltas, chat, entity spawns, block
changes, HUD updates) and accepts RPCs from this adapter (move,
look_at, dig, place, chat, hotbar select, ...).

Luanti is the only one of our four surfaces with a privileged API,
which is what makes its profile ``scriptable=True`` in
``augmentum/game_stream/profiles.py``. The agent benefits from the
richer log without losing the option to fall back to vision for
unanticipated states.

Wire protocol (luanti.v1)
-------------------------
Lua mod -> Python adapter (over WS as JSON-per-frame)::

    {"event": "position", "pos": [x, y, z], "yaw": 1.57, "pitch": 0.0}
    {"event": "block_break", "pos": [...], "node": "default:stone"}
    {"event": "inventory", "items": {"default:dirt": 3, ...}}
    {"event": "chat", "from": "player1", "text": "hello"}
    {"event": "entity_seen", "kind": "mob:pig", "pos": [...]}
    {"event": "hud", "field": "hp", "value": 18}

Python adapter -> Lua mod::

    {"rpc": "move", "dir": "north", "duration_ms": 500}
    {"rpc": "look_at", "target": [x, y, z]}
    {"rpc": "dig", "duration_ms": 200}
    {"rpc": "place", "node": "default:torch"}
    {"rpc": "chat", "text": "..."}

Implementation status
---------------------
Scaffold only. ``caps()`` and resolver bindings are correct;
``start()`` raises until the WebSocket transport is wired.

TODO: ship the Lua mod (``services/game-stream/luanti-agent-mod/``)
and the WS endpoint on the Python side.
"""

from __future__ import annotations

from typing import Literal

from augmentum.game_agent.schema import SurfaceCapsPayload
from augmentum.game_agent.semantic import SemanticInputResolver
from augmentum.game_agent.surfaces.base import EmitEventFn

_DEFAULT_MODALITIES: tuple[str, ...] = ("log", "frame")

# The semantic vocabulary the Luanti mod accepts. Keep this in sync
# with the Lua-side RPC handler; the agent prompt is parameterized by
# whatever caps() returns at runtime so extending this list does not
# require prompt changes.
_DEFAULT_SEMANTICS: tuple[str, ...] = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
    "jump",
    "look_at",
    "dig",
    "place",
    "hotbar_next",
    "hotbar_prev",
    "chat",
)


class LuantiAdapter:
    """Luanti server-mod surface adapter (scaffold)."""

    def __init__(
        self,
        *,
        ws_url: str | None = None,
        log_schema: str = "luanti.v1",
        semantic_inputs: tuple[str, ...] = _DEFAULT_SEMANTICS,
        observation_modalities: tuple[Literal["log", "frame", "ocr", "memory"], ...] = _DEFAULT_MODALITIES,  # type: ignore[assignment]
    ) -> None:
        self._ws_url = ws_url
        self._log_schema = log_schema
        self._semantic_inputs = list(semantic_inputs)
        self._observation_modalities = list(observation_modalities)
        self._resolver = SemanticInputResolver()
        for semantic in self._semantic_inputs:
            self._resolver.bind(semantic, self._make_rpc(semantic))

    @property
    def resolver(self) -> SemanticInputResolver:
        return self._resolver

    def caps(self) -> SurfaceCapsPayload:
        return SurfaceCapsPayload(
            semantic_inputs=self._semantic_inputs,
            log_schema=self._log_schema,
            observation_modalities=self._observation_modalities,  # type: ignore[arg-type]
        )

    async def start(self, emit: EmitEventFn) -> None:
        if self._ws_url is None:
            raise NotImplementedError(
                "LuantiAdapter: ws_url is None; the Lua-mod WebSocket bridge "
                "is not wired. Provide ws_url to enable."
            )
        # TODO: open WebSocket; demux Lua-emitted events into
        # EventPayload(channel="log", data=event_dict) and await emit().
        raise NotImplementedError("LuantiAdapter wire transport TODO")

    async def stop(self) -> None:
        # TODO: close WebSocket.
        return None

    async def snapshot_frame(self) -> bytes | None:
        # TODO: capture from streaming container (xvfb / WebRTC keyframe).
        return None

    def _make_rpc(self, semantic: str):  # type: ignore[no-untyped-def]
        async def _resolver(duration_ms: int) -> None:
            # TODO: send {"rpc": semantic, "duration_ms": duration_ms} on WS.
            _ = duration_ms
            _ = semantic

        return _resolver
