"""js13k surface adapter (scaffold).

Targets HTML5 games served from an in-page ``<iframe>`` -- typically
the catalog in ``augmentum/games/providers/js13k.py``. The Python
adapter on its own cannot reach inside an iframe; it talks to a thin
**browser-side shim** over WebSocket that:

1. Listens for postMessage from the game iframe (the shim has injected
   a small instrumentation script at iframe-load time that posts
   game state when the game emits it -- typically via
   ``window.parent.postMessage(...)``).
2. Hooks ``KeyboardEvent`` and ``MouseEvent`` so the iframe receives
   synthetic events when the adapter says "press jump".
3. Periodically reads the iframe canvas to PNG (``canvas.toDataURL``)
   when the adapter requests a frame.

Wire protocol (js13k.v1)
------------------------
Browser shim -> Python adapter::

    {"kind": "event", "data": {...arbitrary postMessage payload...}}
    {"kind": "dom_mutation", "data": {"selector": "...", "added": 3, "removed": 0}}
    {"kind": "console", "data": {"level": "log", "text": "..."}}
    {"kind": "canvas_hash", "data": {"hash": "...", "dirty_regions": [[x,y,w,h], ...]}}

Python adapter -> browser shim::

    {"action": "key", "code": "Space", "duration_ms": 200}
    {"action": "mouse", "button": "left", "x": 320, "y": 240, "duration_ms": 50}
    {"action": "request_frame"}

Implementation status
---------------------
This module currently:

* Declares :meth:`caps` correctly.
* Binds semantic-input names against a binding table the caller
  supplies; the resolver no-ops on emit until the WebSocket is wired.
* :meth:`start` raises ``NotImplementedError`` if no wire is
  configured -- a deliberate trip-wire so a half-built deployment
  doesn't pretend to play games.

TODO: wire WebSocket transport (probably hono/h3 endpoint mounted in
``augmentum/proxy/game_agent_routes.py``); wire browser shim
distribution (probably a static script under ``ui/scripts/agent/``).
"""

from __future__ import annotations

from collections.abc import Mapping

from augmentum.game_agent.schema import SurfaceCapsPayload
from augmentum.game_agent.semantic import SemanticInputResolver
from augmentum.game_agent.surfaces.base import EmitEventFn

# Default observable channels for a js13k surface. Frames are available
# via canvas readback but cost a postMessage round-trip per pull.
_DEFAULT_MODALITIES: tuple[str, ...] = ("log", "frame")


class Js13kAdapter:
    """js13k iframe surface adapter (scaffold)."""

    def __init__(
        self,
        *,
        semantic_to_key: Mapping[str, str],
        ws_url: str | None = None,
        log_schema: str = "js13k.v1",
    ) -> None:
        """Construct an adapter.

        Parameters
        ----------
        semantic_to_key:
            Per-game binding table, e.g.
            ``{"jump": "Space", "left": "ArrowLeft", ...}``. Values
            are :class:`KeyboardEvent.code` strings.
        ws_url:
            WebSocket endpoint of the browser shim. When ``None``, the
            adapter is in scaffold mode and :meth:`start` will raise.
        """

        self._semantic_to_key = dict(semantic_to_key)
        self._ws_url = ws_url
        self._log_schema = log_schema
        self._resolver = SemanticInputResolver()
        for semantic, _key in self._semantic_to_key.items():
            self._resolver.bind(semantic, self._make_resolver(semantic))

    @property
    def resolver(self) -> SemanticInputResolver:
        return self._resolver

    def caps(self) -> SurfaceCapsPayload:
        return SurfaceCapsPayload(
            semantic_inputs=list(self._semantic_to_key.keys()),
            log_schema=self._log_schema,
            observation_modalities=list(_DEFAULT_MODALITIES),  # type: ignore[arg-type]
        )

    async def start(self, emit: EmitEventFn) -> None:
        if self._ws_url is None:
            raise NotImplementedError(
                "Js13kAdapter: ws_url is None; the WebSocket bridge to the "
                "browser shim is not yet wired. Provide ws_url to enable."
            )
        # TODO: open websockets connection to self._ws_url, demux
        # incoming JSON frames into EventPayload(channel="log", data=...)
        # entries and await emit(...) for each.
        raise NotImplementedError("Js13kAdapter wire transport TODO")

    async def stop(self) -> None:
        # TODO: close websocket; cancel background read task.
        return None

    async def snapshot_frame(self) -> bytes | None:
        # TODO: send {"action": "request_frame"} over the websocket and
        # await the canvas PNG. Return None until wired.
        return None

    # ── Internals ─────────────────────────────────────────────────

    def _make_resolver(self, semantic: str):  # type: ignore[no-untyped-def]
        async def _resolver(duration_ms: int) -> None:
            # TODO: send {"action": "key", "code": self._semantic_to_key[semantic],
            #             "duration_ms": duration_ms} over the websocket.
            _ = duration_ms  # noqa: F841
            _ = self._semantic_to_key[semantic]

        return _resolver
