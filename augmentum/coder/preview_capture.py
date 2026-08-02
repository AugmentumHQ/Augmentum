"""Live-preview screenshot broker.

Heavy WebGL/Three.js pages render in a few seconds in the user's real-GPU
browser but 6-45s+ (or time out) in the coder's headless, GPU-less workspace
Chromium. When the user has the coder preview open, we'd rather capture the
frame their GPU already rendered than re-render it headless.

This module is the server-side rendezvous between two ends that never call each
other directly:

- the ``/ws/coder/preview-capture/{workspace_id}`` WebSocket, held open by the
  coder UI whenever a preview is live (its mere presence == "a GPU frame is
  available for this workspace"), and
- ``browser_screenshot`` running inside a coder turn, which asks the broker for
  a frame and, if one comes back in time, uses it instead of spawning Playwright.

Flow: the tool calls :meth:`capture`, which sends a ``{type:'capture', id, url}``
message down the socket and awaits a future. The UI relays the request into the
preview iframe (a proxy-injected agent does ``canvas.toDataURL()`` on the user's
GPU), relays the data URL back up the socket, and the WS handler calls
:meth:`resolve` to complete the future. Everything runs in the one app event
loop, so a plain ``asyncio.Future`` is the whole synchronization primitive.

If no socket is registered, or the frame doesn't arrive before the timeout, or
the page has no capturable canvas, :meth:`capture` returns ``None`` and the
caller falls back to the (now graceful) headless path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# A registered sender: an async callable that pushes one JSON message to the
# coder UI over its preview-capture WebSocket.
Sender = Callable[[dict[str, Any]], Awaitable[None]]


class PreviewCaptureBroker:
    """Per-process registry mapping a workspace to its live preview socket,
    plus the pending capture futures keyed by request id."""

    def __init__(self) -> None:
        # Only one live preview per workspace at a time (the user's open tab);
        # a re-open replaces the prior sender.
        self._senders: dict[str, Sender] = {}
        self._pending: dict[str, asyncio.Future] = {}
        self._counter = 0

    # --- socket lifecycle (called by the WS route) -----------------------

    def register(self, workspace_id: str, sender: Sender) -> None:
        self._senders[workspace_id] = sender
        log.info("preview_capture.register", workspace_id=workspace_id)

    def unregister(self, workspace_id: str, sender: Sender) -> None:
        # Guard against a stale socket unregistering a newer one that already
        # replaced it.
        if self._senders.get(workspace_id) is sender:
            self._senders.pop(workspace_id, None)
            log.info("preview_capture.unregister", workspace_id=workspace_id)

    def is_connected(self, workspace_id: str) -> bool:
        return workspace_id in self._senders

    # --- request / response ---------------------------------------------

    async def capture(
        self, workspace_id: str, *, url: str = "", timeout: float = 8.0
    ) -> dict[str, Any] | None:
        """Ask the live preview for a frame. Returns the result envelope
        ``{data_url, width, height}`` or ``None`` (no socket / timed out /
        no canvas / error). Never raises."""
        sender = self._senders.get(workspace_id)
        if sender is None:
            return None
        self._counter += 1
        capture_id = f"{workspace_id}:{self._counter}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[capture_id] = fut
        try:
            await sender({"type": "capture", "id": capture_id, "url": url})
        except Exception as exc:  # socket died between check and send
            self._pending.pop(capture_id, None)
            log.warning("preview_capture.send_failed", workspace_id=workspace_id, error=str(exc))
            return None
        try:
            result = await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            return None
        except Exception:
            return None
        finally:
            self._pending.pop(capture_id, None)
        if not isinstance(result, dict) or not result.get("data_url"):
            return None
        return result

    def resolve(self, capture_id: str, payload: dict[str, Any]) -> None:
        """Complete the future for ``capture_id`` (called by the WS route when
        the browser relays a result). No-op if unknown/already done."""
        fut = self._pending.get(capture_id)
        if fut is not None and not fut.done():
            fut.set_result(payload)


# Process-wide singleton — imported by both the WS route and browser.py.
broker = PreviewCaptureBroker()
