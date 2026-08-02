"""DeviceCommandBus — server-initiated, result-returning calls to the phone.

The notification WebSocket (``notifications_routes.subscribe_ws`` +
``NotificationHub``) is the always-on server↔phone channel. It was
push-only; this module makes it a *request/response* channel without a
second socket.

Flow for "list the phone's bluetooth devices":

  1. A companion verb calls :meth:`DeviceCommandBus.request` with an
     ``action`` ("bluetooth_list") and ``params``.
  2. The bus mints a ``request_id``, parks an ``asyncio.Future`` under it,
     and pushes ``{"type": "device_command", "request_id", "action",
     "params"}`` to the user's ``android`` connection via
     :meth:`NotificationHub.send_to_device`.
  3. The phone's foreground service executes natively and sends back
     ``{"type": "device_command_response", "request_id", "result"}`` on
     the same socket. The WS receive-loop calls :meth:`resolve`, which
     completes the future.
  4. ``request`` returns the result (or a typed error: ``not_connected``
     when the phone has no live connection, ``timeout`` when it never
     answers). It never raises — the verb degrades to "I couldn't reach
     your phone."

The correlation pattern (request_id → pending future) mirrors Fabric's
peer coordinator; the difference is the counterparty is a phone, not
another Augmentum node, so it rides the notification hub the phone
already holds open rather than the signed fabric link.

One bus per instance, lazily created on ``app_state.device_command_bus``
via :func:`get_device_bus`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Hard ceiling so a misbehaving caller can't park a future forever even
# if it passes a huge timeout; the config default is well under this.
_MAX_TIMEOUT_S = 30.0


class DeviceCommandBus:
    """Request/response over the notification WS, keyed by request_id."""

    def __init__(self, app_state: Any) -> None:
        # Hold app_state (not the hub) so we always resolve the CURRENT
        # hub — it's created lazily on first WS attach and may not exist
        # when the bus is first constructed.
        self._app_state = app_state
        self._pending: dict[str, asyncio.Future] = {}
        self._seq = 0

    def _hub(self) -> Any:
        return getattr(self._app_state, "notification_hub", None)

    async def request(
        self,
        *,
        user_id: str,
        action: str,
        params: dict[str, Any] | None = None,
        device_type: str = "android",
        timeout: float = 8.0,
    ) -> dict[str, Any]:
        """Send a command to the user's phone and await its reply.

        Returns the phone's ``result`` dict on success, or
        ``{"ok": False, "error": "not_connected" | "timeout" | "no_hub"}``
        on the failure modes. Never raises.
        """
        if not user_id or not action:
            return {"ok": False, "error": "bad_request"}
        hub = self._hub()
        if hub is None:
            return {"ok": False, "error": "no_hub"}

        self._seq += 1
        request_id = f"devcmd-{self._seq}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[request_id] = fut

        try:
            delivered = await hub.send_to_device(
                user_id=user_id,
                device_type=device_type,
                payload={
                    "type": "device_command",
                    "request_id": request_id,
                    "action": action,
                    "params": params or {},
                },
            )
            if not delivered:
                return {"ok": False, "error": "not_connected"}

            capped = max(0.1, min(float(timeout or 8.0), _MAX_TIMEOUT_S))
            try:
                result = await asyncio.wait_for(fut, timeout=capped)
            except TimeoutError:
                log.info(
                    "device_command_timeout",
                    user_id=user_id, action=action, request_id=request_id,
                )
                return {"ok": False, "error": "timeout"}
            if isinstance(result, dict):
                return result
            return {"ok": True, "result": result}
        finally:
            # Always reclaim the slot — success, timeout, or send failure.
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str, result: Any) -> bool:
        """Complete the future for ``request_id`` with the phone's result.

        Called by the WS receive-loop on a ``device_command_response``
        frame. Returns True if a waiter was found (else the request
        already timed out / was reclaimed — a late answer we drop).
        """
        fut = self._pending.get(request_id)
        if fut is None or fut.done():
            return False
        fut.set_result(result)
        return True

    def pending_count(self) -> int:
        return len(self._pending)


def get_device_bus(app_state: Any) -> DeviceCommandBus:
    """Get-or-create the per-instance bus on ``app_state``."""
    bus = getattr(app_state, "device_command_bus", None)
    if bus is None:
        bus = DeviceCommandBus(app_state)
        try:
            app_state.device_command_bus = bus
        except Exception:  # noqa: BLE001 — exotic app_state (tests) — fine
            log.debug("device_bus_attach_failed", exc_info=True)
    return bus
