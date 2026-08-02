"""Action-handler glue between the notification substrate and the offer catalog.

The notification substrate (``augmentum/notifications/actions.py``)
matches a notification's ``channel_id`` to a handler. We register one
handler against the ``system.offer`` channel; it inspects the
notification's ``payload.kind`` + ``target_id`` and dispatches to the
right ``CatalogEntry.accept`` (for the Install action) or to the
suppression store (for Snooze / Never).

``register_offer_action_handler`` is called once at app startup from
``server.py`` (next to the Connect / Coder handler registrations).
"""

from __future__ import annotations

from .system_offer import (
    handle_offer_action,
    register_offer_action_handler,
)

__all__ = [
    "handle_offer_action",
    "register_offer_action_handler",
]
