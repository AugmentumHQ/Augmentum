"""``system.offer`` notification action handler.

Wires the existing notification action-callback path (one handler per
channel pattern) into the offer catalog. Three actions are valid on
every offer notification:

* ``accept`` → run the catalog entry's ``accept`` callable, return its
  result. The notification was already marked read by the route layer;
  we additionally dismiss it on success so the chip flips to "✓
  Installed" state without lingering as pending.
* ``snooze`` ("Not now") → dismiss THIS chip only. No suppression row.
* ``never`` → write a permanent suppression row, dismiss the chip.

"Not now" used to write a 30-day suppression row, which made one tap on
one device mute that capability everywhere for a month — and the chat
fallback then invited a retry that could never succeed (a silent dead
end; see migration 326). A per-turn decline must not carry per-month
consequences. ``never`` remains the only user-driven suppression, because
that word actually means it and is undoable from Settings → Offers.

The handler is admin-scope-aware: an admin-scoped offer surfaced
to a non-admin user returns 403 from the route layer (we re-check
here so direct callers can't bypass it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.notifications.actions import register_action_handler
from augmentum.notifications.store import dismiss as _dismiss_notification
from augmentum.offers.catalog.base import get_entry
from augmentum.offers.store import never
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from augmentum.notifications.store import Notification


log = get_logger(__name__)


OFFER_CHANNEL_PATTERN: str = "system.offer"


def _resolve_conn(request: Request):
    sm = getattr(request.app.state, "state_manager", None)
    if sm is not None and isinstance(getattr(sm, "backend", None), SQLiteBackend):
        return sm.backend.conn
    return None


def _is_admin(request: Request) -> bool:
    """True when the request's authenticated user has admin role.

    The ``user`` object on ``request.scope`` is populated by the auth
    middleware; an absent user is non-admin by definition.
    """

    user = request.scope.get("user")
    if user is None:
        return False
    return bool(getattr(user, "is_admin", False))


async def handle_offer_action(
    notification: Notification, action_id: str, request: Request,
) -> dict[str, Any]:
    """Dispatch ``accept`` / ``snooze`` / ``never`` for an offer notification."""

    payload = notification.payload or {}
    kind = str(payload.get("kind") or "")
    target_id = str(payload.get("target_id") or "")
    scope = str(payload.get("scope") or "user")

    if not kind or not target_id:
        return {
            "ok": False,
            "error": "malformed_offer",
            "detail": "notification payload missing kind / target_id",
        }

    conn = _resolve_conn(request)
    if conn is None:
        return {
            "ok": False,
            "error": "no_backend",
            "detail": "offer actions require a SQLite backend",
        }

    user_id = notification.user_id

    # "Not now" dismisses this one chip and nothing else — no suppression
    # row, so the same offer may surface again next time it's relevant.
    if action_id == "snooze":
        await _dismiss_notification(
            conn, user_id=user_id, notification_id=notification.notification_id,
        )
        return {"ok": True, "action": "snooze", "suppressed": False}

    # Never is the only user-driven suppression write.
    if action_id == "never":
        await never(conn, user_id=user_id, kind=kind, target_id=target_id)
        await _dismiss_notification(
            conn, user_id=user_id, notification_id=notification.notification_id,
        )
        return {"ok": True, "action": "never"}

    if action_id != "accept":
        return {
            "ok": False,
            "error": "unknown_action",
            "detail": f"action {action_id!r} is not valid for an offer",
        }

    # Accept path — auth-scope gate then catalog dispatch.
    if scope == "admin" and not _is_admin(request):
        return {
            "ok": False,
            "error": "forbidden",
            "detail": "admin scope required to accept this offer",
        }

    entry = get_entry(kind, target_id)
    if entry is None:
        return {
            "ok": False,
            "error": "unknown_target",
            "detail": f"{kind!r}/{target_id!r} not in catalog",
        }
    if entry.accept is None:
        return {
            "ok": False,
            "error": "no_accept_handler",
            "detail": f"catalog entry for {kind!r}/{target_id!r} has no accept",
        }

    try:
        result = await entry.accept(payload, request)
    except Exception as exc:
        log.warning(
            "offer_accept_failed",
            kind=kind, target_id=target_id, user_id=user_id,
            error=str(exc)[:200],
        )
        return {
            "ok": False,
            "error": "accept_failed",
            "detail": str(exc)[:200],
        }

    # On success, also dismiss so the chip transitions to its
    # post-accept state without staying in the pending feed.
    if isinstance(result, dict) and result.get("ok", True):
        await _dismiss_notification(
            conn, user_id=user_id, notification_id=notification.notification_id,
        )

    return result if isinstance(result, dict) else {"ok": True, "result": result}


def register_offer_action_handler() -> None:
    """Register against the ``system.offer`` channel.

    Idempotent — re-registering replaces the previous binding cleanly.
    Called from ``server.py`` startup alongside the Connect / Coder
    handlers.
    """

    register_action_handler(OFFER_CHANNEL_PATTERN, handle_offer_action)
