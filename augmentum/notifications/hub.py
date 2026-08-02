"""NotificationHub — in-process WS registry for live feed push.

One ``NotificationHub`` per Augmentum instance, mounted on
``app.state.notification_hub``. Mirrors ``augmentum/connect/hub.py``'s
shape: a lock-protected ``{user_id: list[_Attachment]}`` map, with
attach/detach/dispatch primitives.

A "subscription" here is an attached WebSocket plus the filters the
client passed at attach time:

* ``channel_pattern``  — glob over channel_id. ``'*'`` matches all.
* ``importance_floor`` — drop events below this threshold.

Persistent subscriptions (web push endpoints, future phone APK device
tokens) live in the ``notification_subscriptions`` table and are
handled by a separate dispatcher. The hub only owns the ephemeral
WS-attached connections.

Why ephemeral-only in v1:

* WS sessions disappear on close anyway — persisting them adds churn.
* The notification feed is best-effort on disconnect; missed pushes
  re-surface on the next ``GET /api/notify/feed`` poll on reconnect.
* Persistent subscriptions are a different problem (offline delivery,
  TTL on tokens, retry semantics) — solved by the dispatcher when
  we add web push.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.notifications.store import Notification, NotificationAction
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import WebSocket


log = get_logger(__name__)


@dataclass
class _Attachment:
    """One open WS subscription for one user."""

    ws: WebSocket
    connection_id: str
    user_id: str
    channel_pattern: str
    importance_floor: int
    attached_at: float = field(default_factory=time.time)
    # Client kind, learned from the WS ``hello`` frame ("android",
    # "web", ...). Drives presence ("they're on their phone") and lets
    # the device command bus target ONLY the phone's connection — a
    # bluetooth query must not be sent to a desktop tab. Empty until a
    # hello arrives; web clients that never say hello stay "".
    device_type: str = ""


def _notification_to_dict(n: Notification) -> dict[str, Any]:
    """Wire form of one Notification — what the WS push delivers.

    Matches the shape ``GET /api/notify/feed`` returns so the UI
    can use the same parser for both.
    """

    def _action(a: NotificationAction) -> dict[str, str]:
        return a.to_dict()

    # Resolve the channel's default cue from the catalog so the client
    # knows which sound to play on arrival. Catalog-only (no DB read on
    # the hot dispatch path); per-user sound overrides are a follow-up.
    from augmentum.notifications.catalog import catalog_channel
    _template = catalog_channel(n.channel_id)
    _sound = _template.default_sound if _template else ""

    return {
        "notification_id": n.notification_id,
        "channel_id": n.channel_id,
        "source": n.source,
        "title": n.title,
        "body": n.body,
        "icon": n.icon,
        "importance": n.importance,
        "sound": _sound,
        "thread_id": n.thread_id,
        "actions": [_action(a) for a in n.actions],
        "payload": n.payload,
        "transient": n.transient,
        "expires_at": n.expires_at,
        "created_at": n.created_at,
        "updated_at": n.updated_at,
    }


class NotificationHub:
    """In-memory WS registry + per-user dispatch."""

    def __init__(self) -> None:
        self._by_user: dict[str, list[_Attachment]] = {}
        self._by_conn: dict[str, _Attachment] = {}
        self._lock = asyncio.Lock()
        # Monotonic counter — keeps connection ids greppable in logs
        # without a uuid roundtrip on every connect.
        self._next_conn_seq = 0

    async def attach(
        self, *,
        ws: WebSocket,
        user_id: str,
        channel_pattern: str = "*",
        importance_floor: int = 0,
    ) -> _Attachment:
        """Register an accepted WS for ``user_id`` with subscription filters."""

        if not user_id:
            raise ValueError("user_id is required (per-user isolation)")
        async with self._lock:
            self._next_conn_seq += 1
            conn_id = f"notif-conn-{self._next_conn_seq}"
            att = _Attachment(
                ws=ws,
                connection_id=conn_id,
                user_id=user_id,
                channel_pattern=channel_pattern or "*",
                importance_floor=max(0, int(importance_floor or 0)),
            )
            self._by_user.setdefault(user_id, []).append(att)
            self._by_conn[conn_id] = att
        log.debug(
            "notification_hub_attached",
            user_id=user_id, connection_id=conn_id,
            channel_pattern=att.channel_pattern,
            importance_floor=att.importance_floor,
        )
        return att

    async def set_device_type(self, connection_id: str, device_type: str) -> None:
        """Tag an attachment's client kind from its WS ``hello`` frame.

        Idempotent and best-effort — a missing connection (already
        detached) is a no-op, not an error.
        """
        dt = (device_type or "").strip().lower()[:32]
        if not dt:
            return
        tagged = False
        async with self._lock:
            att = self._by_conn.get(connection_id)
            if att is not None:
                att.device_type = dt
                tagged = True
        # info (not debug): this is the one signal that confirms the phone's
        # always-on socket registered as a command target. Without it,
        # send_to_device finds 0 targets and device verbs report "not connected".
        log.info(
            "notification_hub_device_tagged",
            connection_id=connection_id, device_type=dt, tagged=tagged,
        )

    def device_types(self, user_id: str) -> set[str]:
        """The set of client kinds currently attached for ``user_id``.

        ``{"android"}`` means the phone's foreground service holds a live
        connection — i.e. the user is on/near their phone right now. Read
        live off the connection registry (no staleness): a dropped socket
        removes the attachment, so presence is self-healing.
        """
        return {
            att.device_type
            for att in self._by_user.get(user_id, ())
            if att.device_type
        }

    async def send_to_device(
        self, *, user_id: str, device_type: str, payload: dict[str, Any],
    ) -> int:
        """Push one control frame to a user's connections of one kind.

        Used by the device command bus to deliver a ``device_command`` to
        the phone (and only the phone). Returns the number of successful
        sends — zero means that device kind isn't connected, which the
        caller surfaces as "your phone isn't reachable right now."
        """
        dt = (device_type or "").strip().lower()
        async with self._lock:
            user_conns = list(self._by_user.get(user_id, ()))
            targets = [att for att in user_conns if att.device_type == dt]
        if not targets:
            # Diagnostic: a device verb just failed to deliver. Show whether the
            # user has ANY connections (→ tagging/user-id problem) or none
            # (→ the phone's native service WS isn't registered for this user).
            log.warning(
                "notification_hub_no_device_target",
                user_id=user_id,
                want_device_type=dt,
                conns=len(user_conns),
                present_device_types=sorted({a.device_type for a in user_conns if a.device_type}),
            )
            return 0
        body = json.dumps(payload, separators=(",", ":"))
        delivered = 0
        for att in targets:
            try:
                await att.ws.send_text(body)
                delivered += 1
            except Exception as exc:
                log.warning(
                    "notification_hub_device_send_failed",
                    connection_id=att.connection_id,
                    user_id=user_id, device_type=dt, error=str(exc)[:160],
                )
        return delivered

    async def detach(self, connection_id: str) -> None:
        async with self._lock:
            att = self._by_conn.pop(connection_id, None)
            if att is None:
                return
            connections = self._by_user.get(att.user_id, [])
            try:
                connections.remove(att)
            except ValueError:
                pass
            if not connections:
                self._by_user.pop(att.user_id, None)
        log.debug(
            "notification_hub_detached",
            connection_id=connection_id, user_id=att.user_id,
        )

    async def dispatch(self, *, notification: Notification) -> int:
        """Push one notification to all matching WS for the recipient.

        Returns the number of successful sends. Zero is not an error
        — the user may simply be offline; the row stays in the store
        for the next poll.
        """

        async with self._lock:
            candidates = list(self._by_user.get(notification.user_id, ()))

        # Filter outside the lock — pattern + importance checks are
        # pure functions of the attachment's filters.
        eligible = [
            att for att in candidates
            if att.importance_floor <= notification.importance
            and (
                att.channel_pattern == "*"
                or fnmatch.fnmatchcase(notification.channel_id, att.channel_pattern)
            )
        ]
        if not eligible:
            return 0

        payload = json.dumps(
            {
                "type": "notification",
                "notification": _notification_to_dict(notification),
            },
            separators=(",", ":"),
        )
        delivered = 0
        for att in eligible:
            try:
                await att.ws.send_text(payload)
                delivered += 1
            except Exception as exc:
                # Best-effort — a wedged WS doesn't stop the others.
                # Detach happens via the receive-loop's finally block
                # when the WS actually closes; we just skip it here.
                log.warning(
                    "notification_hub_dispatch_failed",
                    connection_id=att.connection_id,
                    user_id=att.user_id,
                    channel_id=notification.channel_id,
                    error=str(exc)[:160],
                )
        return delivered

    async def broadcast_event(
        self, *, user_id: str, event: dict[str, Any],
    ) -> int:
        """Send a control event to ALL of a user's attached clients.

        Unlike ``dispatch`` (a new notification, filtered by each
        attachment's channel/importance), this is for cross-client state
        sync — "this notification was dismissed/read elsewhere, clear it."
        It ignores per-attachment filters because a dismiss applies to the
        notification regardless of which channels a given tab subscribed
        to. Returns the number of successful sends.
        """

        async with self._lock:
            candidates = list(self._by_user.get(user_id, ()))
        if not candidates:
            return 0
        payload = json.dumps(event, separators=(",", ":"))
        delivered = 0
        for att in candidates:
            try:
                await att.ws.send_text(payload)
                delivered += 1
            except Exception as exc:
                log.warning(
                    "notification_hub_broadcast_failed",
                    connection_id=att.connection_id,
                    user_id=user_id, error=str(exc)[:160],
                )
        return delivered

    def online_user_ids(self) -> list[str]:
        return list(self._by_user.keys())

    def connection_count(self) -> int:
        return len(self._by_conn)


async def publish_and_dispatch(
    conn: Any, *,
    hub: NotificationHub,
    user_id: str,
    channel_id: str,
    source: str,
    title: str,
    body: str = "",
    importance: int | None = None,
    dedupe_key: str = "",
    thread_id: str = "",
    actions: list[NotificationAction] | None = None,
    payload: dict[str, Any] | None = None,
    transient: bool = False,
    expires_at: int | str | None = None,
    icon: str = "",
) -> str:
    """Publish a notification and immediately fan it out to live subscribers.

    Two-tier dispatch:

      1. In-process WS push to every currently-attached client
         (handled by ``hub.dispatch``).
      2. Web Push to every persisted subscription whose filter
         matches — only attempted when the live-WS dispatch reached
         zero clients (i.e. the user is offline). This avoids
         double-notifying a desktop browser that's already attached.

    Returns the notification id. The fan-out is best-effort — a
    failed push to one device doesn't fail the publish itself; the
    persisted row is the source of truth and the next poll surfaces
    the missed event.

    Importing ``publish`` lazily keeps the import graph clean for
    callers that only need the store layer.
    """

    from augmentum.notifications.store import (
        get_notification,
    )
    from augmentum.notifications.store import (
        publish as _publish,
    )

    notification_id = await _publish(
        conn,
        user_id=user_id,
        channel_id=channel_id,
        source=source,
        title=title,
        body=body,
        importance=importance,
        dedupe_key=dedupe_key,
        thread_id=thread_id,
        actions=actions,
        payload=payload,
        transient=transient,
        expires_at=expires_at,
        icon=icon,
    )
    notification = await get_notification(
        conn, user_id=user_id, notification_id=notification_id,
    )
    if notification is None:
        return notification_id

    live_delivered = 0
    try:
        live_delivered = await hub.dispatch(notification=notification)
    except Exception as exc:
        log.warning(
            "notification_dispatch_failed",
            notification_id=notification_id,
            error=str(exc)[:160],
        )

    # Web Push fan-out policy (2026-06-11 revision):
    #
    #   * importance >= HIGH — push to ALL subscriptions, ALWAYS, in
    #     addition to live WS. The old "skip when a WS client got it"
    #     gate had a hole: a tab open on the desk at home counted as
    #     "delivered", so the locked phone in your pocket stayed
    #     silent for a tornado warning. Urgent things go everywhere —
    #     desktop banner AND phone buzz is correct for urgent (the
    #     native-ecosystem behavior); the service worker dedupes by
    #     tag so one device never double-renders.
    #   * below HIGH — original offline-only behavior: skip push when
    #     a live WS client already rendered it, so routine events
    #     don't buzz the phone while you're sitting at the desktop.
    #   * transient toasts never push, at any importance.
    from augmentum.notifications.catalog import IMPORTANCE_HIGH
    always_push = notification.importance >= IMPORTANCE_HIGH
    if (always_push or live_delivered == 0) and not notification.transient:
        try:
            await _dispatch_webpush(
                conn, notification=notification,
            )
        except Exception as exc:
            log.warning(
                "notification_webpush_dispatch_failed",
                notification_id=notification_id,
                error=str(exc)[:160],
            )

    return notification_id


async def _dispatch_webpush(
    conn: Any, *, notification: Notification,
) -> int:
    """Send Web Push to every matching subscription for the recipient.

    Returns the number of subscriptions we successfully pushed to.
    Subscriptions that come back 404/410 are deleted (the user
    revoked permission or uninstalled the browser).
    """

    import asyncio
    import fnmatch
    import json as _json

    from augmentum.notifications.webpush import (
        ensure_vapid_keys,
        send_webpush,
    )

    cur = await conn.execute(
        "SELECT subscription_id, channel_pattern, target_address, "
        "importance_floor "
        "FROM notification_subscriptions "
        "WHERE user_id = ? AND target_kind = 'webpush'",
        (notification.user_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    if not rows:
        return 0

    vapid = await ensure_vapid_keys(conn)

    # Build the push payload once — same content for every device.
    push_payload = {
        "notification_id": notification.notification_id,
        "channel_id": notification.channel_id,
        "title": notification.title,
        "body": notification.body,
        "icon": notification.icon,
        "importance": notification.importance,
        "thread_id": notification.thread_id,
        # Hint to the SW about which action buttons exist + their
        # ids; the SW renders them via NotificationOptions.actions
        # when the platform supports it.
        "actions": [
            {"action": a.id, "title": a.label}
            for a in notification.actions
        ][:2],  # Spec caps at 2 actions per notification.
        "payload": notification.payload,
    }

    delivered = 0
    expired_ids: list[str] = []
    for sub_id, pattern, addr, floor in rows:
        try:
            floor_i = int(floor or 0)
        except (TypeError, ValueError):
            floor_i = 0
        if floor_i > notification.importance:
            continue
        if pattern and pattern != "*" and not fnmatch.fnmatchcase(
            notification.channel_id, pattern,
        ):
            continue

        try:
            keys = _json.loads(addr or "{}")
        except (ValueError, TypeError):
            log.warning(
                "notification_webpush_addr_invalid",
                subscription_id=sub_id,
            )
            continue
        endpoint = str(keys.get("endpoint") or "")
        p256dh = str(keys.get("p256dh") or "")
        auth = str(keys.get("auth") or "")
        if not endpoint or not p256dh or not auth:
            continue

        # send_webpush uses synchronous requests under the hood; run
        # in a thread so we don't block the asyncio loop.
        result = await asyncio.to_thread(
            send_webpush,
            endpoint=endpoint,
            p256dh=p256dh,
            auth=auth,
            payload=push_payload,
            vapid=vapid,
        )
        if result.expired:
            expired_ids.append(sub_id)
            log.info(
                "notification_webpush_subscription_expired",
                subscription_id=sub_id,
                status=result.status,
            )
        elif result.status and 200 <= result.status < 300:
            delivered += 1
        else:
            log.warning(
                "notification_webpush_send_failed",
                subscription_id=sub_id,
                status=result.status,
                error=result.error,
            )

    if expired_ids:
        placeholders = ",".join("?" * len(expired_ids))
        await conn.execute(
            f"DELETE FROM notification_subscriptions "
            f"WHERE subscription_id IN ({placeholders})",
            expired_ids,
        )
        await conn.commit()

    return delivered
