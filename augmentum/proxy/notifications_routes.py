"""Notification substrate — HTTP + WS surface.

Endpoints:

* ``GET  /api/notify/feed``                    — list this user's notifications
* ``GET  /api/notify/channels``                — resolved channels (catalog + overrides)
* ``POST /api/notify/channels/{id}/mute``      — set / clear per-channel mute
* ``POST /api/notify/{id}/read``               — mark read
* ``POST /api/notify/{id}/dismiss``            — dismiss
* ``POST /api/notify/{id}/action/{action_id}`` — invoke an action button
* ``WS   /api/notify/subscribe``               — live push feed

Feature-gated by ``settings.notifications_enabled``. With the flag
off (default), HTTP returns 503 and WS closes with policy-violation —
matching the fabric / Connect routes' posture.

The class-based ``NotificationStore`` wrapper is what these routes
use; tests inject a different conn via the store class without
patching the module-level functions.

See ``docs/superpowers/specs/2026-06-01-notification-substrate-design.md``
for design rationale; ``augmentum/notifications/`` for the store +
hub + action registry.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from augmentum.config import settings
from augmentum.notifications.actions import resolve_handler
from augmentum.notifications.hub import NotificationHub
from augmentum.notifications.store import (
    Notification,
    NotificationChannel,
    dismiss,
    get_notification,
    list_for_user,
    mark_read,
    mute_channel,
    resolved_channels,
)
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.utils.logging import get_logger


log = get_logger(__name__)


router = APIRouter(prefix="/api/notify", tags=["notifications"])


# ── Helpers ───────────────────────────────────────────────────────


def _notifications_enabled() -> bool:
    return bool(getattr(settings, "notifications_enabled", False))


def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    if user is None:
        raise HTTPException(status_code=401, detail="auth required")
    return user.id


def _get_conn(request: Request):
    """Resolve the aiosqlite connection from app state.

    Returns ``None`` when the backend isn't SQLite-backed (tests
    using the in-memory backend). The notifications substrate is
    persistence-bound — without a conn the routes 503 rather than
    silently dropping events.
    """

    sm = getattr(request.app.state, "state_manager", None)
    if sm is not None and isinstance(getattr(sm, "backend", None), SQLiteBackend):
        return sm.backend.conn
    return None


def _require_conn(request: Request):
    conn = _get_conn(request)
    if conn is None:
        raise HTTPException(
            status_code=503,
            detail="notifications require a SQLite backend",
        )
    return conn


def _get_hub(request: Request) -> NotificationHub:
    hub = getattr(request.app.state, "notification_hub", None)
    if hub is None:
        hub = NotificationHub()
        request.app.state.notification_hub = hub
    return hub


async def _broadcast_state(
    request: Request, *, user_id: str, notification_id: str, state: str,
) -> None:
    """Tell the user's OTHER attached clients that a notification changed
    state (read/dismissed) so they clear the live banner instead of leaving
    it stuck until reload. Best-effort: a broadcast miss never fails the
    originating request (the DB row is already authoritative)."""
    try:
        await _get_hub(request).broadcast_event(
            user_id=user_id,
            event={
                "type": "notification_update",
                "notification_id": notification_id,
                "state": state,
            },
        )
    except Exception as exc:  # noqa: BLE001 — sync is opportunistic
        log.warning(
            "notification_state_broadcast_failed",
            notification_id=notification_id, error=str(exc)[:160],
        )


def _notification_to_dict(n: Notification) -> dict[str, Any]:
    return {
        "notification_id": n.notification_id,
        "channel_id": n.channel_id,
        "source": n.source,
        "title": n.title,
        "body": n.body,
        "icon": n.icon,
        "importance": n.importance,
        "thread_id": n.thread_id,
        "actions": [a.to_dict() for a in n.actions],
        "payload": n.payload,
        "transient": n.transient,
        "expires_at": n.expires_at,
        "created_at": n.created_at,
        "updated_at": n.updated_at,
        "delivered_at": n.delivered_at,
        "read_at": n.read_at,
        "dismissed_at": n.dismissed_at,
    }


def _channel_to_dict(c: NotificationChannel) -> dict[str, Any]:
    return {
        "channel_id": c.channel_id,
        "name": c.name,
        "description": c.description,
        "importance": c.importance,
        "default_sound": c.default_sound,
        "muted_until": c.muted_until,
        "user_customized": c.user_customized,
    }


# ── HTTP ─────────────────────────────────────────────────────────


@router.get("/feed")
async def get_feed(
    request: Request,
    include_read: bool = True,
    include_dismissed: bool = False,
    thread_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Return the requesting user's notification feed.

    Defaults to "everything not dismissed" (Slack/Discord
    convention). Set ``include_read=false`` for an unread-only inbox.
    """

    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    items = await list_for_user(
        conn,
        user_id=uid,
        include_read=include_read,
        include_dismissed=include_dismissed,
        thread_id=thread_id,
        limit=int(limit),
    )
    return {"items": [_notification_to_dict(n) for n in items]}


@router.get("/channels")
async def get_channels(request: Request) -> dict[str, Any]:
    """Catalog defaults merged with this user's overrides."""

    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    chs = await resolved_channels(conn, user_id=uid)
    return {"channels": [_channel_to_dict(c) for c in chs]}


class MuteBody(BaseModel):
    until: str = Field(
        default="",
        description="ISO 8601 timestamp to mute until; empty string unmutes.",
    )


@router.post("/channels/{channel_id}/mute")
async def post_mute_channel(
    channel_id: str, body: MuteBody, request: Request,
) -> dict[str, Any]:
    """Set or clear a per-user mute on a channel.

    ``until = ""`` clears any active mute. Far-future date = "muted
    indefinitely". The store lazy-materializes the row from the
    catalog template if no override exists yet.
    """

    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    until_iso = body.until.strip() or None
    await mute_channel(
        conn, user_id=uid, channel_id=channel_id, until_iso=until_iso,
    )
    return {"channel_id": channel_id, "muted_until": until_iso or ""}


@router.post("/{notification_id}/read")
async def post_mark_read(notification_id: str, request: Request) -> dict[str, Any]:
    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    changed = await mark_read(
        conn, user_id=uid, notification_id=notification_id,
    )
    if not changed:
        # Either already read, doesn't exist, or belongs to another
        # user. We don't distinguish in the response — same posture
        # as a 200-with-no-effect REST style and avoids confirming
        # to a hostile caller whether a given id exists.
        existing = await get_notification(
            conn, user_id=uid, notification_id=notification_id,
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="not found")
    else:
        await _broadcast_state(
            request, user_id=uid, notification_id=notification_id, state="read",
        )
    return {"notification_id": notification_id, "read": True}


@router.post("/{notification_id}/dismiss")
async def post_dismiss(notification_id: str, request: Request) -> dict[str, Any]:
    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    changed = await dismiss(
        conn, user_id=uid, notification_id=notification_id,
    )
    if not changed:
        existing = await get_notification(
            conn, user_id=uid, notification_id=notification_id,
        )
        if existing is None:
            raise HTTPException(status_code=404, detail="not found")
    else:
        await _broadcast_state(
            request, user_id=uid, notification_id=notification_id,
            state="dismissed",
        )
    return {"notification_id": notification_id, "dismissed": True}


@router.post("/{notification_id}/action/{action_id}")
async def post_action(
    notification_id: str, action_id: str, request: Request,
) -> dict[str, Any]:
    """Invoke an action button on a notification.

    The notification's ``channel_id`` resolves to a registered handler
    (see ``augmentum/notifications/actions.py``). The handler does
    the subsystem-specific work (route the accept through ConnectHub,
    requeue the failed job, etc.) and returns a dict the UI renders.

    The notification's ``payload`` carries the state the handler
    needs (e.g. ``call_id`` + ``party_id`` for an accept on an
    incoming call). The handler reads it, acts, and returns.
    """

    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)

    notification = await get_notification(
        conn, user_id=uid, notification_id=notification_id,
    )
    if notification is None:
        raise HTTPException(status_code=404, detail="not found")

    # Validate the action exists on this notification. Without this
    # check, a client could invoke any action_id on any notification
    # — even one that doesn't actually appear in the row's actions
    # array — and the handler would have no way to know.
    valid_action_ids = {a.id for a in notification.actions}
    if action_id not in valid_action_ids:
        raise HTTPException(
            status_code=400,
            detail=f"action '{action_id}' not present on this notification",
        )

    handler = resolve_handler(notification.channel_id)
    if handler is None:
        raise HTTPException(
            status_code=404,
            detail=f"no handler registered for channel '{notification.channel_id}'",
        )

    result = await handler(notification, action_id, request)
    # Mark the notification read once the action fires — clicking
    # an action button is implicit acknowledgment.
    await mark_read(conn, user_id=uid, notification_id=notification_id)
    # Acting on one client clears the banner everywhere.
    await _broadcast_state(
        request, user_id=uid, notification_id=notification_id, state="read",
    )

    return {
        "notification_id": notification_id,
        "action_id": action_id,
        "result": result,
    }


# ── Web Push: VAPID + subscriptions ─────────────────────────────


@router.get("/vapid-public-key")
async def get_vapid_public_key(request: Request) -> dict[str, Any]:
    """Return the application's VAPID public key, base64url-encoded.

    The browser uses this as ``applicationServerKey`` in
    ``pushManager.subscribe(...)``. The key is application-wide
    (not per-user) — VAPID identifies the SERVER, not the user.
    Auto-generated + persisted on first call; rotation is operator-
    driven.
    """

    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    # Auth-required, even though the key is non-secret — we don't
    # want unauthenticated probes (lets us correlate with abuse if
    # something starts spamming subscribe attempts).
    _user_id(request)
    conn = _require_conn(request)

    from augmentum.notifications.webpush import get_public_key

    public_key = await get_public_key(conn)
    return {"public_key": public_key}


class WebPushSubscriptionBody(BaseModel):
    """Wire shape posted by the browser after pushManager.subscribe."""

    endpoint: str = Field(..., description="Push service endpoint URL.")
    p256dh: str = Field(..., description="Public key from keys.p256dh, b64url.")
    auth: str = Field(..., description="Auth secret from keys.auth, b64url.")
    channel_pattern: str = Field(
        default="*",
        description="Channel glob to deliver to this subscription.",
    )
    importance_floor: int = Field(
        default=0,
        description="Drop events below this importance.",
    )


@router.post("/subscriptions")
async def post_subscription(
    body: WebPushSubscriptionBody, request: Request,
) -> dict[str, Any]:
    """Register (or refresh) a Web Push subscription for this user.

    Idempotent on (user_id, endpoint): re-posting the same endpoint
    updates the keys + filters in place rather than stacking dupes.
    Endpoints are stable per browser-install per VAPID key, so this
    matches how browsers resubscribe (e.g. after a session restart
    they may report the same endpoint or a fresh one).
    """

    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)

    endpoint = body.endpoint.strip()
    if not endpoint or not endpoint.startswith(("https://", "http://")):
        raise HTTPException(status_code=400, detail="invalid endpoint")
    if not body.p256dh.strip() or not body.auth.strip():
        raise HTTPException(status_code=400, detail="p256dh and auth required")

    # target_address packs the cryptographic material the dispatcher
    # needs to actually send. Using a single column matches the
    # existing subscription schema's "opaque address" model — no
    # migration required.
    import json as _json
    import secrets

    address_payload = _json.dumps({
        "endpoint": endpoint,
        "p256dh": body.p256dh.strip(),
        "auth": body.auth.strip(),
    }, separators=(",", ":"))

    # Idempotent upsert: if this user already has a subscription with
    # the same endpoint, refresh keys + filters; otherwise insert
    # a fresh row.
    cur = await conn.execute(
        "SELECT subscription_id FROM notification_subscriptions "
        "WHERE user_id = ? AND target_kind = 'webpush' "
        "AND json_extract(target_address, '$.endpoint') = ?",
        (uid, endpoint),
    )
    row = await cur.fetchone()
    await cur.close()

    if row is not None:
        sub_id = row[0]
        await conn.execute(
            "UPDATE notification_subscriptions SET "
            "target_address = ?, channel_pattern = ?, importance_floor = ? "
            "WHERE subscription_id = ?",
            (address_payload, body.channel_pattern or "*",
             max(0, int(body.importance_floor or 0)), sub_id),
        )
    else:
        sub_id = f"sub_{secrets.token_hex(8)}"
        await conn.execute(
            "INSERT INTO notification_subscriptions "
            "(subscription_id, user_id, channel_pattern, target_kind, "
            " target_address, importance_floor) "
            "VALUES (?, ?, ?, 'webpush', ?, ?)",
            (sub_id, uid, body.channel_pattern or "*",
             address_payload, max(0, int(body.importance_floor or 0))),
        )
    await conn.commit()
    return {"subscription_id": sub_id, "endpoint": endpoint}


@router.get("/subscriptions")
async def list_subscriptions(request: Request) -> dict[str, Any]:
    """List this user's registered push subscriptions.

    The cryptographic keys aren't returned — they're write-only from
    the client's perspective. Just the endpoint and filters so the
    UI can offer "unsubscribe this device" without leaking secrets
    via the response body.
    """

    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    cur = await conn.execute(
        "SELECT subscription_id, target_kind, target_address, "
        "channel_pattern, importance_floor, created_at "
        "FROM notification_subscriptions "
        "WHERE user_id = ? "
        "ORDER BY created_at DESC",
        (uid,),
    )
    rows = await cur.fetchall()
    await cur.close()

    import json as _json
    out = []
    for r in rows:
        sub_id, kind, addr, pattern, floor, created_at = r
        endpoint = ""
        if kind == "webpush":
            try:
                endpoint = (_json.loads(addr or "{}") or {}).get("endpoint", "")
            except Exception:
                endpoint = ""
        out.append({
            "subscription_id": sub_id,
            "target_kind": kind,
            "endpoint": endpoint,
            "channel_pattern": pattern or "*",
            "importance_floor": int(floor or 0),
            "created_at": created_at or "",
        })
    return {"subscriptions": out}


@router.delete("/subscriptions/{subscription_id}")
async def delete_subscription(
    subscription_id: str, request: Request,
) -> dict[str, Any]:
    """Remove one subscription (e.g. on browser logout)."""

    if not _notifications_enabled():
        raise HTTPException(status_code=503, detail="notifications disabled")
    uid = _user_id(request)
    conn = _require_conn(request)
    cur = await conn.execute(
        "DELETE FROM notification_subscriptions "
        "WHERE subscription_id = ? AND user_id = ?",
        (subscription_id, uid),
    )
    await conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True}


# ── WebSocket ────────────────────────────────────────────────────


async def _handle_inbound_frame(
    websocket: WebSocket, hub: NotificationHub, att: Any, raw: str,
) -> None:
    """Route one inbound WS control frame. Never raises.

    Recognised: ``hello`` (tag device kind → presence) and
    ``device_command_response`` (complete a DeviceCommandBus future).
    Everything else — keepalive pings, unknown types, non-JSON — is a
    deliberate no-op so a chatty or buggy client can't wedge the loop.
    """
    import json

    if not raw:
        return
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return  # plain-text keepalive ping — ignore
    if not isinstance(msg, dict):
        return
    mtype = msg.get("type")
    if mtype == "hello":
        await hub.set_device_type(att.connection_id, str(msg.get("device") or ""))
    elif mtype == "device_command_response":
        request_id = str(msg.get("request_id") or "")
        if not request_id:
            return
        try:
            from augmentum.notifications.device_bus import get_device_bus
            bus = get_device_bus(websocket.app.state)
            bus.resolve(request_id, msg.get("result"))
        except Exception as exc:  # noqa: BLE001 — a stray reply must not kill the WS
            log.warning(
                "device_command_response_failed",
                request_id=request_id, error=str(exc)[:160],
            )


@router.websocket("/subscribe")
async def subscribe_ws(
    websocket: WebSocket,
    channel_pattern: str = "*",
    importance_floor: int = 0,
) -> None:
    """Live notification feed.

    The client connects, optionally narrows by ``channel_pattern``
    glob and ``importance_floor``, and receives a JSON ``{"type":
    "notification", "notification": {...}}`` frame on every new
    publish that matches.

    Inbound frames (2026-06-17): the socket is now read for two control
    messages, so the phone's always-on connection doubles as the
    server↔phone request/response channel without a second socket:

      * ``{"type": "hello", "device": "android"}`` — tags this
        connection's client kind. Presence ("they're on their phone")
        reads it live off the registry.
      * ``{"type": "device_command_response", "request_id", "result"}``
        — completes a :class:`DeviceCommandBus` future (the phone
        answering a ``device_command`` we pushed, e.g. bluetooth_list).

    Any other frame (plain-text keepalive ping, unknown type) is
    ignored. Notification *actions* still go over HTTP, not this socket —
    push-via-WS / actions-via-HTTP separation is preserved; the inbound
    path is control-only, no re-entrancy into the notification store.
    """

    if not _notifications_enabled():
        await websocket.close(code=1008, reason="notifications disabled")
        return

    user = websocket.scope.get("user")
    if user is None:
        await websocket.close(code=1008, reason="auth required")
        return

    await websocket.accept()

    hub = getattr(websocket.app.state, "notification_hub", None)
    if hub is None:
        hub = NotificationHub()
        websocket.app.state.notification_hub = hub

    try:
        att = await hub.attach(
            ws=websocket,
            user_id=user.id,
            channel_pattern=channel_pattern or "*",
            importance_floor=int(importance_floor or 0),
        )
    except ValueError as exc:
        await websocket.close(code=1008, reason=str(exc))
        return

    try:
        while True:
            # Inbound frames are control-only (hello / device response)
            # or keepalive pings. receive_text blocks until close or a
            # frame; we route the two control types and ignore the rest.
            raw = await websocket.receive_text()
            await _handle_inbound_frame(websocket, hub, att, raw)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.warning(
            "notification_ws_error",
            connection_id=att.connection_id,
            user_id=user.id,
            error=str(exc)[:160],
        )
    finally:
        await hub.detach(att.connection_id)
