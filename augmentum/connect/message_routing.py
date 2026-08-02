"""Connect text message routing — dispatches MSG_TEXT_* envelopes.

Sits between the signaling WS receive loop and the message store +
notification substrate. Mirrors call_routing.py's shape so the two
sides of Connect share the same dispatch idiom:

  - resolve peer DID → local user_id (fabric peers deferred)
  - persist BOTH sides' rows under per-user isolation
  - route an ``EVENT_TEXT_RECEIVED`` to the peer if they're online
  - publish a ``connect.message.received`` notification on the peer's
    feed (dedupe by thread so a chatty sender doesn't stack banners)

The ``Reply`` notification quick-action is deferred until the
in-banner reply UI lands — the action would need a body input on
the receiver's banner. For now the banner action simply marks the
message as read; the full reply flow opens the Connect thread panel.

Same-instance only in Phase 1. ``ResolvedPeer(kind="fabric")``
returns a routing error with a ``fabric_routing_pending`` code so
the caller can surface an actionable error to the sender.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.connect.contact_store import is_blocked
from augmentum.connect.contacts import (
    THIS_INSTANCE_SENTINEL,
    display_name_for_did,
    local_did_for,
    resolve_peer_did,
)
from augmentum.connect.message_store import (
    edit_message,
    get_message,
    get_or_create_thread,
    insert_message,
    mark_thread_read,
    new_message_id,
    new_thread_id,
    soft_delete_message,
    stamp_delivered,
)
from augmentum.connect.protocol import (
    EVENT_TEXT_DELETE,
    EVENT_TEXT_DELIVERED,
    EVENT_TEXT_EDIT,
    EVENT_TEXT_REACT,
    EVENT_TEXT_READ,
    EVENT_TEXT_RECEIVED,
    EVENT_TYPING_START,
    EVENT_TYPING_STOP,
    MSG_TEXT_DELETE,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT,
    MSG_TEXT_REACT,
    MSG_TEXT_READ,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    MSG_TYPING_STOP,
    ConnectEnvelope,
)
from augmentum.notifications import (
    IMPORTANCE_DEFAULT,
    NotificationAction,
)
from augmentum.utils.logging import get_logger


if TYPE_CHECKING:
    from augmentum.connect.hub import ConnectHub
    from augmentum.notifications import NotificationHub


log = get_logger(__name__)


# ── Result shape (mirrors CallRoutingResult) ──────────────────────


@dataclass
class MessageRoutingResult:
    """Outcome of a single MSG_TEXT_* dispatch.

    ``routed`` = number of peer WSes that received the event.
    ``notification_id`` = id of the persisted notification on the
    recipient (only set for MSG_TEXT_SEND when the peer was offline
    or no banner was needed).
    ``thread_id`` / ``message_id`` = canonical ids after the server
    minted defaults for any missing fields.
    ``error_code`` is non-empty when routing failed in a way the
    sender should hear about.
    """

    routed: int = 0
    notification_id: str = ""
    thread_id: str = ""
    message_id: str = ""
    error_code: str = ""
    error_message: str = ""


# ── Dispatcher ─────────────────────────────────────────────────────


async def handle_message_envelope(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    notification_hub: "NotificationHub",
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    sender_role: str = "",
    fabric_coordinator: Any = None,
    fabric_identity: Any = None,
    our_attachment_base_url: str = "",
) -> MessageRoutingResult:
    """Route one inbound MSG_TEXT_* envelope.

    The signaling WS loop dispatches every text verb through here.
    Returns a ``MessageRoutingResult``; on error, the caller surfaces
    ``error_code`` back to the sender as an ``EVENT_ERROR`` envelope.

    When ``env.peer`` is a fabric DID (``alice@instance-A``), the
    handler persists the sender's local-side state for the verb (so
    their own UI reflects the change immediately) and dispatches the
    envelope over fabric. The recipient-side mirror + WS event +
    notification fire on the receiving instance via
    ``apply_inbound_fabric_envelope``. ``fabric_coordinator`` is the
    runtime fabric handle from ``app.state.fabric_coordinator``;
    when None (fabric disabled), fabric-bound envelopes return a
    clean ``fabric_unavailable`` error.
    """

    resolved = resolve_peer_did(env.peer)
    if resolved is None:
        return MessageRoutingResult(
            error_code="peer_did_invalid",
            error_message=f"could not parse peer DID '{env.peer}'",
        )
    if resolved.kind == "fabric":
        # A guest is scoped to a LOCAL host grant — it may never reach a
        # cross-instance fabric peer. This check sits BEFORE the fabric
        # dispatch because the local-target ACL gate below is unreachable on
        # this path; without it a guest could message any user on any paired
        # instance (the fabric backdoor).
        if sender_role == "guest":
            return MessageRoutingResult(
                error_code="guest_scope_violation",
                error_message="guests may only reach their host",
            )
        return await _handle_fabric_outbound(
            conn=conn,
            fabric_coordinator=fabric_coordinator,
            fabric_identity=fabric_identity,
            our_attachment_base_url=our_attachment_base_url,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            target_hostname=resolved.address,
        )
    if resolved.kind != "local":
        return MessageRoutingResult(
            error_code="peer_routing_unsupported",
            error_message=f"routing kind '{resolved.kind}' not supported",
        )
    target_user_id = resolved.address
    target_did = local_did_for(target_user_id)

    # Guest ACL gate (Phase 3a) — beside the per-verb block checks below. A
    # role='guest' sender may reach ONLY the host it holds a live grant for;
    # one check here covers every text verb (target is resolved once).
    from augmentum.connect.guest_grant_store import guest_scope_blocked
    if await guest_scope_blocked(
        conn, sender_user_id=sender_user_id, sender_role=sender_role,
        target_user_id=target_user_id,
    ):
        return MessageRoutingResult(
            error_code="guest_scope_violation",
            error_message="guests may only reach their host",
        )

    if env.verb == MSG_TEXT_SEND:
        return await _handle_send(
            conn=conn,
            connect_hub=connect_hub,
            notification_hub=notification_hub,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            target_user_id=target_user_id,
            target_did=target_did,
        )
    if env.verb == MSG_TEXT_READ:
        return await _handle_read(
            conn=conn,
            connect_hub=connect_hub,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            target_user_id=target_user_id,
        )
    if env.verb == MSG_TEXT_DELETE:
        return await _handle_delete(
            conn=conn,
            connect_hub=connect_hub,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            target_user_id=target_user_id,
        )
    if env.verb == MSG_TEXT_EDIT:
        return await _handle_edit(
            conn=conn,
            connect_hub=connect_hub,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            target_user_id=target_user_id,
        )
    if env.verb in (MSG_TYPING_START, MSG_TYPING_STOP):
        return await _handle_typing(
            conn=conn,
            connect_hub=connect_hub,
            env=env,
            sender_did=sender_did,
            target_user_id=target_user_id,
        )
    if env.verb == MSG_TEXT_REACT:
        return await _handle_react(
            conn=conn,
            connect_hub=connect_hub,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            target_user_id=target_user_id,
        )
    if env.verb == MSG_TEXT_DELIVERED:
        return await _handle_delivered(
            conn=conn,
            connect_hub=connect_hub,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            target_user_id=target_user_id,
        )

    return MessageRoutingResult(
        error_code="unsupported_verb",
        error_message=f"verb '{env.verb}' has no text-routing handler",
    )


async def _handle_typing(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    env: ConnectEnvelope,
    sender_did: str,
    target_user_id: str,
) -> MessageRoutingResult:
    """Fan an ephemeral typing event out to the peer.

    No DB writes — typing is presence, not state. Server doesn't even
    log the event (high frequency, low signal). We just translate
    MSG_TYPING_{START,STOP} to EVENT_TYPING_{START,STOP} on the peer
    side and forward thread_id so the receiver knows which thread.
    """

    data = env.data or {}
    thread_id = str(data.get("thread_id") or "")
    if not thread_id:
        return MessageRoutingResult(
            error_code="missing_thread_id",
            error_message=f"{env.verb} requires thread_id",
        )
    event_verb = (
        EVENT_TYPING_START if env.verb == MSG_TYPING_START else EVENT_TYPING_STOP
    )
    # Silent-block typing — the blocker shouldn't see the blockee
    # "typing…" indicator any more than they see the actual message.
    if conn is not None and await is_blocked(
        conn, user_id=target_user_id, peer_did=sender_did,
    ):
        return MessageRoutingResult(routed=0, thread_id=thread_id)
    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=event_verb,
            corr_id=env.corr_id,
            peer=sender_did,
            data={"thread_id": thread_id},
        ),
    )
    return MessageRoutingResult(routed=routed, thread_id=thread_id)


# ── Verb handlers ──────────────────────────────────────────────────


async def _handle_react(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    target_user_id: str,
) -> MessageRoutingResult:
    """Apply or remove an emoji reaction on a message.

    Each user owns a per-side copy of the reaction row (multi-tenant
    pattern). Action defaults to "add" when absent — the wire is
    forward-compat with future verbs (e.g. "toggle") that map to the
    same SQL path.

    Routes ``EVENT_TEXT_REACT`` to the peer with the same payload so
    their UI updates the reaction pill stack live.
    """

    from datetime import UTC, datetime

    data = env.data or {}
    message_id = str(data.get("message_id") or "")
    emoji = str(data.get("emoji") or "").strip()
    thread_id = str(data.get("thread_id") or "")
    action = str(data.get("action") or "add").lower()

    if not message_id:
        return MessageRoutingResult(
            error_code="missing_message_id",
            error_message="react requires message_id",
        )
    if not emoji:
        return MessageRoutingResult(
            error_code="missing_emoji",
            error_message="react requires emoji",
        )
    # Cap emoji length so a malicious or buggy client can't store a
    # huge string in the reaction column.
    emoji = emoji[:32]

    now = datetime.now(UTC).isoformat()

    # Silent-block reactions: sender's own row updates; recipient
    # mirror + WS event skip so the blocker never sees the blockee's
    # reactions appear.
    blocked = await is_blocked(
        conn, user_id=target_user_id, peer_did=sender_did,
    )
    mutate_uids = (sender_user_id,) if blocked else (sender_user_id, target_user_id)

    if action == "remove":
        for uid in mutate_uids:
            await conn.execute(
                "DELETE FROM connect_message_reactions "
                "WHERE message_id = ? AND user_id = ? AND reactor_did = ? AND emoji = ?",
                (message_id, uid, sender_did, emoji),
            )
        await conn.commit()
    else:
        for uid in mutate_uids:
            await conn.execute(
                "INSERT OR IGNORE INTO connect_message_reactions "
                "(message_id, user_id, reactor_did, emoji, reacted_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (message_id, uid, sender_did, emoji, now),
            )
        await conn.commit()

    if blocked:
        return MessageRoutingResult(routed=0, thread_id=thread_id)

    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_TEXT_REACT,
            corr_id=env.corr_id,
            peer=sender_did,
            data={
                "thread_id": thread_id,
                "message_id": message_id,
                "emoji": emoji,
                "action": action,
                "reactor_did": sender_did,
            },
        ),
    )
    return MessageRoutingResult(routed=routed, thread_id=thread_id)


async def _handle_send(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    notification_hub: "NotificationHub",
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    target_user_id: str,
    target_did: str,
) -> MessageRoutingResult:
    from augmentum.notifications.hub import publish_and_dispatch

    data = env.data or {}
    body = str(data.get("body") or "")
    if not body and not data.get("attachment_ref"):
        return MessageRoutingResult(
            error_code="message_empty",
            error_message="text_send requires non-empty body or attachment_ref",
        )

    thread_id = str(data.get("thread_id") or "") or new_thread_id()
    message_id = str(data.get("message_id") or "") or new_message_id()
    fmt = str(data.get("format") or "plain")
    if fmt not in {"plain", "markdown", "voice_note", "embed"}:
        fmt = "plain"
    attachment_ref = str(data.get("attachment_ref") or "")
    # Optional sender-supplied attachment metadata. The wire passes
    # these through unchanged so the recipient's UI can render the
    # right widget without a HEAD round-trip. They aren't persisted;
    # the canonical metadata is on the sender's uploads row, fetched
    # on demand via the attachment route's HEAD.
    attachment_name = str(data.get("attachment_name") or "")
    attachment_mime = str(data.get("attachment_mime") or "")
    attachment_size = int(data.get("attachment_size") or 0) if data.get("attachment_size") else 0
    reply_to = str(data.get("reply_to") or "")
    sent_at = str(data.get("sent_at") or "") or None
    transcript = str(data.get("transcript") or "")

    # 1) Sender's perspective: ensure the thread exists and insert the row.
    sender_thread = await get_or_create_thread(
        conn,
        thread_id=thread_id,
        user_id=sender_user_id,
        peer_did=target_did,
    )
    # If the unique-pair index demoted our thread_id to an existing
    # row's id, use that id from here on — the wire envelope to the
    # peer carries the resolved id so both ends agree on the thread.
    thread_id = sender_thread.thread_id

    await insert_message(
        conn,
        message_id=message_id,
        thread_id=thread_id,
        user_id=sender_user_id,
        sender_did=sender_did,
        body=body,
        format=fmt,
        attachment_ref=attachment_ref,
        reply_to=reply_to,
        sent_at=sent_at,
        transcript=transcript,
    )

    # Silent-block: if the recipient has blocked us, sender's row is
    # already persisted but everything downstream (recipient mirror,
    # WS event, notification) is skipped. Sender's UI will show "sent"
    # but the message never becomes "delivered" or "read" — same
    # observable shape as a permanently-offline peer. Matches WhatsApp
    # / iMessage block semantics: don't tell the sender they're blocked.
    if await is_blocked(
        conn, user_id=target_user_id, peer_did=sender_did,
    ):
        return MessageRoutingResult(
            routed=0,
            thread_id=thread_id,
            message_id=message_id,
        )

    # 2) Recipient's perspective: separate thread row (per-user
    # isolation), then mirror the message. We mint a recipient-side
    # thread_id only if the recipient doesn't already have a thread
    # with the sender — the unique-pair index makes this idempotent.
    recipient_thread = await get_or_create_thread(
        conn,
        thread_id=thread_id,
        user_id=target_user_id,
        peer_did=sender_did,
    )
    recipient_thread_id = recipient_thread.thread_id

    await insert_message(
        conn,
        message_id=message_id,
        thread_id=recipient_thread_id,
        user_id=target_user_id,
        sender_did=sender_did,
        body=body,
        format=fmt,
        attachment_ref=attachment_ref,
        reply_to=reply_to,
        sent_at=sent_at,
        transcript=transcript,
    )

    # 3) NOTE: we used to call ``stamp_delivered`` here for the
    # sender's row. That was a lie — server-stored is not "the peer
    # received the bytes". delivered_at is now stamped by
    # ``_handle_delivered`` when the recipient's UI acks reception
    # (over WS or via the catch-up endpoint).

    # 4) Route EVENT_TEXT_RECEIVED to recipient's live sessions.
    event_data = {
        "thread_id": recipient_thread_id,
        "message_id": message_id,
        "body": body,
        "format": fmt,
        "attachment_ref": attachment_ref,
        "reply_to": reply_to,
        "sent_at": sent_at or "",
        "sender_did": sender_did,
        "transcript": transcript,
    }
    if attachment_ref:
        # Pass-through metadata so the recipient renders the right
        # widget (image preview vs audio player vs file pill) without
        # an extra HEAD request. None of these are stored — they're a
        # render hint only; canonical truth lives on the sender's
        # uploads row, accessible via HEAD on the attachment route.
        if attachment_name:
            event_data["attachment_name"] = attachment_name
        if attachment_mime:
            event_data["attachment_mime"] = attachment_mime
        if attachment_size:
            event_data["attachment_size"] = attachment_size
    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_TEXT_RECEIVED,
            corr_id=env.corr_id,
            peer=sender_did,
            data=event_data,
        ),
    )

    # 5) Publish a notification regardless of routing — the notification
    # store is the inbox even when the peer's signaling WS is open.
    # The UI side chooses whether to suppress the banner when the
    # active thread is already in view; that decision lives client-side
    # so it can react to focus changes without a server round-trip.
    notification_id = ""
    try:
        preview = body[:200] if body else (
            f"[{fmt}]" if attachment_ref or fmt != "plain" else ""
        )
        sender_name = await display_name_for_did(conn, sender_did)
        notification_id = await publish_and_dispatch(
            conn,
            hub=notification_hub,
            user_id=target_user_id,
            channel_id="connect.message",
            source="connect",
            title=sender_name or sender_did,
            body=preview,
            importance=IMPORTANCE_DEFAULT,
            # Dedupe by thread so a fast sender doesn't stack a banner
            # per message — the latest preview replaces in place.
            dedupe_key=f"thread:{recipient_thread_id}",
            thread_id=recipient_thread_id,
            actions=[
                NotificationAction(
                    id="open_thread", label="Open", style="primary",
                ),
                NotificationAction(
                    id="mark_read", label="Mark read", style="default",
                ),
            ],
            payload={
                "thread_id": recipient_thread_id,
                "message_id": message_id,
                "sender_did": sender_did,
                "sender_user_id": sender_user_id,
            },
            transient=False,
            icon="message",
        )
    except Exception as exc:
        log.warning(
            "connect_text_notification_failed",
            thread_id=recipient_thread_id,
            target_user_id=target_user_id,
            error=str(exc)[:160],
        )

    return MessageRoutingResult(
        routed=routed,
        notification_id=notification_id,
        thread_id=thread_id,
        message_id=message_id,
    )


async def _handle_read(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    target_user_id: str,
) -> MessageRoutingResult:
    """Apply a read receipt: clear unread on sender's side, mark every
    matching row as read, and route an ``EVENT_TEXT_READ`` to the
    peer so their UI can update the per-message read indicator.
    """

    data = env.data or {}
    thread_id = str(data.get("thread_id") or "")
    last_read = str(data.get("last_read_message_id") or "")
    if not thread_id:
        return MessageRoutingResult(
            error_code="missing_thread_id",
            error_message="text_read requires thread_id",
        )

    marked = await mark_thread_read(
        conn,
        thread_id=thread_id,
        user_id=sender_user_id,
        last_read_message_id=last_read,
    )
    # Silent-block: if the original sender has blocked us, swallow the
    # read receipt — they shouldn't get any signal from this thread
    # (including "they read my message").
    if await is_blocked(
        conn, user_id=target_user_id, peer_did=sender_did,
    ):
        return MessageRoutingResult(routed=0, thread_id=thread_id)
    # Routing the receipt to the peer is still useful even when no
    # rows changed — the receipt is idempotent and the peer's UI may
    # have been stale.
    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_TEXT_READ,
            corr_id=env.corr_id,
            peer=sender_did,
            data={
                "thread_id": thread_id,
                "last_read_message_id": last_read,
                "reader_did": sender_did,
                "marked": marked,
            },
        ),
    )
    return MessageRoutingResult(routed=routed, thread_id=thread_id)


async def _handle_delivered(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    target_user_id: str,
) -> MessageRoutingResult:
    """Apply a batched delivery receipt.

    ``sender_user_id`` here is the RECIPIENT of the original message —
    they're the one acknowledging receipt. ``target_user_id`` is the
    ORIGINAL sender we route the receipt back to.

    For each id in ``message_ids`` we stamp ``delivered_at`` on the
    original sender's row (if not already stamped) and roll the
    successfully-stamped subset into a single EVENT_TEXT_DELIVERED
    envelope back to the original sender. Already-stamped ids are
    silently included in the routed list so the sender's UI doesn't
    treat a re-ack as a new "first delivery" event — the receipt is
    idempotent end-to-end.
    """

    data = env.data or {}
    thread_id = str(data.get("thread_id") or "")
    raw_ids = data.get("message_ids") or []
    if not isinstance(raw_ids, (list, tuple)):
        return MessageRoutingResult(
            error_code="invalid_message_ids",
            error_message="message_ids must be a list",
        )
    message_ids = [str(mid) for mid in raw_ids if mid]
    if not message_ids:
        return MessageRoutingResult(
            error_code="missing_message_ids",
            error_message="delivered requires at least one message_id",
        )

    # Cap the batch so a buggy client can't ask the server to update
    # tens of thousands of rows in one envelope. 200 is comfortably
    # above any realistic catch-up batch.
    if len(message_ids) > 200:
        message_ids = message_ids[:200]

    # Silent-block: skip stamping AND routing if the original sender
    # has blocked us. Their UI shouldn't ever flip to "delivered" for
    # a thread they've blocked.
    if await is_blocked(
        conn, user_id=target_user_id, peer_did=sender_did,
    ):
        return MessageRoutingResult(routed=0, thread_id=thread_id)

    for mid in message_ids:
        await stamp_delivered(
            conn, message_id=mid, user_id=target_user_id,
        )

    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_TEXT_DELIVERED,
            corr_id=env.corr_id,
            peer=sender_did,
            data={
                "thread_id": thread_id,
                "message_ids": message_ids,
            },
        ),
    )
    return MessageRoutingResult(routed=routed, thread_id=thread_id)


async def _handle_delete(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    target_user_id: str,
) -> MessageRoutingResult:
    """Soft-delete one of the sender's own messages on both sides."""

    data = env.data or {}
    message_id = str(data.get("message_id") or "")
    thread_id = str(data.get("thread_id") or "")
    if not message_id or not thread_id:
        return MessageRoutingResult(
            error_code="missing_message_id",
            error_message="text_delete requires thread_id + message_id",
        )

    # Refuse to delete messages the sender doesn't own — quick guard
    # against a buggy or hostile client trying to wipe peer messages.
    msg = await get_message(
        conn, message_id=message_id, user_id=sender_user_id,
    )
    if msg is None or msg.sender_did != sender_did:
        return MessageRoutingResult(
            error_code="message_not_owned",
            error_message="cannot delete message you did not send",
        )
    # Double-delete should be a clean no-op rather than firing a
    # redundant EVENT_TEXT_DELETE. The store's UPDATE…WHERE deleted_at
    # IS NULL clause makes the DB side idempotent; this guard makes
    # the wire side idempotent too.
    if msg.deleted_at is not None:
        return MessageRoutingResult(
            routed=0, thread_id=thread_id, message_id=message_id,
        )

    await soft_delete_message(
        conn, message_id=message_id, user_id=sender_user_id,
    )
    # Silent-block: skip the recipient-side mutation + WS event when
    # blocked, so the blocker's stored copy and live view aren't
    # touched by the blockee's post-block edits.
    if await is_blocked(
        conn, user_id=target_user_id, peer_did=sender_did,
    ):
        return MessageRoutingResult(
            routed=0, thread_id=thread_id, message_id=message_id,
        )
    await soft_delete_message(
        conn, message_id=message_id, user_id=target_user_id,
    )

    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_TEXT_DELETE,
            corr_id=env.corr_id,
            peer=sender_did,
            data={"thread_id": thread_id, "message_id": message_id},
        ),
    )
    return MessageRoutingResult(
        routed=routed, thread_id=thread_id, message_id=message_id,
    )


async def _handle_edit(
    *,
    conn: Any,
    connect_hub: "ConnectHub",
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    target_user_id: str,
) -> MessageRoutingResult:
    """Edit the sender's own message on both sides."""

    data = env.data or {}
    message_id = str(data.get("message_id") or "")
    thread_id = str(data.get("thread_id") or "")
    new_body = str(data.get("body") or "")
    if not message_id or not thread_id:
        return MessageRoutingResult(
            error_code="missing_message_id",
            error_message="text_edit requires thread_id + message_id",
        )

    msg = await get_message(
        conn, message_id=message_id, user_id=sender_user_id,
    )
    if msg is None or msg.sender_did != sender_did:
        return MessageRoutingResult(
            error_code="message_not_owned",
            error_message="cannot edit message you did not send",
        )
    # Edit-after-delete races: the store's edit_message refuses to
    # touch deleted rows, but firing EVENT_TEXT_EDIT with a body the
    # recipient would render over their tombstone is a live/stored
    # divergence (recipient's UI shows the edited body until reload).
    # Refuse the edit at the routing layer so both wire + DB stay
    # consistent.
    if msg.deleted_at is not None:
        return MessageRoutingResult(
            error_code="message_already_deleted",
            error_message="cannot edit a deleted message",
        )

    await edit_message(
        conn, message_id=message_id, user_id=sender_user_id, body=new_body,
    )
    # Silent-block: same as delete — sender's own row updates, but the
    # blocker's mirror + live view are frozen at the pre-block state.
    if await is_blocked(
        conn, user_id=target_user_id, peer_did=sender_did,
    ):
        return MessageRoutingResult(
            routed=0, thread_id=thread_id, message_id=message_id,
        )
    await edit_message(
        conn, message_id=message_id, user_id=target_user_id, body=new_body,
    )

    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_TEXT_EDIT,
            corr_id=env.corr_id,
            peer=sender_did,
            data={
                "thread_id": thread_id,
                "message_id": message_id,
                "body": new_body,
            },
        ),
    )
    return MessageRoutingResult(
        routed=routed, thread_id=thread_id, message_id=message_id,
    )


# ── Fabric (cross-instance) outbound dispatch ─────────────────────


async def _handle_fabric_outbound(
    *,
    conn: Any,
    fabric_coordinator: Any,
    fabric_identity: Any = None,
    our_attachment_base_url: str = "",
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    target_hostname: str,
) -> MessageRoutingResult:
    """Sender-side persistence + fabric dispatch for one text verb.

    The recipient-side mirror write, WS event, and notification all
    happen on the receiving instance via the inbound dispatcher. We
    persist whatever the SENDER needs locally so their own UI shows
    the verb's effect (their own row inserted, edited, deleted, etc.)
    even if the peer's instance is offline.

    Block-check is intentionally NOT done here — the sender's instance
    has no view of the recipient's blocklist. Block enforcement lives
    on the receiving instance's inbound dispatcher, matching the
    silent-block design (sender's UI sees "sent" either way).
    """
    from augmentum.connect.fabric_transport import dispatch_fabric_envelope

    data = env.data or {}
    thread_id = ""
    message_id = ""
    sent_at: str | None = None

    # Per-verb sender-side local persistence. Each branch sets
    # thread_id / message_id (used in the response shape) and writes
    # the same store rows the local path would.
    if env.verb == MSG_TEXT_SEND:
        body = str(data.get("body") or "")
        if not body and not data.get("attachment_ref"):
            return MessageRoutingResult(
                error_code="message_empty",
                error_message="text_send requires non-empty body or attachment_ref",
            )
        thread_id = str(data.get("thread_id") or "") or new_thread_id()
        message_id = str(data.get("message_id") or "") or new_message_id()
        fmt = str(data.get("format") or "plain")
        if fmt not in {"plain", "markdown", "voice_note", "embed"}:
            fmt = "plain"
        attachment_ref = str(data.get("attachment_ref") or "")
        reply_to = str(data.get("reply_to") or "")
        sent_at = str(data.get("sent_at") or "") or None
        transcript = str(data.get("transcript") or "")

        sender_thread = await get_or_create_thread(
            conn,
            thread_id=thread_id,
            user_id=sender_user_id,
            peer_did=env.peer,
        )
        thread_id = sender_thread.thread_id
        await insert_message(
            conn,
            message_id=message_id,
            thread_id=thread_id,
            user_id=sender_user_id,
            sender_did=sender_did,
            body=body,
            format=fmt,
            attachment_ref=attachment_ref,
            reply_to=reply_to,
            sent_at=sent_at,
            transcript=transcript,
        )
        # Update envelope payload so the inbound dispatcher sees the
        # resolved thread/message ids + sent_at fingerprint. The
        # recipient mints its OWN thread row for the user-pair, so
        # we keep thread_id in the envelope as the sender's view —
        # the inbound dispatcher rederives the recipient's thread.
        env.data["thread_id"] = thread_id
        env.data["message_id"] = message_id
        if sent_at:
            env.data["sent_at"] = sent_at

        # Mint cross-instance attachment fetch token + URL when the
        # message has an attachment and we know our base URL. The
        # recipient's UI uses these to fetch the blob directly from
        # us; without them, the recipient sees the attachment_ref but
        # can't resolve it (local /attachment route would 404).
        if attachment_ref and our_attachment_base_url:
            from augmentum.connect.fabric_transport import sign_attachment_token
            tok = sign_attachment_token(
                identity=fabric_identity, ref=attachment_ref,
            )
            env.data["attachment_token"] = tok
            env.data["attachment_fetch_url"] = (
                f"{our_attachment_base_url.rstrip('/')}"
                f"/api/connect/fabric/attachments/{attachment_ref}"
            )

    elif env.verb == MSG_TEXT_EDIT:
        message_id = str(data.get("message_id") or "")
        thread_id = str(data.get("thread_id") or "")
        new_body = str(data.get("body") or "")
        if not message_id or not thread_id:
            return MessageRoutingResult(
                error_code="missing_message_id",
                error_message="text_edit requires thread_id + message_id",
            )
        msg = await get_message(
            conn, message_id=message_id, user_id=sender_user_id,
        )
        if msg is None or msg.sender_did != sender_did:
            return MessageRoutingResult(
                error_code="message_not_owned",
                error_message="cannot edit message you did not send",
            )
        if msg.deleted_at is not None:
            return MessageRoutingResult(
                error_code="message_already_deleted",
                error_message="cannot edit a deleted message",
            )
        await edit_message(
            conn, message_id=message_id, user_id=sender_user_id, body=new_body,
        )

    elif env.verb == MSG_TEXT_DELETE:
        message_id = str(data.get("message_id") or "")
        thread_id = str(data.get("thread_id") or "")
        if not message_id or not thread_id:
            return MessageRoutingResult(
                error_code="missing_message_id",
                error_message="text_delete requires thread_id + message_id",
            )
        msg = await get_message(
            conn, message_id=message_id, user_id=sender_user_id,
        )
        if msg is None or msg.sender_did != sender_did:
            return MessageRoutingResult(
                error_code="message_not_owned",
                error_message="cannot delete message you did not send",
            )
        if msg.deleted_at is not None:
            # Already deleted — idempotent no-op (don't re-dispatch).
            return MessageRoutingResult(
                routed=0, thread_id=thread_id, message_id=message_id,
            )
        await soft_delete_message(
            conn, message_id=message_id, user_id=sender_user_id,
        )

    elif env.verb == MSG_TEXT_READ:
        thread_id = str(data.get("thread_id") or "")
        last_read = str(data.get("last_read_message_id") or "")
        if not thread_id:
            return MessageRoutingResult(
                error_code="missing_thread_id",
                error_message="text_read requires thread_id",
            )
        await mark_thread_read(
            conn,
            thread_id=thread_id,
            user_id=sender_user_id,
            last_read_message_id=last_read,
        )

    elif env.verb == MSG_TEXT_REACT:
        from datetime import UTC, datetime
        message_id = str(data.get("message_id") or "")
        thread_id = str(data.get("thread_id") or "")
        emoji = str(data.get("emoji") or "").strip()[:32]
        action = str(data.get("action") or "add").lower()
        if not message_id:
            return MessageRoutingResult(
                error_code="missing_message_id",
                error_message="react requires message_id",
            )
        if not emoji:
            return MessageRoutingResult(
                error_code="missing_emoji",
                error_message="react requires emoji",
            )
        now = datetime.now(UTC).isoformat()
        if action == "remove":
            await conn.execute(
                "DELETE FROM connect_message_reactions "
                "WHERE message_id = ? AND user_id = ? AND reactor_did = ? AND emoji = ?",
                (message_id, sender_user_id, sender_did, emoji),
            )
        else:
            await conn.execute(
                "INSERT OR IGNORE INTO connect_message_reactions "
                "(message_id, user_id, reactor_did, emoji, reacted_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (message_id, sender_user_id, sender_did, emoji, now),
            )
        await conn.commit()

    elif env.verb in (
        MSG_TEXT_DELIVERED, MSG_TYPING_START, MSG_TYPING_STOP,
    ):
        # No sender-side persistence — these are purely receipt/
        # presence frames going to the original sender's instance.
        thread_id = str(data.get("thread_id") or "")

    else:
        return MessageRoutingResult(
            error_code="unsupported_verb",
            error_message=f"fabric routing for '{env.verb}' not wired",
        )

    # Dispatch via fabric. Even when the peer is offline the outbox
    # makes it durable; we return "routed=1" because the message is
    # accepted (the sender's UI shows "sent").
    result = await dispatch_fabric_envelope(
        conn,
        coordinator=fabric_coordinator,
        target_hostname=target_hostname,
        source_did=sender_did,
        sender_user_id=sender_user_id,
        sender_party_id="",  # text verbs don't carry party_id
        envelope=env,
    )
    return MessageRoutingResult(
        routed=1 if result.queued else 0,
        thread_id=thread_id,
        message_id=message_id,
        error_code=result.error_code,
        error_message=result.error_message,
    )


# ── Action handler (Notification "Open"/"Mark read" buttons) ──────


async def handle_message_action(
    notification: "Any", action_id: str, request: "Any",
) -> dict[str, Any]:
    """Notification action handler for ``connect.message.*``.

    Both 'open_thread' and 'mark_read' clear the recipient's unread
    counter for the thread; 'open_thread' additionally returns an
    intent hint the UI consumes to surface the thread panel.

    Wired by ``connect_routes.py`` at module import via
    ``register_action_handler('connect.message.*', handle_message_action)``.
    """

    if action_id not in ("open_thread", "mark_read"):
        return {"status": "error", "error": f"unknown action '{action_id}'"}

    payload = notification.payload or {}
    thread_id = str(payload.get("thread_id") or "")
    if not thread_id:
        return {
            "status": "error",
            "error": "notification payload missing thread_id",
        }

    from augmentum.state.backends.sqlite import SQLiteBackend
    sm = getattr(request.app.state, "state_manager", None)
    conn = (
        sm.backend.conn
        if sm is not None and isinstance(getattr(sm, "backend", None), SQLiteBackend)
        else None
    )
    if conn is None:
        return {
            "status": "error",
            "error": "persistence not active",
        }

    marked = await mark_thread_read(
        conn, thread_id=thread_id, user_id=notification.user_id,
    )

    intent = "open_thread" if action_id == "open_thread" else "mark_read"
    return {
        "status": intent,
        "thread_id": thread_id,
        "marked_count": marked,
    }


# Re-export for callers that need to know we use this sentinel; tests
# reference it directly.
__all__ = [
    "MessageRoutingResult",
    "THIS_INSTANCE_SENTINEL",
    "handle_message_action",
    "handle_message_envelope",
]
