"""Inbound Connect-over-fabric envelope dispatcher.

When a paired peer sends us an ``MSG_CONNECT_ENVELOPE`` fabric frame,
the FabricCoordinator's inbound dispatcher (after Ed25519 signature
verification) calls into this module. We:

  1. Re-parse the inner ConnectEnvelope from the payload.
  2. Resolve ``target_did`` to the local user_id via the contacts
     module's DID resolver (must be ``kind="local"`` — if the target
     isn't on this instance, the sending peer is misrouting).
  3. Apply the verb against local state — usually means writing the
     recipient mirror row + firing the corresponding EVENT to the
     local user's signaling WS. Block-list checks still apply
     (silent-block semantics carry across fabric).

Per-verb handlers in this module mirror what the local outbound
handlers in ``message_routing.py`` / ``call_routing.py`` do for
the RECIPIENT side ONLY — the sender row lives on the originating
instance.

Phase 1 ships the dispatch skeleton + a NotImplementedError stub
for each verb so the routing path can be wired before the verb
logic lands. Phase 2 fills in text verbs; Phase 4 fills in call
verbs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.connect.contacts import (
    THIS_INSTANCE_SENTINEL,
    resolve_peer_did,
)
from augmentum.connect.protocol import (
    MSG_ACCEPT,
    MSG_ANSWER,
    MSG_CANDIDATES,
    MSG_DECLINE,
    MSG_HANGUP,
    MSG_INVITE,
    MSG_MUTE_STATE,
    MSG_NEGOTIATE,
    MSG_OFFER,
    MSG_SELECT_ANSWER,
    MSG_TEXT_DELETE,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT,
    MSG_TEXT_REACT,
    MSG_TEXT_READ,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    MSG_TYPING_STOP,
    MSG_VIDEO_STATE,
    ConnectEnvelope,
    deserialise_envelope,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

    from augmentum.connect.hub import ConnectHub
    from augmentum.notifications.hub import NotificationHub

log = get_logger(__name__)


# Set of verbs this dispatcher knows how to handle. Anything outside
# the set gets logged + dropped (defensive — a peer running a future
# protocol version may send verbs we haven't shipped yet).
_TEXT_VERBS = frozenset({
    MSG_TEXT_SEND, MSG_TEXT_READ, MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT, MSG_TEXT_DELETE, MSG_TEXT_REACT,
    MSG_TYPING_START, MSG_TYPING_STOP,
})

_CALL_VERBS = frozenset({
    MSG_INVITE, MSG_ACCEPT, MSG_DECLINE,
    MSG_OFFER, MSG_ANSWER, MSG_CANDIDATES, MSG_NEGOTIATE,
    MSG_HANGUP, MSG_SELECT_ANSWER, MSG_MUTE_STATE,
    MSG_VIDEO_STATE,
})


def _normalise_source_did(
    *, source_did: str, coordinator: Any, sender_node_id: str,
) -> str:
    """Rewrite a ``user@this-instance`` source DID into ``user@<hostname>``.

    The sender's instance puts ``THIS_INSTANCE_SENTINEL`` in the host-part
    because, on the sender's box, that IS the local form. The receiving
    instance wants the sender's real hostname so:

      * its blocklist, contacts, and thread-pair index all key off a
        stable global identifier rather than the sender's local sentinel
      * bidirectional thread reuse with the same ``thread_id`` doesn't
        collide on ``(thread_id, user_id)`` because the unique pair
        ``(user_id, peer_did)`` resolves the existing row first

    Falls through (returns the input unchanged) when the coordinator or
    sender_node_id is missing, the peer isn't in our registry, or the
    DID doesn't have the sentinel suffix. These all preserve forward-
    compatibility with future DID forms.
    """
    if not source_did or coordinator is None or not sender_node_id:
        return source_did
    suffix = f"@{THIS_INSTANCE_SENTINEL}"
    if not source_did.endswith(suffix):
        return source_did
    peer_state_fn = getattr(coordinator, "peer_state", None)
    if not callable(peer_state_fn):
        return source_did
    state = peer_state_fn(sender_node_id)
    paired = getattr(state, "paired", None) if state is not None else None
    hostname = getattr(paired, "hostname", "") if paired is not None else ""
    if not hostname:
        return source_did
    local_part = source_did[: -len(suffix)]
    return f"{local_part}@{hostname}"


async def apply_inbound_fabric_envelope(
    conn: aiosqlite.Connection,
    *,
    connect_hub: ConnectHub | None,
    notification_hub: NotificationHub | None,
    fabric_payload: dict[str, Any],
    coordinator: Any = None,
    sender_node_id: str = "",
) -> dict[str, Any]:
    """Top-level dispatch for one inbound Connect-over-fabric frame.

    ``fabric_payload`` is the dict carried inside MSG_CONNECT_ENVELOPE,
    shape:

        {
            "envelope": "<serialised ConnectEnvelope JSON>",
            "source_did": "alice@instance-A",
            "sender_party_id": "<8-char alnum>",
        }

    ``coordinator`` + ``sender_node_id`` come from the fabric layer (the
    Ed25519-verified peer that delivered this envelope). When both are
    available and the ``source_did`` carries the local-instance sentinel,
    we rewrite it to use the sender's real hostname — see
    :func:`_normalise_source_did`. Tests that don't have a coordinator
    can omit both and the source_did flows through as-is.

    Returns a small result dict for tests + telemetry:
        {"applied": bool, "verb": str, "error": str}

    Never raises — fabric layer wants steady-state on any inbound
    failure (a buggy peer shouldn't tear our coordinator down).
    """

    raw = str(fabric_payload.get("envelope") or "")
    source_did = str(fabric_payload.get("source_did") or "")
    sender_party_id = str(fabric_payload.get("sender_party_id") or "")
    target_user_id = str(fabric_payload.get("target_user_id") or "")
    source_display_name = str(fabric_payload.get("source_display_name") or "")

    source_did = _normalise_source_did(
        source_did=source_did,
        coordinator=coordinator,
        sender_node_id=sender_node_id,
    )

    inner = deserialise_envelope(raw) if raw else None
    if inner is None:
        log.warning(
            "connect_fabric_inbound_malformed",
            payload_keys=list(fabric_payload.keys())[:5],
        )
        return {"applied": False, "verb": "", "error": "malformed"}

    if not source_did:
        return {"applied": False, "verb": inner.verb, "error": "missing_source_did"}

    # If target_user_id wasn't carried explicitly (older sender), try
    # the DID resolver — works only when the target DID's hostname
    # matches "this-instance".
    if not target_user_id:
        resolved = resolve_peer_did(inner.peer)
        if resolved is None or resolved.kind != "local":
            log.warning(
                "connect_fabric_inbound_misroute",
                target_did=inner.peer, source_did=source_did,
                kind=resolved.kind if resolved else "invalid",
            )
            return {"applied": False, "verb": inner.verb, "error": "misroute"}
        target_user_id = resolved.address

    # Federated-PBX admission gate (default-OFF — no change unless an
    # operator enabled fabric_federation_enabled). Enforces the sending
    # instance's denylist/revocation and the deny-by-default stranger
    # posture (knock/private/allowlist) BEFORE any delivery. Known
    # contacts and the open posture flow through untouched. See
    # augmentum/connect/federation_gate.py.
    if conn is not None:
        try:
            from augmentum.connect.federation_gate import (
                gate_inbound,
                gate_result_dict,
            )
            gate = await gate_inbound(
                conn,
                sender_node_id=sender_node_id,
                source_did=source_did,
                target_user_id=target_user_id,
                verb=inner.verb,
                body=str((inner.data or {}).get("body") or ""),
            )
            if not gate.allow:
                log.info(
                    "connect_fabric_inbound_gated",
                    verb=inner.verb, reason=gate.reason, source_did=source_did,
                )
                return gate_result_dict(inner.verb, gate)
        except Exception as exc:
            # On an UNEXPECTED gate error, the safe default depends on intent.
            # Default-off / "open" posture: fail OPEN so a gate bug never drops
            # the working transport for installs that didn't opt into gating.
            # But once an operator has chosen a RESTRICTIVE federation posture,
            # delivering an unvetted stranger frame because the gate threw would
            # silently defeat exactly the protection they asked for — so fail
            # CLOSED there. Either way, log loudly.
            log.warning(
                "connect_fabric_inbound_gate_error",
                error=str(exc)[:200], verb=inner.verb,
            )
            _fed_on = getattr(settings, "fabric_federation_enabled", False)
            _posture = getattr(settings, "fabric_admission_posture", "knock") or "knock"
            if _fed_on and _posture != "open" and inner.verb in (
                MSG_TEXT_SEND, MSG_INVITE,
            ):
                return {
                    "applied": False, "verb": inner.verb,
                    "error": "gate_error_failclosed", "gated": True,
                }

    # Remember the remote peer's human name on first relationship-creating
    # traffic (a message or an incoming call) so the recipient's contacts +
    # call-history lists resolve a username instead of the raw fabric DID.
    # Receipt/typing/signaling frames don't mint a relationship, so we skip
    # them — no auto-contact from a stray typing indicator.
    if source_display_name and conn is not None and inner.verb in (
        MSG_TEXT_SEND, MSG_INVITE,
    ):
        try:
            from augmentum.connect.contact_store import remember_peer_display_name
            await remember_peer_display_name(
                conn, user_id=target_user_id, peer_did=source_did,
                display_name=source_display_name,
            )
        except Exception as exc:  # never break delivery on a name-cache miss
            log.warning("connect_fabric_name_cache_failed", error=str(exc)[:160])

    if inner.verb in _TEXT_VERBS:
        return await _dispatch_text_verb(
            conn,
            connect_hub=connect_hub,
            notification_hub=notification_hub,
            env=inner,
            source_did=source_did,
            target_user_id=target_user_id,
            sender_party_id=sender_party_id,
        )
    if inner.verb in _CALL_VERBS:
        return await _dispatch_call_verb(
            conn,
            connect_hub=connect_hub,
            notification_hub=notification_hub,
            env=inner,
            source_did=source_did,
            target_user_id=target_user_id,
            sender_party_id=sender_party_id,
        )

    log.info(
        "connect_fabric_inbound_unknown_verb",
        verb=inner.verb, source_did=source_did,
    )
    return {"applied": False, "verb": inner.verb, "error": "unknown_verb"}


async def _dispatch_text_verb(
    conn: aiosqlite.Connection,
    *,
    connect_hub: ConnectHub | None,
    notification_hub: NotificationHub | None,
    env: ConnectEnvelope,
    source_did: str,
    target_user_id: str,
    sender_party_id: str,
) -> dict[str, Any]:
    """Per-verb dispatch for text-substrate verbs.

    Applies the recipient-side mutation against the local DB +
    routes the corresponding EVENT_* to the local user's signaling
    WS. Block-list checks happen HERE — the sending instance has
    no view of our local blocklist, so we silently drop downstream
    work when the local user has blocked the remote source.
    """
    from augmentum.connect.contact_store import is_blocked
    from augmentum.connect.message_store import (
        edit_message,
        get_message,
        get_or_create_thread,
        insert_message,
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
    )

    data = env.data or {}

    # Silent-block: target_user_id is the local recipient; check
    # whether they have blocked the source. If blocked, skip every
    # downstream mutation + WS + notification.
    blocked = False
    if conn is not None:
        try:
            blocked = await is_blocked(
                conn, user_id=target_user_id, peer_did=source_did,
            )
        except Exception as exc:
            log.warning(
                "connect_fabric_inbound_block_check_failed",
                error=str(exc)[:160],
            )

    if env.verb == MSG_TEXT_SEND:
        body = str(data.get("body") or "")
        thread_id = str(data.get("thread_id") or "")
        message_id = str(data.get("message_id") or "")
        fmt = str(data.get("format") or "plain")
        attachment_ref = str(data.get("attachment_ref") or "")
        reply_to = str(data.get("reply_to") or "")
        sent_at = str(data.get("sent_at") or "") or None
        transcript = str(data.get("transcript") or "")
        if not thread_id or not message_id:
            return {"applied": False, "verb": env.verb,
                    "error": "missing_ids"}
        if blocked:
            return {"applied": True, "verb": env.verb,
                    "error": "", "blocked": True}
        recipient_thread = await get_or_create_thread(
            conn,
            thread_id=thread_id,
            user_id=target_user_id,
            peer_did=source_did,
        )
        await insert_message(
            conn,
            message_id=message_id,
            thread_id=recipient_thread.thread_id,
            user_id=target_user_id,
            sender_did=source_did,
            body=body,
            format=fmt,
            attachment_ref=attachment_ref,
            reply_to=reply_to,
            sent_at=sent_at,
            transcript=transcript,
        )
        # Cross-instance attachment fetch metadata. The sender's
        # instance issued a fabric-signed token at dispatch time;
        # store it alongside the URL so the recipient's UI can
        # render the attachment via a direct fetch back to the
        # sender's box. Both NULL for plain text messages.
        attachment_token = str(data.get("attachment_token") or "")
        attachment_fetch_url = str(data.get("attachment_fetch_url") or "")
        if attachment_token and attachment_fetch_url:
            await conn.execute(
                """UPDATE connect_messages
                     SET attachment_fetch_url = ?,
                         attachment_fetch_token = ?
                     WHERE message_id = ? AND user_id = ?""",
                (attachment_fetch_url, attachment_token,
                 message_id, target_user_id),
            )
            await conn.commit()
        # Route EVENT_TEXT_RECEIVED to the local recipient's WS.
        # Carry the fabric attachment fetch URL + token through to the
        # client so live-delivered images render without a separate
        # catch-up fetch.
        if connect_hub is not None:
            event_data = {
                "thread_id": recipient_thread.thread_id,
                "message_id": message_id,
                "body": body,
                "format": fmt,
                "attachment_ref": attachment_ref,
                "reply_to": reply_to,
                "sent_at": sent_at or "",
                "sender_did": source_did,
                "transcript": transcript,
            }
            fetch_url = str(data.get("attachment_fetch_url") or "")
            fetch_token = str(data.get("attachment_token") or "")
            if fetch_url and fetch_token:
                event_data["attachment_fetch_url"] = fetch_url
                event_data["attachment_fetch_token"] = fetch_token
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event",
                    verb=EVENT_TEXT_RECEIVED,
                    peer=source_did,
                    data=event_data,
                ),
            )
        # Phase 2 keeps notification publish on the inbound path
        # minimal — the existing notification substrate fires from
        # _handle_send on the LOCAL path; for fabric we omit the
        # banner for now and revisit in Phase 5 polish (cross-
        # instance push needs its own threading to avoid duplicate
        # notifications when the receiver's UI is live).
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb == MSG_TEXT_EDIT:
        message_id = str(data.get("message_id") or "")
        thread_id = str(data.get("thread_id") or "")
        new_body = str(data.get("body") or "")
        if blocked:
            return {"applied": True, "verb": env.verb,
                    "error": "", "blocked": True}
        # Refuse edit-after-delete on the recipient side too —
        # otherwise a slow fabric delivery could overwrite a
        # tombstone with a stale edit.
        msg = await get_message(
            conn, message_id=message_id, user_id=target_user_id,
        )
        if msg is None or msg.deleted_at is not None:
            return {"applied": False, "verb": env.verb,
                    "error": "message_missing_or_deleted"}
        await edit_message(
            conn, message_id=message_id, user_id=target_user_id,
            body=new_body,
        )
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event",
                    verb=EVENT_TEXT_EDIT,
                    peer=source_did,
                    data={
                        "thread_id": thread_id,
                        "message_id": message_id,
                        "body": new_body,
                    },
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb == MSG_TEXT_DELETE:
        message_id = str(data.get("message_id") or "")
        thread_id = str(data.get("thread_id") or "")
        if blocked:
            return {"applied": True, "verb": env.verb,
                    "error": "", "blocked": True}
        msg = await get_message(
            conn, message_id=message_id, user_id=target_user_id,
        )
        if msg is None:
            return {"applied": False, "verb": env.verb,
                    "error": "message_missing"}
        if msg.deleted_at is not None:
            return {"applied": True, "verb": env.verb, "error": ""}
        await soft_delete_message(
            conn, message_id=message_id, user_id=target_user_id,
        )
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event",
                    verb=EVENT_TEXT_DELETE,
                    peer=source_did,
                    data={
                        "thread_id": thread_id,
                        "message_id": message_id,
                    },
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb == MSG_TEXT_READ:
        thread_id = str(data.get("thread_id") or "")
        last_read = str(data.get("last_read_message_id") or "")
        if blocked:
            return {"applied": True, "verb": env.verb,
                    "error": "", "blocked": True}
        # The local user is the ORIGINAL SENDER on this inbound
        # path — the remote (source_did) is the reader. No local
        # row mutation; just fire EVENT_TEXT_READ so their UI
        # updates the per-message read indicator.
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event",
                    verb=EVENT_TEXT_READ,
                    peer=source_did,
                    data={
                        "thread_id": thread_id,
                        "last_read_message_id": last_read,
                        "reader_did": source_did,
                        "marked": 0,
                    },
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb == MSG_TEXT_DELIVERED:
        thread_id = str(data.get("thread_id") or "")
        raw_ids = data.get("message_ids") or []
        if not isinstance(raw_ids, (list, tuple)):
            return {"applied": False, "verb": env.verb,
                    "error": "invalid_message_ids"}
        message_ids = [str(mid) for mid in raw_ids if mid][:200]
        if blocked:
            return {"applied": True, "verb": env.verb,
                    "error": "", "blocked": True}
        # The local user is the original sender; stamp their rows
        # delivered + fan back an EVENT_TEXT_DELIVERED.
        for mid in message_ids:
            await stamp_delivered(
                conn, message_id=mid, user_id=target_user_id,
            )
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event",
                    verb=EVENT_TEXT_DELIVERED,
                    peer=source_did,
                    data={
                        "thread_id": thread_id,
                        "message_ids": message_ids,
                    },
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb == MSG_TEXT_REACT:
        from datetime import UTC, datetime
        message_id = str(data.get("message_id") or "")
        thread_id = str(data.get("thread_id") or "")
        emoji = str(data.get("emoji") or "").strip()[:32]
        action = str(data.get("action") or "add").lower()
        if not message_id or not emoji:
            return {"applied": False, "verb": env.verb,
                    "error": "missing_message_id_or_emoji"}
        if blocked:
            return {"applied": True, "verb": env.verb,
                    "error": "", "blocked": True}
        now = datetime.now(UTC).isoformat()
        if action == "remove":
            await conn.execute(
                "DELETE FROM connect_message_reactions "
                "WHERE message_id = ? AND user_id = ? "
                "AND reactor_did = ? AND emoji = ?",
                (message_id, target_user_id, source_did, emoji),
            )
        else:
            await conn.execute(
                "INSERT OR IGNORE INTO connect_message_reactions "
                "(message_id, user_id, reactor_did, emoji, reacted_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (message_id, target_user_id, source_did, emoji, now),
            )
        await conn.commit()
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event",
                    verb=EVENT_TEXT_REACT,
                    peer=source_did,
                    data={
                        "thread_id": thread_id,
                        "message_id": message_id,
                        "emoji": emoji,
                        "action": action,
                        "reactor_did": source_did,
                    },
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb in (MSG_TYPING_START, MSG_TYPING_STOP):
        thread_id = str(data.get("thread_id") or "")
        if not thread_id or blocked:
            return {"applied": True, "verb": env.verb, "error": "",
                    "blocked": blocked}
        event_verb = (
            EVENT_TYPING_START if env.verb == MSG_TYPING_START
            else EVENT_TYPING_STOP
        )
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event",
                    verb=event_verb,
                    peer=source_did,
                    data={"thread_id": thread_id},
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    return {"applied": False, "verb": env.verb,
            "error": "verb_not_implemented"}


async def _dispatch_call_verb(
    conn: aiosqlite.Connection,
    *,
    connect_hub: ConnectHub | None,
    notification_hub: NotificationHub | None,
    env: ConnectEnvelope,
    source_did: str,
    target_user_id: str,
    sender_party_id: str,
) -> dict[str, Any]:
    """Per-verb dispatch for call signaling verbs.

    Mirrors what the local-path handlers do for the RECIPIENT — the
    sender row lives on the originating instance. Routes the WS event
    to the local user's signaling sockets so their dialer / incoming
    modal / call client reacts to the inbound signaling frame.
    """
    from augmentum.connect.call_routing import (
        _DEFAULT_MODALITIES,
        _cancel_invite_timer_safely,
        _insert_call_session,
        _log_call_event,
        _normalise_modalities,
        _update_call_session_state,
    )
    from augmentum.connect.contacts import local_did_for
    from augmentum.connect.protocol import (
        EVENT_ACCEPT,
        EVENT_ANSWER,
        EVENT_CANDIDATES,
        EVENT_DECLINE,
        EVENT_HANGUP,
        EVENT_INVITE,
        EVENT_MUTE_STATE,
        EVENT_NEGOTIATE,
        EVENT_OFFER,
        EVENT_SELECT_ANSWER,
        EVENT_VIDEO_STATE,
        MSG_ACCEPT,
        MSG_ANSWER,
        MSG_CANDIDATES,
        MSG_DECLINE,
        MSG_HANGUP,
        MSG_INVITE,
        MSG_MUTE_STATE,
        MSG_NEGOTIATE,
        MSG_OFFER,
        MSG_SELECT_ANSWER,
        MSG_VIDEO_STATE,
    )

    data = env.data or {}
    call_id = str(data.get("call_id") or "")
    target_did = local_did_for(target_user_id)

    if env.verb == MSG_INVITE:
        if not call_id:
            return {"applied": False, "verb": env.verb,
                    "error": "missing_call_id"}
        modalities = str(data.get("modalities") or _DEFAULT_MODALITIES)
        # Recipient row only — sender row is on the originating instance.
        await _insert_call_session(
            conn,
            call_id=call_id, user_id=target_user_id,
            initiator_did=source_did, receiver_did=target_did,
            modalities=modalities, state="invited",
        )
        await _log_call_event(
            conn, call_id=call_id, user_id=target_user_id,
            event_type="invited",
            event_data={
                "party_id": sender_party_id, "modalities": modalities,
                "fabric_source": source_did,
            },
        )
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event", verb=EVENT_INVITE,
                    peer=source_did,
                    data={
                        "call_id": call_id, "modalities": modalities,
                        "party_id": sender_party_id,
                    },
                ),
            )
        # Phase 4 minimum: skip notification publish on fabric INVITE
        # — receiver sees the incoming-call modal via WS just fine.
        # Cross-instance notification dedupe needs care; Phase 5
        # polish.
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb in (MSG_HANGUP, MSG_DECLINE):
        if not call_id:
            return {"applied": False, "verb": env.verb,
                    "error": "missing_call_id"}
        state = "ended" if env.verb == MSG_HANGUP else "declined"
        end_reason = (
            str(data.get("reason") or "user_hangup")
            if env.verb == MSG_HANGUP else "declined"
        )
        await _update_call_session_state(
            conn, call_id=call_id, user_id=target_user_id,
            state=state, end_reason=end_reason,
        )
        _cancel_invite_timer_safely(call_id)
        event_verb = (
            EVENT_HANGUP if env.verb == MSG_HANGUP else EVENT_DECLINE
        )
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event", verb=event_verb,
                    peer=source_did,
                    data={
                        "call_id": call_id,
                        "party_id": sender_party_id,
                        **(
                            {"reason": end_reason}
                            if env.verb == MSG_HANGUP else {}
                        ),
                    },
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb == MSG_ACCEPT:
        # Remote peer accepted our INVITE; transition local row to
        # connected + cancel any local timer + fire EVENT_ACCEPT.
        if not call_id:
            return {"applied": False, "verb": env.verb,
                    "error": "missing_call_id"}
        await _update_call_session_state(
            conn, call_id=call_id, user_id=target_user_id,
            state="connected",
        )
        _cancel_invite_timer_safely(call_id)
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event", verb=EVENT_ACCEPT,
                    peer=source_did,
                    data={"call_id": call_id,
                          "party_id": sender_party_id},
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    if env.verb in (
        MSG_OFFER, MSG_ANSWER, MSG_CANDIDATES,
        MSG_NEGOTIATE, MSG_SELECT_ANSWER, MSG_MUTE_STATE,
        MSG_VIDEO_STATE,
    ):
        if call_id:
            await _log_call_event(
                conn, call_id=call_id, user_id=target_user_id,
                event_type=f"signaling.{env.verb}",
                event_data={"party_id": sender_party_id},
            )
        # NEGOTIATE — update local recipient's modalities.
        if env.verb == MSG_NEGOTIATE and call_id:
            new_mods = data.get("modalities")
            if new_mods:
                norm = _normalise_modalities(new_mods)
                await conn.execute(
                    "UPDATE call_sessions SET modalities = ? "
                    "WHERE call_id = ? AND user_id = ?",
                    (norm, call_id, target_user_id),
                )
                await conn.commit()
        event_verb = {
            MSG_OFFER: EVENT_OFFER,
            MSG_ANSWER: EVENT_ANSWER,
            MSG_CANDIDATES: EVENT_CANDIDATES,
            MSG_NEGOTIATE: EVENT_NEGOTIATE,
            MSG_SELECT_ANSWER: EVENT_SELECT_ANSWER,
            MSG_MUTE_STATE: EVENT_MUTE_STATE,
            MSG_VIDEO_STATE: EVENT_VIDEO_STATE,
        }[env.verb]
        if connect_hub is not None:
            await connect_hub.route_to_user(
                target_user_id=target_user_id,
                envelope=ConnectEnvelope(
                    kind="event", verb=event_verb,
                    peer=source_did,
                    data={**data, "party_id": sender_party_id},
                ),
            )
        return {"applied": True, "verb": env.verb, "error": ""}

    return {"applied": False, "verb": env.verb,
            "error": "verb_not_implemented"}
