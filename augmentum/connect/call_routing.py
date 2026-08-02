"""Connect call lifecycle — invite routing + notification publish.

Sits between the signaling WS receive loop and the underlying
substrates (ConnectHub for direct peer routing, NotificationHub +
store for inbox-style surfacing, SQLite for persisted call records).

Phase 1 scope: same-instance routing only. A ``ResolvedPeer`` of kind
``"fabric"`` returns a ``call_error`` with code ``fabric_routing_pending``
so a future fabric-dispatch layer can replace this branch without
breaking the signaling-WS contract.

The action handler for ``connect.call.*`` notifications routes Accept
/ Decline clicks back to the initiator via ConnectHub — the
notification + signaling layers don't otherwise know about each
other, but they share state through this module.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.connect.contact_store import is_blocked
from augmentum.connect.contacts import (
    display_name_for_did,
    local_did_for,
    resolve_peer_did,
)
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
    ConnectEnvelope,
)
from augmentum.notifications import (
    IMPORTANCE_CRITICAL,
    NotificationAction,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request

    from augmentum.connect.hub import ConnectHub
    from augmentum.notifications import Notification, NotificationHub


log = get_logger(__name__)


# ── Constants ────────────────────────────────────────────────────


# Verbs we route directly to the other peer without inspecting the
# payload. The hub does the fan-out; we just transcribe the verb
# from msg-kind to event-kind.
#
# MSG_NEGOTIATE is NOT in this map — it has its own handler so the
# routing layer can persist the new modalities on call_sessions when
# the offer changes audio↔video tracks. See ``_handle_negotiate``.
_PEER_ROUTED_PAIRS: dict[str, str] = {
    MSG_OFFER:         EVENT_OFFER,
    MSG_ANSWER:        EVENT_ANSWER,
    MSG_CANDIDATES:    EVENT_CANDIDATES,
    MSG_HANGUP:        EVENT_HANGUP,
    MSG_DECLINE:       EVENT_DECLINE,
    MSG_SELECT_ANSWER: EVENT_SELECT_ANSWER,
    MSG_MUTE_STATE:    EVENT_MUTE_STATE,
    MSG_VIDEO_STATE:   EVENT_VIDEO_STATE,
    # MSG_ACCEPT is routed by this module too, but it goes through
    # a dedicated path that also publishes a missed-call resolution
    # (planned). For now, route as a plain event.
    MSG_ACCEPT:        EVENT_ACCEPT,
}


# Modality enumerator. The wire shape is a comma-separated string
# (``"audio"`` or ``"audio,video"``) for backward-compat with the
# invite payload. The set form is the canonical representation for
# comparison + transition logging.
_VALID_MODALITIES = frozenset({"audio", "video"})


def _normalise_modalities(value: Any) -> str:
    """Normalise a modality declaration to a canonical comma-joined form.

    Accepts a string ``"audio,video"`` or a list/set ``["video", "audio"]``.
    Returns ``"audio"`` when input is empty/garbage so the call always has
    at least one modality on record.
    """

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = [str(p).strip() for p in value if str(p).strip()]
    else:
        parts = []
    kept = [p for p in parts if p in _VALID_MODALITIES]
    if not kept:
        return "audio"
    # Canonical order: audio first, then video, dedup-stable.
    out: list[str] = []
    for kind in ("audio", "video"):
        if kind in kept and kind not in out:
            out.append(kind)
    return ",".join(out)


# Default invite modalities when the client omits them.
_DEFAULT_MODALITIES = "audio"


# 8-character alnum party_id per Matrix MSC2746. Distinguishes
# sibling devices the same user has online so accepted-from-phone
# vs accepted-from-laptop is resolvable.
def new_party_id() -> str:
    """Cryptographically-random 8-char alnum identifier."""

    return secrets.token_urlsafe(6)[:8]


# Shorter ids for call_id when the client doesn't supply one.
def new_call_id() -> str:
    return secrets.token_urlsafe(9)[:12]


# ── Data shape ───────────────────────────────────────────────────


@dataclass
class CallRoutingResult:
    """What ``handle_signaling_envelope`` produced.

    ``routed`` = number of peer WSes that received the envelope.
    ``notification_id`` = the recipient's notification id if we
    published one (only set on invite). ``error_code`` is non-empty
    when the routing failed in a way the sender should hear about.
    """

    routed: int = 0
    notification_id: str = ""
    call_id: str = ""
    error_code: str = ""
    error_message: str = ""


# ── DB helpers (call_sessions + call_events) ─────────────────────


def _now_iso() -> str:
    # Matches the augmentum convention used elsewhere — UTC ISO 8601.
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat()


async def _insert_call_session(
    conn: Any, *,
    call_id: str,
    user_id: str,
    initiator_did: str,
    receiver_did: str,
    modalities: str,
    state: str,
) -> None:
    """Insert one perspective's call_sessions row.

    Per migration 219: both ends store their own row (strict per-user
    isolation). On same-instance, this happens twice — once per user.
    """

    await conn.execute(
        """INSERT OR IGNORE INTO call_sessions
               (call_id, user_id, initiator_did, receiver_did,
                modalities, state, initiated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (call_id, user_id, initiator_did, receiver_did,
         modalities, state, _now_iso()),
    )
    await conn.commit()


async def _get_call_session_state(
    conn: Any, *, call_id: str, user_id: str,
) -> str:
    """Read the current ``state`` column of one call_sessions row.

    Returns an empty string if no row exists. Used by the accept/decline
    race guard: when MSG_DECLINE arrives from a sibling tab after one
    tab has already accepted, the recipient's row is already
    ``connected`` — the late decline is dropped instead of tearing the
    accepted call down.
    """

    cur = await conn.execute(
        "SELECT state FROM call_sessions "
        "WHERE call_id = ? AND user_id = ?",
        (call_id, user_id),
    )
    row = await cur.fetchone()
    return (row[0] if row else "") or ""


async def _update_call_session_state(
    conn: Any, *, call_id: str, user_id: str, state: str,
    end_reason: str = "",
) -> bool:
    """Move one perspective's call_session into a new state.

    Returns whether the row was updated. Stamps ``connected_at`` on
    transition into ``connected`` and ``ended_at`` on the terminal
    states (``ended``, ``declined``, ``missed``, ``failed``).
    """

    now = _now_iso()
    terminal_states = {"ended", "declined", "missed", "failed"}
    parts = ["state = ?"]
    params: list[Any] = [state]
    if state == "connected":
        parts.append("connected_at = COALESCE(connected_at, ?)")
        params.append(now)
    if state in terminal_states:
        parts.append("ended_at = COALESCE(ended_at, ?)")
        params.append(now)
        if end_reason:
            parts.append("end_reason = ?")
            params.append(end_reason)
    params.extend([call_id, user_id])
    cur = await conn.execute(
        f"UPDATE call_sessions SET {', '.join(parts)} "
        "WHERE call_id = ? AND user_id = ?",
        params,
    )
    await conn.commit()
    return cur.rowcount > 0


async def _log_call_event(
    conn: Any, *, call_id: str, user_id: str,
    event_type: str, event_data: dict[str, Any] | None = None,
) -> None:
    """Append one row to call_events for the audit trail."""

    import json

    await conn.execute(
        """INSERT INTO call_events
               (call_id, user_id, event_type, event_data, occurred_at)
             VALUES (?, ?, ?, ?, ?)""",
        (call_id, user_id, event_type,
         json.dumps(event_data or {}, separators=(",", ":")),
         _now_iso()),
    )
    await conn.commit()


# ── Signaling envelope handler ───────────────────────────────────


async def handle_signaling_envelope(
    *,
    conn: Any,
    connect_hub: ConnectHub,
    notification_hub: NotificationHub,
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    sender_party_id: str,
    sender_role: str = "",
    fabric_coordinator: Any = None,
    sender_connection_id: str = "",
) -> CallRoutingResult:
    """Route one inbound signaling envelope.

    The dispatcher reads ``env.peer`` to find the target user_id and
    routes accordingly. For ``MSG_INVITE`` specifically, it also
    publishes a ``connect.call.incoming`` notification on the
    recipient's feed.

    When the peer DID is fabric-form (``alice@instance-A``), the
    handler persists the sender-side state for the verb (call_sessions
    row for INVITE, terminal transition for HANGUP/DECLINE, call_events
    log for OFFER/ANSWER/CANDIDATES/NEGOTIATE) and dispatches the
    envelope over fabric. The recipient-side mirror + WS event +
    notification fire on the receiving instance.

    Returns a ``CallRoutingResult``. On error, the caller should
    surface ``error_code`` / ``error_message`` back to the sender as
    an ``EVENT_ERROR`` envelope.
    """

    # Resolve peer routing target.
    resolved = resolve_peer_did(env.peer)
    if resolved is None:
        return CallRoutingResult(
            error_code="peer_did_invalid",
            error_message=f"could not parse peer DID '{env.peer}'",
        )
    if resolved.kind == "fabric":
        # A guest is scoped to a LOCAL host grant — never a cross-instance
        # fabric peer. Checked BEFORE the fabric dispatch (the local-target ACL
        # gate below is unreachable here); without it a guest could call any
        # user on any paired instance.
        if sender_role == "guest":
            return CallRoutingResult(
                error_code="guest_scope_violation",
                error_message="guests may only reach their host",
            )
        return await _handle_fabric_signaling_outbound(
            conn=conn,
            fabric_coordinator=fabric_coordinator,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            sender_party_id=sender_party_id,
            target_hostname=resolved.address,
        )
    if resolved.kind != "local":
        return CallRoutingResult(
            error_code="peer_routing_unsupported",
            error_message=f"routing kind '{resolved.kind}' not supported",
        )
    target_user_id = resolved.address

    # Guest ACL gate (Phase 3a) — a role='guest' caller may ring ONLY the host
    # it holds a live grant for; one check covers every signaling verb.
    from augmentum.connect.guest_grant_store import guest_scope_blocked
    if await guest_scope_blocked(
        conn, sender_user_id=sender_user_id, sender_role=sender_role,
        target_user_id=target_user_id,
    ):
        return CallRoutingResult(
            error_code="guest_scope_violation",
            error_message="guests may only reach their host",
        )

    if env.verb == MSG_INVITE:
        return await _handle_invite(
            conn=conn,
            connect_hub=connect_hub,
            notification_hub=notification_hub,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            sender_party_id=sender_party_id,
            target_user_id=target_user_id,
        )

    if env.verb == MSG_NEGOTIATE:
        return await _handle_negotiate(
            conn=conn,
            connect_hub=connect_hub,
            env=env,
            sender_user_id=sender_user_id,
            sender_did=sender_did,
            sender_party_id=sender_party_id,
            target_user_id=target_user_id,
        )

    if env.verb in _PEER_ROUTED_PAIRS:
        event_verb = _PEER_ROUTED_PAIRS[env.verb]
        call_id = str(env.data.get("call_id") or "")

        # MSG_DECLINE race guard: a sibling tab may have already
        # accepted this invite on the same user_id. Without this check,
        # the late decline from tab 2 transitions both rows to
        # ``declined`` and routes EVENT_DECLINE to the caller, which
        # tears down the WebRTC call tab 1 just established. Drop the
        # decline and echo EVENT_ACCEPT back to the declining tab so
        # its modal closes silently.
        if env.verb == MSG_DECLINE and call_id:
            current = await _get_call_session_state(
                conn, call_id=call_id, user_id=sender_user_id,
            )
            if current == "connected":
                await _log_call_event(
                    conn, call_id=call_id, user_id=sender_user_id,
                    event_type="decline_after_accept_dropped",
                    event_data={"party_id": sender_party_id},
                )
                # Echo an EVENT_ACCEPT to the declining tab so its
                # incoming-modal sees the call_id as resolved and
                # closes. Send to ALL of sender's tabs (idempotent —
                # other tabs already closed when the original accept
                # echoed). ``peer`` is the call's initiator DID, which
                # we read from the row.
                cur = await conn.execute(
                    "SELECT initiator_did FROM call_sessions "
                    "WHERE call_id = ? AND user_id = ? LIMIT 1",
                    (call_id, sender_user_id),
                )
                row = await cur.fetchone()
                initiator_did = (row[0] if row else "") or ""
                await connect_hub.route_to_user(
                    target_user_id=sender_user_id,
                    envelope=ConnectEnvelope(
                        kind="event",
                        verb=EVENT_ACCEPT,
                        corr_id=env.corr_id,
                        peer=initiator_did,
                        data={
                            "call_id": call_id,
                            "resolved_by": "sibling",
                        },
                    ),
                )
                return CallRoutingResult(routed=0, call_id=call_id)

        # Persist a call_events row when we have a call_id in payload.
        if call_id:
            await _log_call_event(
                conn, call_id=call_id, user_id=sender_user_id,
                event_type=f"signaling.{env.verb}",
                event_data={"party_id": sender_party_id},
            )

        # Terminal state transitions for hangup/decline (initiator side).
        if env.verb == MSG_HANGUP and call_id:
            await _update_call_session_state(
                conn, call_id=call_id, user_id=sender_user_id,
                state="ended",
                end_reason=str(env.data.get("reason") or ""),
            )
            await _update_call_session_state(
                conn, call_id=call_id, user_id=target_user_id,
                state="ended",
                end_reason=str(env.data.get("reason") or ""),
            )
            _cancel_invite_timer_safely(call_id)
        elif env.verb == MSG_DECLINE and call_id:
            await _update_call_session_state(
                conn, call_id=call_id, user_id=sender_user_id,
                state="declined", end_reason="declined",
            )
            await _update_call_session_state(
                conn, call_id=call_id, user_id=target_user_id,
                state="declined", end_reason="declined",
            )
            _cancel_invite_timer_safely(call_id)
        elif env.verb == MSG_ACCEPT and call_id:
            # Transition both perspectives to ``connected`` so a later
            # MSG_DECLINE from a sibling tab can be recognised as
            # stale by the race guard above. The ``connected`` state
            # is also what the notification-action HTTP path writes.
            await _update_call_session_state(
                conn, call_id=call_id, user_id=sender_user_id,
                state="connected",
            )
            await _update_call_session_state(
                conn, call_id=call_id, user_id=target_user_id,
                state="connected",
            )
            _cancel_invite_timer_safely(call_id)

        routed = await connect_hub.route_to_user(
            target_user_id=target_user_id,
            envelope=ConnectEnvelope(
                kind="event",
                verb=event_verb,
                corr_id=env.corr_id,
                peer=sender_did,  # event.from for the recipient
                data={**env.data, "party_id": sender_party_id},
            ),
        )

        # Sibling-tab fanout: when the user accepts or declines on
        # one tab, their other tabs need to dismiss the ringing modal
        # and stop the ringtone. The hub fans the same EVENT_* back
        # to sender_user_id's other connections, with the originating
        # tab excluded so it doesn't echo to itself. Same-device
        # browser tabs also get an instant client-side dismiss via
        # BroadcastChannel (see ui/scripts/connect/incoming-modal.js);
        # this server-side echo is the cross-device path and the
        # source of truth when a sibling tab has stale state.
        if env.verb in (MSG_ACCEPT, MSG_DECLINE) and call_id:
            try:
                await connect_hub.route_to_user(
                    target_user_id=sender_user_id,
                    envelope=ConnectEnvelope(
                        kind="event",
                        verb=event_verb,
                        corr_id=env.corr_id,
                        peer=env.peer,  # the caller's DID
                        data={
                            "call_id": call_id,
                            "resolved_by": "sibling",
                        },
                    ),
                    exclude_connection_id=sender_connection_id,
                )
            except Exception as exc:
                log.warning(
                    "connect_sibling_fanout_failed",
                    call_id=call_id, verb=env.verb,
                    error=str(exc)[:160],
                )

        return CallRoutingResult(routed=routed, call_id=call_id)

    return CallRoutingResult(
        error_code="unsupported_verb",
        error_message=f"verb '{env.verb}' has no routing handler",
    )


async def _handle_invite(
    *,
    conn: Any,
    connect_hub: ConnectHub,
    notification_hub: NotificationHub,
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    sender_party_id: str,
    target_user_id: str,
) -> CallRoutingResult:
    """Build the full invite fan-out.

    Steps:
      1. Mint call_id if the client didn't supply one.
      2. Insert call_sessions rows for both perspectives.
      3. Log an invite event for the audit trail.
      4. Route EVENT_INVITE to the recipient's signaling WS.
      5. Publish ``connect.call.incoming`` notification with accept
         + decline action buttons; payload carries the routing
         state the action handler needs.
    """

    from augmentum.connect.contacts import local_did_for
    from augmentum.notifications.hub import publish_and_dispatch

    call_id = str(env.data.get("call_id") or "") or new_call_id()
    modalities = str(env.data.get("modalities") or _DEFAULT_MODALITIES)
    target_did = local_did_for(target_user_id)

    # Sender row: state="ringing" (waiting for callee).
    await _insert_call_session(
        conn,
        call_id=call_id, user_id=sender_user_id,
        initiator_did=sender_did, receiver_did=target_did,
        modalities=modalities, state="ringing",
    )

    # Silent-block: target has blocked caller. Skip recipient row +
    # WS event + notification + missed-call timer; transition the
    # caller's row straight to ``missed`` so their UI clears the
    # "ringing" state quickly. From the caller's perspective this
    # looks similar to a peer whose ringing immediately timed out —
    # the block is never revealed.
    if await is_blocked(
        conn, user_id=target_user_id, peer_did=sender_did,
    ):
        await _update_call_session_state(
            conn, call_id=call_id, user_id=sender_user_id,
            state="missed", end_reason="no_answer",
        )
        await _log_call_event(
            conn, call_id=call_id, user_id=sender_user_id,
            event_type="invited",
            event_data={"target_user_id": target_user_id, "modalities": modalities},
        )
        return CallRoutingResult(routed=0, call_id=call_id)

    # Recipient row: state="invited" (UI should ring on attach).
    await _insert_call_session(
        conn,
        call_id=call_id, user_id=target_user_id,
        initiator_did=sender_did, receiver_did=target_did,
        modalities=modalities, state="invited",
    )
    await _log_call_event(
        conn, call_id=call_id, user_id=sender_user_id,
        event_type="invited",
        event_data={"target_user_id": target_user_id, "modalities": modalities},
    )

    # Route EVENT_INVITE to the recipient's active signaling WS.
    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_INVITE,
            corr_id=env.corr_id,
            peer=sender_did,
            data={
                **env.data,
                "call_id": call_id,
                "modalities": modalities,
                "party_id": sender_party_id,
            },
        ),
    )

    # Publish connect.call.incoming notification. The payload carries
    # the state the action handler needs to route accept/decline back.
    notification_id = ""
    try:
        sender_name = await display_name_for_did(conn, sender_did)
        notification_id = await publish_and_dispatch(
            conn,
            hub=notification_hub,
            user_id=target_user_id,
            channel_id="connect.call.incoming",
            source="connect",
            title=f"Call from {sender_name or sender_did}",
            body=f"{modalities}",
            importance=IMPORTANCE_CRITICAL,
            dedupe_key=call_id,
            thread_id=call_id,
            actions=[
                NotificationAction(
                    id="accept", label="Accept", style="primary",
                ),
                NotificationAction(
                    id="decline", label="Decline", style="danger",
                ),
            ],
            payload={
                "call_id": call_id,
                "initiator_did": sender_did,
                "initiator_user_id": sender_user_id,
                "initiator_party_id": sender_party_id,
                "modalities": modalities,
            },
            transient=False,
            icon="phone",
        )
    except Exception as exc:
        log.warning(
            "connect_invite_notification_failed",
            call_id=call_id, target_user_id=target_user_id,
            error=str(exc)[:160],
        )

    # Arm the missed-call timer. Cancelled on accept/decline/hangup;
    # fires _mark_missed after lifetime_ms if no one acts.
    try:
        from augmentum.connect.call_lifecycle import arm_invite_timer
        from augmentum.connect.protocol import DEFAULT_INVITE_LIFETIME_MS

        lifetime_ms = int(env.data.get("lifetime") or DEFAULT_INVITE_LIFETIME_MS)
        arm_invite_timer(
            conn=conn,
            connect_hub=connect_hub,
            notification_hub=notification_hub,
            call_id=call_id,
            initiator_user_id=sender_user_id,
            initiator_did=sender_did,
            recipient_user_id=target_user_id,
            recipient_did=target_did,
            lifetime_ms=lifetime_ms,
        )
    except Exception as exc:
        log.warning(
            "connect_invite_timer_arm_failed",
            call_id=call_id, error=str(exc)[:160],
        )

    return CallRoutingResult(
        routed=routed,
        notification_id=notification_id,
        call_id=call_id,
    )


async def _handle_negotiate(
    *,
    conn: Any,
    connect_hub: ConnectHub,
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    sender_party_id: str,
    target_user_id: str,
) -> CallRoutingResult:
    """Mid-call renegotiation.

    Triggered when one side wants to change media configuration without
    placing a new call — adding video to an audio-only call, dropping
    video back, or (future) starting a screen-share track.

    Per the protocol's ``MSG_NEGOTIATE`` doc, ``data`` carries:

      ``description`` — the SDP offer or answer driving the change
      ``modalities``  — the new modality declaration (optional)

    The wire shape is intentionally tolerant: a peer can renegotiate
    without declaring modalities (e.g. codec swap), in which case we
    pass the SDP through and don't touch ``call_sessions.modalities``.
    The state row only changes when the declaration is present and
    different from what we have on file. That keeps the routing layer
    out of SDP parsing while still maintaining accurate per-side
    modality state for the call history surface.

    The recipient receives ``EVENT_NEGOTIATE`` with the same payload;
    their UI is expected to do the SDP offer/answer dance on the
    existing peer connection (no new RTCPeerConnection).
    """

    call_id = str(env.data.get("call_id") or "")
    if not call_id:
        return CallRoutingResult(
            error_code="missing_call_id",
            error_message="negotiate requires call_id",
        )

    raw_modalities = env.data.get("modalities")
    declared_modalities = (
        _normalise_modalities(raw_modalities) if raw_modalities is not None else ""
    )

    # When a new modality declaration arrives, apply it to BOTH
    # perspectives. The receiver hasn't accepted yet (they'd reject
    # via the UI), but the row is the source-of-truth for the
    # call-history surface so we keep both sides aligned.
    old_modalities = ""
    if declared_modalities:
        cur = await conn.execute(
            "SELECT modalities FROM call_sessions "
            "WHERE call_id = ? AND user_id = ?",
            (call_id, sender_user_id),
        )
        row = await cur.fetchone()
        old_modalities = (row[0] if row else "") or ""
        if old_modalities != declared_modalities:
            await conn.execute(
                "UPDATE call_sessions SET modalities = ? "
                "WHERE call_id = ? AND user_id IN (?, ?)",
                (declared_modalities, call_id, sender_user_id, target_user_id),
            )
            await conn.commit()

    await _log_call_event(
        conn, call_id=call_id, user_id=sender_user_id,
        event_type="renegotiate",
        event_data={
            "party_id": sender_party_id,
            "modalities": declared_modalities or old_modalities,
            "previous_modalities": old_modalities,
            "description_type": str(
                (env.data.get("description") or {}).get("type") or ""
            ),
        },
    )

    routed = await connect_hub.route_to_user(
        target_user_id=target_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_NEGOTIATE,
            corr_id=env.corr_id,
            peer=sender_did,
            data={**env.data, "party_id": sender_party_id},
        ),
    )
    return CallRoutingResult(routed=routed, call_id=call_id)


# ── Notification action handler ──────────────────────────────────


def _cancel_invite_timer_safely(call_id: str) -> None:
    """Best-effort timer cancel — only fires when the lifecycle module
    is loaded. Routing remains functional even when the timer module
    is absent (e.g. in tests that don't bootstrap the asyncio loop)."""
    try:
        from augmentum.connect.call_lifecycle import cancel_invite_timer
        cancel_invite_timer(call_id)
    except Exception:
        pass


async def handle_call_action(
    notification: Notification, action_id: str, request: Request,
) -> dict[str, Any]:
    """Action handler for ``connect.call.*`` notifications.

    Reads the notification's payload to find the call_id + initiator
    routing info, updates ``call_sessions`` for both perspectives,
    then routes an ``EVENT_ACCEPT`` or ``EVENT_DECLINE`` back through
    ConnectHub so the initiator's UI can react.

    Wired up by ``connect_routes.py`` at app startup via
    ``register_action_handler('connect.call.*', handle_call_action)``.
    """

    if action_id not in ("accept", "decline"):
        return {
            "status": "error",
            "error": f"unknown action '{action_id}'",
        }

    payload = notification.payload or {}
    call_id = str(payload.get("call_id") or "")
    initiator_did = str(payload.get("initiator_did") or "")
    initiator_user_id = str(payload.get("initiator_user_id") or "")
    if not call_id or not initiator_user_id or not initiator_did:
        return {
            "status": "error",
            "error": "notification payload missing call routing state",
        }

    connect_hub = getattr(request.app.state, "connect_hub", None)
    if connect_hub is None:
        return {
            "status": "error",
            "error": "connect signaling not active",
        }

    # Resolve the conn from app state so we can update call_sessions.
    from augmentum.state.backends.sqlite import SQLiteBackend
    sm = getattr(request.app.state, "state_manager", None)
    conn = (
        sm.backend.conn
        if sm is not None and isinstance(getattr(sm, "backend", None), SQLiteBackend)
        else None
    )

    recipient_user_id = notification.user_id
    receiver_did = local_did_for(recipient_user_id)

    # Resolve initiator DID for the sibling-fanout payload — the
    # banner-click path is HTTP, so we don't have one in scope.
    initiator_did_for_sibling = ""
    if conn is not None:
        cur = await conn.execute(
            "SELECT initiator_did FROM call_sessions "
            "WHERE call_id = ? AND user_id = ? LIMIT 1",
            (call_id, recipient_user_id),
        )
        row = await cur.fetchone()
        initiator_did_for_sibling = (row[0] if row else "") or ""

    if action_id == "accept":
        _cancel_invite_timer_safely(call_id)
        # Both perspectives transition to 'connected'.
        if conn is not None:
            await _update_call_session_state(
                conn, call_id=call_id, user_id=initiator_user_id,
                state="connected",
            )
            await _update_call_session_state(
                conn, call_id=call_id, user_id=recipient_user_id,
                state="connected",
            )
            await _log_call_event(
                conn, call_id=call_id, user_id=recipient_user_id,
                event_type="accepted",
                event_data={"action_id": action_id},
            )
        # Route an EVENT_ACCEPT back to the initiator.
        delivered = await connect_hub.route_to_user(
            target_user_id=initiator_user_id,
            envelope=ConnectEnvelope(
                kind="event",
                verb=EVENT_ACCEPT,
                peer=receiver_did,
                data={"call_id": call_id},
            ),
        )
        # Fan to recipient's sibling tabs so their incoming-modals
        # dismiss. HTTP path has no originating connection_id so we
        # fan to every tab; same-tab idempotently no-ops.
        try:
            await connect_hub.route_to_user(
                target_user_id=recipient_user_id,
                envelope=ConnectEnvelope(
                    kind="event",
                    verb=EVENT_ACCEPT,
                    peer=initiator_did_for_sibling,
                    data={"call_id": call_id, "resolved_by": "sibling"},
                ),
            )
        except Exception as exc:
            log.warning(
                "connect_sibling_fanout_failed",
                call_id=call_id, verb="accept", path="notification",
                error=str(exc)[:160],
            )
        return {
            "status": "accepted",
            "call_id": call_id,
            "delivered_to_initiator": delivered,
        }

    # Decline.
    _cancel_invite_timer_safely(call_id)
    if conn is not None:
        await _update_call_session_state(
            conn, call_id=call_id, user_id=initiator_user_id,
            state="declined", end_reason="declined",
        )
        await _update_call_session_state(
            conn, call_id=call_id, user_id=recipient_user_id,
            state="declined", end_reason="declined",
        )
        await _log_call_event(
            conn, call_id=call_id, user_id=recipient_user_id,
            event_type="declined",
            event_data={"action_id": action_id},
        )
    delivered = await connect_hub.route_to_user(
        target_user_id=initiator_user_id,
        envelope=ConnectEnvelope(
            kind="event",
            verb=EVENT_DECLINE,
            peer=receiver_did,
            data={"call_id": call_id, "reason": "declined"},
        ),
    )
    try:
        await connect_hub.route_to_user(
            target_user_id=recipient_user_id,
            envelope=ConnectEnvelope(
                kind="event",
                verb=EVENT_DECLINE,
                peer=initiator_did_for_sibling,
                data={
                    "call_id": call_id,
                    "reason": "declined",
                    "resolved_by": "sibling",
                },
            ),
        )
    except Exception as exc:
        log.warning(
            "connect_sibling_fanout_failed",
            call_id=call_id, verb="decline", path="notification",
            error=str(exc)[:160],
        )
    return {
        "status": "declined",
        "call_id": call_id,
        "delivered_to_initiator": delivered,
    }


# ── Fabric (cross-instance) outbound signaling dispatch ───────────


async def _handle_fabric_signaling_outbound(
    *,
    conn: Any,
    fabric_coordinator: Any,
    env: ConnectEnvelope,
    sender_user_id: str,
    sender_did: str,
    sender_party_id: str,
    target_hostname: str,
) -> CallRoutingResult:
    """Sender-side persistence + fabric dispatch for one call verb.

    Per-verb behaviour:
      * INVITE — insert sender's call_sessions row (state=ringing),
        log invited event, arm missed-call timer locally so the
        sender's own UI gets a missed-call notification if the
        remote peer never answers.
      * HANGUP / DECLINE — terminal transition on sender's row,
        cancel any local invite timer.
      * ACCEPT — cancel sender-side timer (the call is no longer
        unanswered from our perspective once we accept on the other
        side; ACCEPT going outbound means we're echoing accept to
        the remote initiator).
      * OFFER / ANSWER / CANDIDATES / NEGOTIATE / SELECT_ANSWER —
        log a call_event row for the audit trail; no state change.

    All branches then dispatch the envelope over fabric. The remote
    instance's inbound handler does the recipient-side equivalent.
    """
    from augmentum.connect.fabric_transport import dispatch_fabric_envelope

    data = env.data or {}
    call_id = str(data.get("call_id") or "")
    target_did = env.peer  # fabric DID, e.g. "bob@instance-B"

    if env.verb == MSG_INVITE:
        call_id = call_id or new_call_id()
        modalities = str(data.get("modalities") or _DEFAULT_MODALITIES)
        # Sender row only. Recipient row appears on remote instance.
        await _insert_call_session(
            conn,
            call_id=call_id, user_id=sender_user_id,
            initiator_did=sender_did, receiver_did=target_did,
            modalities=modalities, state="ringing",
        )
        await _log_call_event(
            conn, call_id=call_id, user_id=sender_user_id,
            event_type="invited",
            event_data={
                "party_id": sender_party_id,
                "modalities": modalities,
                "fabric_target": target_hostname,
            },
        )
        # Arm the missed-call timer on the sender's side. If the
        # remote peer's instance is reachable, the timer is cancelled
        # by the inbound EVENT_ACCEPT/EVENT_DECLINE/EVENT_HANGUP that
        # fabric pushes back. Re-mint call_id into env.data so the
        # remote sees it.
        env.data["call_id"] = call_id
        env.data["modalities"] = modalities
        # Note: missed-call timer is NOT armed on the sender side for
        # fabric calls in v1. The remote recipient's instance arms its
        # own timer (single-instance pattern). If the remote instance
        # is unreachable AND the call goes unanswered, the sender's UI
        # is responsible for surfacing the stall — covered by the
        # ringing-state stall heuristic the dialer already runs.

    elif env.verb in (MSG_HANGUP, MSG_DECLINE):
        if call_id:
            state = "ended" if env.verb == MSG_HANGUP else "declined"
            end_reason = (
                str(data.get("reason") or "user_hangup")
                if env.verb == MSG_HANGUP else "declined"
            )
            await _update_call_session_state(
                conn, call_id=call_id, user_id=sender_user_id,
                state=state, end_reason=end_reason,
            )
            _cancel_invite_timer_safely(call_id)
            await _log_call_event(
                conn, call_id=call_id, user_id=sender_user_id,
                event_type=f"signaling.{env.verb}",
                event_data={"party_id": sender_party_id},
            )

    elif env.verb == MSG_ACCEPT:
        if call_id:
            _cancel_invite_timer_safely(call_id)
            await _log_call_event(
                conn, call_id=call_id, user_id=sender_user_id,
                event_type=f"signaling.{env.verb}",
                event_data={"party_id": sender_party_id},
            )

    elif env.verb in (
        MSG_OFFER, MSG_ANSWER, MSG_CANDIDATES,
        MSG_NEGOTIATE, MSG_SELECT_ANSWER, MSG_MUTE_STATE,
        MSG_VIDEO_STATE,
    ):
        if call_id:
            await _log_call_event(
                conn, call_id=call_id, user_id=sender_user_id,
                event_type=f"signaling.{env.verb}",
                event_data={"party_id": sender_party_id},
            )
        # NEGOTIATE — update local sender's modalities mirror.
        if env.verb == MSG_NEGOTIATE and call_id:
            new_mods = data.get("modalities")
            if new_mods:
                norm = _normalise_modalities(new_mods)
                await conn.execute(
                    "UPDATE call_sessions SET modalities = ? "
                    "WHERE call_id = ? AND user_id = ?",
                    (norm, call_id, sender_user_id),
                )
                await conn.commit()

    else:
        return CallRoutingResult(
            error_code="unsupported_verb",
            error_message=f"fabric routing for '{env.verb}' not wired",
        )

    result = await dispatch_fabric_envelope(
        conn,
        coordinator=fabric_coordinator,
        target_hostname=target_hostname,
        source_did=sender_did,
        sender_user_id=sender_user_id,
        sender_party_id=sender_party_id,
        envelope=env,
    )

    return CallRoutingResult(
        routed=1 if result.queued else 0,
        call_id=call_id,
        error_code=result.error_code,
        error_message=result.error_message,
    )


