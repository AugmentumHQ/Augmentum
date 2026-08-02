"""ConnectHub — in-process WS connection registry + presence.

One ``ConnectHub`` per Augmentum instance, mounted on ``app.state``.
Tracks ``{user_id → list[WebSocket]}`` so the signaling endpoint can:

* Notify contacts when a user comes online / drops off (presence).
* Route invite/offer/answer/ICE/hangup envelopes from one peer to
  another on the same instance.
* Detect orphaned connections (a WS closes without going through
  the normal hangup path).

A single user may have multiple WS connections at once — e.g. desktop
+ phone both in Connect mode — so the registry value is a list. When
routing inbound events for that user, the hub fans the event to every
connection it has for that user_id.

Phase 1 scope (this scaffold):

* Same-instance routing only. Cross-instance (fabric peer) routing
  is the next layer up — it'll dispatch via the fabric session pipe
  rather than the local hub, and re-enter this hub on the receiving
  instance. The dispatch shim lives in connect_routes' WS receive
  loop, not here.

* No persistence of presence — the hub is in-memory. If the process
  restarts, every Connect WS reconnects and re-announces.

* No quality / rate-limiting — Phase 1 trusts the small user base.

What this hub does NOT do:

* Mint TURN credentials — that's ``augmentum/calling/turn_credentials``.
* Persist call_sessions / call_events / connect_messages rows — the
  signaling route does that around the hub's routing calls.
* Resolve ``peer_did`` to a user_id — that's the connect_contacts
  table; the route layer translates DID → user_id before calling
  ``ConnectHub.route_to_user``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.connect.protocol import (
    EVENT_PRESENCE_UPDATE,
    ConnectEnvelope,
    serialise_envelope,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import WebSocket


log = get_logger(__name__)


@dataclass
class _Attachment:
    """One open WS for one user."""

    ws: WebSocket
    connection_id: str
    user_id: str
    user_did: str
    attached_at: float = field(default_factory=time.time)


class ConnectHub:
    """In-memory presence + routing registry.

    Methods are async because the underlying WS sends are async; the
    in-memory map itself is touched under a lock so concurrent
    attach/detach from many WS connections doesn't race.
    """

    def __init__(self) -> None:
        # user_id → list of attached WS handles (multiple devices OK).
        self._by_user: dict[str, list[_Attachment]] = {}
        # connection_id → attachment, for fast detach.
        self._by_conn: dict[str, _Attachment] = {}
        self._lock = asyncio.Lock()
        # Monotonic counter for connection_ids — keeps logs greppable
        # without requiring a uuid roundtrip on every WS connect.
        self._next_conn_seq = 0
        # Optional async hook ``sink(user_id: str, online: bool)`` invoked on
        # the first-connect / last-disconnect transitions so a DB-backed
        # presence store can persist last-seen. Kept optional + best-effort so
        # the hub stays DB-agnostic and a sink failure never breaks attach.
        self._presence_sink = None

    def set_presence_sink(self, sink) -> None:
        """Install the persistent-presence hook (see ``_presence_sink``)."""
        self._presence_sink = sink

    async def _emit_presence_sink(self, user_id: str, online: bool) -> None:
        if self._presence_sink is None:
            return
        try:
            await self._presence_sink(user_id, online)
        except Exception:
            log.warning("connect_presence_sink_failed", user_id=user_id, exc_info=True)

    async def attach(
        self, *, ws: WebSocket, user_id: str, user_did: str,
    ) -> _Attachment:
        """Register a freshly-accepted WS for ``user_id``."""

        async with self._lock:
            self._next_conn_seq += 1
            conn_id = f"conn-{self._next_conn_seq}"
            att = _Attachment(
                ws=ws, connection_id=conn_id, user_id=user_id, user_did=user_did,
            )
            self._by_user.setdefault(user_id, []).append(att)
            self._by_conn[conn_id] = att
            is_first_connection = len(self._by_user[user_id]) == 1
        if is_first_connection:
            log.info(
                "connect_hub_user_online",
                user_id=user_id, connection_id=conn_id,
            )
            # Phase 1 placeholder: presence broadcast to contacts is
            # the next layer. Hook stays here so the call site (route
            # WS handler) doesn't need to know whether broadcast is
            # wired yet.
            await self._broadcast_presence(user_did, "online")
            await self._emit_presence_sink(user_id, True)
        return att

    async def detach(self, connection_id: str) -> None:
        """Remove a closed WS."""

        async with self._lock:
            att = self._by_conn.pop(connection_id, None)
            if att is None:
                return
            connections = self._by_user.get(att.user_id, [])
            try:
                connections.remove(att)
            except ValueError:
                pass
            became_offline = not connections
            if became_offline:
                self._by_user.pop(att.user_id, None)
        if became_offline:
            log.info(
                "connect_hub_user_offline",
                user_id=att.user_id, connection_id=connection_id,
            )
            await self._broadcast_presence(att.user_did, "offline")
            await self._emit_presence_sink(att.user_id, False)

    async def route_to_user(
        self, *, target_user_id: str, envelope: ConnectEnvelope,
        exclude_connection_id: str = "",
    ) -> int:
        """Send ``envelope`` to every active WS for a user. Returns count.

        Zero connections is not an error here — the route layer
        decides whether that means "store as missed call" or
        "fail-closed with error to sender".

        ``exclude_connection_id`` skips one specific connection — used
        when echoing a sender-side action to its sibling tabs so the
        originating tab doesn't receive its own broadcast back.
        """

        async with self._lock:
            targets = [
                att for att in self._by_user.get(target_user_id, ())
                if att.connection_id != exclude_connection_id
            ]
        if not targets:
            return 0
        payload = serialise_envelope(envelope)
        delivered = 0
        for att in targets:
            try:
                await att.ws.send_text(payload)
                delivered += 1
            except Exception as exc:
                log.warning(
                    "connect_hub_route_failed",
                    target_user_id=target_user_id,
                    connection_id=att.connection_id,
                    verb=envelope.verb,
                    error=str(exc)[:160],
                )
        return delivered

    def online_user_ids(self) -> list[str]:
        """Snapshot of currently-online user_ids (no lock — best-effort)."""

        return list(self._by_user.keys())

    def is_online(self, user_id: str) -> bool:
        """Whether a user has at least one active WS connection."""

        return user_id in self._by_user

    async def _broadcast_presence(self, user_did: str, status: str) -> None:
        """Notify everyone else about a user's presence change.

        Phase 1 stub: blasts to every connected peer regardless of
        contact relationship. The contact-scoped variant (only notify
        peers who have this user in their connect_contacts) is the
        next iteration once the contacts table is being populated.
        """

        async with self._lock:
            targets = [
                att
                for atts in self._by_user.values()
                for att in atts
                if att.user_did != user_did
            ]
        if not targets:
            return
        env = ConnectEnvelope(
            kind="event",
            verb=EVENT_PRESENCE_UPDATE,
            peer=user_did,
            data={"peer_did": user_did, "status": status},
        )
        payload = serialise_envelope(env)
        for att in targets:
            try:
                await att.ws.send_text(payload)
            except Exception:
                # Presence is best-effort; a failed send doesn't get
                # retried — the next legitimate event for this peer
                # will surface the dead connection and clean it up.
                pass
