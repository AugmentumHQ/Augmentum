"""Phase 5 resilience tests — peer outage + reconnect drain semantics.

Builds on the Phase 1 substrate + Phase 2 round-trip patterns to
verify the full offline-then-online sequence:

  * Peer disconnect mid-conversation: messages queue durably.
  * Peer reconnect: queued envelopes drain in insertion order and
    appear on the recipient instance.
  * Bidirectional outbox: both A→B and B→A queued; both drain when
    the link is restored.
  * Permanent disconnect: after MAX_OUTBOX_ATTEMPTS retries, the row
    is evicted (with log warning) — won't accumulate forever.

These pin the durability contract that turns Connect over fabric from
"best-effort online-only delivery" into "messages eventually arrive
when the peer comes back".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.connect.contacts import local_did_for
from augmentum.connect.fabric_inbound import apply_inbound_fabric_envelope
from augmentum.connect.fabric_transport import (
    MAX_OUTBOX_ATTEMPTS,
    drain_outbox_for_peer,
)
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import handle_message_envelope
from augmentum.connect.protocol import (
    EVENT_TEXT_RECEIVED,
    MSG_TEXT_SEND,
    ConnectEnvelope,
)
from augmentum.notifications.hub import NotificationHub


CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql",
).read_text(encoding="utf-8")
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql",
).read_text(encoding="utf-8")
OUTBOX_MIGRATION = Path(
    "augmentum/state/migrations/241_connect_fabric_outbox.sql",
).read_text(encoding="utf-8")


ALICE_ID = "alice"
BOB_ID = "bob"
ALICE_DID_LOCAL = local_did_for(ALICE_ID)
BOB_DID_LOCAL = local_did_for(BOB_ID)
ALICE_DID_REMOTE = "alice@instance-A"
BOB_DID_REMOTE = "bob@instance-B"


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


class PipedCoordinator:
    def __init__(
        self, *,
        other_node_id: str, other_hostname: str,
        self_node_id: str = "", self_hostname: str = "",
    ) -> None:
        paired = MagicMock(node_id=other_node_id, hostname=other_hostname)
        self._peers = {other_node_id: MagicMock(paired=paired)}
        self.target_conn: Any | None = None
        self.target_connect_hub: ConnectHub | None = None
        self.target_notification_hub: NotificationHub | None = None
        self.target_coordinator: PipedCoordinator | None = None
        self.connected = True
        self.self_node_id = self_node_id
        self.self_hostname = self_hostname

    def peer_state(self, node_id: str):
        return self._peers.get(node_id)

    async def send_to_peer(
        self, node_id: str, *, msg_type: str, payload: dict,
    ) -> bool:
        if not self.connected or self.target_conn is None:
            return False
        await apply_inbound_fabric_envelope(
            self.target_conn,
            connect_hub=self.target_connect_hub,
            notification_hub=self.target_notification_hub,
            fabric_payload=payload,
            coordinator=self.target_coordinator,
            sender_node_id=self.self_node_id,
        )
        return True


@pytest.fixture
async def two_instances():
    a_conn = await aiosqlite.connect(":memory:")
    b_conn = await aiosqlite.connect(":memory:")
    for c in (a_conn, b_conn):
        await c.executescript(CONNECT_MIGRATION)
        await c.executescript(NOTIFICATIONS_MIGRATION)
        await c.execute(
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY, description TEXT, "
            " applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        )
        await c.executescript(OUTBOX_MIGRATION)
        await c.commit()
    a_hub, b_hub = ConnectHub(), ConnectHub()
    a_notif, b_notif = NotificationHub(), NotificationHub()
    a_ws, b_ws = FakeWS(), FakeWS()
    await a_hub.attach(ws=a_ws, user_id=ALICE_ID, user_did=ALICE_DID_LOCAL)
    await b_hub.attach(ws=b_ws, user_id=BOB_ID, user_did=BOB_DID_LOCAL)
    a_ws.sent.clear()
    b_ws.sent.clear()

    a_coord = PipedCoordinator(
        other_node_id="peer-B", other_hostname="instance-B",
        self_node_id="peer-A", self_hostname="instance-A",
    )
    a_coord.target_conn = b_conn
    a_coord.target_connect_hub = b_hub
    a_coord.target_notification_hub = b_notif

    b_coord = PipedCoordinator(
        other_node_id="peer-A", other_hostname="instance-A",
        self_node_id="peer-B", self_hostname="instance-B",
    )
    b_coord.target_conn = a_conn
    b_coord.target_connect_hub = a_hub
    b_coord.target_notification_hub = a_notif

    a_coord.target_coordinator = b_coord
    b_coord.target_coordinator = a_coord

    yield {
        "a_conn": a_conn, "a_hub": a_hub, "a_notif": a_notif, "a_ws": a_ws,
        "a_coord": a_coord,
        "b_conn": b_conn, "b_hub": b_hub, "b_notif": b_notif, "b_ws": b_ws,
        "b_coord": b_coord,
    }
    await a_conn.close()
    await b_conn.close()


def _events(ws: FakeWS, verb: str) -> list[dict]:
    return [
        json.loads(s) for s in ws.sent
        if json.loads(s).get("event") == verb
    ]


# ── Offline → reconnect drain ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_three_messages_queued_offline_drain_in_order_on_reconnect(
    two_instances,
) -> None:
    state = two_instances
    state["a_coord"].connected = False  # B's instance is down

    for i in range(3):
        await handle_message_envelope(
            conn=state["a_conn"], connect_hub=state["a_hub"],
            notification_hub=state["a_notif"],
            env=ConnectEnvelope(
                kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
                data={
                    "thread_id": "t-q", "message_id": f"m-q{i}",
                    "body": f"queued-{i}",
                },
            ),
            sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
            fabric_coordinator=state["a_coord"],
        )

    # Outbox has 3 entries.
    cur = await state["a_conn"].execute(
        "SELECT COUNT(*) FROM connect_fabric_outbox",
    )
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 3

    # Bob's instance has nothing.
    cur = await state["b_conn"].execute(
        "SELECT COUNT(*) FROM connect_messages",
    )
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 0

    # Peer comes back online — drain.
    state["a_coord"].connected = True
    counters = await drain_outbox_for_peer(
        state["a_conn"], coordinator=state["a_coord"], node_id="peer-B",
    )
    assert counters == {"sent": 3, "still_queued": 0, "exhausted": 0}

    # All 3 messages now on Bob's instance, in order.
    cur = await state["b_conn"].execute(
        "SELECT message_id, body FROM connect_messages WHERE user_id = ? "
        "ORDER BY sent_at, message_id",
        (BOB_ID,),
    )
    rows = await cur.fetchall()
    await cur.close()
    bodies = [r[1] for r in rows]
    assert bodies == ["queued-0", "queued-1", "queued-2"]

    # Bob's WS got 3 EVENT_TEXT_RECEIVED.
    received = _events(state["b_ws"], EVENT_TEXT_RECEIVED)
    assert len(received) == 3


@pytest.mark.asyncio
async def test_bidirectional_outbox_drain(two_instances) -> None:
    """Both A→B and B→A queue while the link is down. After reconnect,
    both directions drain independently."""

    state = two_instances
    state["a_coord"].connected = False
    state["b_coord"].connected = False

    # Both directions reuse the same logical thread id ``t-shared``.
    # Post DID-normalisation, the (user_id, peer_did) unique index
    # collapses the bidirectional inserts onto a single thread row per
    # instance — A↔B is a single conversation even when both sides
    # initiate fresh sends while disconnected.
    for i in range(2):
        await handle_message_envelope(
            conn=state["a_conn"], connect_hub=state["a_hub"],
            notification_hub=state["a_notif"],
            env=ConnectEnvelope(
                kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
                data={
                    "thread_id": "t-shared",
                    "message_id": f"a{i}", "body": f"a{i}",
                },
            ),
            sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
            fabric_coordinator=state["a_coord"],
        )
    for i in range(2):
        await handle_message_envelope(
            conn=state["b_conn"], connect_hub=state["b_hub"],
            notification_hub=state["b_notif"],
            env=ConnectEnvelope(
                kind="msg", verb=MSG_TEXT_SEND, peer=ALICE_DID_REMOTE,
                data={
                    "thread_id": "t-shared",
                    "message_id": f"b{i}", "body": f"b{i}",
                },
            ),
            sender_user_id=BOB_ID, sender_did=BOB_DID_LOCAL,
            fabric_coordinator=state["b_coord"],
        )

    # Both outboxes have 2 entries each.
    for c in (state["a_conn"], state["b_conn"]):
        cur = await c.execute("SELECT COUNT(*) FROM connect_fabric_outbox")
        (n,) = await cur.fetchone()
        await cur.close()
        assert n == 2

    # Reconnect both sides + drain both.
    state["a_coord"].connected = True
    state["b_coord"].connected = True
    a_counters = await drain_outbox_for_peer(
        state["a_conn"], coordinator=state["a_coord"], node_id="peer-B",
    )
    b_counters = await drain_outbox_for_peer(
        state["b_conn"], coordinator=state["b_coord"], node_id="peer-A",
    )
    assert a_counters["sent"] == 2
    assert b_counters["sent"] == 2

    # Alice's messages arrived on Bob's instance (recipient rows).
    # Filter by sender_did so we ignore Bob's own outbound rows that
    # also live in his DB. After normalisation Bob stores Alice's DID
    # in its remote form.
    cur = await state["b_conn"].execute(
        "SELECT message_id FROM connect_messages "
        "WHERE user_id = ? AND sender_did = ? ORDER BY sent_at, message_id",
        (BOB_ID, ALICE_DID_REMOTE),
    )
    alice_msgs_on_bob = [r[0] for r in await cur.fetchall()]
    await cur.close()
    assert alice_msgs_on_bob == ["a0", "a1"]

    # Bob's messages arrived on Alice's instance.
    cur = await state["a_conn"].execute(
        "SELECT message_id FROM connect_messages "
        "WHERE user_id = ? AND sender_did = ? ORDER BY sent_at, message_id",
        (ALICE_ID, BOB_DID_REMOTE),
    )
    bob_msgs_on_alice = [r[0] for r in await cur.fetchall()]
    await cur.close()
    assert bob_msgs_on_alice == ["b0", "b1"]


@pytest.mark.asyncio
async def test_permanent_peer_disappearance_caps_attempts(
    two_instances,
) -> None:
    """A row stuck for MAX_OUTBOX_ATTEMPTS attempts is evicted — won't
    accumulate forever even if the peer never comes back."""

    state = two_instances
    state["a_coord"].connected = False

    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
            data={
                "thread_id": "t-zombie", "message_id": "m-zombie",
                "body": "into the void",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )

    # Fast-forward attempts to one short of the cap.
    await state["a_conn"].execute(
        "UPDATE connect_fabric_outbox SET attempts = ?",
        (MAX_OUTBOX_ATTEMPTS - 1,),
    )
    await state["a_conn"].commit()

    # Drain one more time; peer still down.
    counters = await drain_outbox_for_peer(
        state["a_conn"], coordinator=state["a_coord"], node_id="peer-B",
    )
    assert counters["exhausted"] == 1
    assert counters["sent"] == 0

    cur = await state["a_conn"].execute(
        "SELECT COUNT(*) FROM connect_fabric_outbox",
    )
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 0


@pytest.mark.asyncio
async def test_drain_preserves_outbox_when_peer_intermittent(
    two_instances,
) -> None:
    """Peer comes back briefly then drops — outbox row stays + attempts
    increments, but message isn't lost."""

    state = two_instances
    state["a_coord"].connected = False
    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-blip", "message_id": "m-blip", "body": "?"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )

    # Drain while still offline.
    counters = await drain_outbox_for_peer(
        state["a_conn"], coordinator=state["a_coord"], node_id="peer-B",
    )
    assert counters["still_queued"] == 1

    # Now actually reconnect + drain — message arrives.
    state["a_coord"].connected = True
    counters = await drain_outbox_for_peer(
        state["a_conn"], coordinator=state["a_coord"], node_id="peer-B",
    )
    assert counters["sent"] == 1

    cur = await state["b_conn"].execute(
        "SELECT body FROM connect_messages WHERE message_id = ?",
        ("m-blip",),
    )
    (body,) = await cur.fetchone()
    await cur.close()
    assert body == "?"
