"""Phase 2 round-trip tests — 8 text verbs across a fabric hop.

Each test stands up TWO in-memory instances (Alice's box + Bob's box),
each with its own DB + ConnectHub + NotificationHub. A
``PipedCoordinator`` pretends to be the fabric coordinator on BOTH
sides — when one side sends an envelope, the piped coordinator
re-injects it into the other side's inbound dispatcher (mimicking the
fabric WS transport). This lets us drive the full outbound +
fabric-hop + inbound flow with no real network involved.

Coverage:
  * SEND: Alice@inst-A → Bob@inst-B. Recipient row appears on inst-B;
    Bob's WS receives EVENT_TEXT_RECEIVED with the right sender_did.
  * EDIT: Alice edits her message; Bob's mirror updates + EVENT_TEXT_EDIT
    fires on Bob's WS.
  * DELETE: Alice deletes; Bob's mirror is tombstoned + EVENT_TEXT_DELETE.
  * READ: Bob reads Alice's messages; Alice receives EVENT_TEXT_READ.
  * DELIVERED: Bob acks; Alice's row stamps delivered_at + EVENT.
  * REACT: Alice reacts; Bob mirror gets reaction + EVENT_TEXT_REACT.
  * TYPING_START / TYPING_STOP: Bob receives EVENT_TYPING_*.

Block enforcement across fabric is a separate test class — silent-block
applies on the inbound side (Bob has blocked Alice → Bob's instance
sees the SEND but drops mirror + WS + notification).
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
from augmentum.connect.hub import ConnectHub
from augmentum.connect.message_routing import handle_message_envelope
from augmentum.connect.protocol import (
    EVENT_TEXT_DELETE,
    EVENT_TEXT_DELIVERED,
    EVENT_TEXT_EDIT,
    EVENT_TEXT_READ,
    EVENT_TEXT_RECEIVED,
    EVENT_TEXT_REACT,
    EVENT_TYPING_START,
    EVENT_TYPING_STOP,
    MSG_TEXT_DELETE,
    MSG_TEXT_DELIVERED,
    MSG_TEXT_EDIT,
    MSG_TEXT_READ,
    MSG_TEXT_REACT,
    MSG_TEXT_SEND,
    MSG_TYPING_START,
    MSG_TYPING_STOP,
    ConnectEnvelope,
)
from augmentum.notifications.hub import NotificationHub


CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql",
).read_text(encoding="utf-8")
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql",
).read_text(encoding="utf-8")
REACTIONS_MIGRATION = Path(
    "augmentum/state/migrations/233_connect_message_reactions.sql",
).read_text(encoding="utf-8")
OUTBOX_MIGRATION = Path(
    "augmentum/state/migrations/241_connect_fabric_outbox.sql",
).read_text(encoding="utf-8")


ALICE_ID = "alice"
BOB_ID = "bob"
ALICE_DID_REMOTE = "alice@instance-A"  # As seen from Bob's instance
BOB_DID_REMOTE = "bob@instance-B"  # As seen from Alice's instance
ALICE_DID_LOCAL = local_did_for(ALICE_ID)  # alice@this-instance (on Alice's box)
BOB_DID_LOCAL = local_did_for(BOB_ID)  # bob@this-instance (on Bob's box)


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


async def _make_instance() -> tuple[aiosqlite.Connection, ConnectHub, NotificationHub]:
    """Create a fresh in-memory instance — DB + hubs."""
    conn = await aiosqlite.connect(":memory:")
    # schema_version table is needed by every migration's tail INSERT.
    await conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, description TEXT, "
        " applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    )
    await conn.executescript(CONNECT_MIGRATION)
    await conn.executescript(NOTIFICATIONS_MIGRATION)
    await conn.executescript(REACTIONS_MIGRATION)
    await conn.executescript(OUTBOX_MIGRATION)
    await conn.commit()
    return conn, ConnectHub(), NotificationHub()


class PipedCoordinator:
    """Fabric coordinator stub that pipes outbound frames directly
    into the OTHER instance's inbound dispatcher.

    Pretends to be a single paired peer (the other instance). Wires
    up the in-process connection so tests can drive both sides without
    a real WebSocket. Failure modes (peer offline) are toggled via
    ``connected``.

    ``self_node_id`` + ``self_hostname`` describe how THIS coordinator's
    box appears to the OTHER side. When piping an envelope across, we
    pass them as the ``sender_node_id`` + (via the target's peer
    registry) hostname so the DID-normalisation path on the receiver
    can rewrite ``source_did`` away from the ``@this-instance`` sentinel.
    ``target_coordinator`` is the target's own coordinator (so the
    receiver looks up the sender's hostname in ITS peer registry —
    mirroring how the real fabric layer works).
    """

    def __init__(
        self, *,
        other_node_id: str = "peer-other",
        other_hostname: str = "instance-other",
        self_node_id: str = "",
        self_hostname: str = "",
        connected: bool = True,
    ) -> None:
        paired = MagicMock(node_id=other_node_id, hostname=other_hostname)
        self._peers = {other_node_id: MagicMock(paired=paired)}
        self.connected = connected
        self.self_node_id = self_node_id
        self.self_hostname = self_hostname
        self.target_conn: aiosqlite.Connection | None = None
        self.target_connect_hub: ConnectHub | None = None
        self.target_notification_hub: NotificationHub | None = None
        self.target_coordinator: PipedCoordinator | None = None

    def peer_state(self, node_id: str):
        return self._peers.get(node_id)

    async def send_to_peer(
        self, node_id: str, *, msg_type: str, payload: dict,
    ) -> bool:
        if not self.connected:
            return False
        if self.target_conn is None:
            return False
        # Re-inject into the target instance's inbound dispatcher.
        await apply_inbound_fabric_envelope(
            self.target_conn,
            connect_hub=self.target_connect_hub,
            notification_hub=self.target_notification_hub,
            fabric_payload=payload,
            coordinator=self.target_coordinator,
            sender_node_id=self.self_node_id,
        )
        return True


def _last(ws: FakeWS) -> dict[str, Any]:
    assert ws.sent, "expected at least one frame on the WS"
    return json.loads(ws.sent[-1])


# ── Round-trip helpers ────────────────────────────────────────────────


async def _setup_two_instances() -> dict[str, Any]:
    """Stand up Alice's instance (inst-A) + Bob's instance (inst-B)
    with hubs cross-piped via PipedCoordinator."""

    a_conn, a_hub, a_notif = await _make_instance()
    b_conn, b_hub, b_notif = await _make_instance()
    a_ws = FakeWS()
    b_ws = FakeWS()
    await a_hub.attach(ws=a_ws, user_id=ALICE_ID, user_did=ALICE_DID_LOCAL)
    await b_hub.attach(ws=b_ws, user_id=BOB_ID, user_did=BOB_DID_LOCAL)
    a_ws.sent.clear()
    b_ws.sent.clear()

    # Coordinator on Alice's side: outbound goes to Bob's instance.
    # ``self_node_id`` is how Bob sees Alice on the wire.
    a_coord = PipedCoordinator(
        other_node_id="peer-B", other_hostname="instance-B",
        self_node_id="peer-A", self_hostname="instance-A",
    )
    a_coord.target_conn = b_conn
    a_coord.target_connect_hub = b_hub
    a_coord.target_notification_hub = b_notif

    # Coordinator on Bob's side: outbound goes to Alice's instance.
    b_coord = PipedCoordinator(
        other_node_id="peer-A", other_hostname="instance-A",
        self_node_id="peer-B", self_hostname="instance-B",
    )
    b_coord.target_conn = a_conn
    b_coord.target_connect_hub = a_hub
    b_coord.target_notification_hub = a_notif

    a_coord.target_coordinator = b_coord
    b_coord.target_coordinator = a_coord

    return {
        "a_conn": a_conn, "a_hub": a_hub, "a_notif": a_notif, "a_ws": a_ws,
        "a_coord": a_coord,
        "b_conn": b_conn, "b_hub": b_hub, "b_notif": b_notif, "b_ws": b_ws,
        "b_coord": b_coord,
    }


@pytest.fixture
async def two_instances():
    state = await _setup_two_instances()
    yield state
    await state["a_conn"].close()
    await state["b_conn"].close()


# ── SEND ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_send_creates_recipient_row_and_fires_event(two_instances) -> None:
    """Alice@inst-A sends → Bob's row appears on inst-B + WS event fires."""

    state = two_instances
    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
        data={
            "thread_id": "t-1", "message_id": "m-1",
            "body": "hello from alice", "format": "plain",
        },
    )
    result = await handle_message_envelope(
        conn=state["a_conn"],
        connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=env,
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    assert result.error_code == "", result.error_message
    assert result.routed == 1

    # Sender's row on inst-A.
    cur = await state["a_conn"].execute(
        "SELECT body, sender_did FROM connect_messages "
        "WHERE message_id = ? AND user_id = ?",
        ("m-1", ALICE_ID),
    )
    a_row = await cur.fetchone()
    await cur.close()
    assert a_row == ("hello from alice", ALICE_DID_LOCAL)

    # Recipient's row on inst-B.
    cur = await state["b_conn"].execute(
        "SELECT body, sender_did FROM connect_messages "
        "WHERE message_id = ? AND user_id = ?",
        ("m-1", BOB_ID),
    )
    b_row = await cur.fetchone()
    await cur.close()
    assert b_row is not None
    assert b_row[0] == "hello from alice"
    # Bob sees Alice's REMOTE DID as the sender — the fabric layer
    # normalises ``alice@this-instance`` to ``alice@instance-A`` using
    # the sender's hostname from Bob's paired-peer registry.
    assert b_row[1] == ALICE_DID_REMOTE

    # Bob's WS got EVENT_TEXT_RECEIVED.
    received = _last(state["b_ws"])
    assert received["event"] == EVENT_TEXT_RECEIVED
    assert received["data"]["body"] == "hello from alice"
    assert received["from"] == ALICE_DID_REMOTE


@pytest.mark.asyncio
async def test_fabric_send_when_peer_offline_queues_in_outbox(two_instances) -> None:
    """Bob's instance is offline → Alice's local row persists, outbox
    holds the dispatch, no Bob row created yet."""

    state = two_instances
    state["a_coord"].connected = False

    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
        data={"thread_id": "t-q", "message_id": "m-q", "body": "queued msg"},
    )
    result = await handle_message_envelope(
        conn=state["a_conn"],
        connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=env,
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    # Queued (durable) counts as routed=1 — the sender's UI sees
    # "sent" and the outbox guarantees delivery once the peer
    # reconnects. Delivery is best-effort; the routing-level success
    # signal is acceptance, not real-time delivery.
    assert result.routed == 1
    assert result.error_code == ""

    # Sender row exists on inst-A.
    cur = await state["a_conn"].execute(
        "SELECT COUNT(*) FROM connect_messages WHERE message_id = ?",
        ("m-q",),
    )
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 1

    # Outbox has one entry.
    cur = await state["a_conn"].execute(
        "SELECT COUNT(*) FROM connect_fabric_outbox",
    )
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 1

    # Bob's instance has no row yet.
    cur = await state["b_conn"].execute(
        "SELECT COUNT(*) FROM connect_messages",
    )
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 0


# ── EDIT ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_edit_propagates_new_body(two_instances) -> None:
    """SEND then EDIT — Bob's mirror updates + EVENT_TEXT_EDIT fires."""

    state = two_instances
    # First send.
    await handle_message_envelope(
        conn=state["a_conn"],
        connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1", "message_id": "m-e",
                  "body": "original", "format": "plain"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    state["b_ws"].sent.clear()

    # Edit.
    await handle_message_envelope(
        conn=state["a_conn"],
        connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_EDIT, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1", "message_id": "m-e",
                  "body": "edited"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )

    # Bob's row updated.
    cur = await state["b_conn"].execute(
        "SELECT body FROM connect_messages WHERE message_id = ?",
        ("m-e",),
    )
    (body,) = await cur.fetchone()
    await cur.close()
    assert body == "edited"

    edit_env = _last(state["b_ws"])
    assert edit_env["event"] == EVENT_TEXT_EDIT
    assert edit_env["data"]["body"] == "edited"


# ── DELETE ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_delete_tombstones_recipient_row(two_instances) -> None:
    state = two_instances
    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1", "message_id": "m-d",
                  "body": "ephemeral"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    state["b_ws"].sent.clear()

    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELETE, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1", "message_id": "m-d"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )

    cur = await state["b_conn"].execute(
        "SELECT deleted_at FROM connect_messages WHERE message_id = ?",
        ("m-d",),
    )
    (deleted_at,) = await cur.fetchone()
    await cur.close()
    assert deleted_at is not None

    del_env = _last(state["b_ws"])
    assert del_env["event"] == EVENT_TEXT_DELETE


# ── READ ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_read_routes_back_to_original_sender(two_instances) -> None:
    """Alice sends, Bob reads → Alice receives EVENT_TEXT_READ."""

    state = two_instances
    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1", "message_id": "m-r", "body": "yo"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    state["a_ws"].sent.clear()

    # Bob reads — Bob's instance dispatches READ to Alice's instance.
    await handle_message_envelope(
        conn=state["b_conn"], connect_hub=state["b_hub"],
        notification_hub=state["b_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_READ, peer=ALICE_DID_REMOTE,
            data={"thread_id": "t-1", "last_read_message_id": "m-r"},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID_LOCAL,
        fabric_coordinator=state["b_coord"],
    )

    read_env = _last(state["a_ws"])
    assert read_env["event"] == EVENT_TEXT_READ
    assert read_env["data"]["last_read_message_id"] == "m-r"
    assert read_env["data"]["reader_did"] == BOB_DID_REMOTE


# ── DELIVERED ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_delivered_stamps_sender_row(two_instances) -> None:
    """Bob acks → Alice's row stamps delivered_at + EVENT_TEXT_DELIVERED."""

    state = two_instances
    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1", "message_id": "m-D", "body": "yo"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    state["a_ws"].sent.clear()

    # Alice's row has delivered_at = NULL pre-ack.
    cur = await state["a_conn"].execute(
        "SELECT delivered_at FROM connect_messages WHERE message_id = ?",
        ("m-D",),
    )
    (pre_delivered,) = await cur.fetchone()
    await cur.close()
    assert pre_delivered is None

    # Bob's instance dispatches DELIVERED.
    await handle_message_envelope(
        conn=state["b_conn"], connect_hub=state["b_hub"],
        notification_hub=state["b_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_DELIVERED, peer=ALICE_DID_REMOTE,
            data={"thread_id": "t-1", "message_ids": ["m-D"]},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID_LOCAL,
        fabric_coordinator=state["b_coord"],
    )

    # Alice's row stamped delivered.
    cur = await state["a_conn"].execute(
        "SELECT delivered_at FROM connect_messages WHERE message_id = ?",
        ("m-D",),
    )
    (post_delivered,) = await cur.fetchone()
    await cur.close()
    assert post_delivered is not None

    # Alice's WS got the event.
    delivered_env = _last(state["a_ws"])
    assert delivered_env["event"] == EVENT_TEXT_DELIVERED
    assert delivered_env["data"]["message_ids"] == ["m-D"]


# ── REACT ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_react_propagates_emoji(two_instances) -> None:
    state = two_instances
    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1", "message_id": "m-x", "body": "react me"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    state["b_ws"].sent.clear()

    # Alice reacts.
    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_REACT, peer=BOB_DID_REMOTE,
            data={
                "thread_id": "t-1", "message_id": "m-x",
                "emoji": "👍", "action": "add",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )

    # Sender's reaction row on inst-A.
    cur = await state["a_conn"].execute(
        "SELECT COUNT(*) FROM connect_message_reactions "
        "WHERE message_id = ? AND user_id = ? AND emoji = ?",
        ("m-x", ALICE_ID, "👍"),
    )
    (n_a,) = await cur.fetchone()
    await cur.close()
    assert n_a == 1

    # Recipient's reaction row on inst-B (mirror).
    cur = await state["b_conn"].execute(
        "SELECT COUNT(*) FROM connect_message_reactions "
        "WHERE message_id = ? AND user_id = ? AND emoji = ?",
        ("m-x", BOB_ID, "👍"),
    )
    (n_b,) = await cur.fetchone()
    await cur.close()
    assert n_b == 1

    react_env = _last(state["b_ws"])
    assert react_env["event"] == EVENT_TEXT_REACT
    assert react_env["data"]["emoji"] == "👍"


# ── TYPING ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_typing_start_stop_fires_events(two_instances) -> None:
    state = two_instances
    state["b_ws"].sent.clear()

    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TYPING_START, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    start_env = _last(state["b_ws"])
    assert start_env["event"] == EVENT_TYPING_START
    assert start_env["data"]["thread_id"] == "t-1"

    state["b_ws"].sent.clear()
    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TYPING_STOP, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-1"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )
    stop_env = _last(state["b_ws"])
    assert stop_env["event"] == EVENT_TYPING_STOP


# ── Block enforcement across fabric ───────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_send_silently_blocked_when_recipient_blocked_sender(
    two_instances,
) -> None:
    """Bob has blocked Alice → SEND succeeds on Alice's side (queued in
    her DB + dispatched) but on Bob's side the mirror + WS + event
    are skipped."""

    state = two_instances

    # Bob's instance blocks Alice's REMOTE did. The block list keys
    # off the post-normalised hostname-form DID — same shape Bob's
    # contacts UI would have stored.
    await state["b_conn"].execute(
        """INSERT INTO connect_contacts
              (contact_id, user_id, peer_did, blocked)
            VALUES ('c-blk', ?, ?, 1)""",
        (BOB_ID, ALICE_DID_REMOTE),
    )
    await state["b_conn"].commit()

    state["b_ws"].sent.clear()

    await handle_message_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_TEXT_SEND, peer=BOB_DID_REMOTE,
            data={"thread_id": "t-blk", "message_id": "m-blk",
                  "body": "should be silently dropped"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        fabric_coordinator=state["a_coord"],
    )

    # Sender's row exists on inst-A.
    cur = await state["a_conn"].execute(
        "SELECT COUNT(*) FROM connect_messages WHERE message_id = ?",
        ("m-blk",),
    )
    (n_a,) = await cur.fetchone()
    await cur.close()
    assert n_a == 1

    # Bob's instance has NO recipient row + NO WS event.
    cur = await state["b_conn"].execute(
        "SELECT COUNT(*) FROM connect_messages WHERE message_id = ?",
        ("m-blk",),
    )
    (n_b,) = await cur.fetchone()
    await cur.close()
    assert n_b == 0
    assert state["b_ws"].sent == []
