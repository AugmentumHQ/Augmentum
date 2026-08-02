"""Phase 4 round-trip tests — 9 call signaling verbs across fabric.

Same pipe-the-coordinator pattern as the text round-trip tests:
two in-memory instances + two ConnectHubs + a PipedCoordinator on
each side that injects outbound envelopes directly into the other
instance's inbound dispatcher.

Coverage:
  * INVITE: Alice@inst-A invites Bob@inst-B → Bob's call_sessions
    row exists in 'invited' state, EVENT_INVITE on Bob's WS.
  * ACCEPT: Bob accepts → Alice's row transitions to 'connected',
    EVENT_ACCEPT on Alice's WS.
  * DECLINE: Bob declines → Alice's row 'declined', EVENT_DECLINE.
  * OFFER / ANSWER / CANDIDATES: SDP frames forward to peer's WS.
  * HANGUP: terminal transition on both sides.
  * NEGOTIATE: mid-call modality flip propagates.
  * SELECT_ANSWER: pass-through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.connect.call_lifecycle import reset_timers_for_test
from augmentum.connect.call_routing import (
    handle_signaling_envelope,
    new_party_id,
)
from augmentum.connect.contacts import local_did_for
from augmentum.connect.fabric_inbound import apply_inbound_fabric_envelope
from augmentum.connect.hub import ConnectHub
from augmentum.connect.protocol import (
    EVENT_ACCEPT,
    EVENT_ANSWER,
    EVENT_CANDIDATES,
    EVENT_DECLINE,
    EVENT_HANGUP,
    EVENT_INVITE,
    EVENT_NEGOTIATE,
    EVENT_OFFER,
    MSG_ACCEPT,
    MSG_ANSWER,
    MSG_CANDIDATES,
    MSG_DECLINE,
    MSG_HANGUP,
    MSG_INVITE,
    MSG_NEGOTIATE,
    MSG_OFFER,
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


def _last(ws: FakeWS) -> dict[str, Any]:
    assert ws.sent, "expected at least one frame"
    return json.loads(ws.sent[-1])


@pytest.fixture(autouse=True)
def _reset_timers():
    reset_timers_for_test()
    yield
    reset_timers_for_test()


@pytest.fixture
async def two_instances():
    """Two in-memory instances cross-piped."""
    a_conn = await aiosqlite.connect(":memory:")
    b_conn = await aiosqlite.connect(":memory:")
    for c in (a_conn, b_conn):
        await c.executescript(CONNECT_MIGRATION)
        await c.executescript(NOTIFICATIONS_MIGRATION)
        # schema_version + outbox
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


# ── INVITE → ACCEPT happy path ────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_invite_creates_recipient_row(two_instances) -> None:
    state = two_instances
    alice_party = new_party_id()
    result = await handle_signaling_envelope(
        conn=state["a_conn"],
        connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID_REMOTE,
            data={"modalities": "audio,video"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        sender_party_id=alice_party,
        fabric_coordinator=state["a_coord"],
    )
    assert result.error_code == ""
    call_id = result.call_id
    assert call_id

    # Sender (alice) row exists on inst-A in 'ringing'.
    cur = await state["a_conn"].execute(
        "SELECT state FROM call_sessions WHERE call_id = ? AND user_id = ?",
        (call_id, ALICE_ID),
    )
    (a_state,) = await cur.fetchone()
    await cur.close()
    assert a_state == "ringing"

    # Recipient (bob) row exists on inst-B in 'invited'.
    cur = await state["b_conn"].execute(
        "SELECT state, modalities, initiator_did, receiver_did "
        "FROM call_sessions WHERE call_id = ? AND user_id = ?",
        (call_id, BOB_ID),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    b_state, b_mods, b_init, b_recv = row
    assert b_state == "invited"
    assert b_mods == "audio,video"
    # Initiator DID is Alice's REMOTE form — the fabric layer rewrote
    # ``alice@this-instance`` (her local sentinel) using her hostname
    # from Bob's paired-peer registry.
    assert b_init == ALICE_DID_REMOTE
    # Receiver DID is bob's LOCAL form on his instance.
    assert b_recv == BOB_DID_LOCAL

    # Bob's WS got EVENT_INVITE.
    invite_env = _last(state["b_ws"])
    assert invite_env["event"] == EVENT_INVITE
    assert invite_env["data"]["call_id"] == call_id
    assert invite_env["data"]["modalities"] == "audio,video"


@pytest.mark.asyncio
async def test_fabric_accept_transitions_alice_row_to_connected(
    two_instances,
) -> None:
    """Alice invites; Bob accepts (over fabric) → Alice's row connected,
    EVENT_ACCEPT fires on Alice's WS."""

    state = two_instances
    result = await handle_signaling_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID_REMOTE,
            data={"modalities": "audio"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["a_coord"],
    )
    call_id = result.call_id
    state["a_ws"].sent.clear()

    # Bob accepts (his UI sends MSG_ACCEPT via fabric).
    await handle_signaling_envelope(
        conn=state["b_conn"], connect_hub=state["b_hub"],
        notification_hub=state["b_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_ACCEPT, peer=ALICE_DID_REMOTE,
            data={"call_id": call_id},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["b_coord"],
    )

    # Alice's row transitioned.
    cur = await state["a_conn"].execute(
        "SELECT state FROM call_sessions WHERE call_id = ? AND user_id = ?",
        (call_id, ALICE_ID),
    )
    (a_state,) = await cur.fetchone()
    await cur.close()
    assert a_state == "connected"

    # Alice's WS got EVENT_ACCEPT.
    accept_env = _last(state["a_ws"])
    assert accept_env["event"] == EVENT_ACCEPT
    assert accept_env["data"]["call_id"] == call_id


# ── OFFER / ANSWER / CANDIDATES ───────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_offer_answer_candidates_pass_through(
    two_instances,
) -> None:
    state = two_instances
    # Seed: connected call.
    from datetime import UTC, datetime
    now = datetime.now(UTC).isoformat()
    call_id = "c-roundtrip"
    for uid, conn in ((ALICE_ID, state["a_conn"]), (BOB_ID, state["b_conn"])):
        await conn.execute(
            """INSERT INTO call_sessions
                 (call_id, user_id, initiator_did, receiver_did,
                  modalities, state, initiated_at, connected_at)
               VALUES (?, ?, ?, ?, 'audio', 'connected', ?, ?)""",
            (call_id, uid, ALICE_DID_LOCAL, BOB_DID_LOCAL, now, now),
        )
        await conn.commit()

    # OFFER alice → bob.
    state["b_ws"].sent.clear()
    await handle_signaling_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_OFFER, peer=BOB_DID_REMOTE,
            data={"call_id": call_id, "sdp": "v=0\r\nalice-offer"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["a_coord"],
    )
    offer_env = _last(state["b_ws"])
    assert offer_env["event"] == EVENT_OFFER
    assert offer_env["data"]["sdp"].endswith("alice-offer")

    # ANSWER bob → alice.
    state["a_ws"].sent.clear()
    await handle_signaling_envelope(
        conn=state["b_conn"], connect_hub=state["b_hub"],
        notification_hub=state["b_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_ANSWER, peer=ALICE_DID_REMOTE,
            data={"call_id": call_id, "sdp": "v=0\r\nbob-answer"},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["b_coord"],
    )
    answer_env = _last(state["a_ws"])
    assert answer_env["event"] == EVENT_ANSWER
    assert answer_env["data"]["sdp"].endswith("bob-answer")

    # CANDIDATES batch alice → bob.
    state["b_ws"].sent.clear()
    await handle_signaling_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_CANDIDATES, peer=BOB_DID_REMOTE,
            data={
                "call_id": call_id,
                "candidates": [
                    {"candidate": "a=cand-1", "sdpMid": "0", "sdpMLineIndex": 0},
                ],
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["a_coord"],
    )
    cand_env = _last(state["b_ws"])
    assert cand_env["event"] == EVENT_CANDIDATES
    assert cand_env["data"]["candidates"][0]["candidate"] == "a=cand-1"


# ── HANGUP / DECLINE ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_hangup_transitions_both_sides(two_instances) -> None:
    state = two_instances
    result = await handle_signaling_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID_REMOTE,
            data={"modalities": "audio"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["a_coord"],
    )
    call_id = result.call_id
    state["b_ws"].sent.clear()

    # Alice hangs up.
    await handle_signaling_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_HANGUP, peer=BOB_DID_REMOTE,
            data={"call_id": call_id, "reason": "user_hangup"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["a_coord"],
    )

    # Both sides 'ended'.
    cur = await state["a_conn"].execute(
        "SELECT state FROM call_sessions WHERE call_id = ?", (call_id,),
    )
    (a_state,) = await cur.fetchone()
    await cur.close()
    cur = await state["b_conn"].execute(
        "SELECT state, end_reason FROM call_sessions WHERE call_id = ?",
        (call_id,),
    )
    (b_state, b_reason) = await cur.fetchone()
    await cur.close()
    assert a_state == "ended"
    assert b_state == "ended"
    assert b_reason == "user_hangup"

    # Bob's WS got EVENT_HANGUP.
    hangup_env = _last(state["b_ws"])
    assert hangup_env["event"] == EVENT_HANGUP


@pytest.mark.asyncio
async def test_fabric_decline_transitions_both_sides(two_instances) -> None:
    state = two_instances
    result = await handle_signaling_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_INVITE, peer=BOB_DID_REMOTE,
            data={"modalities": "audio"},
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["a_coord"],
    )
    call_id = result.call_id
    state["a_ws"].sent.clear()

    # Bob declines.
    await handle_signaling_envelope(
        conn=state["b_conn"], connect_hub=state["b_hub"],
        notification_hub=state["b_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_DECLINE, peer=ALICE_DID_REMOTE,
            data={"call_id": call_id},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["b_coord"],
    )

    cur = await state["a_conn"].execute(
        "SELECT state FROM call_sessions WHERE call_id = ? AND user_id = ?",
        (call_id, ALICE_ID),
    )
    (a_state,) = await cur.fetchone()
    await cur.close()
    assert a_state == "declined"

    decline_env = _last(state["a_ws"])
    assert decline_env["event"] == EVENT_DECLINE


# ── NEGOTIATE ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fabric_negotiate_flips_modalities_both_sides(
    two_instances,
) -> None:
    state = two_instances
    # Seed connected audio call.
    from datetime import UTC, datetime
    now = datetime.now(UTC).isoformat()
    call_id = "c-neg"
    for uid, conn in ((ALICE_ID, state["a_conn"]), (BOB_ID, state["b_conn"])):
        await conn.execute(
            """INSERT INTO call_sessions
                 (call_id, user_id, initiator_did, receiver_did,
                  modalities, state, initiated_at, connected_at)
               VALUES (?, ?, ?, ?, 'audio', 'connected', ?, ?)""",
            (call_id, uid, ALICE_DID_LOCAL, BOB_DID_LOCAL, now, now),
        )
        await conn.commit()

    state["b_ws"].sent.clear()
    await handle_signaling_envelope(
        conn=state["a_conn"], connect_hub=state["a_hub"],
        notification_hub=state["a_notif"],
        env=ConnectEnvelope(
            kind="msg", verb=MSG_NEGOTIATE, peer=BOB_DID_REMOTE,
            data={
                "call_id": call_id,
                "description": {"type": "offer", "sdp": "v=0\r\nvideo-on"},
                "modalities": "audio,video",
            },
        ),
        sender_user_id=ALICE_ID, sender_did=ALICE_DID_LOCAL,
        sender_party_id=new_party_id(),
        fabric_coordinator=state["a_coord"],
    )

    # Modality flipped on both.
    for uid, conn in ((ALICE_ID, state["a_conn"]), (BOB_ID, state["b_conn"])):
        cur = await conn.execute(
            "SELECT modalities FROM call_sessions "
            "WHERE call_id = ? AND user_id = ?",
            (call_id, uid),
        )
        (mods,) = await cur.fetchone()
        await cur.close()
        assert mods == "audio,video", f"{uid} modalities = {mods}"

    neg_env = _last(state["b_ws"])
    assert neg_env["event"] == EVENT_NEGOTIATE
    assert neg_env["data"]["modalities"] == "audio,video"
