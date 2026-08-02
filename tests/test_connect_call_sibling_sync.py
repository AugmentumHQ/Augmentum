"""Cross-tab / sibling-device sync for Connect call accept/decline.

A user with several browser tabs (or laptop + phone) open in Connect
has one WS attachment per surface. EVENT_INVITE fans to all of them
so they all ring; accept/decline on one tab must dismiss the others
without tearing down the accepted call.

The bug this guards against: tab 1 accepts, tabs 2/3 still ringing,
declining on tab 2 routes EVENT_DECLINE to the caller and transitions
the row to ``declined`` — tab 1's WebRTC handshake dies.

The fix has two layers:

* Server-side echo (``ConnectHub.route_to_user(exclude_connection_id=)``):
  MSG_ACCEPT / MSG_DECLINE from one tab fan EVENT_ACCEPT / EVENT_DECLINE
  to that user's other tabs so their incoming-modal dismisses. This
  also covers cross-device (laptop ↔ phone) which can't share a
  BroadcastChannel.

* Race guard: a late MSG_DECLINE that arrives after a sibling has
  already accepted is dropped — the recipient row stays ``connected``
  and the caller does NOT see EVENT_DECLINE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from augmentum.connect.call_routing import (
    handle_signaling_envelope,
    new_party_id,
)
from augmentum.connect.contacts import local_did_for
from augmentum.connect.hub import ConnectHub
from augmentum.connect.protocol import (
    EVENT_ACCEPT,
    EVENT_DECLINE,
    MSG_ACCEPT,
    MSG_DECLINE,
    MSG_INVITE,
    ConnectEnvelope,
)
from augmentum.notifications.hub import NotificationHub

CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql"
).read_text()
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql"
).read_text()


ALICE_ID = "alice"
BOB_ID = "bob"
ALICE_DID = local_did_for(ALICE_ID)
BOB_DID = local_did_for(BOB_ID)


class FakeWS:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        await c.executescript(CONNECT_MIGRATION)
        await c.executescript(NOTIFICATIONS_MIGRATION)
        await c.commit()
        yield c


def _events_of(ws: FakeWS, verb: str) -> list[dict[str, Any]]:
    """All envelopes the WS received whose ``event`` field matches verb."""

    out: list[dict[str, Any]] = []
    for raw in ws.sent:
        try:
            env = json.loads(raw)
        except Exception:
            continue
        if env.get("type") == "event" and env.get("event") == verb:
            out.append(env)
    return out


async def _seed_invite(
    *, conn, connect_hub: ConnectHub, notification_hub: NotificationHub,
    bob_tabs: list[FakeWS], alice_ws: FakeWS,
) -> str:
    """Send the INVITE from Alice → Bob and return the call_id."""

    party = new_party_id()
    invite = ConnectEnvelope(
        kind="msg", verb=MSG_INVITE, corr_id="alice-1",
        peer=BOB_DID, data={"modalities": "audio"},
    )
    result = await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=invite,
        sender_user_id=ALICE_ID,
        sender_did=ALICE_DID,
        sender_party_id=party,
    )
    assert result.error_code == ""
    assert result.call_id
    # Every Bob tab got EVENT_INVITE.
    assert result.routed == len(bob_tabs)
    # Clear the sent buffers so subsequent assertions only see the
    # accept/decline-driven events.
    for ws in bob_tabs:
        ws.sent.clear()
    alice_ws.sent.clear()
    return result.call_id


# ── Tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_accept_fans_to_sibling_tabs_excluding_origin(conn) -> None:
    """Tab 1 accepts → Tabs 2 & 3 receive EVENT_ACCEPT; Tab 1 does not."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_ws = FakeWS()
    tab1, tab2, tab3 = FakeWS(), FakeWS(), FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    att1 = await connect_hub.attach(ws=tab1, user_id=BOB_ID, user_did=BOB_DID)
    await connect_hub.attach(ws=tab2, user_id=BOB_ID, user_did=BOB_DID)
    await connect_hub.attach(ws=tab3, user_id=BOB_ID, user_did=BOB_DID)

    call_id = await _seed_invite(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        bob_tabs=[tab1, tab2, tab3], alice_ws=alice_ws,
    )

    # Tab 1 accepts via WS.
    accept = ConnectEnvelope(
        kind="msg", verb=MSG_ACCEPT, peer=ALICE_DID,
        data={"call_id": call_id},
    )
    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=accept,
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=new_party_id(),
        sender_connection_id=att1.connection_id,
    )

    # Alice's tab got EVENT_ACCEPT (existing caller-side routing).
    assert len(_events_of(alice_ws, EVENT_ACCEPT)) == 1

    # Tabs 2 & 3 got an EVENT_ACCEPT echo with resolved_by=sibling.
    for tab, label in [(tab2, "tab2"), (tab3, "tab3")]:
        echoes = _events_of(tab, EVENT_ACCEPT)
        assert len(echoes) == 1, f"{label} missing sibling echo"
        assert echoes[0]["data"]["call_id"] == call_id
        assert echoes[0]["data"].get("resolved_by") == "sibling"

    # The originating tab does NOT get its own echo.
    assert _events_of(tab1, EVENT_ACCEPT) == []

    # Recipient row is now ``connected`` so a stale decline can be
    # recognised by the race guard.
    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = ? AND user_id = ?",
        (call_id, BOB_ID),
    )
    row = await cur.fetchone()
    assert row[0] == "connected"


@pytest.mark.asyncio
async def test_decline_after_sibling_accept_is_dropped(conn) -> None:
    """The exact bug: tab 1 accepted, tab 2 then declines.

    Expected: tab 2's modal closes via EVENT_ACCEPT echo; the call
    stays connected; Alice does NOT see EVENT_DECLINE.
    """

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_ws = FakeWS()
    tab1, tab2 = FakeWS(), FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    att1 = await connect_hub.attach(ws=tab1, user_id=BOB_ID, user_did=BOB_DID)
    att2 = await connect_hub.attach(ws=tab2, user_id=BOB_ID, user_did=BOB_DID)

    call_id = await _seed_invite(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        bob_tabs=[tab1, tab2], alice_ws=alice_ws,
    )

    # Tab 1 accepts.
    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_ACCEPT, peer=ALICE_DID,
            data={"call_id": call_id},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=new_party_id(),
        sender_connection_id=att1.connection_id,
    )
    alice_ws.sent.clear()
    tab1.sent.clear()
    tab2.sent.clear()

    # Tab 2 now decline-clicks, oblivious to tab 1's accept (its echo
    # may be in flight or it may have raced ahead).
    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_DECLINE, peer=ALICE_DID,
            data={"call_id": call_id, "reason": "declined"},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=new_party_id(),
        sender_connection_id=att2.connection_id,
    )

    # The caller MUST NOT see a decline — the call is connected.
    assert _events_of(alice_ws, EVENT_DECLINE) == []

    # Tab 2 receives an EVENT_ACCEPT echo (resolved_by=sibling) so its
    # modal closes silently rather than the user discovering they
    # accidentally killed the call.
    echoes = _events_of(tab2, EVENT_ACCEPT)
    assert len(echoes) == 1
    assert echoes[0]["data"].get("resolved_by") == "sibling"
    assert echoes[0]["data"]["call_id"] == call_id

    # The recipient row stays ``connected`` — the late decline did
    # NOT transition state.
    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = ? AND user_id = ?",
        (call_id, BOB_ID),
    )
    assert (await cur.fetchone())[0] == "connected"

    # The caller's row also stays ``connected``.
    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = ? AND user_id = ?",
        (call_id, ALICE_ID),
    )
    assert (await cur.fetchone())[0] == "connected"


@pytest.mark.asyncio
async def test_decline_first_fans_to_siblings_and_caller(conn) -> None:
    """No sibling accept yet — decline proceeds normally + fans to siblings."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_ws = FakeWS()
    tab1, tab2, tab3 = FakeWS(), FakeWS(), FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    att1 = await connect_hub.attach(ws=tab1, user_id=BOB_ID, user_did=BOB_DID)
    await connect_hub.attach(ws=tab2, user_id=BOB_ID, user_did=BOB_DID)
    await connect_hub.attach(ws=tab3, user_id=BOB_ID, user_did=BOB_DID)

    call_id = await _seed_invite(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        bob_tabs=[tab1, tab2, tab3], alice_ws=alice_ws,
    )

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_DECLINE, peer=ALICE_DID,
            data={"call_id": call_id, "reason": "declined"},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=new_party_id(),
        sender_connection_id=att1.connection_id,
    )

    # Alice sees the decline (caller-side routing).
    assert len(_events_of(alice_ws, EVENT_DECLINE)) == 1

    # Siblings tab2, tab3 get the decline echo so their modals close.
    for tab in (tab2, tab3):
        echoes = _events_of(tab, EVENT_DECLINE)
        assert len(echoes) == 1
        assert echoes[0]["data"].get("resolved_by") == "sibling"

    # Origin tab1 does NOT get its own echo.
    assert _events_of(tab1, EVENT_DECLINE) == []

    # Both rows are declined.
    cur = await conn.execute(
        "SELECT user_id, state FROM call_sessions WHERE call_id = ?",
        (call_id,),
    )
    by_user = {u: s for u, s in await cur.fetchall()}
    assert by_user[BOB_ID] == "declined"
    assert by_user[ALICE_ID] == "declined"


@pytest.mark.asyncio
async def test_single_tab_accept_no_sibling_fanout(conn) -> None:
    """One WS only — no sibling exists, nothing changes for that path."""

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_ws = FakeWS()
    tab1 = FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    att1 = await connect_hub.attach(ws=tab1, user_id=BOB_ID, user_did=BOB_DID)

    call_id = await _seed_invite(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        bob_tabs=[tab1], alice_ws=alice_ws,
    )

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_ACCEPT, peer=ALICE_DID,
            data={"call_id": call_id},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=new_party_id(),
        sender_connection_id=att1.connection_id,
    )

    # Alice sees the accept.
    assert len(_events_of(alice_ws, EVENT_ACCEPT)) == 1
    # tab1 is the originating connection — it does NOT receive a
    # sibling echo of its own action.
    assert _events_of(tab1, EVENT_ACCEPT) == []


@pytest.mark.asyncio
async def test_ws_accept_transitions_row_to_connected(conn) -> None:
    """Regression: previously only the HTTP notification-action path
    transitioned state to ``connected``. The race guard for stale
    declines needs the WS accept path to do it too.
    """

    connect_hub = ConnectHub()
    notification_hub = NotificationHub()

    alice_ws, tab1 = FakeWS(), FakeWS()
    await connect_hub.attach(ws=alice_ws, user_id=ALICE_ID, user_did=ALICE_DID)
    att1 = await connect_hub.attach(ws=tab1, user_id=BOB_ID, user_did=BOB_DID)

    call_id = await _seed_invite(
        conn=conn, connect_hub=connect_hub,
        notification_hub=notification_hub,
        bob_tabs=[tab1], alice_ws=alice_ws,
    )

    await handle_signaling_envelope(
        conn=conn,
        connect_hub=connect_hub,
        notification_hub=notification_hub,
        env=ConnectEnvelope(
            kind="msg", verb=MSG_ACCEPT, peer=ALICE_DID,
            data={"call_id": call_id},
        ),
        sender_user_id=BOB_ID, sender_did=BOB_DID,
        sender_party_id=new_party_id(),
        sender_connection_id=att1.connection_id,
    )

    cur = await conn.execute(
        "SELECT user_id, state FROM call_sessions WHERE call_id = ?",
        (call_id,),
    )
    by_user = {u: s for u, s in await cur.fetchall()}
    assert by_user[BOB_ID] == "connected"
    assert by_user[ALICE_ID] == "connected"


@pytest.mark.asyncio
async def test_route_to_user_exclude_connection_id_drops_origin() -> None:
    """Unit-level guard on ConnectHub's new exclude parameter."""

    hub = ConnectHub()
    ws_a, ws_b, ws_c = FakeWS(), FakeWS(), FakeWS()
    att_a = await hub.attach(ws=ws_a, user_id="u1", user_did="u1@h")
    await hub.attach(ws=ws_b, user_id="u1", user_did="u1@h")
    await hub.attach(ws=ws_c, user_id="u1", user_did="u1@h")

    env = ConnectEnvelope(
        kind="event", verb=EVENT_ACCEPT, peer="alice",
        data={"call_id": "c1"},
    )
    delivered = await hub.route_to_user(
        target_user_id="u1", envelope=env,
        exclude_connection_id=att_a.connection_id,
    )
    assert delivered == 2
    assert ws_a.sent == []
    assert len(ws_b.sent) == 1
    assert len(ws_c.sent) == 1
