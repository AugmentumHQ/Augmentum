"""Missed-call detection timer tests.

Uses short lifetimes (50ms) so tests don't sleep the full 60s default.
The timer registry is process-local; reset_timers_for_test cleans up
between cases.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from augmentum.connect.call_lifecycle import (
    active_timer_count,
    arm_invite_timer,
    cancel_invite_timer,
    recover_stale_invites_on_startup,
    reset_timers_for_test,
)
from augmentum.connect.call_routing import (
    _insert_call_session,
)
from augmentum.connect.hub import ConnectHub
from augmentum.connect.protocol import (
    EVENT_HANGUP,
)
from augmentum.notifications.hub import NotificationHub


CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql"
).read_text()
NOTIFICATIONS_MIGRATION = Path(
    "augmentum/state/migrations/221_notification_substrate.sql"
).read_text()


ALICE = "alice"
BOB = "bob"
ALICE_DID = "alice@this-instance"
BOB_DID = "bob@this-instance"


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


@pytest.fixture(autouse=True)
def _reset_timers():
    reset_timers_for_test()
    yield
    reset_timers_for_test()


async def _seed_invite(conn, call_id: str) -> None:
    await _insert_call_session(
        conn, call_id=call_id, user_id=ALICE,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
        modalities="audio", state="ringing",
    )
    await _insert_call_session(
        conn, call_id=call_id, user_id=BOB,
        initiator_did=ALICE_DID, receiver_did=BOB_DID,
        modalities="audio", state="invited",
    )


# ── Timer arming + cancellation ──────────────────────────────────


@pytest.mark.asyncio
async def test_arm_creates_timer(conn) -> None:
    hub = ConnectHub()
    notif_hub = NotificationHub()
    arm_invite_timer(
        conn=conn, connect_hub=hub, notification_hub=notif_hub,
        call_id="c1",
        initiator_user_id=ALICE, initiator_did=ALICE_DID,
        recipient_user_id=BOB, recipient_did=BOB_DID,
        lifetime_ms=10000,  # don't fire — we cancel below
    )
    assert active_timer_count() == 1
    assert cancel_invite_timer("c1") is True
    assert active_timer_count() == 0


@pytest.mark.asyncio
async def test_cancel_unknown_call_is_noop() -> None:
    assert cancel_invite_timer("not-a-call") is False


@pytest.mark.asyncio
async def test_rearm_replaces_existing_timer(conn) -> None:
    hub = ConnectHub()
    notif_hub = NotificationHub()
    for _ in range(3):
        arm_invite_timer(
            conn=conn, connect_hub=hub, notification_hub=notif_hub,
            call_id="c1",
            initiator_user_id=ALICE, initiator_did=ALICE_DID,
            recipient_user_id=BOB, recipient_did=BOB_DID,
            lifetime_ms=10000,
        )
    assert active_timer_count() == 1


# ── Timer fires → missed-call transition ─────────────────────────


@pytest.mark.asyncio
async def test_timer_fires_marks_both_sides_missed(conn) -> None:
    await _seed_invite(conn, "c1")

    hub = ConnectHub()
    notif_hub = NotificationHub()
    alice_ws = FakeWS()
    await hub.attach(ws=alice_ws, user_id=ALICE, user_did=ALICE_DID)
    alice_ws.sent.clear()  # ignore presence frame

    arm_invite_timer(
        conn=conn, connect_hub=hub, notification_hub=notif_hub,
        call_id="c1",
        initiator_user_id=ALICE, initiator_did=ALICE_DID,
        recipient_user_id=BOB, recipient_did=BOB_DID,
        lifetime_ms=50,
    )
    await asyncio.sleep(0.2)  # let timer fire + writes settle

    cur = await conn.execute(
        "SELECT user_id, state, end_reason FROM call_sessions "
        "WHERE call_id = ? ORDER BY user_id",
        ("c1",),
    )
    rows = await cur.fetchall()
    by_user = {r[0]: (r[1], r[2]) for r in rows}
    assert by_user[ALICE] == ("missed", "timeout")
    assert by_user[BOB] == ("missed", "timeout")

    # Alice's signaling WS received EVENT_HANGUP{reason:'missed'}
    assert alice_ws.sent, "expected EVENT_HANGUP to reach Alice"
    last = json.loads(alice_ws.sent[-1])
    assert last["event"] == EVENT_HANGUP
    assert last["data"]["call_id"] == "c1"
    assert last["data"]["reason"] == "missed"


@pytest.mark.asyncio
async def test_timer_does_not_fire_when_state_already_advanced(conn) -> None:
    """If accept/decline ran before the timer expired (e.g. asyncio race),
    the timer should bail rather than wrongly stamping 'missed'."""

    await _seed_invite(conn, "c1")

    hub = ConnectHub()
    notif_hub = NotificationHub()
    arm_invite_timer(
        conn=conn, connect_hub=hub, notification_hub=notif_hub,
        call_id="c1",
        initiator_user_id=ALICE, initiator_did=ALICE_DID,
        recipient_user_id=BOB, recipient_did=BOB_DID,
        lifetime_ms=50,
    )
    # Race: pretend accept advanced the state before the timer fires.
    await conn.execute(
        "UPDATE call_sessions SET state = 'connected' WHERE call_id = ?",
        ("c1",),
    )
    await conn.commit()

    await asyncio.sleep(0.2)

    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = ?",
        ("c1",),
    )
    rows = await cur.fetchall()
    for (state,) in rows:
        # State should remain 'connected', not be overwritten as 'missed'.
        assert state == "connected"


@pytest.mark.asyncio
async def test_cancelled_timer_does_not_fire(conn) -> None:
    await _seed_invite(conn, "c1")

    hub = ConnectHub()
    notif_hub = NotificationHub()
    arm_invite_timer(
        conn=conn, connect_hub=hub, notification_hub=notif_hub,
        call_id="c1",
        initiator_user_id=ALICE, initiator_did=ALICE_DID,
        recipient_user_id=BOB, recipient_did=BOB_DID,
        lifetime_ms=50,
    )
    cancel_invite_timer("c1")
    await asyncio.sleep(0.2)

    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = ?",
        ("c1",),
    )
    rows = await cur.fetchall()
    # No transition fired — both sides still in their initial states.
    states = {r[0] for r in rows}
    assert "missed" not in states


@pytest.mark.asyncio
async def test_missed_call_publishes_notification(conn) -> None:
    await _seed_invite(conn, "c1")

    hub = ConnectHub()
    notif_hub = NotificationHub()
    bob_notif_ws = FakeWS()
    await notif_hub.attach(ws=bob_notif_ws, user_id=BOB)

    arm_invite_timer(
        conn=conn, connect_hub=hub, notification_hub=notif_hub,
        call_id="c1",
        initiator_user_id=ALICE, initiator_did=ALICE_DID,
        recipient_user_id=BOB, recipient_did=BOB_DID,
        lifetime_ms=50,
    )
    await asyncio.sleep(0.2)

    assert bob_notif_ws.sent
    push = json.loads(bob_notif_ws.sent[-1])
    assert push["type"] == "notification"
    assert push["notification"]["channel_id"] == "connect.call.missed"
    assert push["notification"]["payload"]["call_id"] == "c1"


# ── Restart recovery sweep ───────────────────────────────────────
#
# In-memory invite timers don't survive a process death. Without the
# startup sweep, rows caught mid-ring when augmentum was killed sat in
# 'ringing' / 'invited' forever and polluted the calls history. The
# sweep ages out anything older than ``grace_ms`` (defaulting to the
# invite lifetime, so genuinely live calls survive a hot reload).


@pytest.mark.asyncio
async def test_recover_marks_old_ringing_rows_missed(conn) -> None:
    # Seed two stale rows (both sides) with initiated_at well past
    # the grace window.
    await conn.execute(
        """INSERT INTO call_sessions
              (call_id, user_id, initiator_did, receiver_did,
               modalities, state, initiated_at)
            VALUES (?, ?, ?, ?, 'audio', 'ringing',
                    datetime('now', '-2 days'))""",
        ("stale", ALICE, ALICE_DID, BOB_DID),
    )
    await conn.execute(
        """INSERT INTO call_sessions
              (call_id, user_id, initiator_did, receiver_did,
               modalities, state, initiated_at)
            VALUES (?, ?, ?, ?, 'audio', 'invited',
                    datetime('now', '-2 days'))""",
        ("stale", BOB, ALICE_DID, BOB_DID),
    )
    await conn.commit()

    count = await recover_stale_invites_on_startup(conn)
    assert count == 2

    cur = await conn.execute(
        "SELECT state, end_reason FROM call_sessions WHERE call_id = 'stale'"
    )
    for state, end_reason in await cur.fetchall():
        assert state == "missed"
        assert end_reason == "restart_recovery"


@pytest.mark.asyncio
async def test_recover_preserves_in_grace_window_invites(conn) -> None:
    # A row inserted "now" is within grace — must survive the sweep so
    # a real in-flight INVITE isn't killed if augmentum is restarted
    # mid-ring legitimately. (Grace window = full invite lifetime.)
    await conn.execute(
        """INSERT INTO call_sessions
              (call_id, user_id, initiator_did, receiver_did,
               modalities, state, initiated_at)
            VALUES (?, ?, ?, ?, 'audio', 'ringing', CURRENT_TIMESTAMP)""",
        ("fresh", ALICE, ALICE_DID, BOB_DID),
    )
    await conn.commit()

    count = await recover_stale_invites_on_startup(conn)
    assert count == 0

    cur = await conn.execute(
        "SELECT state FROM call_sessions WHERE call_id = 'fresh'"
    )
    row = await cur.fetchone()
    assert row[0] == "ringing"


@pytest.mark.asyncio
async def test_recover_leaves_terminal_states_alone(conn) -> None:
    # Already-ended rows must not be re-touched; their ended_at
    # carries the real timestamp.
    await conn.execute(
        """INSERT INTO call_sessions
              (call_id, user_id, initiator_did, receiver_did,
               modalities, state, end_reason, initiated_at, ended_at)
            VALUES (?, ?, ?, ?, 'audio', 'ended', 'local_hangup',
                    datetime('now', '-3 days'),
                    datetime('now', '-3 days', '+1 minute'))""",
        ("done", ALICE, ALICE_DID, BOB_DID),
    )
    await conn.commit()

    count = await recover_stale_invites_on_startup(conn)
    assert count == 0

    cur = await conn.execute(
        "SELECT state, end_reason FROM call_sessions WHERE call_id = 'done'"
    )
    row = await cur.fetchone()
    assert row[0] == "ended"
    assert row[1] == "local_hangup"
