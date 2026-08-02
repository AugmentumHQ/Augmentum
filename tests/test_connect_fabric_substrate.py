"""Phase 1 substrate tests for Wedge B (Connect over fabric).

Covers:
  * Outbox round-trip (queued → drained on reconnect).
  * Outbound dispatch happy path + every failure shape.
  * Inbound dispatch: malformed payload, missing fields, misroute, and
    verb-class routing (text → NotImplementedError stub, call →
    NotImplementedError stub — Phase 2 / Phase 4 fill them).

No real fabric WS is involved — we use FakeCoordinator stubs that
expose ``send_to_peer`` + ``_peers`` so the transport believes it's
talking to a real coordinator. End-to-end tests across actual fabric
sockets land in Phase 2 once the per-verb logic exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import aiosqlite
import pytest

from augmentum.connect.contacts import local_did_for
from augmentum.connect.fabric_inbound import apply_inbound_fabric_envelope
from augmentum.connect.fabric_transport import (
    MAX_OUTBOX_ATTEMPTS,
    dispatch_fabric_envelope,
    drain_outbox_for_peer,
)
from augmentum.connect.hub import ConnectHub
from augmentum.connect.protocol import (
    MSG_TEXT_SEND,
    ConnectEnvelope,
    serialise_envelope,
)

CONNECT_MIGRATION = Path(
    "augmentum/state/migrations/219_connect_substrate.sql",
).read_text(encoding="utf-8")
OUTBOX_MIGRATION = Path(
    "augmentum/state/migrations/241_connect_fabric_outbox.sql",
).read_text(encoding="utf-8")


@pytest.fixture
async def conn():
    async with aiosqlite.connect(":memory:") as c:
        # Schema_version table is needed by both migrations' INSERT.
        await c.execute(
            "CREATE TABLE IF NOT EXISTS schema_version "
            "(version INTEGER PRIMARY KEY, description TEXT, "
            " applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        )
        await c.executescript(CONNECT_MIGRATION)
        await c.executescript(OUTBOX_MIGRATION)
        await c.commit()
        yield c


class FakeCoordinator:
    """Minimal coordinator stub with the surface the transport uses.

    ``connected`` toggles whether ``send_to_peer`` returns True (live
    flush) or False (peer offline → outbox stays queued).
    """

    def __init__(self, *, node_id: str = "peer-B",
                 hostname: str = "instance-B", connected: bool = True) -> None:
        paired = MagicMock(node_id=node_id, hostname=hostname)
        self._peers = {node_id: MagicMock(paired=paired)}
        self.connected = connected
        self.sent: list[tuple[str, str, dict]] = []  # (node_id, msg_type, payload)
        self.fail_send = False  # toggle to simulate write errors

    def peer_state(self, node_id: str):
        return self._peers.get(node_id)

    async def send_to_peer(
        self, node_id: str, *, msg_type: str, payload: dict,
    ) -> bool:
        if self.fail_send:
            return False
        if not self.connected:
            return False
        self.sent.append((node_id, msg_type, dict(payload)))
        return True


def _make_text_send_envelope(
    *, body: str = "hello fabric", to_did: str = "bob@instance-B",
) -> ConnectEnvelope:
    return ConnectEnvelope(
        kind="msg",
        verb=MSG_TEXT_SEND,
        peer=to_did,
        data={
            "thread_id": "t-1",
            "message_id": "m-fabric-1",
            "body": body,
            "format": "text",
        },
    )


# ── Outbound: dispatch happy path ─────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_happy_path_deletes_outbox_row(conn) -> None:
    """Peer is connected → envelope flushed → outbox row deleted."""

    coord = FakeCoordinator(connected=True)
    env = _make_text_send_envelope()

    result = await dispatch_fabric_envelope(
        conn,
        coordinator=coord,
        target_hostname="instance-B",
        source_did=local_did_for("alice"),
        sender_user_id="alice",
        sender_party_id="party-x",
        envelope=env,
    )

    assert result.queued is True
    assert result.delivered is True
    assert result.error_code == ""

    # Outbox row is gone.
    cur = await conn.execute("SELECT COUNT(*) FROM connect_fabric_outbox")
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 0

    # Coordinator saw exactly one send.
    assert len(coord.sent) == 1
    target_node, msg_type, payload = coord.sent[0]
    assert target_node == "peer-B"
    assert msg_type == "connect_envelope"
    assert payload["source_did"] == local_did_for("alice")
    assert payload["sender_party_id"] == "party-x"
    inner = json.loads(payload["envelope"])
    assert inner["msg"] == MSG_TEXT_SEND
    assert inner["data"]["body"] == "hello fabric"


@pytest.mark.asyncio
async def test_dispatch_peer_disconnected_queues(conn) -> None:
    """Peer offline → envelope persisted in outbox, queued for drain."""

    coord = FakeCoordinator(connected=False)
    env = _make_text_send_envelope()

    result = await dispatch_fabric_envelope(
        conn,
        coordinator=coord,
        target_hostname="instance-B",
        source_did=local_did_for("alice"),
        sender_user_id="alice",
        sender_party_id="party-x",
        envelope=env,
    )

    assert result.queued is True
    assert result.delivered is False

    cur = await conn.execute(
        "SELECT target_node_id, sender_user_id, attempts, last_error "
        "FROM connect_fabric_outbox",
    )
    rows = await cur.fetchall()
    await cur.close()
    assert len(rows) == 1
    target_node, sender_uid, attempts, last_err = rows[0]
    assert target_node == "peer-B"
    assert sender_uid == "alice"
    assert attempts == 1
    assert last_err == "peer_unreachable"


@pytest.mark.asyncio
async def test_dispatch_no_coordinator_returns_fabric_unavailable(conn) -> None:
    """Fabric disabled → clean error, no outbox row."""

    result = await dispatch_fabric_envelope(
        conn,
        coordinator=None,
        target_hostname="instance-B",
        source_did=local_did_for("alice"),
        sender_user_id="alice",
        sender_party_id="party-x",
        envelope=_make_text_send_envelope(),
    )
    assert result.queued is False
    assert result.delivered is False
    assert result.error_code == "fabric_unavailable"

    cur = await conn.execute("SELECT COUNT(*) FROM connect_fabric_outbox")
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 0


@pytest.mark.asyncio
async def test_dispatch_unknown_peer_returns_fabric_peer_unknown(conn) -> None:
    """Hostname not in fabric_nodes → fabric_peer_unknown, no outbox row."""

    coord = FakeCoordinator(node_id="peer-B", hostname="instance-B")
    result = await dispatch_fabric_envelope(
        conn,
        coordinator=coord,
        target_hostname="instance-Z",  # unpaired
        source_did=local_did_for("alice"),
        sender_user_id="alice",
        sender_party_id="party-x",
        envelope=_make_text_send_envelope(),
    )
    assert result.queued is False
    assert result.error_code == "fabric_peer_unknown"

    cur = await conn.execute("SELECT COUNT(*) FROM connect_fabric_outbox")
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 0


# ── Drain ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drain_flushes_queued_envelopes_in_order(conn) -> None:
    """3 queued envelopes drain in queued_at order on reconnect."""

    coord = FakeCoordinator(connected=False)
    for i in range(3):
        await dispatch_fabric_envelope(
            conn, coordinator=coord, target_hostname="instance-B",
            source_did=local_did_for("alice"),
            sender_user_id="alice",
            sender_party_id="party-x",
            envelope=_make_text_send_envelope(body=f"msg-{i}"),
        )

    # Peer reconnects.
    coord.connected = True
    counters = await drain_outbox_for_peer(
        conn, coordinator=coord, node_id="peer-B",
    )

    assert counters == {"sent": 3, "still_queued": 0, "exhausted": 0}
    cur = await conn.execute("SELECT COUNT(*) FROM connect_fabric_outbox")
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 0

    # Order preserved.
    bodies = [
        json.loads(p["envelope"])["data"]["body"]
        for _, _, p in coord.sent
    ]
    assert bodies == ["msg-0", "msg-1", "msg-2"]


@pytest.mark.asyncio
async def test_drain_still_failing_increments_attempts(conn) -> None:
    """If the peer comes up briefly then drops again, attempts bumps."""

    coord = FakeCoordinator(connected=False)
    await dispatch_fabric_envelope(
        conn, coordinator=coord, target_hostname="instance-B",
        source_did=local_did_for("alice"),
        sender_user_id="alice",
        sender_party_id="party-x",
        envelope=_make_text_send_envelope(),
    )

    # Reconnect, but send_to_peer still fails for some reason.
    coord.fail_send = True
    counters = await drain_outbox_for_peer(
        conn, coordinator=coord, node_id="peer-B",
    )
    assert counters == {"sent": 0, "still_queued": 1, "exhausted": 0}

    cur = await conn.execute(
        "SELECT attempts FROM connect_fabric_outbox LIMIT 1",
    )
    (attempts,) = await cur.fetchone()
    await cur.close()
    # First dispatch attempt + one drain attempt = 2.
    assert attempts == 2


@pytest.mark.asyncio
async def test_drain_exhausts_after_max_attempts(conn) -> None:
    """Row whose attempts >= MAX_OUTBOX_ATTEMPTS is dropped on next fail."""

    coord = FakeCoordinator(connected=False)
    await dispatch_fabric_envelope(
        conn, coordinator=coord, target_hostname="instance-B",
        source_did=local_did_for("alice"),
        sender_user_id="alice",
        sender_party_id="party-x",
        envelope=_make_text_send_envelope(),
    )

    # Pre-bump attempts to one short of the cap.
    await conn.execute(
        "UPDATE connect_fabric_outbox SET attempts = ?",
        (MAX_OUTBOX_ATTEMPTS - 1,),
    )
    await conn.commit()

    coord.fail_send = True
    counters = await drain_outbox_for_peer(
        conn, coordinator=coord, node_id="peer-B",
    )
    assert counters == {"sent": 0, "still_queued": 0, "exhausted": 1}

    cur = await conn.execute("SELECT COUNT(*) FROM connect_fabric_outbox")
    (n,) = await cur.fetchone()
    await cur.close()
    assert n == 0


@pytest.mark.asyncio
async def test_drain_corrupt_row_evicted(conn) -> None:
    """A row with un-parseable envelope JSON gets cleaned up rather
    than blocking the drain queue indefinitely."""

    coord = FakeCoordinator(connected=True)
    await conn.execute(
        """INSERT INTO connect_fabric_outbox
              (id, target_node_id, sender_user_id, envelope_json,
               queued_at, attempts)
            VALUES ('bogus', 'peer-B', 'alice', 'not-json{', 0, 0)""",
    )
    await conn.commit()

    counters = await drain_outbox_for_peer(
        conn, coordinator=coord, node_id="peer-B",
    )
    assert counters == {"sent": 0, "still_queued": 0, "exhausted": 1}


# ── Inbound dispatch ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inbound_malformed_envelope_returns_clean_error(conn) -> None:
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=ConnectHub(),
        notification_hub=None,
        fabric_payload={"envelope": "not-a-json"},
    )
    assert res == {"applied": False, "verb": "", "error": "malformed"}


@pytest.mark.asyncio
async def test_inbound_missing_source_did_rejected(conn) -> None:
    env = _make_text_send_envelope()
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=ConnectHub(),
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            # source_did intentionally missing
            "sender_party_id": "party-x",
        },
    )
    assert res["applied"] is False
    assert res["error"] == "missing_source_did"


@pytest.mark.asyncio
async def test_inbound_misroute_rejected(conn) -> None:
    """Inner envelope.peer points at a fabric DID, not a local one —
    means the sender misrouted; we refuse to apply."""

    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND,
        peer="elsewhere@some-other-instance",  # fabric DID, not local
        data={"thread_id": "t-1", "message_id": "m-1", "body": "x"},
    )
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=ConnectHub(),
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            "source_did": local_did_for("alice"),
            "sender_party_id": "party-x",
        },
    )
    assert res["applied"] is False
    assert res["error"] == "misroute"


@pytest.mark.asyncio
async def test_inbound_text_verb_routes_through_dispatcher(conn) -> None:
    """Text verb routes to the per-verb dispatcher and applies. Was a
    Phase-1 NotImplementedError stub; Phase 2 wired the real logic."""

    env = _make_text_send_envelope(to_did=local_did_for("bob"))
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=ConnectHub(),
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            "source_did": local_did_for("alice"),
            "target_user_id": "bob",
            "sender_party_id": "party-x",
        },
    )
    assert res["applied"] is True
    assert res["verb"] == MSG_TEXT_SEND


@pytest.mark.asyncio
async def test_inbound_call_verb_routes_through_dispatcher(conn) -> None:
    """Call verb routes through the dispatcher and applies. Was a
    Phase-1 NotImplementedError stub; Phase 4 wired real call logic."""

    from augmentum.connect.protocol import MSG_INVITE

    env = ConnectEnvelope(
        kind="msg", verb=MSG_INVITE, peer=local_did_for("bob"),
        data={"call_id": "c-1", "modalities": "audio"},
    )
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=ConnectHub(),
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            "source_did": local_did_for("alice"),
            "target_user_id": "bob",
            "sender_party_id": "party-x",
        },
    )
    assert res["applied"] is True
    assert res["verb"] == MSG_INVITE


# ── DID normalisation at the fabric boundary ──────────────────────────


@pytest.mark.asyncio
async def test_inbound_rewrites_source_did_via_sender_node_id(conn) -> None:
    """Source DID arrives in the sender's local-sentinel form
    (``alice@this-instance``) and the inbound dispatcher rewrites it to
    use the sender's hostname from THIS instance's paired-peer registry
    (``alice@instance-A``).

    Asserts the rewrite by reading back the recipient's mirror row +
    the WS event Bob's UI receives — both must show the hostname-form
    DID, not the sentinel.
    """
    from augmentum.connect.hub import ConnectHub as _Hub
    from augmentum.connect.protocol import EVENT_TEXT_RECEIVED

    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, payload: str) -> None:
            self.sent.append(payload)

    hub = _Hub()
    ws = _FakeWS()
    await hub.attach(ws=ws, user_id="bob", user_did=local_did_for("bob"))
    ws.sent.clear()

    # Bob's instance has Alice's peer registered as ``node-A`` ↔
    # hostname ``instance-A``.
    coord = FakeCoordinator(node_id="node-A", hostname="instance-A")
    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer=local_did_for("bob"),
        data={"thread_id": "t-norm", "message_id": "m-norm", "body": "hi"},
    )
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=hub,
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            "source_did": local_did_for("alice"),  # alice@this-instance
            "target_user_id": "bob",
            "sender_party_id": "party-x",
        },
        coordinator=coord,
        sender_node_id="node-A",
    )
    assert res["applied"] is True

    cur = await conn.execute(
        "SELECT sender_did FROM connect_messages "
        "WHERE message_id = ? AND user_id = ?",
        ("m-norm", "bob"),
    )
    (sender_did_stored,) = await cur.fetchone()
    await cur.close()
    assert sender_did_stored == "alice@instance-A"

    received = json.loads(ws.sent[-1])
    assert received["event"] == EVENT_TEXT_RECEIVED
    assert received["from"] == "alice@instance-A"


@pytest.mark.asyncio
async def test_inbound_falls_through_when_sender_node_id_unknown(conn) -> None:
    """Unknown sender_node_id → no rewrite, source_did flows through
    as-is. Defensive: a misbehaving peer claiming an unknown identity
    shouldn't cause inbound dispatch to crash; the verbs apply with
    the original source_did (and the unique-pair index on contacts
    will still scope to that string)."""

    coord = FakeCoordinator(node_id="node-A", hostname="instance-A")
    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer=local_did_for("bob"),
        data={
            "thread_id": "t-unk", "message_id": "m-unk", "body": "hi",
        },
    )
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=ConnectHub(),
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            "source_did": local_did_for("alice"),
            "target_user_id": "bob",
            "sender_party_id": "party-x",
        },
        coordinator=coord,
        sender_node_id="node-UNKNOWN",
    )
    assert res["applied"] is True
    cur = await conn.execute(
        "SELECT sender_did FROM connect_messages "
        "WHERE message_id = ? AND user_id = ?",
        ("m-unk", "bob"),
    )
    (sender_did_stored,) = await cur.fetchone()
    await cur.close()
    # No rewrite — original sentinel form persists.
    assert sender_did_stored == local_did_for("alice")


@pytest.mark.asyncio
async def test_inbound_does_not_rewrite_already_normalised_did(conn) -> None:
    """Source DID is already a hostname-form (the sender's instance was
    properly configured with its hostname) → the normaliser leaves it
    alone. Guards against double-rewriting if a future sender ships the
    fully-qualified DID directly."""

    coord = FakeCoordinator(node_id="node-A", hostname="instance-A")
    env = ConnectEnvelope(
        kind="msg", verb=MSG_TEXT_SEND, peer=local_did_for("bob"),
        data={"thread_id": "t-fq", "message_id": "m-fq", "body": "hi"},
    )
    res = await apply_inbound_fabric_envelope(
        conn,
        connect_hub=ConnectHub(),
        notification_hub=None,
        fabric_payload={
            "envelope": serialise_envelope(env),
            "source_did": "alice@instance-A",  # already FQ
            "target_user_id": "bob",
            "sender_party_id": "party-x",
        },
        coordinator=coord,
        sender_node_id="node-A",
    )
    assert res["applied"] is True
    cur = await conn.execute(
        "SELECT sender_did FROM connect_messages "
        "WHERE message_id = ? AND user_id = ?",
        ("m-fq", "bob"),
    )
    (sender_did_stored,) = await cur.fetchone()
    await cur.close()
    assert sender_did_stored == "alice@instance-A"


def test_normalise_source_did_unit_pure_function() -> None:
    """Pure-function unit tests on the normaliser (no DB / WS)."""
    from augmentum.connect.fabric_inbound import _normalise_source_did

    coord = FakeCoordinator(node_id="node-A", hostname="instance-A")

    # Happy path: sentinel rewritten to hostname-form.
    assert _normalise_source_did(
        source_did="alice@this-instance",
        coordinator=coord, sender_node_id="node-A",
    ) == "alice@instance-A"

    # Unknown sender_node_id → no rewrite.
    assert _normalise_source_did(
        source_did="alice@this-instance",
        coordinator=coord, sender_node_id="node-OTHER",
    ) == "alice@this-instance"

    # Missing coordinator → no rewrite.
    assert _normalise_source_did(
        source_did="alice@this-instance",
        coordinator=None, sender_node_id="node-A",
    ) == "alice@this-instance"

    # Empty sender_node_id → no rewrite.
    assert _normalise_source_did(
        source_did="alice@this-instance",
        coordinator=coord, sender_node_id="",
    ) == "alice@this-instance"

    # Non-sentinel host → passed through unchanged.
    assert _normalise_source_did(
        source_did="alice@other-host",
        coordinator=coord, sender_node_id="node-A",
    ) == "alice@other-host"

    # Empty DID → no-op.
    assert _normalise_source_did(
        source_did="",
        coordinator=coord, sender_node_id="node-A",
    ) == ""
