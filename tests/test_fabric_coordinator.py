"""Tests for the in-memory fabric coordinator.

The coordinator's contract:

  - initialise() loads paired peers from fabric_nodes
  - attach_connection succeeds for known peers, fails for unknown
  - re-attach closes the old socket (replace-on-reconnect)
  - detach_connection clears the live socket but keeps the peer
    registered
  - register/unregister_paired_peer updates the registry
  - peer_count returns the right aggregates
  - next_local_seq is monotonic
  - record_heartbeat updates last-seen and last-seq invariants
"""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_auth import PairedPeer
from augmentum.state.settings_store import SettingsStore


async def _make_env() -> tuple[aiosqlite.Connection, SettingsStore, FabricIdentity, FabricCoordinator]:
    """Set up an in-memory DB with the fabric_nodes schema + identity."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.execute(
        """CREATE TABLE fabric_nodes (
            id TEXT PRIMARY KEY,
            hostname TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'peer',
            pubkey_ed25519 TEXT NOT NULL,
            pubkey_fingerprint TEXT NOT NULL,
            addr TEXT NOT NULL DEFAULT '',
            tier TEXT NOT NULL DEFAULT 'local',
            fabric_share_enabled INTEGER NOT NULL DEFAULT 1,
            paired_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen_at TEXT, icon TEXT NOT NULL DEFAULT '')"""
    )
    await conn.commit()
    store = SettingsStore(conn)
    identity = await FabricIdentity.from_settings_store(store)
    coordinator = FabricCoordinator(identity, conn)
    return conn, store, identity, coordinator


def _fake_peer(node_id: str, role: str = "peer") -> PairedPeer:
    return PairedPeer(
        node_id=node_id,
        hostname=f"host-{node_id[:4]}",
        role=role,
        pubkey_b64="dGVzdC1wdWJrZXkta2V5",  # not a real key; coordinator doesn't verify
        fingerprint=f"SHA256:fp{node_id[:8]}",
        addr="192.168.1.10:6443",
        tier="local",
        fabric_share_enabled=True,
        paired_at="2026-05-15 22:00:00",
        last_seen_at=None,
    )


class _FakeWebSocket:
    """Minimal duck-typed WebSocket for coordinator tests.

    Records close() + send_text() calls so tests can assert the
    coordinator emitted the right envelope. Phase 9 added send_text;
    everything before that only needed close().
    """

    def __init__(self) -> None:
        self.closed = False
        self.close_reason: str | None = None
        self.sent_frames: list[str] = []

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_reason = reason

    async def send_text(self, frame: str) -> None:
        # Mimic Starlette's failure mode if a test reaches send after
        # close. The coordinator catches + absorbs this as "peer gone".
        if self.closed:
            raise ConnectionError("socket already closed")
        self.sent_frames.append(frame)


@pytest.mark.asyncio
async def test_initialise_loads_paired_peers():
    conn, _, _, coordinator = await _make_env()
    try:
        # Seed two paired peers in the DB directly.
        for nid in ("peer-aaa", "peer-bbb"):
            await conn.execute(
                """INSERT INTO fabric_nodes
                   (id, hostname, role, pubkey_ed25519, pubkey_fingerprint, addr)
                   VALUES (?, ?, 'peer', 'pk', 'fp', '192.168.1.10:6443')""",
                (nid, f"host-{nid}"),
            )
        await conn.commit()
        await coordinator.initialise()
        assert set(coordinator.known_peer_ids()) == {"peer-aaa", "peer-bbb"}
        assert coordinator.connected_peer_ids() == []
        assert coordinator.peer_count() == {"total": 2, "connected": 0, "offline": 2}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_attach_succeeds_for_known_peer():
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("known-peer"))
        ws = _FakeWebSocket()
        assert await coordinator.attach_connection("known-peer", ws) is True
        assert coordinator.peer_state("known-peer").connected is True
        assert coordinator.peer_state("known-peer").socket is ws
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_attach_fails_for_unknown_peer():
    conn, _, _, coordinator = await _make_env()
    try:
        ws = _FakeWebSocket()
        assert await coordinator.attach_connection("never-paired", ws) is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_reattach_closes_old_socket():
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("p"))
        old_ws = _FakeWebSocket()
        await coordinator.attach_connection("p", old_ws)
        new_ws = _FakeWebSocket()
        await coordinator.attach_connection("p", new_ws)
        # Old socket should have been closed on replacement.
        assert old_ws.closed is True
        # New socket is now the live one.
        assert coordinator.peer_state("p").socket is new_ws
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_detach_marks_offline_but_keeps_registered():
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("p"))
        ws = _FakeWebSocket()
        await coordinator.attach_connection("p", ws)
        await coordinator.detach_connection("p")
        # Peer still in registry, but no live socket.
        assert "p" in coordinator.known_peer_ids()
        assert coordinator.peer_state("p").connected is False
        assert coordinator.peer_state("p").socket is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_detach_clears_advertised_capabilities():
    """Phase 6.x hygiene: a disconnected peer's advertised
    capabilities should be dropped. find_peers_with_capability
    already filters by connection, but the matrix-view UI consumer
    iterates state.capabilities directly — and would otherwise show
    "loaded · 4 slots" for a peer that's been offline for hours.
    """
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("p"))
        ws = _FakeWebSocket()
        await coordinator.attach_connection("p", ws)
        coordinator.record_remote_capabilities("p", [{
            "kind": "llm.inference", "schema_version": 1,
            "model_id": "some-model", "model_family": "qwen3",
            "params_b": 7.0, "ctx_max": 8192, "loaded": True,
            "free_slots": 4,
        }])
        # Sanity: cap was recorded.
        assert len(coordinator.peer_state("p").capabilities) == 1

        await coordinator.detach_connection("p")
        # Cap list is now empty — a reconnect would repopulate via
        # the next heartbeat.
        assert coordinator.peer_state("p").capabilities == []
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_detach_writes_last_seen_at_to_db_and_memory():
    """Phase 6.x: last_seen_at must be persisted to fabric_nodes AND
    reflected in the in-memory PairedPeer (so /peers responses don't
    serve stale data until a coordinator restart).
    """
    conn, _, _, coordinator = await _make_env()
    try:
        # Persist a row in fabric_nodes the UPDATE can hit.
        await conn.execute(
            """INSERT INTO fabric_nodes
               (id, pubkey_ed25519, pubkey_fingerprint)
               VALUES ('p', 'pk', 'fp')""",
        )
        await conn.commit()

        await coordinator.register_paired_peer(_fake_peer("p"))
        ws = _FakeWebSocket()
        await coordinator.attach_connection("p", ws)
        await coordinator.detach_connection("p")

        # In-memory PairedPeer now carries a last_seen_at timestamp.
        paired = coordinator.peer_state("p").paired
        assert paired is not None
        assert paired.last_seen_at  # truthy ISO string

        # DB row matches.
        cur = await conn.execute(
            "SELECT last_seen_at FROM fabric_nodes WHERE id='p'",
        )
        row = await cur.fetchone()
        assert row is not None
        assert row[0] == paired.last_seen_at
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_unregister_removes_and_closes():
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("p"))
        ws = _FakeWebSocket()
        await coordinator.attach_connection("p", ws)
        await coordinator.unregister_peer("p")
        # Peer gone from registry, socket closed.
        assert "p" not in coordinator.known_peer_ids()
        assert ws.closed is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_peer_count_aggregates():
    conn, _, _, coordinator = await _make_env()
    try:
        for nid in ("a", "b", "c"):
            await coordinator.register_paired_peer(_fake_peer(nid))
        # Connect one of three.
        ws = _FakeWebSocket()
        await coordinator.attach_connection("b", ws)
        counts = coordinator.peer_count()
        assert counts == {"total": 3, "connected": 1, "offline": 2}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_next_local_seq_is_monotonic():
    conn, _, _, coordinator = await _make_env()
    try:
        seqs = [coordinator.next_local_seq() for _ in range(5)]
        assert seqs == [1, 2, 3, 4, 5]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_record_heartbeat_updates_last_seq():
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("p"))
        coordinator.record_heartbeat("p", 42)
        assert coordinator.peer_state("p").last_seq_received == 42
        # Stale (lower) seq doesn't roll back the watermark.
        coordinator.record_heartbeat("p", 7)
        assert coordinator.peer_state("p").last_seq_received == 42
        # Higher seq advances.
        coordinator.record_heartbeat("p", 100)
        assert coordinator.peer_state("p").last_seq_received == 100
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_shutdown_closes_all_active_sockets():
    conn, _, _, coordinator = await _make_env()
    try:
        sockets = []
        for nid in ("a", "b"):
            await coordinator.register_paired_peer(_fake_peer(nid))
            ws = _FakeWebSocket()
            sockets.append(ws)
            await coordinator.attach_connection(nid, ws)
        await coordinator.shutdown()
        assert all(s.closed for s in sockets)
        # State reset.
        for nid in ("a", "b"):
            assert coordinator.peer_state(nid).connected is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_send_to_peer_emits_signed_envelope():
    """Phase 9.2: coordinator.send_to_peer writes a signed envelope
    via the attached WS. Verifies the frame is JSON, carries the
    msg_type + payload, and has a non-empty signature (the receiver
    side's envelope verification is exercised in protocol tests).
    """
    import json as _json

    conn, _, identity, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("peer-x"))
        ws = _FakeWebSocket()
        await coordinator.attach_connection("peer-x", ws)

        ok = await coordinator.send_to_peer(
            "peer-x",
            msg_type="cancel_request",
            payload={"request_id": "req-123"},
        )
        assert ok is True
        assert len(ws.sent_frames) == 1
        frame = _json.loads(ws.sent_frames[0])
        # Wire keys are short-form (see FabricEnvelope.to_wire):
        # t=msg_type, from=sender_node_id, sig=signature.
        assert frame["t"] == "cancel_request"
        assert frame["payload"] == {"request_id": "req-123"}
        assert frame["from"] == identity.node_id
        assert frame["sig"]  # non-empty base64
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_send_to_peer_returns_false_when_unknown():
    """A peer we've never paired returns False, no crash."""
    conn, _, _, coordinator = await _make_env()
    try:
        ok = await coordinator.send_to_peer(
            "never-paired", msg_type="cancel_request", payload={"request_id": "r"},
        )
        assert ok is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_send_to_peer_returns_false_when_offline():
    """Peer is registered but currently disconnected → False (no
    exception, no buffered send). The cancel-on-cancel contract is
    best-effort; the caller treats False as "couldn't push, peer
    will sort itself out via TCP-close fallback".
    """
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("p-off"))
        # Never attached → connected is False
        ok = await coordinator.send_to_peer(
            "p-off", msg_type="cancel_request", payload={"request_id": "r"},
        )
        assert ok is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_inflight_registry_register_cancel_unregister():
    """Phase 9.3: register an in-flight task by request_id; a later
    cancel_inflight finds it + calls .cancel(); unregister drops it
    from the table.
    """
    conn, _, _, coordinator = await _make_env()
    try:
        # asyncio.Task() is hard to construct directly; use a
        # MagicMock with a .cancel attribute. The coordinator only
        # calls .cancel() — duck typing is fine.
        from unittest.mock import MagicMock
        fake_task = MagicMock()

        coordinator.register_inflight("req-abc", fake_task)
        cancelled = coordinator.cancel_inflight("req-abc")
        assert cancelled is True
        fake_task.cancel.assert_called_once()

        # Second call returns False (already popped).
        assert coordinator.cancel_inflight("req-abc") is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_handle_inbound_envelope_records_lifecycle_events():
    """Phase 9-lifecycle: MSG_JOB_* envelopes accumulate per-request
    in _peer_call_events. Consumer reads via peer_call_events (peek)
    or drain_peer_call_events (consuming read).
    """
    from augmentum.fabric.protocol import (
        MSG_JOB_COMPLETED,
        MSG_JOB_STARTED,
        FabricEnvelope,
    )

    conn, _, identity_a, coordinator = await _make_env()
    try:
        # Started + completed for one request_id.
        started = FabricEnvelope.build(
            msg_type=MSG_JOB_STARTED, seq=1, sender_node_id="peer-b",
            payload={"request_id": "req-1", "kind": "llm.inference",
                     "path": "/v1/chat/completions"},
            signing_key=identity_a.private_key,
        )
        completed = FabricEnvelope.build(
            msg_type=MSG_JOB_COMPLETED, seq=2, sender_node_id="peer-b",
            payload={"request_id": "req-1", "ok": True},
            signing_key=identity_a.private_key,
        )
        coordinator.handle_inbound_envelope(started)
        coordinator.handle_inbound_envelope(completed)

        events = coordinator.peer_call_events("req-1")
        assert len(events) == 2
        assert events[0]["msg_type"] == "job_started"
        assert events[0]["kind"] == "llm.inference"
        assert events[1]["msg_type"] == "job_completed"

        # Drain returns the list + clears storage.
        drained = coordinator.drain_peer_call_events("req-1")
        assert len(drained) == 2
        assert coordinator.peer_call_events("req-1") == []
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_handle_inbound_envelope_dispatches_cancel():
    """The shared dispatcher used by both client.py + fabric_routes
    read loops: MSG_CANCEL_REQUEST envelopes call cancel_inflight
    on the matching request_id, AND every envelope updates
    heartbeat seq + applies capability payloads.
    """
    from unittest.mock import MagicMock

    from augmentum.fabric.protocol import (
        MSG_CANCEL_REQUEST,
        FabricEnvelope,
    )

    conn, _, identity_a, coordinator = await _make_env()
    try:
        # Register a fake peer + register an in-flight request
        # to make the cancel observable.
        await coordinator.register_paired_peer(_fake_peer("peer-b"))
        ws = _FakeWebSocket()
        await coordinator.attach_connection("peer-b", ws)
        fake_task = MagicMock()
        coordinator.register_inflight("req-xyz", fake_task)

        # Build a signed envelope FROM peer-b TO us with a
        # MSG_CANCEL_REQUEST. (We can sign with our own identity
        # since the dispatcher doesn't re-verify — that already
        # happened in the read loop.)
        env = FabricEnvelope.build(
            msg_type=MSG_CANCEL_REQUEST, seq=42,
            sender_node_id="peer-b",
            payload={"request_id": "req-xyz"},
            signing_key=identity_a.private_key,
        )
        coordinator.handle_inbound_envelope(env)

        # Heartbeat seq was applied (via record_heartbeat).
        assert coordinator.peer_state("peer-b").last_seq_received == 42
        # The in-flight task was cancelled.
        fake_task.cancel.assert_called_once()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_invalidate_peer_capability_drops_matching_entry():
    """Phase 9-failure-surfacing: when a peer 404s with "model not
    found", FabricBackend asks the coordinator to drop that single
    LLM advertisement. Verify the drop is precise — siblings stay.
    """
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("p"))
        ws = _FakeWebSocket()
        await coordinator.attach_connection("p", ws)
        # Advertise two LLMs + one image model on the peer.
        coordinator.record_remote_capabilities("p", [
            {"kind": "llm.inference", "schema_version": 1,
             "model_id": "flux-dev", "model_family": "flux"},
            {"kind": "llm.inference", "schema_version": 1,
             "model_id": "qwen-72b", "model_family": "qwen3"},
            {"kind": "image.generation", "schema_version": 1,
             "model_id": "sdxl", "family": "sdxl"},
        ])
        assert len(coordinator.peer_state("p").capabilities) == 3

        # Drop just the flux-dev LLM.
        dropped = coordinator.invalidate_peer_capability(
            "p", kind="llm.inference", model_id="flux-dev",
        )
        assert dropped is True
        caps = coordinator.peer_state("p").capabilities
        assert len(caps) == 2
        # qwen-72b LLM survives; sdxl image survives.
        names = {(c.kind, getattr(c, "model_id", "")) for c in caps}
        assert names == {
            ("llm.inference", "qwen-72b"),
            ("image.generation", "sdxl"),
        }
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_invalidate_peer_capability_unknown_peer_safe():
    """Unknown node_id returns False, never raises."""
    conn, _, _, coordinator = await _make_env()
    try:
        result = coordinator.invalidate_peer_capability(
            "never-paired", kind="llm.inference", model_id="x",
        )
        assert result is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_inflight_registry_unknown_id_safe():
    """cancel_inflight for a request_id that was never registered
    returns False, never raises. Important because the WS read loop
    calls cancel_inflight on every MSG_CANCEL_REQUEST envelope —
    races where the request finished before the cancel arrived
    are normal and must not log-spam.
    """
    conn, _, _, coordinator = await _make_env()
    try:
        assert coordinator.cancel_inflight("never-registered") is False
        # Empty request_id short-circuits silently.
        coordinator.register_inflight("", object())
        coordinator.unregister_inflight("")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_send_to_peer_absorbs_send_exceptions():
    """If the socket raises mid-send (peer disconnected during the
    await), the coordinator logs + returns False instead of bubbling
    the exception to the caller. Cancellation paths must never
    propagate transport errors into chat-egress paths.
    """
    conn, _, _, coordinator = await _make_env()
    try:
        await coordinator.register_paired_peer(_fake_peer("p-drop"))
        ws = _FakeWebSocket()
        await coordinator.attach_connection("p-drop", ws)
        # Simulate the peer dropping between lookup + send by closing
        # the socket; _FakeWebSocket.send_text raises ConnectionError
        # when closed.
        await ws.close()
        ok = await coordinator.send_to_peer(
            "p-drop", msg_type="cancel_request", payload={"request_id": "r"},
        )
        assert ok is False
    finally:
        await conn.close()
