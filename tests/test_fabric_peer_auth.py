"""Tests for the pairing flow + signature verification.

Pins:

  - build/verify roundtrip of a pair request succeeds
  - fingerprint mismatch is detected (operator typed wrong fp)
  - own-fingerprint mismatch is detected (request sent to wrong node)
  - timestamp out of window is rejected (replay defense)
  - signature tampering is detected
  - persist_pairing writes a fabric_nodes row + is idempotent
  - load_paired_peers / lookup_peer_pubkey roundtrip
"""
from __future__ import annotations

import time

import aiosqlite
import pytest

from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.peer_auth import (
    PairRequest,
    PairRequestError,
    build_pair_request,
    load_paired_peers,
    lookup_peer_pubkey,
    persist_pairing,
    verify_pair_request,
)
from augmentum.state.settings_store import SettingsStore


async def _make_db() -> tuple[aiosqlite.Connection, SettingsStore]:
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
    return conn, SettingsStore(conn)


@pytest.mark.asyncio
async def test_build_and_verify_roundtrip():
    # Two separate identities representing two peers about to pair.
    conn_a, store_a = await _make_db()
    conn_b, store_b = await _make_db()
    try:
        identity_a = await FabricIdentity.from_settings_store(store_a)
        identity_b = await FabricIdentity.from_settings_store(store_b)
        # A targets B's fingerprint (what the operator typed on A's UI).
        req = build_pair_request(
            identity=identity_a,
            hostname="host-a",
            target_fingerprint_hint=identity_b.fingerprint,
        )
        # B verifies the request against its own fingerprint.
        verify_pair_request(req, own_fingerprint=identity_b.fingerprint)
    finally:
        await conn_a.close()
        await conn_b.close()


@pytest.mark.asyncio
async def test_verify_rejects_wrong_target_fingerprint():
    """If the operator typed the wrong fingerprint into peer A's UI
    (e.g. pasted peer C's fingerprint while trying to pair with B),
    peer B catches the mismatch and refuses the request.
    """
    conn_a, store_a = await _make_db()
    conn_b, store_b = await _make_db()
    try:
        identity_a = await FabricIdentity.from_settings_store(store_a)
        identity_b = await FabricIdentity.from_settings_store(store_b)
        # A targets a *different* fingerprint by mistake.
        req = build_pair_request(
            identity=identity_a,
            hostname="host-a",
            target_fingerprint_hint="SHA256:wrong00000000000000000000000000",
        )
        # B's check fires.
        with pytest.raises(PairRequestError, match="fingerprint"):
            verify_pair_request(req, own_fingerprint=identity_b.fingerprint)
    finally:
        await conn_a.close()
        await conn_b.close()


@pytest.mark.asyncio
async def test_verify_rejects_stale_timestamp():
    conn_a, store_a = await _make_db()
    conn_b, store_b = await _make_db()
    try:
        identity_a = await FabricIdentity.from_settings_store(store_a)
        identity_b = await FabricIdentity.from_settings_store(store_b)
        req = build_pair_request(
            identity=identity_a,
            hostname="host-a",
            target_fingerprint_hint=identity_b.fingerprint,
        )
        # Build a tampered request with an old timestamp; re-sign so
        # the signature itself isn't what catches it.
        from augmentum.fabric.peer_auth import _pair_canonical_bytes
        import base64
        old_ts = int(time.time()) - 3600  # 1h old, well outside the 5min window
        canonical = _pair_canonical_bytes(
            sender_node_id=req.sender_node_id,
            hostname=req.hostname,
            pubkey_b64=req.pubkey_b64,
            fingerprint_hint=req.fingerprint_hint,
            role=req.role,
            timestamp=old_ts,
        )
        new_sig = identity_a.sign(canonical)
        stale = PairRequest(
            sender_node_id=req.sender_node_id,
            hostname=req.hostname,
            pubkey_b64=req.pubkey_b64,
            fingerprint_hint=req.fingerprint_hint,
            role=req.role,
            timestamp=old_ts,
            signature=base64.b64encode(new_sig).decode("ascii"),
        )
        with pytest.raises(PairRequestError, match="timestamp"):
            verify_pair_request(stale, own_fingerprint=identity_b.fingerprint)
    finally:
        await conn_a.close()
        await conn_b.close()


@pytest.mark.asyncio
async def test_verify_rejects_tampered_signature():
    conn_a, store_a = await _make_db()
    conn_b, store_b = await _make_db()
    try:
        identity_a = await FabricIdentity.from_settings_store(store_a)
        identity_b = await FabricIdentity.from_settings_store(store_b)
        req = build_pair_request(
            identity=identity_a,
            hostname="host-a",
            target_fingerprint_hint=identity_b.fingerprint,
        )
        # Tamper the hostname AFTER signing. Replace the request
        # with a version whose hostname differs but signature unchanged.
        tampered = PairRequest(
            sender_node_id=req.sender_node_id,
            hostname="other-host",  # tampered
            pubkey_b64=req.pubkey_b64,
            fingerprint_hint=req.fingerprint_hint,
            role=req.role,
            timestamp=req.timestamp,
            signature=req.signature,
        )
        with pytest.raises(PairRequestError, match="signature"):
            verify_pair_request(tampered, own_fingerprint=identity_b.fingerprint)
    finally:
        await conn_a.close()
        await conn_b.close()


@pytest.mark.asyncio
async def test_persist_and_lookup_roundtrip():
    conn_a, store_a = await _make_db()
    conn_b, store_b = await _make_db()
    try:
        identity_a = await FabricIdentity.from_settings_store(store_a)
        identity_b = await FabricIdentity.from_settings_store(store_b)
        req = build_pair_request(
            identity=identity_a,
            hostname="host-a",
            target_fingerprint_hint=identity_b.fingerprint,
        )
        verify_pair_request(req, own_fingerprint=identity_b.fingerprint)
        paired = await persist_pairing(conn_b, req=req, addr="192.168.1.10:6443")
        # Stored row matches request.
        assert paired.node_id == identity_a.node_id
        assert paired.pubkey_b64 == identity_a.public_key_b64
        # SECURITY: inbound pair lands PENDING (fabric_share_enabled=0). A
        # signed pair request is NOT operator consent, and /api/fabric/pair is
        # unauthenticated, so lookup_peer_pubkey (the data-plane identity gate)
        # must NOT resolve a pending peer — a self-enrolled attacker can't
        # authenticate envelopes until an operator approves.
        assert paired.fabric_share_enabled is False
        assert await lookup_peer_pubkey(conn_b, identity_a.node_id) is None
        # After operator approval (enable), the pinned key resolves.
        await conn_b.execute(
            "UPDATE fabric_nodes SET fabric_share_enabled = 1 WHERE id = ?",
            (identity_a.node_id,),
        )
        await conn_b.commit()
        pub = await lookup_peer_pubkey(conn_b, identity_a.node_id)
        assert pub == identity_a.public_key_b64
        # load_paired_peers sees the row.
        peers = await load_paired_peers(conn_b)
        assert len(peers) == 1
        assert peers[0].node_id == identity_a.node_id
    finally:
        await conn_a.close()
        await conn_b.close()


@pytest.mark.asyncio
async def test_persist_pairing_is_idempotent():
    """A second pair from the same node_id refreshes the row instead
    of erroring (operator may re-pair after a keypair rotation).
    """
    conn_a, store_a = await _make_db()
    conn_b, store_b = await _make_db()
    try:
        identity_a = await FabricIdentity.from_settings_store(store_a)
        identity_b = await FabricIdentity.from_settings_store(store_b)
        # First pair.
        req1 = build_pair_request(
            identity=identity_a, hostname="host-a-v1",
            target_fingerprint_hint=identity_b.fingerprint,
        )
        verify_pair_request(req1, own_fingerprint=identity_b.fingerprint)
        await persist_pairing(conn_b, req=req1, addr="192.168.1.10:6443")

        # Second pair from same node with different hostname.
        req2 = build_pair_request(
            identity=identity_a, hostname="host-a-v2",
            target_fingerprint_hint=identity_b.fingerprint,
        )
        verify_pair_request(req2, own_fingerprint=identity_b.fingerprint)
        await persist_pairing(conn_b, req=req2, addr="192.168.1.11:6443")

        # Only ONE row in fabric_nodes (idempotent upsert).
        peers = await load_paired_peers(conn_b)
        assert len(peers) == 1
        # Refreshed values reflect the second request.
        assert peers[0].hostname == "host-a-v2"
        assert peers[0].addr == "192.168.1.11:6443"
    finally:
        await conn_a.close()
        await conn_b.close()


@pytest.mark.asyncio
async def test_lookup_peer_pubkey_returns_none_for_unknown():
    conn, _ = await _make_db()
    try:
        result = await lookup_peer_pubkey(conn, "never-paired-node-id")
        assert result is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_persist_remote_node_records_icon():
    """Phase 8: outbound pair-with-remote persists the operator-picked
    icon for this peer's row. Used by the UI peer list, matrix view,
    and in-chat peer badge.
    """
    from augmentum.fabric.peer_auth import persist_remote_node

    conn, _ = await _make_db()
    try:
        paired = await persist_remote_node(
            conn,
            node_id="peer-rocket", hostname="rocket-host", role="peer",
            pubkey_b64="dGVzdA==", addr="192.168.1.20:6443",
            icon="🚀",
        )
        assert paired.icon == "🚀"
        # Reload from DB to confirm the column round-tripped.
        peers = await load_paired_peers(conn)
        assert len(peers) == 1
        assert peers[0].icon == "🚀"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_persist_remote_node_preserves_icon_on_empty_repair():
    """If a re-pair comes in with empty icon (e.g. inbound /pair from
    the remote side, which doesn't know our local label), the existing
    operator-chosen icon must NOT be wiped. The ON CONFLICT clause uses
    a CASE to keep the prior value when the incoming icon is empty.
    """
    from augmentum.fabric.peer_auth import persist_remote_node

    conn, _ = await _make_db()
    try:
        # First pair: operator labels the peer with 🏎.
        await persist_remote_node(
            conn, node_id="peer-rc", hostname="rc-host", role="peer",
            pubkey_b64="dGVzdA==", addr="192.168.1.20:6443", icon="🏎",
        )
        # Re-pair comes through (e.g. fingerprint refresh) with no icon.
        await persist_remote_node(
            conn, node_id="peer-rc", hostname="rc-host", role="peer",
            pubkey_b64="dGVzdA==", addr="192.168.1.20:6443", icon="",
        )
        # The original icon survived.
        peers = await load_paired_peers(conn)
        assert peers[0].icon == "🏎"

        # But a re-pair WITH a new icon DOES overwrite (the operator
        # explicitly chose to re-label).
        await persist_remote_node(
            conn, node_id="peer-rc", hostname="rc-host", role="peer",
            pubkey_b64="dGVzdA==", addr="192.168.1.20:6443", icon="🐢",
        )
        peers = await load_paired_peers(conn)
        assert peers[0].icon == "🐢"
    finally:
        await conn.close()
