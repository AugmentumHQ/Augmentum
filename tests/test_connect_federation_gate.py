"""Live-path admission gate for Connect-over-fabric (federation_gate.py).

Pins the safe-wiring contract:
  - DEFAULT-OFF: with fabric_federation_enabled False, every frame is
    allowed (existing installs unchanged).
  - when enabled: a denylisted/revoked sending instance is dropped;
    a known contact flows through; an unknown stranger is gated by
    posture (open=allow, private/allowlist=drop, knock=queued+dropped).
  - non-relationship verbs are never stranger-gated.
"""
from __future__ import annotations

import base64

import aiosqlite
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.config import settings
from augmentum.connect import federation_gate as fg
from augmentum.connect.protocol import MSG_TEXT_READ, MSG_TEXT_SEND
from augmentum.fabric.caller_id import authoritative_source_did
from augmentum.fabric.knock import list_pending
from augmentum.fabric.revocation import add_denylist

_SENDER_NODE = "nodeB123"


async def _make_db() -> tuple[aiosqlite.Connection, str]:
    """DB with the tables the gate touches + a pinned sender instance.
    Returns (conn, sender_instance_did)."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    for mig in (
        "165_fabric_identity.sql",       # fabric_nodes
        "219_connect_substrate.sql",     # connect_contacts
        "289_fabric_peer_identities.sql",
        "290_fabric_knocks.sql",
        "291_fabric_revocations.sql",
    ):
        with open(f"augmentum/state/migrations/{mig}") as f:
            await conn.executescript(f.read())
    # Pin the sending instance's key in fabric_nodes (as a verified pair).
    priv = Ed25519PrivateKey.generate()
    pub_b64 = base64.b64encode(priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )).decode()
    await conn.execute(
        "INSERT INTO fabric_nodes (id, pubkey_ed25519, pubkey_fingerprint) "
        "VALUES (?, ?, ?)",
        (_SENDER_NODE, pub_b64, "SHA256:test"),
    )
    await conn.commit()
    return conn, authoritative_source_did(pub_b64)


async def _add_contact(conn, user_id, peer_did):
    await conn.execute(
        "INSERT INTO connect_contacts (contact_id, user_id, peer_did) "
        "VALUES (?, ?, ?)",
        (f"c-{peer_did}", user_id, peer_did),
    )
    await conn.commit()


@pytest.fixture
def federation_on(monkeypatch):
    monkeypatch.setattr(settings, "fabric_federation_enabled", True, raising=False)
    monkeypatch.setattr(settings, "fabric_admission_posture", "knock", raising=False)


@pytest.mark.asyncio
async def test_default_off_allows_everything(monkeypatch):
    monkeypatch.setattr(settings, "fabric_federation_enabled", False, raising=False)
    conn, _ = await _make_db()
    try:
        r = await fg.gate_inbound(
            conn, sender_node_id=_SENDER_NODE, source_did="x@hostB",
            target_user_id="u1", verb=MSG_TEXT_SEND, body="hi",
        )
        assert r.allow is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_denylisted_instance_dropped(federation_on):
    conn, instance_did = await _make_db()
    try:
        await add_denylist(conn, did_key=instance_did, reason="abuse")
        r = await fg.gate_inbound(
            conn, sender_node_id=_SENDER_NODE, source_did="x@hostB",
            target_user_id="u1", verb=MSG_TEXT_SEND, body="hi",
        )
        assert r.allow is False
        assert r.reason == "instance_denylisted"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_known_contact_flows_through(federation_on):
    conn, _ = await _make_db()
    try:
        await _add_contact(conn, "u1", "alice@hostB")
        r = await fg.gate_inbound(
            conn, sender_node_id=_SENDER_NODE, source_did="alice@hostB",
            target_user_id="u1", verb=MSG_TEXT_SEND, body="hi",
        )
        assert r.allow is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_unknown_stranger_knock_posture_queues(federation_on):
    conn, _ = await _make_db()
    try:
        r = await fg.gate_inbound(
            conn, sender_node_id=_SENDER_NODE, source_did="stranger@hostB",
            target_user_id="u1", verb=MSG_TEXT_SEND, body="let me in",
        )
        assert r.allow is False and r.knocked is True
        # A knock was queued for u1 (intro withheld).
        pending = await list_pending(conn, to_user_id="u1")
        assert len(pending) == 1
        assert pending[0].from_handle == "stranger@hostB"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_private_posture_drops_stranger(monkeypatch):
    monkeypatch.setattr(settings, "fabric_federation_enabled", True, raising=False)
    monkeypatch.setattr(settings, "fabric_admission_posture", "private", raising=False)
    conn, _ = await _make_db()
    try:
        r = await fg.gate_inbound(
            conn, sender_node_id=_SENDER_NODE, source_did="stranger@hostB",
            target_user_id="u1", verb=MSG_TEXT_SEND, body="hi",
        )
        assert r.allow is False and r.knocked is False
        assert r.reason == "posture_private"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_open_posture_allows_stranger(monkeypatch):
    monkeypatch.setattr(settings, "fabric_federation_enabled", True, raising=False)
    monkeypatch.setattr(settings, "fabric_admission_posture", "open", raising=False)
    conn, _ = await _make_db()
    try:
        r = await fg.gate_inbound(
            conn, sender_node_id=_SENDER_NODE, source_did="stranger@hostB",
            target_user_id="u1", verb=MSG_TEXT_SEND, body="hi",
        )
        assert r.allow is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_non_relationship_verb_not_gated(federation_on):
    # A read receipt from an unknown sender isn't stranger-gated (it
    # references an existing thread or no-ops).
    conn, _ = await _make_db()
    try:
        r = await fg.gate_inbound(
            conn, sender_node_id=_SENDER_NODE, source_did="stranger@hostB",
            target_user_id="u1", verb=MSG_TEXT_READ,
        )
        assert r.allow is True
    finally:
        await conn.close()
