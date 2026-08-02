"""Tests for the Phase 4.y outbound pair client.

Covers the operator-driven outbound flow that complements the
existing inbound /pair receiver:

  - Successful round-trip persists the remote into fabric_nodes
  - The wire payload is the signed PairRequest the receiver expects
  - Fingerprint mismatch in remote response is rejected (defensive
    cross-check)
  - Remote 4xx is surfaced operator-facing
  - Connection error becomes OutboundPairError, not a leaked httpx
    exception

The client is exercised directly with a stub httpx client; the
route handler in fabric_routes.py is tested separately.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest

from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.pair_client import (
    OutboundPairError,
    initiate_pair_with_remote,
)
from augmentum.fabric.peer_auth import _fingerprint_from_b64
from augmentum.state.settings_store import SettingsStore


async def _make_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.execute(
        """CREATE TABLE fabric_nodes (
            id TEXT PRIMARY KEY, hostname TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'peer',
            pubkey_ed25519 TEXT NOT NULL, pubkey_fingerprint TEXT NOT NULL,
            addr TEXT NOT NULL DEFAULT '', tier TEXT NOT NULL DEFAULT 'local',
            fabric_share_enabled INTEGER NOT NULL DEFAULT 1,
            paired_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen_at TEXT, icon TEXT NOT NULL DEFAULT '')"""
    )
    await conn.commit()
    return conn


def _ok_response(remote_pubkey_b64: str) -> MagicMock:
    """A stub httpx response mimicking the real /api/fabric/pair
    success body (this_node + paired wrappers).
    """
    fp = _fingerprint_from_b64(remote_pubkey_b64)
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "ok": True,
        "this_node": {
            "node_id": "remote-node-xyz",
            "public_key": remote_pubkey_b64,
            "fingerprint": fp,
        },
        "paired": {
            "node_id": "local-node-abc",
            "role": "peer",
            "addr": "192.168.1.20:6443",
        },
    }
    return resp


@pytest.mark.asyncio
async def test_pair_success_persists_remote_into_db():
    """Happy path: remote returns a clean 200, we persist the row,
    refresh succeeds end-to-end.
    """
    local_db = await _make_db()
    remote_db = await _make_db()
    try:
        local_id = await FabricIdentity.from_settings_store(SettingsStore(local_db))
        remote_id = await FabricIdentity.from_settings_store(SettingsStore(remote_db))

        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_ok_response(remote_id.public_key_b64))

        paired = await initiate_pair_with_remote(
            identity=local_id, hostname="local-host",
            remote_url="https://remote.local",
            expected_fingerprint=remote_id.fingerprint,
            remote_addr="192.168.1.20:6443",
            db=local_db, http_client=fake_client,
        )

        # Returned dataclass matches what we POSTed about + persisted.
        assert paired.node_id == "remote-node-xyz"
        assert paired.fingerprint == remote_id.fingerprint
        assert paired.addr == "192.168.1.20:6443"

        # DB row was inserted.
        cursor = await local_db.execute(
            "SELECT id, pubkey_ed25519 FROM fabric_nodes WHERE id = ?",
            ("remote-node-xyz",),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[1] == remote_id.public_key_b64
    finally:
        await local_db.close()
        await remote_db.close()


@pytest.mark.asyncio
async def test_pair_wire_payload_is_signed_pair_request():
    """The body we POST to /api/fabric/pair must be the JSON form of a
    PairRequest with the operator-supplied fingerprint as
    fingerprint_hint, plus an optional addr field. Without this the
    receiver's verify_pair_request would never validate.
    """
    local_db = await _make_db()
    remote_db = await _make_db()
    try:
        local_id = await FabricIdentity.from_settings_store(SettingsStore(local_db))
        remote_id = await FabricIdentity.from_settings_store(SettingsStore(remote_db))

        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_ok_response(remote_id.public_key_b64))

        await initiate_pair_with_remote(
            identity=local_id, hostname="my-host",
            remote_url="https://remote.local",
            expected_fingerprint=remote_id.fingerprint,
            remote_addr="192.168.1.20:6443",
            db=local_db, http_client=fake_client,
        )

        call_args = fake_client.post.call_args
        endpoint = call_args[0][0]
        assert endpoint.endswith("/api/fabric/pair")

        body = call_args[1]["json"]
        # Required PairRequest fields are present.
        assert body["sender_node_id"] == local_id.node_id
        assert body["hostname"] == "my-host"
        assert body["pubkey_b64"] == local_id.public_key_b64
        # The hint is what the operator typed — must equal the REMOTE's
        # fingerprint so the receiver's same-node check passes.
        assert body["fingerprint_hint"] == remote_id.fingerprint
        assert body["role"] == "peer"
        assert isinstance(body["timestamp"], int)
        assert body["signature"]  # non-empty base64
        # Our addr (for the remote to store back) is included.
        assert body["addr"] == "192.168.1.20:6443"
    finally:
        await local_db.close()
        await remote_db.close()


@pytest.mark.asyncio
async def test_fingerprint_mismatch_in_response_rejects_pair():
    """If the remote returns 200 but its this_node.fingerprint doesn't
    match what the operator typed, refuse to persist. This is the
    defensive cross-check against a hostile remote that played fair
    on inbound /pair but tried to spoof its identity in the response.
    """
    local_db = await _make_db()
    remote_db = await _make_db()
    try:
        local_id = await FabricIdentity.from_settings_store(SettingsStore(local_db))
        remote_id = await FabricIdentity.from_settings_store(SettingsStore(remote_db))

        # Build a response with a DIFFERENT fingerprint than what the
        # operator pasted. The remote returned a valid 200 — the
        # mismatch is the only signal.
        wrong_resp = _ok_response(remote_id.public_key_b64)
        wrong_resp.json.return_value["this_node"]["fingerprint"] = "SHA256:ffffffffffffffff"
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=wrong_resp)

        with pytest.raises(OutboundPairError) as excinfo:
            await initiate_pair_with_remote(
                identity=local_id, hostname="local",
                remote_url="https://remote.local",
                expected_fingerprint=remote_id.fingerprint,
                remote_addr="192.168.1.20:6443",
                db=local_db, http_client=fake_client,
            )
        assert "fingerprint mismatch" in str(excinfo.value).lower()

        # And nothing was persisted.
        cursor = await local_db.execute("SELECT COUNT(*) FROM fabric_nodes")
        count = (await cursor.fetchone())[0]
        assert count == 0
    finally:
        await local_db.close()
        await remote_db.close()


@pytest.mark.asyncio
async def test_remote_4xx_surfaced_operator_facing():
    """Remote returns 400 with a JSON detail — surface it verbatim so
    the operator sees WHY (e.g. "fingerprint mismatch on remote side").
    """
    local_db = await _make_db()
    try:
        local_id = await FabricIdentity.from_settings_store(SettingsStore(local_db))

        bad_resp = MagicMock(spec=httpx.Response)
        bad_resp.status_code = 400
        bad_resp.json.return_value = {
            "detail": (
                "fingerprint mismatch: the requesting peer thinks they are "
                "pairing with a different node than this one."
            ),
        }
        bad_resp.text = json.dumps(bad_resp.json.return_value)
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=bad_resp)

        with pytest.raises(OutboundPairError) as excinfo:
            await initiate_pair_with_remote(
                identity=local_id, hostname="local",
                remote_url="https://remote.local",
                expected_fingerprint="SHA256:doesnotmatter",
                remote_addr="192.168.1.20:6443",
                db=local_db, http_client=fake_client,
            )
        msg = str(excinfo.value)
        assert "fingerprint mismatch" in msg.lower()
        assert "400" in msg
    finally:
        await local_db.close()


@pytest.mark.asyncio
async def test_remote_unreachable_becomes_outbound_pair_error():
    """Connection refused / timeout from httpx becomes OutboundPairError
    so the route layer can map it to a clean 502. Without this the
    raw httpx exception would propagate up and the UI would see a 500.
    """
    local_db = await _make_db()
    try:
        local_id = await FabricIdentity.from_settings_store(SettingsStore(local_db))

        fake_client = MagicMock()
        fake_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused"),
        )

        with pytest.raises(OutboundPairError) as excinfo:
            await initiate_pair_with_remote(
                identity=local_id, hostname="local",
                remote_url="https://192.168.1.99",
                expected_fingerprint="SHA256:abc",
                remote_addr="192.168.1.99:6443",
                db=local_db, http_client=fake_client,
            )
        assert "unreachable" in str(excinfo.value).lower()
    finally:
        await local_db.close()
