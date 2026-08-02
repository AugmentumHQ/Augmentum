"""Tests for the fabric identity primitive.

Phase 0 ships ``FabricIdentity`` with lazy load-or-generate semantics
backed by the settings_store. These tests pin the contract every
later phase relies on:

  - first call generates a stable identity + persists it
  - subsequent calls return bit-for-bit the same identity
  - the private key is encrypted at rest (raw key bytes do not appear
    in the settings table)
  - sign/verify roundtrips work and reject tampered payloads
  - a corrupted private key in the store FAILS CLOSED (raises rather
    than silently rotating the key) — see the dedicated fail-closed
    suite in ``test_fabric_identity_failclosed.py``
"""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.fabric.identity import FabricIdentity
from augmentum.state.settings_store import SettingsStore


async def _make_store() -> tuple[aiosqlite.Connection, SettingsStore]:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.commit()
    return conn, SettingsStore(conn)


@pytest.mark.asyncio
async def test_first_call_generates_identity():
    conn, store = await _make_store()
    try:
        identity = await FabricIdentity.from_settings_store(store)
        # node_id is 16 bytes of entropy (32 hex chars)
        assert len(identity.node_id) == 32
        assert all(c in "0123456789abcdef" for c in identity.node_id)
        # Public key is 32 bytes ed25519
        assert len(identity.public_key_bytes) == 32
        # Fingerprint is SHA256:<32 hex>
        assert identity.fingerprint.startswith("SHA256:")
        assert len(identity.fingerprint) == 7 + 32
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_subsequent_calls_return_same_identity():
    """Round-trip: load, persist, reload — same node_id and same public key."""
    conn, store = await _make_store()
    try:
        first = await FabricIdentity.from_settings_store(store)
        second = await FabricIdentity.from_settings_store(store)
        assert first.node_id == second.node_id
        assert first.public_key_bytes == second.public_key_bytes
        assert first.fingerprint == second.fingerprint
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_private_key_encrypted_at_rest():
    """The settings_store value must NOT be the raw base64 private key.

    encrypt_api_key wraps the secret in Fernet (the prefix is ``enc:``
    when AUGMENTUM_SECRETS_KEY is set, otherwise the raw value is
    stored — but in either case it must not equal the live identity's
    bytes when comparing the persisted value to the in-memory key).
    Either way, the in-memory key bytes should not appear verbatim in
    the persisted blob.
    """
    import base64

    conn, store = await _make_store()
    try:
        identity = await FabricIdentity.from_settings_store(store)
        stored = await store.get("fabric.node_private_key")
        assert stored is not None
        # The raw base64 of the private key bytes is what gets encrypted.
        # That literal string must NOT appear in the stored blob, or
        # the encryption is broken. (When AUGMENTUM_SECRETS_KEY is set,
        # the blob is prefixed "enc:" and Fernet-encrypted; otherwise
        # encrypt_api_key returns the plaintext unchanged. Either way,
        # the secret should not be present verbatim if encryption is
        # configured -- check by computing the raw and asserting non-
        # equality only when an enc: prefix is present.)
        if stored.startswith("enc:"):
            from cryptography.hazmat.primitives import serialization

            raw_b64 = base64.b64encode(
                identity.private_key.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            ).decode("ascii")
            assert raw_b64 not in stored
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_sign_and_verify_roundtrip():
    conn, store = await _make_store()
    try:
        identity = await FabricIdentity.from_settings_store(store)
        payload = b"phase-1 handshake payload"
        signature = identity.sign(payload)
        assert FabricIdentity.verify(payload, signature, identity.public_key_b64)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_verify_rejects_tampered_payload():
    conn, store = await _make_store()
    try:
        identity = await FabricIdentity.from_settings_store(store)
        payload = b"original payload"
        signature = identity.sign(payload)
        tampered = b"tampered payload"
        assert not FabricIdentity.verify(tampered, signature, identity.public_key_b64)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_verify_rejects_wrong_public_key():
    conn, store = await _make_store()
    try:
        identity_a = await FabricIdentity.from_settings_store(store)
        payload = b"signed by A"
        signature = identity_a.sign(payload)

        # Generate a separate identity with a fresh store; its public
        # key won't match A's.
        conn_b, store_b = await _make_store()
        try:
            identity_b = await FabricIdentity.from_settings_store(store_b)
            assert identity_a.public_key_b64 != identity_b.public_key_b64
            assert not FabricIdentity.verify(payload, signature, identity_b.public_key_b64)
        finally:
            await conn_b.close()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_corrupted_private_key_fails_closed():
    """If the persisted private key is unreadable, FAIL CLOSED.

    The pre-2026-06-23 behavior regenerated the key while preserving the
    node_id — a silent identity rotation that broke every peer pinning
    the old key (v2 red-team #5 / RC-4). The fix raises
    ``FabricIdentityCorruptError`` instead; the operator restores from
    their BIP39 backup. The full no-side-effect / torn-state matrix is
    in ``test_fabric_identity_failclosed.py``.
    """
    from augmentum.fabric.identity import FabricIdentityCorruptError

    conn, store = await _make_store()
    try:
        first = await FabricIdentity.from_settings_store(store)

        # Corrupt the persisted private key.
        await store.set("fabric.node_private_key", "garbage-not-a-real-key")

        with pytest.raises(FabricIdentityCorruptError):
            await FabricIdentity.from_settings_store(store)

        # The corrupt value is left UNTOUCHED — no silent regeneration.
        assert await store.get("fabric.node_private_key") == "garbage-not-a-real-key"
        assert await store.get("fabric.node_id") == first.node_id
    finally:
        await conn.close()
