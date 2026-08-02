"""Fail-closed identity contract (Connect federated-PBX P0).

The pre-2026-06-23 loader silently regenerated the Ed25519 key when the
stored one was unreadable, preserving the node_id — a silent identity
rotation that broke every peer pinning the old key (v2 red-team #5 /
RC-4). These tests pin the corrected contract:

  - true first boot (empty store) still mints cleanly.
  - a clean stored identity still loads unchanged (no false positives).
  - a corrupt key raises ``FabricIdentityCorruptError`` AND leaves the
    store byte-for-byte unchanged (no rotation side effect).
  - a torn / half-present state (only one of node_id / key) raises.
  - the BIP39 backup of a live identity round-trips back to the same key
    (the operator's recovery path actually works), and did_key is a
    well-formed Ed25519 did:key.
"""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.fabric.didkey import is_ed25519_did
from augmentum.fabric.identity import (
    FabricIdentity,
    FabricIdentityCorruptError,
)
from augmentum.fabric.recovery import mnemonic_to_key
from augmentum.state.settings_store import SettingsStore

_KEY_NODE_ID = "fabric.node_id"
_KEY_PRIVATE_KEY = "fabric.node_private_key"


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
async def test_true_first_boot_mints():
    conn, store = await _make_store()
    try:
        identity = await FabricIdentity.from_settings_store(store)
        assert len(identity.public_key_bytes) == 32
        # Persisted both halves.
        assert await store.get(_KEY_NODE_ID) == identity.node_id
        assert await store.get(_KEY_PRIVATE_KEY)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_clean_identity_loads_unchanged():
    conn, store = await _make_store()
    try:
        first = await FabricIdentity.from_settings_store(store)
        second = await FabricIdentity.from_settings_store(store)
        assert first.public_key_bytes == second.public_key_bytes
        assert first.node_id == second.node_id
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_corrupt_key_raises_with_no_side_effect():
    conn, store = await _make_store()
    try:
        first = await FabricIdentity.from_settings_store(store)
        before = await store.get(_KEY_PRIVATE_KEY)

        await store.set(_KEY_PRIVATE_KEY, "garbage-not-a-real-key")
        with pytest.raises(FabricIdentityCorruptError):
            await FabricIdentity.from_settings_store(store)

        # CRITICAL: the store was not mutated by the failed load.
        assert await store.get(_KEY_PRIVATE_KEY) == "garbage-not-a-real-key"
        assert await store.get(_KEY_NODE_ID) == first.node_id
        assert before != "garbage-not-a-real-key"  # sanity on the fixture
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_half_present_node_id_only_raises():
    conn, store = await _make_store()
    try:
        await store.set(_KEY_NODE_ID, "deadbeef" * 4)
        with pytest.raises(FabricIdentityCorruptError):
            await FabricIdentity.from_settings_store(store)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_half_present_key_only_raises():
    conn, store = await _make_store()
    try:
        await store.set(_KEY_PRIVATE_KEY, "garbage")
        with pytest.raises(FabricIdentityCorruptError):
            await FabricIdentity.from_settings_store(store)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_did_key_is_well_formed():
    conn, store = await _make_store()
    try:
        identity = await FabricIdentity.from_settings_store(store)
        assert is_ed25519_did(identity.did_key)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mnemonic_backup_recovers_same_key():
    conn, store = await _make_store()
    try:
        from cryptography.hazmat.primitives import serialization

        identity = await FabricIdentity.from_settings_store(store)
        phrase = identity.mnemonic_backup()
        assert len(phrase.split()) == 24

        recovered = mnemonic_to_key(phrase)
        live = identity.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        assert recovered == live
    finally:
        await conn.close()
