"""Device-bundle store + chain validation (Connect E2E P2).

The server's job is integrity: it must store a bundle only if every
device binding chains to the bundle's master and vouches for that
device's own subkey. A forged/mismatched bundle is rejected whole.
"""
from __future__ import annotations

import aiosqlite
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric.author_binding import mint_binding
from augmentum.fabric.device_bundle import (
    DeviceBundleError,
    get_bundle,
    put_bundle,
    validate_bundle,
)
from augmentum.fabric.didkey import encode_ed25519_did


def _key():
    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv, encode_ed25519_did(raw)


def _bundle(master_priv, master_did, *, sub_did=None):
    sub_priv, sub_did2 = _key()
    sub_did = sub_did or sub_did2
    binding = mint_binding(
        master_sign=master_priv.sign, master_did=master_did,
        subkey_did=sub_did, issued_at=1,
    )
    return [{
        "subkey_did": sub_did, "sealing_pub_b64": "AAAA",
        "binding": binding, "label": "phone",
    }]


async def _db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    with open("augmentum/state/migrations/294_fabric_device_bundles.sql") as f:
        await conn.executescript(f.read())
    await conn.commit()
    return conn


def test_validate_good_bundle():
    mp, md = _key()
    b = validate_bundle(md, _bundle(mp, md))
    assert b.master_did == md and len(b.devices) == 1


def test_validate_rejects_binding_for_wrong_master():
    mp, md = _key()
    attacker, attacker_did = _key()
    # Binding signed by a DIFFERENT master than the bundle claims.
    _, sub_did = _key()
    bad_binding = mint_binding(
        master_sign=attacker.sign, master_did=attacker_did,
        subkey_did=sub_did, issued_at=1,
    )
    devices = [{"subkey_did": sub_did, "sealing_pub_b64": "AAAA", "binding": bad_binding}]
    with pytest.raises(DeviceBundleError):
        validate_bundle(md, devices)


def test_validate_rejects_binding_for_other_subkey():
    mp, md = _key()
    _, real_sub = _key()
    _, other_sub = _key()
    # Binding vouches for other_sub, but the device claims real_sub.
    binding = mint_binding(
        master_sign=mp.sign, master_did=md, subkey_did=other_sub, issued_at=1,
    )
    devices = [{"subkey_did": real_sub, "sealing_pub_b64": "AAAA", "binding": binding}]
    with pytest.raises(DeviceBundleError):
        validate_bundle(md, devices)


def test_validate_rejects_empty():
    _, md = _key()
    with pytest.raises(DeviceBundleError):
        validate_bundle(md, [])


@pytest.mark.asyncio
async def test_put_and_get_round_trip():
    conn = await _db()
    try:
        mp, md = _key()
        await put_bundle(conn, user_id="u1", master_did=md, devices=_bundle(mp, md))
        got = await get_bundle(conn, user_id="u1")
        assert got is not None and got.master_did == md
        assert got.devices[0]["label"] == "phone"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_put_rejects_forged_and_anon():
    conn = await _db()
    try:
        mp, md = _key()
        attacker, attacker_did = _key()
        _, sub = _key()
        forged = [{"subkey_did": sub, "sealing_pub_b64": "AAAA",
                   "binding": mint_binding(master_sign=attacker.sign,
                                           master_did=attacker_did,
                                           subkey_did=sub, issued_at=1)}]
        with pytest.raises(DeviceBundleError):
            await put_bundle(conn, user_id="u1", master_did=md, devices=forged)
        with pytest.raises(DeviceBundleError):
            await put_bundle(conn, user_id="", master_did=md, devices=_bundle(mp, md))
    finally:
        await conn.close()
