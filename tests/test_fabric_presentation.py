"""The human-friendly presentation + connect-code UX layer.

  - trust states map to plain-language label/hint/tone (no jargon).
  - identity_code is stable, grouped, and readable.
  - friendly_error never leaks a class name.
  - connect codes use an unambiguous alphabet, round-trip, and are
    tolerant of how a human types them.
"""
from __future__ import annotations

import aiosqlite
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric.connect_codes import (
    create_code,
    format_code,
    normalize_code,
    resolve_code,
)
from augmentum.fabric.didkey import encode_ed25519_did
from augmentum.fabric.presentation import (
    connection_blurb,
    connection_presentation,
    friendly_error,
    identity_code,
)


def _did():
    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return encode_ed25519_did(raw)


# ── presentation ─────────────────────────────────────────────────────


def test_trust_states_are_plain_language():
    verified = connection_presentation({"verified": True})
    assert verified["state"] == "verified"
    assert verified["label"] == "Verified"
    assert verified["tone"] == "good"

    unverified = connection_presentation({"verified": False})
    assert unverified["state"] == "unverified"
    assert "Verify" in unverified["hint"]
    assert unverified["tone"] == "warn"

    changed = connection_presentation({"verified": True, "key_changed": True})
    assert changed["state"] == "changed"
    assert changed["tone"] == "alert"
    # No technical terms leak into user copy.
    for v in (verified, unverified, changed):
        blob = (v["label"] + v["hint"]).lower()
        assert "did:key" not in blob and "tofu" not in blob and "sas" not in blob


def test_identity_code_stable_and_grouped():
    did = _did()
    code = identity_code(did)
    assert code == identity_code(did)            # stable
    assert len(code) == 14 and code.count("-") == 2   # AAAA-BBBB-CCCC
    assert identity_code(_did()) != code         # distinct per key
    assert identity_code("not-a-did") == ""      # graceful on garbage


def test_friendly_error_never_leaks_internals():
    assert "block" in friendly_error("instance_denylisted").lower()
    assert friendly_error("ContactCardError")  # mapped
    # An unknown technical string falls back to calm copy, not the raw string.
    out = friendly_error("KeyError: 'x'")
    assert "KeyError" not in out and out


def test_connection_blurb():
    blurb = connection_blurb({"display_name": "Alice", "verified": True})
    assert blurb == "Alice · Verified"


# ── connect codes ────────────────────────────────────────────────────


def test_code_alphabet_is_unambiguous():
    # No 0/O/1/I/L/U anywhere in a generated code.
    raw = normalize_code("K7P29QX4")
    assert not (set(raw) & set("01ILOU"))


def test_format_and_normalize_round_trip():
    assert format_code("K7P29QX4") == "K7P2-9QX4"
    assert normalize_code("k7p2-9qx4") == "K7P29QX4"
    assert normalize_code(" K7P2 9QX4 ") == "K7P29QX4"


async def _db():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    with open("augmentum/state/migrations/293_fabric_connect_codes.sql") as f:
        await conn.executescript(f.read())
    await conn.commit()
    return conn


@pytest.mark.asyncio
async def test_connect_code_create_and_resolve():
    conn = await _db()
    try:
        card = {"card": {"instance_did_key": _did(), "v": 1}, "profile": {"display_name": "Alice"}}
        code = await create_code(conn, user_id="u1", card=card)
        # Same card reuses the same code (idempotent).
        assert await create_code(conn, user_id="u1", card=card) == code
        # Resolve via the human-typed grouped/lowercase form.
        got = await resolve_code(conn, code=format_code(code).lower())
        assert got["profile"]["display_name"] == "Alice"
        assert await resolve_code(conn, code="ZZZZ9999") is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_connect_code_expiry():
    conn = await _db()
    try:
        code = await create_code(
            conn, user_id="u1", card={"x": 1}, expires_at=100,
        )
        assert await resolve_code(conn, code=code, now=50) is not None
        assert await resolve_code(conn, code=code, now=200) is None  # expired
    finally:
        await conn.close()
