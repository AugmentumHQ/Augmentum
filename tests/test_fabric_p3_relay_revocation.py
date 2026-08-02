"""P3: sealed relay envelopes + revocation/denylist.

Relay sealing (the most red-teamed piece):
  - sign-then-seal round-trips; recipient recovers an AUTHENTICATED
    source_did (RSC-1).
  - the relay (anyone without the recipient's X25519 priv) learns
    nothing and can't open it.
  - tampering ciphertext / swapping the inner signature fails.
  - separate X25519 sealing key, pyca-only (RSC-2/RSC-3) — no PyNaCl.
  - replay window rejects stale/duplicate seq.

Revocation/denylist:
  - a self-signed tombstone verifies; a third-party forgery doesn't.
  - recording requires a valid tombstone; is_revoked/is_denied work;
    subscription unsubscribe is clean.
"""
from __future__ import annotations

import base64

import aiosqlite
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric.didkey import encode_ed25519_did
from augmentum.fabric.relay_seal import (
    ReplayWindow,
    SealError,
    generate_sealing_key,
    seal,
    sealing_pub_b64,
    unseal,
)
from augmentum.fabric.revocation import (
    RevocationError,
    add_denylist,
    is_denied,
    is_revoked,
    mint_revocation,
    record_revocation,
    successor_of,
    unsubscribe_source,
    verify_revocation,
)


def _ed_identity():
    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return encode_ed25519_did(raw), priv.sign


# ── relay sealing ────────────────────────────────────────────────────


def test_seal_unseal_round_trip_authenticated():
    source_did, sign = _ed_identity()
    recipient = generate_sealing_key()
    sealed = seal(
        payload={"text": "hello over the relay"},
        recipient_sealing_pub_b64=sealing_pub_b64(recipient),
        origin_sign=sign, source_did=source_did, seq=1, ts=1718000000,
    )
    opened = unseal(sealed, recipient_sealing_priv=recipient)
    assert opened["source_did"] == source_did   # AUTHENTICATED, not asserted
    assert opened["payload"] == {"text": "hello over the relay"}
    assert opened["seq"] == 1


def test_relay_cannot_read_or_open():
    source_did, sign = _ed_identity()
    recipient = generate_sealing_key()
    sealed = seal(
        payload={"secret": "metadata"},
        recipient_sealing_pub_b64=sealing_pub_b64(recipient),
        origin_sign=sign, source_did=source_did, seq=1, ts=1,
    )
    # The relay sees only opaque fields — no plaintext substring.
    blob = repr(sealed)
    assert "secret" not in blob and "metadata" not in blob
    # A different key (the relay's own) cannot open it.
    attacker = generate_sealing_key()
    with pytest.raises(SealError):
        unseal(sealed, recipient_sealing_priv=attacker)


def test_tampered_ciphertext_fails():
    source_did, sign = _ed_identity()
    recipient = generate_sealing_key()
    sealed = seal(
        payload={"x": 1}, recipient_sealing_pub_b64=sealing_pub_b64(recipient),
        origin_sign=sign, source_did=source_did, seq=1, ts=1,
    )
    ct = bytearray(base64.b64decode(sealed["ct"]))
    ct[0] ^= 0xFF
    sealed["ct"] = base64.b64encode(bytes(ct)).decode("ascii")
    with pytest.raises(SealError):
        unseal(sealed, recipient_sealing_priv=recipient)


def test_forged_inner_signature_rejected():
    # Attacker seals to the recipient (anyone can encrypt) but claims a
    # source_did they don't hold the Ed25519 key for → inner sig fails.
    victim_did, _ = _ed_identity()
    _, attacker_sign = _ed_identity()
    recipient = generate_sealing_key()
    sealed = seal(
        payload={"x": 1}, recipient_sealing_pub_b64=sealing_pub_b64(recipient),
        origin_sign=attacker_sign, source_did=victim_did, seq=1, ts=1,
    )
    with pytest.raises(SealError):
        unseal(sealed, recipient_sealing_priv=recipient)


def test_surreptitious_reseal_rejected():
    # A legitimate recipient (Bob) holds an origin-signed message. He
    # re-seals the SAME origin-signed inner (recipient_seal = Bob) to
    # Carol's key. Carol can DECRYPT the copy, but the signed
    # recipient_seal names Bob, not her → rejected. Origin authenticity
    # does not let Bob forge the origin's intended audience.
    import os

    from cryptography.hazmat.primitives import serialization as ser
    from cryptography.hazmat.primitives.asymmetric import x25519
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    from augmentum.fabric import relay_seal as rs
    from augmentum.fabric.canonical import canonical_bytes

    source_did, sign = _ed_identity()
    bob = generate_sealing_key()
    carol = generate_sealing_key()

    # The inner that the origin signed FOR BOB.
    inner_for_bob = {
        "ctx": rs._SEAL_CTX, "source_did": source_did,
        "recipient_seal": sealing_pub_b64(bob), "seq": 1, "ts": 1,
        "payload": {"m": 1},
    }
    origin_sig = base64.b64encode(sign(canonical_bytes(inner_for_bob))).decode()
    inner_bytes = canonical_bytes({**inner_for_bob, "origin_sig": origin_sig})

    # Bob re-seals those exact bytes to Carol's X25519 key.
    eph = x25519.X25519PrivateKey.generate()
    eph_pub = eph.public_key().public_bytes(
        encoding=ser.Encoding.Raw, format=ser.PublicFormat.Raw)
    carol_pub = x25519.X25519PublicKey.from_public_bytes(
        base64.b64decode(sealing_pub_b64(carol)))
    key = rs._derive_key(eph.exchange(carol_pub), eph_pub)
    aad = canonical_bytes({"v": 1, "eph": base64.b64encode(eph_pub).decode()})
    nonce = os.urandom(12)
    ct = ChaCha20Poly1305(key).encrypt(nonce, inner_bytes, aad)
    forwarded = {
        "v": 1, "eph_pub": base64.b64encode(eph_pub).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ct": base64.b64encode(ct).decode(),
    }
    with pytest.raises(SealError):
        unseal(forwarded, recipient_sealing_priv=carol)


def test_replay_window():
    w = ReplayWindow()
    assert w.check_and_advance("did:a", 1) is True
    assert w.check_and_advance("did:a", 2) is True
    assert w.check_and_advance("did:a", 2) is False   # duplicate
    assert w.check_and_advance("did:a", 1) is False   # stale
    assert w.check_and_advance("did:b", 1) is True     # independent source


# ── revocation / denylist ────────────────────────────────────────────


def test_self_signed_revocation_verifies():
    did, sign = _ed_identity()
    successor, _ = _ed_identity()
    tomb = mint_revocation(
        sign=sign, revoked_did_key=did, reason="compromised",
        issued_at=1718000000, supersedes_to=successor,
    )
    assert verify_revocation(tomb) == did


def test_third_party_revocation_rejected():
    victim_did, _ = _ed_identity()
    _, attacker_sign = _ed_identity()
    # Attacker tries to revoke the victim's key by signing with their own.
    tomb = mint_revocation(
        sign=attacker_sign, revoked_did_key=victim_did, reason="evil",
        issued_at=1,
    )
    with pytest.raises(RevocationError):
        verify_revocation(tomb)


async def _make_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    with open("augmentum/state/migrations/291_fabric_revocations.sql") as f:
        await conn.executescript(f.read())
    await conn.commit()
    return conn


@pytest.mark.asyncio
async def test_record_and_query_revocation():
    conn = await _make_db()
    try:
        did, sign = _ed_identity()
        successor, _ = _ed_identity()
        tomb = mint_revocation(
            sign=sign, revoked_did_key=did, reason="rotate",
            issued_at=1, supersedes_to=successor,
        )
        await record_revocation(conn, tomb)
        assert await is_revoked(conn, did) is True
        assert await is_denied(conn, did) is True   # revoked ⇒ denied
        assert await successor_of(conn, did) == successor
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_record_rejects_forged_tombstone():
    conn = await _make_db()
    try:
        victim_did, _ = _ed_identity()
        _, attacker_sign = _ed_identity()
        tomb = mint_revocation(
            sign=attacker_sign, revoked_did_key=victim_did, reason="x", issued_at=1,
        )
        with pytest.raises(RevocationError):
            await record_revocation(conn, tomb)
        # Nothing was stored.
        assert await is_revoked(conn, victim_did) is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_denylist_subscription_unsubscribe():
    conn = await _make_db()
    try:
        spammer, _ = _ed_identity()
        publisher, _ = _ed_identity()
        await add_denylist(conn, did_key=spammer, reason="spam", source=publisher)
        assert await is_denied(conn, spammer) is True
        # Clean unsubscribe from that publisher's list removes it.
        removed = await unsubscribe_source(conn, source=publisher)
        assert removed == 1
        assert await is_denied(conn, spammer) is False
    finally:
        await conn.close()
