"""Contact-card mint/parse + ceremony + verified-state pin (P1).

Pins the federation trust root:
  - a minted card parses + verifies against its own instance did:key.
  - tampering any field (incl. swapping the key) breaks verification.
  - the SAS / safety-number is order-independent and binds the author
    key (AK-1): changing the author key changes the number.
  - TOFU pin starts unverified; ceremony upgrades it; key-change for a
    known handle is detected.
"""
from __future__ import annotations

import base64

import aiosqlite
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric import ceremony
from augmentum.fabric.contact_card import (
    ContactCardError,
    mint_card,
    parse_card,
)
from augmentum.fabric.didkey import encode_ed25519_did
from augmentum.fabric.peer_identity_store import (
    detect_key_change,
    mark_verified,
    pin_peer,
)


def _make_identity() -> tuple[str, object]:
    """Return (did_key, sign_callable) for a fresh Ed25519 key."""
    priv = Ed25519PrivateKey.generate()
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return encode_ed25519_did(pub_raw), priv.sign


def _make_card() -> tuple[dict, str]:
    did, sign = _make_identity()
    author_did, _ = _make_identity()
    card = mint_card(
        sign=sign,
        instance_did_key=did,
        endpoint="https://alice.example:6443",
        author_did_key=author_did,
        handle="alice@alice.example",
        token="tok-abc",
        issued_at=1718000000,
    )
    return card, did


def test_mint_and_parse_round_trip():
    card, did = _make_card()
    parsed = parse_card(card)
    assert parsed.instance_did_key == did
    assert parsed.handle == "alice@alice.example"
    assert parsed.token == "tok-abc"


def test_tampered_endpoint_fails_verification():
    card, _ = _make_card()
    card["endpoint"] = "https://attacker.example"
    with pytest.raises(ContactCardError):
        parse_card(card)


def test_swapped_key_fails_verification():
    # Attacker substitutes a key they control but can't re-sign as the
    # original issuer → verification against the new key fails because the
    # signature was over the old payload.
    card, _ = _make_card()
    other_did, _ = _make_identity()
    card["instance_did_key"] = other_did
    with pytest.raises(ContactCardError):
        parse_card(card)


def test_missing_signature_rejected():
    card, _ = _make_card()
    del card["sig"]
    with pytest.raises(ContactCardError):
        parse_card(card)


def test_bad_signature_rejected():
    card, _ = _make_card()
    card["sig"] = base64.b64encode(b"\x00" * 64).decode("ascii")
    with pytest.raises(ContactCardError):
        parse_card(card)


# ── ceremony ─────────────────────────────────────────────────────────


def test_sas_is_order_independent():
    a, _ = _make_identity()
    b, _ = _make_identity()
    # Alice computes (self=a, peer=b); Bob computes (self=b, peer=a).
    alice = ceremony.sas_words(a, b)
    bob = ceremony.sas_words(b, a)
    assert alice == bob
    assert len(alice) == 4


def test_safety_number_order_independent():
    a, _ = _make_identity()
    b, _ = _make_identity()
    assert ceremony.safety_number(a, b) == ceremony.safety_number(b, a)


def test_author_key_is_bound_into_sas():
    # AK-1: swapping the author key must change the SAS, or a malicious
    # host could substitute it invisibly after verification.
    a, _ = _make_identity()
    b, _ = _make_identity()
    author1, _ = _make_identity()
    author2, _ = _make_identity()
    sas1 = ceremony.sas_words(a, b, self_author=author1, peer_author="")
    sas2 = ceremony.sas_words(a, b, self_author=author2, peer_author="")
    assert sas1 != sas2


def test_different_peers_differ():
    a, _ = _make_identity()
    b, _ = _make_identity()
    c, _ = _make_identity()
    assert ceremony.sas_words(a, b) != ceremony.sas_words(a, c)


def test_verify_match_accepts_words_and_number():
    a, _ = _make_identity()
    b, _ = _make_identity()
    spoken = " ".join(ceremony.sas_words(b, a))
    number = ceremony.safety_number(b, a)
    assert ceremony.verify_match(a, b, spoken)
    assert ceremony.verify_match(a, b, number)
    assert not ceremony.verify_match(a, b, "wrong words here now")


# ── pin / verified-state store ───────────────────────────────────────


async def _make_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    with open("augmentum/state/migrations/289_fabric_peer_identities.sql") as f:
        sql = f.read()
    await conn.executescript(sql)
    await conn.commit()
    return conn


@pytest.mark.asyncio
async def test_pin_starts_unverified_then_verifies():
    conn = await _make_db()
    try:
        did, _ = _make_identity()
        pinned = await pin_peer(
            conn, user_id="u1", peer_did_key=did, handle="alice@host",
            source="card",
        )
        assert pinned.verified is False
        assert pinned.trust_label == "pinned, not verified"

        verified = await mark_verified(
            conn, user_id="u1", peer_did_key=did, method="sas",
        )
        assert verified.verified is True
        assert verified.verified_method == "sas"
        assert verified.trust_label == "verified"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_pin_is_per_user():
    conn = await _make_db()
    try:
        did, _ = _make_identity()
        await pin_peer(conn, user_id="u1", peer_did_key=did, handle="a@h")
        await mark_verified(conn, user_id="u1", peer_did_key=did, method="qr")
        # u2 pinning the same key starts fresh/unverified — no trust bleed.
        p2 = await pin_peer(conn, user_id="u2", peer_did_key=did, handle="a@h")
        assert p2.verified is False
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_repin_preserves_verified_state():
    conn = await _make_db()
    try:
        did, _ = _make_identity()
        await pin_peer(conn, user_id="u1", peer_did_key=did, endpoint="https://old")
        await mark_verified(conn, user_id="u1", peer_did_key=did, method="sas")
        # Endpoint moved → re-pin; verification must survive.
        again = await pin_peer(
            conn, user_id="u1", peer_did_key=did, endpoint="https://new",
        )
        assert again.verified is True
        assert again.endpoint == "https://new"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_repin_with_new_author_key_drops_verification():
    # AK-1: the author key is bound into the ceremony, so a re-pin that
    # carries a DIFFERENT author key must drop back to unverified — a host
    # can't swap the author key after verification and keep the badge.
    conn = await _make_db()
    try:
        did, _ = _make_identity()
        author1, _ = _make_identity()
        author2, _ = _make_identity()
        await pin_peer(conn, user_id="u1", peer_did_key=did, author_did_key=author1)
        await mark_verified(conn, user_id="u1", peer_did_key=did, method="sas")
        # Re-pin with a different author key → verification dropped.
        again = await pin_peer(
            conn, user_id="u1", peer_did_key=did, author_did_key=author2,
        )
        assert again.verified is False
        assert again.author_did_key == author2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_detect_key_change_for_known_handle():
    conn = await _make_db()
    try:
        old_did, _ = _make_identity()
        new_did, _ = _make_identity()
        await pin_peer(conn, user_id="u1", peer_did_key=old_did, handle="alice@host")
        changed = await detect_key_change(
            conn, user_id="u1", handle="alice@host", new_did_key=new_did,
        )
        assert old_did in changed
        # Same key → no change reported.
        same = await detect_key_change(
            conn, user_id="u1", handle="alice@host", new_did_key=old_did,
        )
        assert same == []
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_pin_rejects_anon_user():
    conn = await _make_db()
    try:
        did, _ = _make_identity()
        with pytest.raises(ValueError):
            await pin_peer(conn, user_id="", peer_did_key=did)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_verify_unpinned_raises():
    conn = await _make_db()
    try:
        did, _ = _make_identity()
        with pytest.raises(ValueError):
            await mark_verified(conn, user_id="u1", peer_did_key=did, method="sas")
    finally:
        await conn.close()
