"""P4: optional directory Number/descriptor + bounded PoW + receipts.

  - Number is self-certifying (re-derivable; Luhn-checked) and ≥80 bits.
  - a descriptor verifies only if its signature AND its Number↔key
    binding hold; tampering either fails (trustless directory).
  - PoW is target-bound, expiring, difficulty-capped, with a
    difficulty-0 accessibility waiver.
  - receipts ack only CONTIGUOUS sequence (a gap caps the ack).
"""
from __future__ import annotations

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric import pow as fabric_pow
from augmentum.fabric.didkey import encode_ed25519_did
from augmentum.fabric.directory import (
    DescriptorError,
    derive_number,
    format_number,
    mint_descriptor,
    verify_descriptor,
    verify_number,
)
from augmentum.fabric.receipts import (
    ReceiptError,
    contiguous_high_water,
    mint_receipt,
    verify_receipt,
)


def _identity():
    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return encode_ed25519_did(raw), priv.sign


# ── Number ───────────────────────────────────────────────────────────


def test_number_is_self_certifying():
    did, _ = _identity()
    number = derive_number(did)
    assert len(number) == 25            # 24 payload + Luhn
    assert verify_number(did, number)
    # grouping-tolerant
    assert verify_number(did, format_number(number))


def test_number_rejects_wrong_key():
    did_a, _ = _identity()
    did_b, _ = _identity()
    assert not verify_number(did_b, derive_number(did_a))


def test_number_width_is_at_least_80_bits():
    # 24 decimal digits ~ 79.7 bits — non-grindable, the v1 lesson.
    assert 10 ** 24 > 2 ** 79


# ── descriptor (trustless directory) ─────────────────────────────────


def test_descriptor_round_trip():
    did, sign = _identity()
    desc = mint_descriptor(
        sign=sign, did_key=did, endpoint="https://host:6443", issued_at=1,
    )
    out = verify_descriptor(desc)
    assert out["did_key"] == did
    assert out["endpoint"] == "https://host:6443"
    assert verify_number(did, out["number"])


def test_descriptor_tampered_endpoint_fails():
    did, sign = _identity()
    desc = mint_descriptor(sign=sign, did_key=did, endpoint="https://real", issued_at=1)
    desc["endpoint"] = "https://evil"   # malicious directory edit
    with pytest.raises(DescriptorError):
        verify_descriptor(desc)


def test_descriptor_swapped_number_fails():
    did, sign = _identity()
    other, _ = _identity()
    desc = mint_descriptor(sign=sign, did_key=did, endpoint="https://x", issued_at=1)
    desc["number"] = derive_number(other)   # wrong Number for this key
    with pytest.raises(DescriptorError):
        verify_descriptor(desc)


# ── PoW ──────────────────────────────────────────────────────────────


def test_pow_solve_and_verify():
    ch = fabric_pow.make_challenge(target="did:recipient", issued_at=1000, difficulty=10)
    sol = fabric_pow.solve(ch)
    assert fabric_pow.verify(ch, sol, now=1100)


def test_pow_difficulty_is_capped():
    ch = fabric_pow.make_challenge(target="t", issued_at=0, difficulty=999)
    assert ch.difficulty == fabric_pow.MAX_DIFFICULTY_BITS


def test_pow_expires():
    ch = fabric_pow.make_challenge(target="t", issued_at=0, difficulty=8)
    sol = fabric_pow.solve(ch)
    assert fabric_pow.verify(ch, sol, now=ch.ttl_s - 1)
    assert not fabric_pow.verify(ch, sol, now=ch.ttl_s + 1)   # too late


def test_pow_accessibility_waiver():
    ch = fabric_pow.make_challenge(target="t", issued_at=0, difficulty=0)
    # difficulty 0 = the can't-compute fallback: any solution verifies in TTL.
    assert fabric_pow.verify(ch, "anything", now=10)


def test_pow_wrong_solution_fails():
    ch = fabric_pow.make_challenge(target="t", issued_at=0, difficulty=12)
    assert not fabric_pow.verify(ch, "not-a-solution", now=10)


# ── SEC-7: signed (server-issued) challenges ─────────────────────────


def test_signed_challenge_round_trip():
    server_did, server_sign = _identity()
    ch = fabric_pow.make_challenge(target="did:recipient", issued_at=1000, difficulty=10)
    signed = fabric_pow.sign_challenge(ch, sign=server_sign, issuer_did=server_did)
    # Client solves the challenge it was handed.
    opened = fabric_pow.open_signed_challenge(
        signed, expected_issuer_did=server_did, now=1100)
    sol = fabric_pow.solve(opened)
    assert fabric_pow.verify_signed_solution(
        signed, sol, expected_issuer_did=server_did, now=1100)


def test_client_forged_challenge_rejected():
    # The SEC-7 attack: an attacker mints their OWN difficulty-0 challenge
    # and "solves" it. The server must refuse it because it wasn't signed
    # by the server's key.
    server_did, _ = _identity()
    forged = {
        "ctx": "augmentum-fabric-pow-challenge-v1",
        "issuer_did": server_did, "target": "victim", "nonce": "deadbeef",
        "difficulty": 0, "issued_at": 0, "ttl_s": 300,
        "sig": "AAAA",  # not a real signature from the server
    }
    with pytest.raises(fabric_pow.PowChallengeError):
        fabric_pow.open_signed_challenge(forged, expected_issuer_did=server_did, now=10)
    assert not _safe_verify(forged, "0", server_did, 10)


def test_signed_challenge_wrong_issuer_rejected():
    server_did, server_sign = _identity()
    other_did, _ = _identity()
    ch = fabric_pow.make_challenge(target="t", issued_at=0, difficulty=8)
    signed = fabric_pow.sign_challenge(ch, sign=server_sign, issuer_did=server_did)
    # A verifier expecting a different issuer must reject it.
    with pytest.raises(fabric_pow.PowChallengeError):
        fabric_pow.open_signed_challenge(signed, expected_issuer_did=other_did, now=10)


def test_signed_challenge_expires():
    server_did, server_sign = _identity()
    ch = fabric_pow.make_challenge(target="t", issued_at=0, difficulty=8)
    signed = fabric_pow.sign_challenge(ch, sign=server_sign, issuer_did=server_did)
    with pytest.raises(fabric_pow.PowChallengeError):
        fabric_pow.open_signed_challenge(
            signed, expected_issuer_did=server_did, now=ch.ttl_s + 5)


def test_signed_challenge_single_use():
    server_did, server_sign = _identity()
    ch = fabric_pow.make_challenge(target="t", issued_at=0, difficulty=8)
    signed = fabric_pow.sign_challenge(ch, sign=server_sign, issuer_did=server_did)
    sol = fabric_pow.solve(fabric_pow.open_signed_challenge(
        signed, expected_issuer_did=server_did, now=10))
    consumed = fabric_pow.ConsumedNonces()
    assert fabric_pow.verify_signed_solution(
        signed, sol, expected_issuer_did=server_did, now=10, consumed=consumed)
    # Replaying the same solved challenge is rejected.
    assert not fabric_pow.verify_signed_solution(
        signed, sol, expected_issuer_did=server_did, now=10, consumed=consumed)


def _safe_verify(signed, sol, issuer, now) -> bool:
    try:
        return fabric_pow.verify_signed_solution(
            signed, sol, expected_issuer_did=issuer, now=now)
    except fabric_pow.PowChallengeError:
        return False


# ── receipts (contiguous) ────────────────────────────────────────────


def test_contiguous_high_water():
    assert contiguous_high_water([1, 2, 3]) == 3
    assert contiguous_high_water([1, 2, 4]) == 2     # gap at 3 caps it
    assert contiguous_high_water([2, 3]) == 0        # missing 1
    assert contiguous_high_water([]) == 0


def test_receipt_round_trip():
    rdid, rsign = _identity()
    sdid, _ = _identity()
    rec = mint_receipt(
        sign=rsign, recipient_did=rdid, source_did=sdid, up_to_seq=5, ts=1,
    )
    assert verify_receipt(rec, expected_recipient_did=rdid) == 5


def test_receipt_wrong_recipient_rejected():
    rdid, rsign = _identity()
    sdid, _ = _identity()
    other, _ = _identity()
    rec = mint_receipt(sign=rsign, recipient_did=rdid, source_did=sdid, up_to_seq=5, ts=1)
    with pytest.raises(ReceiptError):
        verify_receipt(rec, expected_recipient_did=other)
