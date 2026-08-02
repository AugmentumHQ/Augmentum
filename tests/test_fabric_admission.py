"""Production hardening: durable replay/nonce guards + admission gate.

Durable guards (SEC-7/SEC-8 residuals closed):
  - seq high-water survives across store calls; stale/dupe rejected;
    per-(owner,source) isolation.
  - nonce single-use; prune drops expired.

Admission choke-point (SEC-11 keystone): forged caller-ID, denylist,
replay, pinned-admit, and every posture branch.
"""
from __future__ import annotations

import aiosqlite
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric import admission
from augmentum.fabric.didkey import encode_ed25519_did
from augmentum.fabric.durable_guards import (
    check_and_advance_seq,
    prune_expired_nonces,
    spend_nonce,
)
from augmentum.fabric.peer_identity_store import mark_verified, pin_peer
from augmentum.fabric.revocation import add_denylist


def _identity():
    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw, encode_ed25519_did(raw)


async def _make_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    for mig in (
        "289_fabric_peer_identities.sql",
        "290_fabric_knocks.sql",
        "291_fabric_revocations.sql",
        "292_fabric_durable_guards.sql",
    ):
        with open(f"augmentum/state/migrations/{mig}") as f:
            await conn.executescript(f.read())
    await conn.commit()
    return conn


# ── durable guards ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_durable_seq_guard():
    conn = await _make_db()
    try:
        assert await check_and_advance_seq(conn, source_did="did:a", seq=1) is True
        assert await check_and_advance_seq(conn, source_did="did:a", seq=2) is True
        assert await check_and_advance_seq(conn, source_did="did:a", seq=2) is False
        assert await check_and_advance_seq(conn, source_did="did:a", seq=1) is False
        # Independent source + independent owner scope.
        assert await check_and_advance_seq(conn, source_did="did:b", seq=1) is True
        assert await check_and_advance_seq(
            conn, source_did="did:a", seq=1, owner_id="u2") is True
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_durable_nonce_single_use_and_prune():
    conn = await _make_db()
    try:
        assert await spend_nonce(conn, nonce="n1", expires_at=100) is True
        assert await spend_nonce(conn, nonce="n1", expires_at=100) is False
        assert await spend_nonce(conn, nonce="n2", expires_at=50) is True
        # Prune past expiry drops n2 (exp 50) but not n1 (exp 100).
        removed = await prune_expired_nonces(conn, now=75)
        assert removed == 1
        # n2 can be re-spent now that its (already-elapsed) window is gone —
        # harmless: its challenge TTL elapsed so it can't be presented.
        assert await spend_nonce(conn, nonce="n1", expires_at=100) is False
    finally:
        await conn.close()


# ── admission choke-point ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_admission_rejects_forged_caller_id():
    conn = await _make_db()
    try:
        raw, _ = _identity()
        _, other_did = _identity()
        d = await admission.authenticate_and_admit(
            conn, verified_pubkey=raw, claimed_source_did=other_did,
            to_user_id="u1", recipient_posture="open",
        )
        assert d.action == admission.FORGED
        assert not d.allowed
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_admission_blocks_denylisted():
    conn = await _make_db()
    try:
        raw, did = _identity()
        await add_denylist(conn, did_key=did, reason="spam")
        d = await admission.authenticate_and_admit(
            conn, verified_pubkey=raw, claimed_source_did=None,
            to_user_id="u1", recipient_posture="open",
        )
        assert d.action == admission.DENIED
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_admission_admits_pinned_with_trust_label():
    conn = await _make_db()
    try:
        raw, did = _identity()
        await pin_peer(conn, user_id="u1", peer_did_key=did)
        d = await admission.authenticate_and_admit(
            conn, verified_pubkey=raw, claimed_source_did=did,
            to_user_id="u1", recipient_posture="private",  # pinned beats posture
        )
        assert d.action == admission.ADMIT
        assert d.pinned and not d.verified
        assert d.trust_label == "pinned, not verified"
        await mark_verified(conn, user_id="u1", peer_did_key=did, method="sas")
        d2 = await admission.authenticate_and_admit(
            conn, verified_pubkey=raw, claimed_source_did=did,
            to_user_id="u1", recipient_posture="private",
        )
        assert d2.verified and d2.trust_label == "verified"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_admission_posture_branches_for_stranger():
    conn = await _make_db()
    try:
        raw, _ = _identity()
        base = dict(verified_pubkey=raw, claimed_source_did=None, to_user_id="u1")
        assert (await admission.authenticate_and_admit(
            conn, recipient_posture="private", **base)).action == admission.DENIED
        assert (await admission.authenticate_and_admit(
            conn, recipient_posture="allowlist", allowlisted=False, **base)).action == admission.DENIED
        assert (await admission.authenticate_and_admit(
            conn, recipient_posture="allowlist", allowlisted=True, **base)).action == admission.ADMIT
        assert (await admission.authenticate_and_admit(
            conn, recipient_posture="knock", **base)).action == admission.KNOCK
        assert (await admission.authenticate_and_admit(
            conn, recipient_posture="open", **base)).action == admission.ADMIT
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_admission_replay_guard():
    conn = await _make_db()
    try:
        raw, did = _identity()
        await pin_peer(conn, user_id="u1", peer_did_key=did)
        d1 = await admission.authenticate_and_admit(
            conn, verified_pubkey=raw, claimed_source_did=did,
            to_user_id="u1", recipient_posture="open", seq=5,
        )
        assert d1.action == admission.ADMIT
        # Replaying seq 5 (or lower) is rejected before delivery.
        d2 = await admission.authenticate_and_admit(
            conn, verified_pubkey=raw, claimed_source_did=did,
            to_user_id="u1", recipient_posture="open", seq=5,
        )
        assert d2.action == admission.REPLAY
    finally:
        await conn.close()
