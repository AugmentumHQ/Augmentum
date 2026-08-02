"""P2: authoritative caller-ID + knock admission + author bindings.

  - caller-ID is the envelope-verified signer; a forged body source_did
    is rejected, a different-string-same-key claim is accepted.
  - knocks are posture-gated, deny-by-default, rate-limited on source
    key + IP, intro withheld until accept, accept TOFU-pins.
  - master→subkey author bindings verify only against the pinned master;
    tamper/forge → broken badge.
"""
from __future__ import annotations

import aiosqlite
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from augmentum.fabric.author_binding import (
    BADGE_BROKEN,
    BADGE_MASTER_BOUND,
    BADGE_UNBOUND,
    AuthorBindingError,
    author_badge,
    mint_binding,
    verify_binding,
)
from augmentum.fabric.caller_id import (
    CallerIdForgeryError,
    assert_caller,
    authoritative_source_did,
)
from augmentum.fabric.didkey import encode_ed25519_did
from augmentum.fabric.knock import (
    KnockRefused,
    accept_knock,
    list_pending,
    reject_knock,
    submit_knock,
)


def _identity():
    priv = Ed25519PrivateKey.generate()
    raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw, encode_ed25519_did(raw), priv.sign


# ── caller-ID ────────────────────────────────────────────────────────


def test_caller_id_from_verified_key():
    raw, did, _ = _identity()
    assert authoritative_source_did(raw) == did


def test_caller_id_accepts_matching_claim():
    raw, did, _ = _identity()
    assert assert_caller(raw, did) == did
    # absent claim is fine — stamped authoritatively
    assert assert_caller(raw, None) == did


def test_caller_id_rejects_forged_claim():
    raw, _, _ = _identity()
    _, other_did, _ = _identity()
    with pytest.raises(CallerIdForgeryError):
        assert_caller(raw, other_did)


# ── knock admission ──────────────────────────────────────────────────


async def _make_db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    for mig in (
        "289_fabric_peer_identities.sql",
        "290_fabric_knocks.sql",
    ):
        with open(f"augmentum/state/migrations/{mig}") as f:
            await conn.executescript(f.read())
    await conn.commit()
    return conn


@pytest.mark.asyncio
async def test_knock_private_posture_refused():
    conn = await _make_db()
    try:
        _, did, _ = _identity()
        with pytest.raises(KnockRefused) as ei:
            await submit_knock(
                conn, to_user_id="u1", from_did_key=did, posture="private",
            )
        assert ei.value.reason == "posture_private"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_knock_allowlist_blocks_unknown():
    conn = await _make_db()
    try:
        _, did, _ = _identity()
        with pytest.raises(KnockRefused) as ei:
            await submit_knock(
                conn, to_user_id="u1", from_did_key=did,
                posture="allowlist", allowlisted=False,
            )
        assert ei.value.reason == "not_allowlisted"
        # allowlisted passes
        k = await submit_knock(
            conn, to_user_id="u1", from_did_key=did,
            posture="allowlist", allowlisted=True,
        )
        assert k.status == "pending"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_knock_intro_withheld_until_accept():
    conn = await _make_db()
    try:
        _, did, _ = _identity()
        await submit_knock(
            conn, to_user_id="u1", from_did_key=did, posture="knock",
            from_handle="stranger@host", intro_text="secret intro",
        )
        pending = await list_pending(conn, to_user_id="u1")
        assert len(pending) == 1
        # The Knock dataclass surfaced to the list carries NO intro field.
        assert not hasattr(pending[0], "intro_text")

        revealed = await accept_knock(
            conn, to_user_id="u1", knock_id=pending[0].id,
        )
        assert revealed["intro_text"] == "secret intro"  # revealed only now
        # Accept TOFU-pinned the source (unverified).
        assert revealed["pinned"].verified is False
        assert revealed["pinned"].source == "knock"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_knock_source_rate_limit():
    conn = await _make_db()
    try:
        _, did, _ = _identity()
        await submit_knock(conn, to_user_id="u1", from_did_key=did, posture="knock")
        # Second pending knock from the same source is refused.
        with pytest.raises(KnockRefused) as ei:
            await submit_knock(conn, to_user_id="u2", from_did_key=did, posture="knock")
        assert ei.value.reason == "source_rate_limited"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_knock_ip_rate_limit():
    conn = await _make_db()
    try:
        # 5 distinct sources from one IP is the ceiling; the 6th refuses.
        for i in range(5):
            _, did, _ = _identity()
            await submit_knock(
                conn, to_user_id=f"u{i}", from_did_key=did,
                posture="knock", src_ip="203.0.113.7",
            )
        _, did6, _ = _identity()
        with pytest.raises(KnockRefused) as ei:
            await submit_knock(
                conn, to_user_id="u9", from_did_key=did6,
                posture="knock", src_ip="203.0.113.7",
            )
        assert ei.value.reason == "ip_rate_limited"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_knock_reject():
    conn = await _make_db()
    try:
        _, did, _ = _identity()
        k = await submit_knock(conn, to_user_id="u1", from_did_key=did, posture="knock")
        assert await reject_knock(conn, to_user_id="u1", knock_id=k.id) is True
        assert await list_pending(conn, to_user_id="u1") == []
        # Rejecting frees the source rate-limit slot.
        again = await submit_knock(conn, to_user_id="u1", from_did_key=did, posture="knock")
        assert again.status == "pending"
    finally:
        await conn.close()


# ── author bindings ──────────────────────────────────────────────────


def test_binding_round_trip():
    _, master_did, master_sign = _identity()
    _, subkey_did, _ = _identity()
    binding = mint_binding(
        master_sign=master_sign, master_did=master_did,
        subkey_did=subkey_did, issued_at=1718000000,
    )
    assert verify_binding(binding, expected_master_did=master_did) == subkey_did
    assert author_badge(binding, expected_master_did=master_did) == BADGE_MASTER_BOUND


def test_binding_wrong_master_rejected():
    _, master_did, master_sign = _identity()
    _, subkey_did, _ = _identity()
    _, attacker_master, _ = _identity()
    binding = mint_binding(
        master_sign=master_sign, master_did=master_did,
        subkey_did=subkey_did, issued_at=1,
    )
    # A verifier who pinned a DIFFERENT master must not accept it.
    with pytest.raises(AuthorBindingError):
        verify_binding(binding, expected_master_did=attacker_master)
    assert author_badge(binding, expected_master_did=attacker_master) == BADGE_BROKEN


def test_binding_tamper_breaks():
    _, master_did, master_sign = _identity()
    _, subkey_did, _ = _identity()
    _, other_sub, _ = _identity()
    binding = mint_binding(
        master_sign=master_sign, master_did=master_did,
        subkey_did=subkey_did, issued_at=1,
    )
    binding["subkey_did"] = other_sub  # swap the vouched-for subkey
    with pytest.raises(AuthorBindingError):
        verify_binding(binding, expected_master_did=master_did)
    assert author_badge(binding, expected_master_did=master_did) == BADGE_BROKEN


def test_no_binding_is_unbound():
    _, master_did, _ = _identity()
    assert author_badge(None, expected_master_did=master_did) == BADGE_UNBOUND
