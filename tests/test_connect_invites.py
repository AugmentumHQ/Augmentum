"""Invite store + claim flow — Connect open-access onboarding (Phase 1).

Covers the persisted invite lifecycle (mint → preview → consume → revoke) and
the claim wiring the route performs: consume → create_user(email) → mark
claimed → mutual contacts.
"""

from __future__ import annotations

import pytest

from augmentum.auth.invite_store import (
    consume_invite,
    create_invite,
    invite_status,
    list_invites,
    mark_claimed,
    preview_invite,
    revoke_invite,
)
from augmentum.auth.session_manager import SessionManager
from augmentum.config import settings
from augmentum.connect.contact_store import get_contact
from augmentum.connect.contacts import local_did_for
from augmentum.state.backends.sqlite import SQLiteBackend


async def _env():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    sm = SessionManager(backend._conn)
    inviter = await sm.create_user("operator", "supersecret", role="admin")
    return backend, sm, backend._conn, inviter


@pytest.mark.asyncio
async def test_create_and_preview_invite():
    backend, sm, conn, inviter = await _env()
    try:
        inv = await create_invite(conn, inviter_user_id=inviter.id)
        assert inv["token"] and inv["status"] == "active"

        preview = await preview_invite(conn, inv["token"])
        assert preview is not None
        assert preview["status"] == "active"
        assert preview["role"] == "user"
        # Inviter display name surfaces for the join page's "X invited you".
        assert preview["inviter_display_name"] == "operator"
        assert preview["instance_handle"]  # always non-empty (sentinel fallback)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_preview_unknown_token_returns_none():
    backend, sm, conn, inviter = await _env()
    try:
        assert await preview_invite(conn, "not-a-real-token") is None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_consume_is_single_use():
    backend, sm, conn, inviter = await _env()
    try:
        inv = await create_invite(conn, inviter_user_id=inviter.id, max_uses=1)
        first = await consume_invite(conn, inv["token"])
        assert first is not None
        # Second consume on a 1-use invite fails (the conditional UPDATE
        # increments only while use_count < max_uses).
        assert await consume_invite(conn, inv["token"]) is None
        preview = await preview_invite(conn, inv["token"])
        assert preview["status"] == "used"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_multi_use_consumes_n_times():
    backend, sm, conn, inviter = await _env()
    try:
        inv = await create_invite(conn, inviter_user_id=inviter.id, max_uses=3)
        assert await consume_invite(conn, inv["token"]) is not None
        assert await consume_invite(conn, inv["token"]) is not None
        assert await consume_invite(conn, inv["token"]) is not None
        assert await consume_invite(conn, inv["token"]) is None  # 4th exceeds cap
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_expired_invite_cannot_be_consumed():
    backend, sm, conn, inviter = await _env()
    try:
        inv = await create_invite(conn, inviter_user_id=inviter.id)
        # Force a past expiry directly (create_invite only writes future ones).
        await conn.execute(
            "UPDATE auth_invites SET expires_at = '2000-01-01 00:00:00' WHERE id = ?",
            (inv["id"],),
        )
        await conn.commit()
        assert await consume_invite(conn, inv["token"]) is None
        preview = await preview_invite(conn, inv["token"])
        assert preview["status"] == "expired"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_revoked_invite_cannot_be_consumed():
    backend, sm, conn, inviter = await _env()
    try:
        inv = await create_invite(conn, inviter_user_id=inviter.id)
        assert await revoke_invite(conn, invite_id=inv["id"]) is True
        assert await consume_invite(conn, inv["token"]) is None
        preview = await preview_invite(conn, inv["token"])
        assert preview["status"] == "revoked"
        # Double-revoke is a no-op (returns False).
        assert await revoke_invite(conn, invite_id=inv["id"]) is False
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_revoke_scoped_to_inviter():
    backend, sm, conn, inviter = await _env()
    try:
        other = await sm.create_user("someone", "supersecret")
        inv = await create_invite(conn, inviter_user_id=inviter.id)
        # A different creator can't revoke it.
        assert await revoke_invite(conn, invite_id=inv["id"], inviter_user_id=other.id) is False
        # The real creator can.
        assert await revoke_invite(conn, invite_id=inv["id"], inviter_user_id=inviter.id) is True
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_claim_flow_creates_account_with_email_and_mutual_contacts():
    """Simulates what POST /api/auth/invite/{token}/claim does."""
    backend, sm, conn, inviter = await _env()
    try:
        inv = await create_invite(
            conn, inviter_user_id=inviter.id, invitee_email="newbie@example.com",
        )
        consumed = await consume_invite(conn, inv["token"])
        assert consumed is not None

        new_user = await sm.create_user(
            "newbie", "supersecret", role=consumed["role"],
            email=consumed["invitee_email"],
        )
        assert new_user.email == "newbie@example.com"

        await mark_claimed(conn, token_hash=consumed["token_hash"], claimed_user_id=new_user.id)

        # Route auto-adds mutual contacts.
        from augmentum.connect.contact_store import ensure_contact
        await ensure_contact(conn, user_id=new_user.id, peer_did=local_did_for(inviter.id), discovery_source="invite")
        await ensure_contact(conn, user_id=inviter.id, peer_did=local_did_for(new_user.id), discovery_source="invite")

        # Both sides now see each other.
        c1 = await get_contact(conn, user_id=new_user.id, peer_did=local_did_for(inviter.id))
        c2 = await get_contact(conn, user_id=inviter.id, peer_did=local_did_for(new_user.id))
        assert c1 is not None and c2 is not None

        # Invite now records its claimant.
        invites = await list_invites(conn, inviter_user_id=inviter.id)
        assert invites[0]["claimed_at"] != ""
        assert invites[0]["status"] == "used"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_claim_records_ip_for_reaccess():
    backend, sm, conn, inviter = await _env()
    try:
        inv = await create_invite(conn, inviter_user_id=inviter.id)
        consumed = await consume_invite(conn, inv["token"])
        await mark_claimed(
            conn, token_hash=consumed["token_hash"],
            claimed_user_id=inviter.id, claimed_ip="203.0.113.7",
        )
        invites = await list_invites(conn, inviter_user_id=inviter.id)
        # The admin can read back the claiming IP to pin a reconnect link to it.
        assert invites[0]["claimed_ip"] == "203.0.113.7"
    finally:
        await backend.close()


def test_invite_status_ordering():
    # Revoked wins even when also expired.
    row = {"revoked_at": "2020-01-01 00:00:00", "expires_at": "2019-01-01 00:00:00",
           "use_count": 0, "max_uses": 1}
    assert invite_status(row) == "revoked"


@pytest.mark.asyncio
async def test_invalid_kind_and_role_rejected():
    backend, sm, conn, inviter = await _env()
    try:
        with pytest.raises(ValueError):
            await create_invite(conn, inviter_user_id=inviter.id, kind="bogus")
        with pytest.raises(ValueError):
            await create_invite(conn, inviter_user_id=inviter.id, role="superuser")
    finally:
        await backend.close()


@pytest.fixture(autouse=True)
def _reset_handle(monkeypatch):
    # Keep instance_handle deterministic regardless of test env.
    monkeypatch.setattr(settings, "connect_instance_handle", "", raising=False)
    monkeypatch.setattr(settings, "augmentum_public_host", "", raising=False)
