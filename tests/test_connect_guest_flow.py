"""Guest claim → session → revoke flow (Connect Phase 3a).

Exercises the logic the /guest routes wrap (claim issues a grant; the durable
token mints a scoped session; revoke is a total kill-switch) at the store +
SessionManager level — the routes are thin wrappers over exactly this.
"""

from __future__ import annotations

import pytest

from augmentum.auth.invite_store import consume_invite, create_invite
from augmentum.auth.session_manager import SessionManager
from augmentum.connect.guest_grant_store import (
    create_grant,
    get_by_token,
    grant_is_live,
    is_guest_of,
    revoke,
)
from augmentum.state.backends.sqlite import SQLiteBackend


async def _env():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    sm = SessionManager(backend._conn)
    host = await sm.create_user("host", "supersecret")
    return backend, backend._conn, sm, host


@pytest.mark.asyncio
async def test_external_guest_claim_issues_grant_and_session_then_revoke_kills_all():
    backend, conn, sm, host = await _env()
    try:
        # --- claim path: external_guest invite → guest account + grant ---
        inv = await create_invite(
            conn, inviter_user_id=host.id, kind="external_guest", role="guest",
        )
        consumed = await consume_invite(conn, inv["token"])
        assert consumed is not None and consumed["kind"] == "external_guest"

        guest = await sm.create_user("visitor", "supersecret", role="guest")
        grant = await create_grant(
            conn, host_user_id=host.id, host_did=f"{host.id}@this-instance",
            guest_user_id=guest.id, guest_did=f"{guest.id}@this-instance", scopes="text",
        )
        durable_token = grant["token"]

        # --- /guest/session: durable token → scoped guest session ---
        row = await get_by_token(conn, durable_token)
        assert row is not None and grant_is_live(row)
        g = await sm.get_user_by_id(guest.id)
        assert g is not None and g.is_active and g.role == "guest"
        session = await sm.create_session(guest.id, source="connect-guest")
        assert await sm.validate_token(session) is not None  # session works
        assert await is_guest_of(conn, guest_user_id=guest.id, host_user_id=host.id) is True

        # --- /guests/{id}/revoke: the single kill-switch cascade ---
        revoked = await revoke(conn, grant_id=grant["grant_id"], host_user_id=host.id)
        assert revoked is not None
        await sm.revoke_all_sessions(guest.id)
        await sm.update_user(guest.id, is_active=False)

        # After revoke: grant dead, session dead, account disabled, ACL dead.
        assert not grant_is_live(await get_by_token(conn, durable_token))
        assert await sm.validate_token(session) is None
        assert (await sm.get_user_by_id(guest.id)).is_active is False
        assert await is_guest_of(conn, guest_user_id=guest.id, host_user_id=host.id) is False
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_revoked_grant_token_will_not_mint_a_session():
    backend, conn, sm, host = await _env()
    try:
        guest = await sm.create_user("visitor", "supersecret", role="guest")
        grant = await create_grant(
            conn, host_user_id=host.id, host_did=f"{host.id}@this-instance",
            guest_user_id=guest.id, guest_did=f"{guest.id}@this-instance",
        )
        await revoke(conn, grant_id=grant["grant_id"], host_user_id=host.id)
        # The /guest/session check: get_by_token still resolves, but not live →
        # the route returns 410 and never creates a session.
        row = await get_by_token(conn, grant["token"])
        assert row is not None and grant_is_live(row) is False
    finally:
        await backend.close()
