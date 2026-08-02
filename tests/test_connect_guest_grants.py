"""Durable guest-pass grant store (Connect Phase 3a)."""

from __future__ import annotations

import pytest

from augmentum.auth.session_manager import SessionManager
from augmentum.connect.guest_grant_store import (
    create_grant,
    get_by_token,
    grant_allows,
    is_guest_of,
    list_for_host,
    normalize_scopes,
    revoke,
    set_scopes,
    touch_last_used,
)
from augmentum.state.backends.sqlite import SQLiteBackend


async def _env():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    sm = SessionManager(backend._conn)
    host = await sm.create_user("host", "supersecret")
    # "guest" is a reserved username; a real guest gets a generated handle.
    guest = await sm.create_user("visitor", "supersecret", role="guest")
    return backend, backend._conn, sm, host, guest


def test_normalize_scopes():
    assert normalize_scopes("text,call") == "text,call"
    assert normalize_scopes(["call"]) == "text,call"     # text always present
    assert normalize_scopes("") == "text"                # default
    assert normalize_scopes("call,bogus") == "text,call"  # unknown dropped
    assert normalize_scopes("text") == "text"


@pytest.mark.asyncio
async def test_create_and_resolve_by_token():
    backend, conn, sm, host, guest = await _env()
    try:
        g = await create_grant(
            conn, host_user_id=host.id, host_did="host@h",
            guest_user_id=guest.id, guest_did="guest@h", scopes="text,call",
        )
        assert g["token"] and g["scopes"] == "text,call" and g["revoked"] is False

        row = await get_by_token(conn, g["token"])
        assert row is not None and row["grant_id"] == g["grant_id"]
        assert "token" not in row  # only the hash is stored; raw never returned again
        assert await get_by_token(conn, "not-a-token") is None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_is_guest_of_and_scope_gate():
    backend, conn, sm, host, guest = await _env()
    try:
        other = await sm.create_user("other", "supersecret")
        await create_grant(
            conn, host_user_id=host.id, host_did="host@h",
            guest_user_id=guest.id, guest_did="guest@h", scopes="text",
        )
        # Guest may reach its host, not anyone else.
        assert await is_guest_of(conn, guest_user_id=guest.id, host_user_id=host.id) is True
        assert await is_guest_of(conn, guest_user_id=guest.id, host_user_id=other.id) is False
        # Scope gate: text granted, call not.
        assert await grant_allows(conn, guest_user_id=guest.id, host_user_id=host.id, scope="text") is True
        assert await grant_allows(conn, guest_user_id=guest.id, host_user_id=host.id, scope="call") is False
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_set_scopes_widens_call():
    backend, conn, sm, host, guest = await _env()
    try:
        g = await create_grant(
            conn, host_user_id=host.id, host_did="host@h",
            guest_user_id=guest.id, guest_did="guest@h", scopes="text",
        )
        assert await grant_allows(conn, guest_user_id=guest.id, host_user_id=host.id, scope="call") is False
        assert await set_scopes(conn, grant_id=g["grant_id"], host_user_id=host.id, scopes="text,call") is True
        assert await grant_allows(conn, guest_user_id=guest.id, host_user_id=host.id, scope="call") is True
        # Wrong host can't change it.
        assert await set_scopes(conn, grant_id=g["grant_id"], host_user_id="usr_nope", scopes="text") is False
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_revoke_kills_access_and_is_host_scoped():
    backend, conn, sm, host, guest = await _env()
    try:
        g = await create_grant(
            conn, host_user_id=host.id, host_did="host@h",
            guest_user_id=guest.id, guest_did="guest@h", scopes="text,call",
        )
        # A different host can't revoke it.
        assert await revoke(conn, grant_id=g["grant_id"], host_user_id="usr_nope") is None
        # The owning host can; returns the row for the caller's cascade.
        revoked = await revoke(conn, grant_id=g["grant_id"], host_user_id=host.id)
        assert revoked is not None and revoked["guest_user_id"] == guest.id
        # After revoke: token dead, ACL dead.
        assert await is_guest_of(conn, guest_user_id=guest.id, host_user_id=host.id) is False
        # Double-revoke is a no-op.
        assert await revoke(conn, grant_id=g["grant_id"], host_user_id=host.id) is None
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_list_for_host_redacts_token_and_flags_revoked():
    backend, conn, sm, host, guest = await _env()
    try:
        g = await create_grant(
            conn, host_user_id=host.id, host_did="host@h",
            guest_user_id=guest.id, guest_did="guest@h",
        )
        await touch_last_used(conn, grant_id=g["grant_id"])
        listed = await list_for_host(conn, host_user_id=host.id)
        assert len(listed) == 1
        entry = listed[0]
        assert "token_hash" not in entry and "token" not in entry
        assert entry["guest_user_id"] == guest.id and entry["last_used_at"]
        assert entry["revoked"] is False

        await revoke(conn, grant_id=g["grant_id"], host_user_id=host.id)
        assert (await list_for_host(conn, host_user_id=host.id))[0]["revoked"] is True
    finally:
        await backend.close()
