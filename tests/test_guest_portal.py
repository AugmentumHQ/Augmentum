"""Guest portal: pending registration -> admin confirm -> IP allowlist."""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.connect import guest_portal as gp


async def _db() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, "
        "description TEXT, applied_at TEXT DEFAULT (datetime('now')))"
    )
    for mig in ("296_guest_portal.sql", "297_guest_device_reconnect.sql"):
        with open(f"augmentum/state/migrations/{mig}") as f:
            await conn.executescript(f.read())
    await conn.commit()
    return conn


@pytest.mark.asyncio
async def test_pending_then_confirm_allowlists_ip():
    conn = await _db()
    try:
        reg = await gp.register_pending(
            conn, inviter_user_id="host", guest_user_id="guest1",
            display_name="Sam", requested_ip="203.0.113.5", scopes="text,call",
        )
        assert reg.status == gp.PENDING
        # Guest can't reach yet (no allowlist, not confirmed).
        assert await gp.ip_allowed(conn, guest_user_id="guest1", ip="203.0.113.5") is False
        assert await gp.is_confirmed(conn, guest_user_id="guest1") is False
        # Admin sees it pending.
        pend = await gp.list_pending(conn, inviter_user_id="host")
        assert len(pend) == 1 and pend[0].guest_user_id == "guest1"

        # Admin confirms -> IP allowlisted + confirmed.
        await gp.confirm(conn, registration_id=reg.registration_id, admin_user_id="host")
        assert await gp.is_confirmed(conn, guest_user_id="guest1") is True
        assert await gp.ip_allowed(conn, guest_user_id="guest1", ip="203.0.113.5") is True
        # A different IP is still blocked (fail-closed) until the admin adds it.
        assert await gp.ip_allowed(conn, guest_user_id="guest1", ip="198.51.100.9") is False
        assert await gp.list_pending(conn, inviter_user_id="host") == []
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_only_inviting_host_can_confirm():
    conn = await _db()
    try:
        reg = await gp.register_pending(
            conn, inviter_user_id="host", guest_user_id="g", requested_ip="1.2.3.4",
        )
        with pytest.raises(ValueError):
            await gp.confirm(conn, registration_id=reg.registration_id, admin_user_id="someone_else")
        # double-confirm guarded
        await gp.confirm(conn, registration_id=reg.registration_id, admin_user_id="host")
        with pytest.raises(ValueError):
            await gp.confirm(conn, registration_id=reg.registration_id, admin_user_id="host")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_confirm_with_extra_ips():
    conn = await _db()
    try:
        reg = await gp.register_pending(
            conn, inviter_user_id="host", guest_user_id="g", requested_ip="1.1.1.1",
        )
        await gp.confirm(conn, registration_id=reg.registration_id,
                         admin_user_id="host", extra_ips=["2.2.2.2", ""])
        assert await gp.ip_allowed(conn, guest_user_id="g", ip="1.1.1.1") is True
        assert await gp.ip_allowed(conn, guest_user_id="g", ip="2.2.2.2") is True
        assert await gp.ip_allowed(conn, guest_user_id="g", ip="") is False  # fail-closed
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_device_threads_through_for_reconnection():
    # The device id carried at register is what later gives the guest an
    # IP-independent, device-bound session (reconnect from any network).
    conn = await _db()
    try:
        reg = await gp.register_pending(
            conn, inviter_user_id="host", guest_user_id="g",
            requested_ip="1.1.1.1", device_id="dev-abc",
            device_public_key="pk123",
        )
        assert reg.device_id == "dev-abc"
        got = await gp.get_registration(conn, registration_id=reg.registration_id)
        assert got.device_id == "dev-abc" and got.device_public_key == "pk123"
        # The login path looks the device up to bind the session.
        assert await gp.device_for_guest(conn, guest_user_id="g") == "dev-abc"
        assert await gp.device_for_guest(conn, guest_user_id="nobody") == ""
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_deny():
    conn = await _db()
    try:
        reg = await gp.register_pending(
            conn, inviter_user_id="host", guest_user_id="g", requested_ip="1.1.1.1",
        )
        assert await gp.deny(conn, registration_id=reg.registration_id, admin_user_id="host") is True
        assert await gp.list_pending(conn, inviter_user_id="host") == []
        assert await gp.is_confirmed(conn, guest_user_id="g") is False
    finally:
        await conn.close()
