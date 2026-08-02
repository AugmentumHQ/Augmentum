"""Guest comms portal: admin-confirmed registration + IP allowlist.

The gate between "someone registered from an invite link" and "this guest
can use the portal." Flow:

  register_pending()  — the invitee's claim creates a PENDING registration
                        (their scoped guest account exists but is gated).
  list_pending()      — the admin reviews who's waiting.
  confirm()           — the admin's final step: allowlists the registration
                        IP and marks the registration confirmed. The caller
                        then mints the guest grant (existing guest_grant_store)
                        so scopes (text/call/video) apply.
  deny()              — reject a registration.
  ip_allowed()        — enforce: a confirmed guest reaches the portal only
                        from an allowlisted IP.

Sits on top of the existing guest-grant ACL (who a guest may reach); this
module adds the human approval + IP allowlist the operator asked for.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.connect.guest_grant_store import normalize_scopes
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

PENDING = "pending"
CONFIRMED = "confirmed"
DENIED = "denied"


@dataclass(frozen=True)
class GuestRegistration:
    registration_id: str
    inviter_user_id: str
    guest_user_id: str
    display_name: str
    requested_ip: str
    scopes: str
    status: str
    device_id: str = ""
    device_public_key: str = ""


def _row(r) -> GuestRegistration:
    return GuestRegistration(
        registration_id=r[0], inviter_user_id=r[1], guest_user_id=r[2],
        display_name=r[3], requested_ip=r[4], scopes=r[5], status=r[6],
        device_id=r[7] if len(r) > 7 else "",
        device_public_key=r[8] if len(r) > 8 else "",
    )


_COLS = ("registration_id, inviter_user_id, guest_user_id, display_name, "
         "requested_ip, scopes, status, device_id, device_public_key")


async def register_pending(
    conn: aiosqlite.Connection,
    *,
    inviter_user_id: str,
    guest_user_id: str,
    display_name: str = "",
    requested_ip: str = "",
    scopes: str = "text",
    invite_token_hash: str = "",
    device_id: str = "",
    device_public_key: str = "",
) -> GuestRegistration:
    """Create a PENDING registration awaiting the admin's confirm step.

    ``device_id`` / ``device_public_key`` carry the guest's web-device
    identity so confirm can register a trusted device (the basis for the
    IP-independent reconnect-from-anywhere session)."""
    if not inviter_user_id or not guest_user_id:
        raise ValueError("register_pending requires inviter + guest user ids")
    reg_id = uuid.uuid4().hex
    scopes = normalize_scopes(scopes)
    await conn.execute(
        "INSERT INTO guest_registrations "
        "(registration_id, invite_token_hash, inviter_user_id, guest_user_id, "
        " display_name, requested_ip, scopes, device_id, device_public_key) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (reg_id, invite_token_hash, inviter_user_id, guest_user_id,
         display_name, requested_ip, scopes, device_id, device_public_key),
    )
    await conn.commit()
    log.info("guest_registration_pending", inviter=inviter_user_id, guest=guest_user_id)
    return GuestRegistration(
        registration_id=reg_id, inviter_user_id=inviter_user_id,
        guest_user_id=guest_user_id, display_name=display_name,
        requested_ip=requested_ip, scopes=scopes, status=PENDING,
        device_id=device_id, device_public_key=device_public_key,
    )


async def list_pending(
    conn: aiosqlite.Connection, *, inviter_user_id: str,
) -> list[GuestRegistration]:
    """Pending registrations awaiting this admin/host's confirmation."""
    cur = await conn.execute(
        f"SELECT {_COLS} FROM guest_registrations "
        "WHERE inviter_user_id=? AND status=? ORDER BY created_at ASC",
        (inviter_user_id, PENDING),
    )
    return [_row(r) for r in await cur.fetchall()]


async def get_registration(
    conn: aiosqlite.Connection, *, registration_id: str,
) -> GuestRegistration | None:
    cur = await conn.execute(
        f"SELECT {_COLS} FROM guest_registrations WHERE registration_id=?",
        (registration_id,),
    )
    row = await cur.fetchone()
    return _row(row) if row else None


async def confirm(
    conn: aiosqlite.Connection,
    *,
    registration_id: str,
    admin_user_id: str,
    extra_ips: list[str] | None = None,
) -> GuestRegistration:
    """The admin's final step: allowlist the registration IP (+ any extras)
    and mark the registration confirmed. The caller mints the guest grant.

    Only the inviting host may confirm their own pending registration.
    Raises ValueError otherwise."""
    reg = await get_registration(conn, registration_id=registration_id)
    if reg is None:
        raise ValueError("no such registration")
    if reg.inviter_user_id != admin_user_id:
        raise ValueError("only the inviting host may confirm this registration")
    if reg.status != PENDING:
        raise ValueError(f"registration is already {reg.status}")

    ips = {reg.requested_ip, *(extra_ips or [])}
    for ip in ips:
        if ip:
            await allow_ip(conn, guest_user_id=reg.guest_user_id, ip=ip, added_by=admin_user_id)
    await conn.execute(
        "UPDATE guest_registrations SET status=?, decided_at=datetime('now'), "
        "decided_by=? WHERE registration_id=?",
        (CONFIRMED, admin_user_id, registration_id),
    )
    await conn.commit()
    log.info("guest_registration_confirmed", registration_id=registration_id,
             guest=reg.guest_user_id, ips=len([i for i in ips if i]))
    return GuestRegistration(**{**reg.__dict__, "status": CONFIRMED})


async def invite_hash_for_registration(
    conn: aiosqlite.Connection, *, registration_id: str,
) -> str:
    """The originating invite's token hash for a registration ('' if none).

    Lifecycle hook plumbing: portal confirm/deny release the invite's
    ephemeral public tunnel, and only the hash is stored (raw tokens are
    unrecoverable by design)."""
    cur = await conn.execute(
        "SELECT invite_token_hash FROM guest_registrations WHERE registration_id=?",
        (registration_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    return str(row[0]) if row and row[0] else ""


async def deny(
    conn: aiosqlite.Connection, *, registration_id: str, admin_user_id: str,
) -> bool:
    """Deny a pending registration (the host's call)."""
    cur = await conn.execute(
        "UPDATE guest_registrations SET status=?, decided_at=datetime('now'), "
        "decided_by=? WHERE registration_id=? AND inviter_user_id=? AND status=?",
        (DENIED, admin_user_id, registration_id, admin_user_id, PENDING),
    )
    await conn.commit()
    return cur.rowcount > 0


# ── IP allowlist ─────────────────────────────────────────────────────


async def allow_ip(
    conn: aiosqlite.Connection, *, guest_user_id: str, ip: str, added_by: str = "",
) -> None:
    await conn.execute(
        "INSERT OR IGNORE INTO guest_ip_allowlist (guest_user_id, ip, added_by) "
        "VALUES (?, ?, ?)",
        (guest_user_id, ip, added_by),
    )
    await conn.commit()


async def ip_allowed(
    conn: aiosqlite.Connection, *, guest_user_id: str, ip: str,
) -> bool:
    """True iff this guest may reach the portal from ``ip``. An empty ip is
    never allowed (fail closed)."""
    if not ip:
        return False
    cur = await conn.execute(
        "SELECT 1 FROM guest_ip_allowlist WHERE guest_user_id=? AND ip=? LIMIT 1",
        (guest_user_id, ip),
    )
    return await cur.fetchone() is not None


async def is_confirmed(
    conn: aiosqlite.Connection, *, guest_user_id: str,
) -> bool:
    """True iff this guest has a confirmed registration."""
    return await registration_state(conn, guest_user_id=guest_user_id) == CONFIRMED


async def device_for_guest(
    conn: aiosqlite.Connection, *, guest_user_id: str,
) -> str:
    """The web device_id from this guest's most recent registration (''
    if none) — used to bind their session to the device so the host can
    revoke that one device."""
    cur = await conn.execute(
        "SELECT device_id FROM guest_registrations WHERE guest_user_id=? "
        "ORDER BY created_at DESC LIMIT 1",
        (guest_user_id,),
    )
    row = await cur.fetchone()
    return row[0] if row and row[0] else ""


async def registration_state(
    conn: aiosqlite.Connection, *, guest_user_id: str,
) -> str:
    """The guest's portal registration state: 'none' | 'pending' |
    'confirmed' | 'denied'. 'none' means they aren't a portal guest at all
    (e.g. a cast guest), so the portal gate doesn't apply to them — the
    login gate uses this to avoid blocking non-portal guests."""
    cur = await conn.execute(
        "SELECT status FROM guest_registrations WHERE guest_user_id=? "
        "ORDER BY created_at DESC LIMIT 1",
        (guest_user_id,),
    )
    row = await cur.fetchone()
    return row[0] if row else "none"
