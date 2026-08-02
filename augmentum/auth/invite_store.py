"""Persisted invite store — self-claim account onboarding (Connect Phase 1).

An invite lets a host operator mint a link a person can use to create their
own account (choosing their own password) and land in Connect with the
inviter already a contact. See migration ``282_auth_invites.sql`` and the
design doc ``docs/superpowers/specs/2026-06-20-connect-comms-platform-design.md``.

Security model: only the SHA-256 hash of the raw token is stored. The raw
token is returned exactly once (to the creator) and carried in the link.
Validity (not-revoked, within expiry, uses remaining) is enforced at claim
time, and consumption is atomic (a single conditional ``UPDATE`` increments
``use_count`` so two simultaneous claims can't both succeed on a 1-use invite).
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover
    import aiosqlite

log = structlog.get_logger(__name__)

# SQLite's datetime('now') format, in UTC — match it so string comparison of
# expires_at against datetime('now') is correct.
_SQLITE_DT = "%Y-%m-%d %H:%M:%S"

VALID_KINDS = ("account_claim", "external_guest")
VALID_ROLES = ("user", "guest")

_COLUMNS = (
    "id", "token_hash", "inviter_user_id", "kind", "role", "invitee_email",
    "handle_hint", "max_uses", "use_count", "created_at", "expires_at",
    "claimed_at", "claimed_user_id", "revoked_at", "claimed_ip",
)


def _hash_token(raw: str) -> str:
    return hashlib.sha256((raw or "").encode("utf-8")).hexdigest()


def hash_token(raw: str) -> str:
    """Public alias — the invite token's storage/ref hash."""
    return _hash_token(raw)


def tunnel_ref_for_hash(token_hash: str) -> str:
    """The tunnel-manager ref key for an invite, derived from its HASH.

    Lifecycle hooks (portal confirm/deny, revoke, fully-used) only hold the
    hash; the mint site holds the raw token. Both derive the SAME ref via the
    hash so ensure/release pair up. Never key refs on the raw token."""
    return (token_hash or "")[:12]


def tunnel_ref_for_token(raw_token: str) -> str:
    """The tunnel-manager ref key for an invite, from the RAW token."""
    return tunnel_ref_for_hash(_hash_token(raw_token))


def _now_str() -> str:
    return datetime.now(UTC).strftime(_SQLITE_DT)


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {col: row[idx] for idx, col in enumerate(_COLUMNS)}


def invite_status(row: dict[str, Any]) -> str:
    """Return the lifecycle status of an invite row dict.

    One of ``revoked`` | ``expired`` | ``used`` | ``active``. Order matters:
    a revoked-and-expired invite reports ``revoked`` (the operator action wins).
    """
    if row.get("revoked_at"):
        return "revoked"
    expires_at = row.get("expires_at") or ""
    if expires_at and expires_at <= _now_str():
        return "expired"
    if int(row.get("use_count") or 0) >= int(row.get("max_uses") or 1):
        return "used"
    return "active"


async def create_invite(
    conn: aiosqlite.Connection,
    *,
    inviter_user_id: str,
    kind: str = "account_claim",
    role: str = "user",
    invitee_email: str = "",
    handle_hint: str = "",
    max_uses: int = 1,
    ttl_hours: int = 168,
) -> dict[str, Any]:
    """Mint an invite and return a dict including the raw token (shown once).

    ``ttl_hours <= 0`` mints a non-expiring invite. ``max_uses`` is clamped to
    at least 1. Raises ``ValueError`` on an unknown kind/role.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"invalid invite kind: {kind!r}")
    if role not in VALID_ROLES:
        raise ValueError(f"invalid invite role: {role!r}")
    invite_id = f"inv_{secrets.token_hex(8)}"
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    max_uses = max(1, int(max_uses))
    expires_at = ""
    if ttl_hours and ttl_hours > 0:
        expires_at = (
            datetime.now(UTC) + timedelta(hours=int(ttl_hours))
        ).strftime(_SQLITE_DT)

    await conn.execute(
        """INSERT INTO auth_invites
               (id, token_hash, inviter_user_id, kind, role, invitee_email,
                handle_hint, max_uses, expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (invite_id, token_hash, inviter_user_id, kind, role,
         (invitee_email or "").strip(), (handle_hint or "").strip(),
         max_uses, expires_at),
    )
    await conn.commit()
    log.info(
        "connect_invite_created",
        invite_id=invite_id, inviter_user_id=inviter_user_id, kind=kind,
        role=role, max_uses=max_uses, expires_at=expires_at or "never",
    )
    return {
        "id": invite_id,
        "token": raw_token,  # raw — returned ONCE, never stored
        "inviter_user_id": inviter_user_id,
        "kind": kind,
        "role": role,
        "invitee_email": (invitee_email or "").strip(),
        "handle_hint": (handle_hint or "").strip(),
        "max_uses": max_uses,
        "use_count": 0,
        "expires_at": expires_at,
        "status": "active",
    }


async def _fetch_by_hash(
    conn: aiosqlite.Connection, token_hash: str,
) -> dict[str, Any] | None:
    cur = await conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM auth_invites WHERE token_hash = ?",
        (token_hash,),
    )
    row = await cur.fetchone()
    await cur.close()
    return _row_to_dict(row) if row else None


async def preview_invite(
    conn: aiosqlite.Connection, raw_token: str,
) -> dict[str, Any] | None:
    """Public, non-consuming preview of an invite by raw token.

    Returns ``None`` when the token matches nothing (caller → 404). Otherwise
    a public-safe dict: status, role, kind, an email hint, and the inviter's
    display name + this instance's handle (for the join page's "X invited you
    to <instance>" line). Never leaks the token hash or inviter user_id.
    """
    row = await _fetch_by_hash(conn, _hash_token(raw_token))
    if row is None:
        return None
    from augmentum.connect.contacts import instance_handle

    inviter_name = ""
    cur = await conn.execute(
        "SELECT COALESCE(NULLIF(display_name, ''), username) FROM users WHERE id = ?",
        (row["inviter_user_id"],),
    )
    nrow = await cur.fetchone()
    await cur.close()
    if nrow and nrow[0]:
        inviter_name = str(nrow[0])
    return {
        "status": invite_status(row),
        "kind": row["kind"],
        "role": row["role"],
        "invitee_email": row["invitee_email"],
        "handle_hint": row["handle_hint"],
        "inviter_display_name": inviter_name,
        "instance_handle": instance_handle(),
    }


async def consume_invite(
    conn: aiosqlite.Connection, raw_token: str,
) -> dict[str, Any] | None:
    """Atomically consume one use of an invite if it is currently valid.

    The conditional ``UPDATE`` increments ``use_count`` only while the invite
    is un-revoked, un-expired, and has uses remaining — so two concurrent
    claims on a 1-use invite can't both win. Returns the invite row dict on
    success (caller then provisions the account), else ``None``.
    """
    token_hash = _hash_token(raw_token)
    now = _now_str()
    cur = await conn.execute(
        """UPDATE auth_invites
              SET use_count = use_count + 1
            WHERE token_hash = ?
              AND revoked_at = ''
              AND use_count < max_uses
              AND (expires_at = '' OR expires_at > ?)""",
        (token_hash, now),
    )
    consumed = cur.rowcount
    await cur.close()
    if not consumed:
        await conn.rollback()
        return None
    await conn.commit()
    return await _fetch_by_hash(conn, token_hash)


async def mark_claimed(
    conn: aiosqlite.Connection, *, token_hash: str, claimed_user_id: str,
    claimed_ip: str = "",
) -> None:
    """Stamp claimed_at / claimed_user_id / claimed_ip after provisioning.

    Idempotent-ish: only stamps the first claim (won't overwrite an existing
    claimed_user_id), so a multi-use invite records its first claimant. The
    ``claimed_ip`` (the recipient's address, via Cf-Connecting-Ip through a
    tunnel) is what the admin later whitelists to grant re-access.
    """
    await conn.execute(
        """UPDATE auth_invites
              SET claimed_at = ?, claimed_user_id = ?, claimed_ip = ?
            WHERE token_hash = ? AND claimed_user_id = ''""",
        (_now_str(), claimed_user_id, (claimed_ip or "").strip(), token_hash),
    )
    await conn.commit()


async def list_invites(
    conn: aiosqlite.Connection, *, inviter_user_id: str | None = None,
) -> list[dict[str, Any]]:
    """List invites (public-safe), newest first.

    ``inviter_user_id=None`` returns all (admin overview); a user id scopes to
    that creator's invites. The raw token is NOT returned (it's unrecoverable
    by design — only its hash is stored).
    """
    if inviter_user_id is None:
        cur = await conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM auth_invites ORDER BY created_at DESC",
        )
    else:
        cur = await conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM auth_invites "
            "WHERE inviter_user_id = ? ORDER BY created_at DESC",
            (inviter_user_id,),
        )
    rows = await cur.fetchall()
    await cur.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        d = _row_to_dict(row)
        out.append({
            "id": d["id"],
            "inviter_user_id": d["inviter_user_id"],
            "kind": d["kind"],
            "role": d["role"],
            "invitee_email": d["invitee_email"],
            "handle_hint": d["handle_hint"],
            "max_uses": d["max_uses"],
            "use_count": d["use_count"],
            "created_at": d["created_at"],
            "expires_at": d["expires_at"],
            "claimed_at": d["claimed_at"],
            "claimed_ip": d["claimed_ip"],
            "status": invite_status(d),
        })
    return out


async def set_join_base(
    conn: aiosqlite.Connection, *, token_hash: str, join_base: str,
) -> None:
    """Record the public base URL chosen at mint time (guest gateway).

    The QR endpoint must reproduce the IDENTICAL bundle URL the mint response
    returned (an ephemeral tunnel base can't be re-derived from the QR
    request's own Host header). Column added by migration 313.
    """
    await conn.execute(
        "UPDATE auth_invites SET join_base = ? WHERE token_hash = ?",
        ((join_base or "").strip().rstrip("/"), token_hash),
    )
    await conn.commit()


async def get_join_base(conn: aiosqlite.Connection, raw_token: str) -> str:
    """The mint-time public base for a raw token ('' when none recorded)."""
    cur = await conn.execute(
        "SELECT join_base FROM auth_invites WHERE token_hash = ?",
        (_hash_token(raw_token),),
    )
    row = await cur.fetchone()
    await cur.close()
    return str(row[0]) if row and row[0] else ""


async def token_hash_for_id(
    conn: aiosqlite.Connection, invite_id: str,
) -> str:
    """The token hash for an invite id ('' when unknown).

    Used by lifecycle hooks (tunnel release on revoke) that only hold the row
    id — the raw token is unrecoverable by design, so the hash prefix is the
    stable ref key shared with the tunnel manager.
    """
    cur = await conn.execute(
        "SELECT token_hash FROM auth_invites WHERE id = ?", (invite_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    return str(row[0]) if row and row[0] else ""


async def revoke_invite(
    conn: aiosqlite.Connection, *, invite_id: str,
    inviter_user_id: str | None = None,
) -> bool:
    """Revoke an invite. ``inviter_user_id`` (when given) scopes the revoke to
    that creator so a non-admin can only revoke their own. Returns True if a
    row was revoked.
    """
    if inviter_user_id is None:
        cur = await conn.execute(
            "UPDATE auth_invites SET revoked_at = ? WHERE id = ? AND revoked_at = ''",
            (_now_str(), invite_id),
        )
    else:
        cur = await conn.execute(
            "UPDATE auth_invites SET revoked_at = ? "
            "WHERE id = ? AND inviter_user_id = ? AND revoked_at = ''",
            (_now_str(), invite_id, inviter_user_id),
        )
    changed = cur.rowcount
    await cur.close()
    await conn.commit()
    return bool(changed)
