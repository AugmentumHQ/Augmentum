"""Durable guest-pass grant store (Connect Phase 3a).

A thin CRUD layer over ``connect_guest_grants`` (migration 286), mirroring the
idioms of ``augmentum/auth/invite_store.py``: only the SHA-256 hash of the
durable surface token is stored; the raw token is returned once and lives in the
saved PWA. A grant is the durable, revocable relationship between a host and a
scoped ``role='guest'`` account — revoking it (``revoke``) is the host's single
kill-switch. See the design doc
``docs/superpowers/specs/2026-06-21-connect-durable-guest-surface-design.md``.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:  # pragma: no cover
    import aiosqlite

log = structlog.get_logger(__name__)

VALID_SCOPES = ("text", "call")
_SQLITE_DT = "%Y-%m-%d %H:%M:%S"

_COLUMNS = (
    "grant_id", "user_id", "host_did", "guest_user_id", "guest_did",
    "token_hash", "scopes", "created_at", "last_used_at", "revoked_at",
)


def _hash_token(raw: str) -> str:
    return hashlib.sha256((raw or "").encode("utf-8")).hexdigest()


def _now_str() -> str:
    return datetime.now(UTC).strftime(_SQLITE_DT)


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {col: row[idx] for idx, col in enumerate(_COLUMNS)}


def normalize_scopes(scopes: Any) -> str:
    """Coerce a scopes value (list or comma string) to a canonical csv.

    Keeps only known scopes, preserves text→call order, always includes ``text``
    (a guest with no usable scope is meaningless). Defaults to ``"text"``.
    """
    if isinstance(scopes, str):
        items = [s.strip().lower() for s in scopes.split(",")]
    elif isinstance(scopes, list | tuple):
        items = [str(s).strip().lower() for s in scopes]
    else:
        items = []
    keep = [s for s in VALID_SCOPES if s in items]
    if "text" not in keep:
        keep = ["text", *keep]
    return ",".join(keep)


def _public_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Host-safe grant view (never leaks the token hash)."""
    return {
        "grant_id": d["grant_id"],
        "host_user_id": d["user_id"],
        "host_did": d["host_did"],
        "guest_user_id": d["guest_user_id"],
        "guest_did": d["guest_did"],
        "scopes": d["scopes"],
        "created_at": d["created_at"],
        "last_used_at": d["last_used_at"],
        "revoked": bool(d["revoked_at"]),
    }


def grant_is_live(row: dict[str, Any]) -> bool:
    """A grant is usable iff it exists and isn't revoked (grants don't expire —
    they're durable until the host revokes them)."""
    return not row.get("revoked_at")


async def create_grant(
    conn: aiosqlite.Connection,
    *,
    host_user_id: str,
    host_did: str,
    guest_user_id: str,
    guest_did: str,
    scopes: Any = "text",
) -> dict[str, Any]:
    """Mint a grant + its durable token. Returns a dict incl. the raw token once."""
    grant_id = f"grt_{secrets.token_hex(8)}"
    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    scope_csv = normalize_scopes(scopes)
    await conn.execute(
        """INSERT INTO connect_guest_grants
               (grant_id, user_id, host_did, guest_user_id, guest_did,
                token_hash, scopes)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (grant_id, host_user_id, host_did, guest_user_id, guest_did,
         token_hash, scope_csv),
    )
    await conn.commit()
    log.info(
        "connect_guest_grant_created",
        grant_id=grant_id, host_user_id=host_user_id,
        guest_user_id=guest_user_id, scopes=scope_csv,
    )
    return {
        "grant_id": grant_id,
        "token": raw_token,  # raw — returned ONCE, never stored
        "host_user_id": host_user_id,
        "host_did": host_did,
        "guest_user_id": guest_user_id,
        "guest_did": guest_did,
        "scopes": scope_csv,
        "revoked": False,
    }


async def _fetch_one(conn: aiosqlite.Connection, where: str, params: tuple) -> dict[str, Any] | None:
    cur = await conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM connect_guest_grants WHERE {where}",
        params,
    )
    row = await cur.fetchone()
    await cur.close()
    return _row_to_dict(row) if row else None


async def get_by_token(conn: aiosqlite.Connection, raw_token: str) -> dict[str, Any] | None:
    """Resolve a durable token to its grant row (None if unknown)."""
    return await _fetch_one(conn, "token_hash = ?", (_hash_token(raw_token),))


async def get_grant(conn: aiosqlite.Connection, *, grant_id: str) -> dict[str, Any] | None:
    return await _fetch_one(conn, "grant_id = ?", (grant_id,))


async def is_guest_of(
    conn: aiosqlite.Connection, *, guest_user_id: str, host_user_id: str,
) -> bool:
    """True when a LIVE grant authorizes ``guest_user_id`` to reach ``host_user_id``.

    The hot-path check the ACL gate calls: a guest may message/call only a host
    it holds an un-revoked grant for.
    """
    cur = await conn.execute(
        """SELECT 1 FROM connect_guest_grants
            WHERE guest_user_id = ? AND user_id = ? AND revoked_at = '' LIMIT 1""",
        (guest_user_id, host_user_id),
    )
    row = await cur.fetchone()
    await cur.close()
    return row is not None


async def guest_scope_blocked(
    conn: aiosqlite.Connection, *,
    sender_user_id: str, sender_role: str, target_user_id: str,
) -> bool:
    """ACL gate: True when a GUEST sender may NOT reach ``target_user_id``.

    Fast-returns False for non-guests (so normal traffic is untouched). For a
    ``role='guest'`` sender, blocks unless a live grant authorizes reaching that
    host. Sits beside the existing block check in the routing handlers.
    """
    if (sender_role or "") != "guest":
        return False
    return not await is_guest_of(
        conn, guest_user_id=sender_user_id, host_user_id=target_user_id,
    )


async def grant_allows(
    conn: aiosqlite.Connection, *, guest_user_id: str, host_user_id: str, scope: str,
) -> bool:
    """Like :func:`is_guest_of` but also requires the named scope (text|call)."""
    cur = await conn.execute(
        """SELECT scopes FROM connect_guest_grants
            WHERE guest_user_id = ? AND user_id = ? AND revoked_at = '' LIMIT 1""",
        (guest_user_id, host_user_id),
    )
    row = await cur.fetchone()
    await cur.close()
    if not row:
        return False
    return scope in (row[0] or "").split(",")


async def list_for_host(
    conn: aiosqlite.Connection, *, host_user_id: str,
) -> list[dict[str, Any]]:
    """Host-side guest list (newest first), token-hash redacted."""
    cur = await conn.execute(
        f"SELECT {', '.join(_COLUMNS)} FROM connect_guest_grants "
        "WHERE user_id = ? ORDER BY created_at DESC",
        (host_user_id,),
    )
    rows = await cur.fetchall()
    await cur.close()
    return [_public_dict(_row_to_dict(r)) for r in rows]


async def touch_last_used(conn: aiosqlite.Connection, *, grant_id: str) -> None:
    await conn.execute(
        "UPDATE connect_guest_grants SET last_used_at = ? WHERE grant_id = ?",
        (_now_str(), grant_id),
    )
    await conn.commit()


async def set_scopes(
    conn: aiosqlite.Connection, *, grant_id: str, host_user_id: str, scopes: Any,
) -> bool:
    """Narrow/widen a grant's scopes (host-scoped). Returns True if updated."""
    cur = await conn.execute(
        "UPDATE connect_guest_grants SET scopes = ? "
        "WHERE grant_id = ? AND user_id = ? AND revoked_at = ''",
        (normalize_scopes(scopes), grant_id, host_user_id),
    )
    changed = cur.rowcount
    await cur.close()
    await conn.commit()
    return bool(changed)


async def revoke(
    conn: aiosqlite.Connection, *, grant_id: str, host_user_id: str,
) -> dict[str, Any] | None:
    """Revoke a grant (host-scoped). Returns the grant dict (so the caller can
    cascade: revoke the guest's sessions, release any door, drop push) or None
    if not found / not owned / already revoked."""
    row = await _fetch_one(
        conn, "grant_id = ? AND user_id = ?", (grant_id, host_user_id),
    )
    if row is None or row["revoked_at"]:
        return None
    await conn.execute(
        "UPDATE connect_guest_grants SET revoked_at = ? WHERE grant_id = ?",
        (_now_str(), grant_id),
    )
    await conn.commit()
    log.info("connect_guest_grant_revoked", grant_id=grant_id, host_user_id=host_user_id)
    return row
