"""Key revocation tombstones + subscribable abuse denylist (P3).

Two related defenses, both keyed on the canonical did:key:

  * **Revocation tombstone** — a signed statement that an identity key is
    retired (compromise, rotation, succession). Self-signed by the
    revoked key (proves the holder is retiring it) and optionally names a
    successor. Any pinned contact that sees a valid tombstone for a key
    it pinned must drop that key to unverified and refuse it.
  * **Denylist** — operator/community abuse blocks. Subscribable: an
    instance can import another instance's published denylist, tagged
    with the publisher's did:key for provenance and clean unsubscribe.

Delivery (D4 removed the directory): tombstones are served from
``.well-known`` and pushed to already-known peers. This module is the
crypto + store; the serving/push routes are thin wrappers over it.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from augmentum.fabric.canonical import canonical_bytes
from augmentum.fabric.contact_card import normalize_did
from augmentum.fabric.didkey import decode_ed25519_did
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)

_REVOCATION_VERSION = 1


class RevocationError(ValueError):
    """Raised when a revocation tombstone is malformed or unsigned/forged."""


def mint_revocation(
    *,
    sign,
    revoked_did_key: str,
    reason: str,
    issued_at: int,
    supersedes_to: str = "",
) -> dict[str, Any]:
    """Build a self-signed revocation tombstone.

    ``sign`` MUST be the revoked key's own signer (self-revocation proves
    possession). ``supersedes_to`` optionally names the successor key the
    contact should accept instead.
    """
    decode_ed25519_did(revoked_did_key)
    if supersedes_to:
        decode_ed25519_did(supersedes_to)
    statement = {
        "ctx": "augmentum-fabric-revocation-v1",  # domain separation
        "v": _REVOCATION_VERSION,
        "revoked_did_key": revoked_did_key,
        "reason": reason,
        "supersedes_to": supersedes_to,
        "issued_at": int(issued_at),
    }
    sig = sign(canonical_bytes(statement))
    return {**statement, "sig": base64.b64encode(sig).decode("ascii")}


def verify_revocation(tombstone: dict[str, Any]) -> str:
    """Verify a tombstone is self-signed by the revoked key. Returns the
    revoked did:key. Raises :class:`RevocationError` on any failure.

    Self-signature is the trust anchor: only the holder of the revoked
    key (or whoever compromised it — who can also just stop using it)
    can mint a valid self-revocation, so a third party can't revoke your
    key out from under you.
    """
    if not isinstance(tombstone, dict) or tombstone.get("v") != _REVOCATION_VERSION:
        raise RevocationError("unsupported or malformed tombstone")
    sig_b64 = tombstone.get("sig")
    if not isinstance(sig_b64, str) or not sig_b64:
        raise RevocationError("tombstone missing signature")
    revoked = str(tombstone.get("revoked_did_key", ""))
    statement = {
        "ctx": "augmentum-fabric-revocation-v1",  # domain separation
        "v": _REVOCATION_VERSION,
        "revoked_did_key": revoked,
        "reason": str(tombstone.get("reason", "")),
        "supersedes_to": str(tombstone.get("supersedes_to", "")),
        "issued_at": int(tombstone.get("issued_at", 0)),
    }
    try:
        pub_raw = decode_ed25519_did(revoked)
        Ed25519PublicKey.from_public_bytes(pub_raw).verify(
            base64.b64decode(sig_b64), canonical_bytes(statement)
        )
    except Exception as exc:
        raise RevocationError("tombstone signature verification failed") from exc
    return revoked


# ── store ────────────────────────────────────────────────────────────


async def record_revocation(
    conn: aiosqlite.Connection, tombstone: dict[str, Any],
) -> str:
    """Verify then persist a tombstone. Returns the revoked did:key.
    Idempotent on the revoked key. Raises if the tombstone is invalid —
    we never store an unverified revocation (an attacker can't poison the
    denylist with a forged tombstone)."""
    revoked = verify_revocation(tombstone)
    canonical = normalize_did(revoked)
    await conn.execute(
        "INSERT OR REPLACE INTO fabric_revocations "
        "(revoked_did_key, reason, supersedes_to, tombstone_json, issued_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            canonical,
            str(tombstone.get("reason", "")),
            str(tombstone.get("supersedes_to", "")),
            json.dumps(tombstone, separators=(",", ":")),
            int(tombstone.get("issued_at", 0)),
        ),
    )
    await conn.commit()
    log.info("fabric_revocation_recorded", revoked_did_key=canonical)
    return canonical


async def is_revoked(conn: aiosqlite.Connection, did_key: str) -> bool:
    canonical = normalize_did(did_key)
    cur = await conn.execute(
        "SELECT 1 FROM fabric_revocations WHERE revoked_did_key=?", (canonical,)
    )
    return await cur.fetchone() is not None


async def successor_of(conn: aiosqlite.Connection, did_key: str) -> str:
    """Return the successor did:key a revocation names, or '' if none."""
    canonical = normalize_did(did_key)
    cur = await conn.execute(
        "SELECT supersedes_to FROM fabric_revocations WHERE revoked_did_key=?",
        (canonical,),
    )
    row = await cur.fetchone()
    return row[0] if row and row[0] else ""


# ── denylist ─────────────────────────────────────────────────────────


async def add_denylist(
    conn: aiosqlite.Connection,
    *,
    did_key: str,
    reason: str = "",
    source: str = "local",
) -> None:
    """Add (or import) a denylist entry. ``source`` is 'local' or the
    publishing instance's did:key for an imported subscription."""
    canonical = normalize_did(did_key)
    await conn.execute(
        "INSERT OR REPLACE INTO fabric_denylist (did_key, reason, source) "
        "VALUES (?, ?, ?)",
        (canonical, reason, source),
    )
    await conn.commit()


async def is_denied(conn: aiosqlite.Connection, did_key: str) -> bool:
    """True if the key is on the denylist from ANY source (revoked keys
    count too — a retired key should never be reachable)."""
    canonical = normalize_did(did_key)
    cur = await conn.execute(
        "SELECT 1 FROM fabric_denylist WHERE did_key=? LIMIT 1", (canonical,)
    )
    if await cur.fetchone() is not None:
        return True
    return await is_revoked(conn, canonical)


async def unsubscribe_source(conn: aiosqlite.Connection, *, source: str) -> int:
    """Drop all denylist entries imported from ``source`` (clean unsub).
    Returns the number removed. Won't touch 'local' entries unless
    explicitly asked."""
    cur = await conn.execute(
        "DELETE FROM fabric_denylist WHERE source=?", (source,)
    )
    await conn.commit()
    return cur.rowcount
