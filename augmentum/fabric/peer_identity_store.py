"""TOFU pin + verified-state store for federated peer identities (P1).

CRUD over ``fabric_peer_identities`` (migration 289). This is the
security-state layer: who this user has pinned, what did:key it resolves
to, and whether the human verified it out-of-band.

Every function is user-scoped (``user_id``) and refuses the anon row —
TOFU is per-user and must never write into a shared/empty user bucket.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.fabric.contact_card import normalize_did
from augmentum.fabric.didkey import did_equal
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


@dataclass(frozen=True)
class PeerIdentity:
    id: str
    user_id: str
    peer_did_key: str
    handle: str
    endpoint: str
    author_did_key: str
    verified: bool
    verified_method: str
    source: str

    @property
    def trust_label(self) -> str:
        """Human trust label the UI MUST render (D1-01).

        Never collapse these into one — the difference between
        "verified" and "pinned, not verified" is the entire security
        contract of contact-card federation.
        """
        return "verified" if self.verified else "pinned, not verified"


def _row_to_identity(row) -> PeerIdentity:
    return PeerIdentity(
        id=row[0],
        user_id=row[1],
        peer_did_key=row[2],
        handle=row[3],
        endpoint=row[4],
        author_did_key=row[5],
        verified=bool(row[6]),
        verified_method=row[7],
        source=row[9],
    )


_COLS = (
    "id, user_id, peer_did_key, handle, endpoint, author_did_key, "
    "verified, verified_method, verified_at, source"
)


async def pin_peer(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    peer_did_key: str,
    handle: str = "",
    endpoint: str = "",
    author_did_key: str = "",
    source: str = "card",
) -> PeerIdentity:
    """TOFU-pin a peer identity for ``user_id`` (verified=False).

    Idempotent on (user_id, peer_did_key): re-pinning the same key
    refreshes the mutable handle/endpoint/author but PRESERVES the
    verified-state (you don't re-verify just because the endpoint moved).
    The did:key is normalised so byte-comparison stays reliable.

    Raises ValueError on an empty user_id (anon-row guard) or a malformed
    did:key.
    """
    if not user_id:
        raise ValueError("pin_peer requires a non-empty user_id")
    canonical = normalize_did(peer_did_key)  # raises on malformed
    author_canonical = normalize_did(author_did_key) if author_did_key else ""

    existing = await get_peer(conn, user_id=user_id, peer_did_key=canonical)
    if existing is not None:
        # The author key is folded into the verification ceremony (AK-1).
        # If a re-pin carries a DIFFERENT non-empty author key than the one
        # that was verified, the prior verification no longer covers the
        # current key material — drop back to unverified and force a fresh
        # ceremony. Otherwise a malicious host could swap the author key
        # after verification and keep the "verified" badge. An empty
        # incoming author (no new info) leaves verified-state untouched.
        author_changed = bool(
            author_canonical
            and existing.author_did_key
            and not did_equal(existing.author_did_key, author_canonical)
        )
        if author_changed and existing.verified:
            log.warning(
                "fabric_peer_author_key_changed_unverifying",
                user_id=user_id, peer_did_key=canonical,
            )
            await conn.execute(
                "UPDATE fabric_peer_identities SET handle=?, endpoint=?, "
                "author_did_key=?, verified=0, verified_method='', "
                "verified_at=NULL, updated_at=datetime('now') "
                "WHERE user_id=? AND peer_did_key=?",
                (handle, endpoint, author_canonical, user_id, canonical),
            )
        else:
            # Preserve verified-state for benign refreshes (endpoint moved,
            # same/empty author key).
            new_author = author_canonical or existing.author_did_key
            await conn.execute(
                "UPDATE fabric_peer_identities SET handle=?, endpoint=?, "
                "author_did_key=?, updated_at=datetime('now') "
                "WHERE user_id=? AND peer_did_key=?",
                (handle, endpoint, new_author, user_id, canonical),
            )
        await conn.commit()
        refreshed = await get_peer(conn, user_id=user_id, peer_did_key=canonical)
        assert refreshed is not None
        return refreshed

    new_id = uuid.uuid4().hex
    await conn.execute(
        "INSERT INTO fabric_peer_identities "
        "(id, user_id, peer_did_key, handle, endpoint, author_did_key, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (new_id, user_id, canonical, handle, endpoint, author_canonical, source),
    )
    await conn.commit()
    log.info(
        "fabric_peer_pinned",
        user_id=user_id, peer_did_key=canonical, source=source,
    )
    fresh = await get_peer(conn, user_id=user_id, peer_did_key=canonical)
    assert fresh is not None
    return fresh


async def get_peer(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    peer_did_key: str,
) -> PeerIdentity | None:
    canonical = normalize_did(peer_did_key)
    cur = await conn.execute(
        f"SELECT {_COLS} FROM fabric_peer_identities "
        "WHERE user_id=? AND peer_did_key=?",
        (user_id, canonical),
    )
    row = await cur.fetchone()
    return _row_to_identity(row) if row else None


async def mark_verified(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    peer_did_key: str,
    method: str,
) -> PeerIdentity:
    """Upgrade a pin to verified after a successful ceremony.

    ``method`` is 'sas' (voice) or 'qr' (text). Raises ValueError if the
    peer isn't pinned (you can't verify an identity you never saw).
    """
    if method not in ("sas", "qr"):
        raise ValueError(f"unknown verification method {method!r}")
    canonical = normalize_did(peer_did_key)
    existing = await get_peer(conn, user_id=user_id, peer_did_key=canonical)
    if existing is None:
        raise ValueError("cannot verify an unpinned peer identity")
    await conn.execute(
        "UPDATE fabric_peer_identities SET verified=1, verified_method=?, "
        "verified_at=datetime('now'), updated_at=datetime('now') "
        "WHERE user_id=? AND peer_did_key=?",
        (method, user_id, canonical),
    )
    await conn.commit()
    log.info(
        "fabric_peer_verified",
        user_id=user_id, peer_did_key=canonical, method=method,
    )
    result = await get_peer(conn, user_id=user_id, peer_did_key=canonical)
    assert result is not None
    return result


async def detect_key_change(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    handle: str,
    new_did_key: str,
) -> list[str]:
    """Return existing pinned did:keys for ``handle`` that DIFFER from
    ``new_did_key`` — the "safety number changed" signal.

    A non-empty result means the user previously knew this handle under a
    different identity key: either a legitimate key rotation/succession
    or an active impersonation. The caller surfaces a key-change warning
    and forces re-verification; it must NOT silently re-pin.
    """
    if not handle:
        return []
    canonical_new = normalize_did(new_did_key)
    cur = await conn.execute(
        "SELECT peer_did_key FROM fabric_peer_identities "
        "WHERE user_id=? AND handle=?",
        (user_id, handle),
    )
    rows = await cur.fetchall()
    return [
        r[0] for r in rows
        if not did_equal(r[0], canonical_new)
    ]
