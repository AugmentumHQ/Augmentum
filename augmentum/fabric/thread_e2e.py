"""Per-thread E2E state (Connect E2E P3).

Tiny store over ``connect_thread_e2e``: whether a user has end-to-end
encryption on for a given thread, and the verified peer master key the
client seals to. Absence = host-trusted (the default, untouched path).

The server stores/forwards E2E message bodies opaquely (they're sealed
JSON); this flag is what tells the client to seal on send + decrypt on
receive, and what lets surfaces skip content-features (notification
preview, search) for a thread whose content they can't read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


@dataclass(frozen=True)
class ThreadE2E:
    enabled: bool
    peer_master_did: str


async def set_e2e(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    thread_id: str,
    enabled: bool,
    peer_master_did: str = "",
) -> ThreadE2E:
    """Enable/disable E2E for ``user_id``'s copy of ``thread_id``.

    Records the peer master the client will seal to (and refuse to seal if
    a fetched bundle's master differs). Raises on an empty user/thread."""
    if not user_id or not thread_id:
        raise ValueError("set_e2e requires user_id and thread_id")
    await conn.execute(
        "INSERT INTO connect_thread_e2e (thread_id, user_id, enabled, peer_master_did, updated_at) "
        "VALUES (?, ?, ?, ?, datetime('now')) "
        "ON CONFLICT(thread_id, user_id) DO UPDATE SET "
        "enabled=excluded.enabled, peer_master_did=excluded.peer_master_did, "
        "updated_at=datetime('now')",
        (thread_id, user_id, 1 if enabled else 0, peer_master_did),
    )
    await conn.commit()
    log.info("connect_thread_e2e_set", user_id=user_id, thread_id=thread_id, enabled=enabled)
    return ThreadE2E(enabled=enabled, peer_master_did=peer_master_did)


async def get_e2e(
    conn: aiosqlite.Connection, *, user_id: str, thread_id: str,
) -> ThreadE2E:
    """Return the E2E state for a thread (absent => disabled, host-trusted)."""
    cur = await conn.execute(
        "SELECT enabled, peer_master_did FROM connect_thread_e2e "
        "WHERE thread_id=? AND user_id=?",
        (thread_id, user_id),
    )
    row = await cur.fetchone()
    if row is None:
        return ThreadE2E(enabled=False, peer_master_did="")
    return ThreadE2E(enabled=bool(row[0]), peer_master_did=row[1])


async def is_e2e(
    conn: aiosqlite.Connection, *, user_id: str, thread_id: str,
) -> bool:
    """True iff this user has E2E on for the thread. Use to skip
    content-features (notification preview, search) on unreadable threads."""
    state = await get_e2e(conn, user_id=user_id, thread_id=thread_id)
    return state.enabled
