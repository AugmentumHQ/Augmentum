"""Durable anti-replay state for the live federated path (SEC-7/SEC-8).

The in-memory ``relay_seal.ReplayWindow`` and ``pow.ConsumedNonces`` are
fine for a single process lifetime, but a restart reopens both windows —
a real production gap the security review flagged. These functions back
the same guarantees with ``fabric_replay_watermarks`` /
``fabric_consumed_nonces`` (migration 292) so replay protection survives
restarts.

The live inbound path uses THESE, not the in-memory classes. The
in-memory classes remain for unit tests and ephemeral contexts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


async def check_and_advance_seq(
    conn: aiosqlite.Connection,
    *,
    source_did: str,
    seq: int,
    owner_id: str = "",
) -> bool:
    """Durable per-(owner, source) monotonic replay guard.

    Returns True if ``seq`` is fresh (strictly greater than the stored
    high-water for this owner+source) and atomically records it; False if
    it's a replay/stale. Survives restarts.

    The UPSERT only advances the high-water when the new seq is greater,
    so concurrent/duplicate frames can't lower it. We read-then-write
    inside the same connection; the caller serialises per-connection.
    """
    cur = await conn.execute(
        "SELECT high_seq FROM fabric_replay_watermarks "
        "WHERE owner_id=? AND source_did=?",
        (owner_id, source_did),
    )
    row = await cur.fetchone()
    if row is not None and seq <= int(row[0]):
        return False
    await conn.execute(
        "INSERT INTO fabric_replay_watermarks (owner_id, source_did, high_seq) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(owner_id, source_did) DO UPDATE SET "
        "high_seq=excluded.high_seq, updated_at=datetime('now') "
        "WHERE excluded.high_seq > fabric_replay_watermarks.high_seq",
        (owner_id, source_did, int(seq)),
    )
    await conn.commit()
    return True


async def spend_nonce(
    conn: aiosqlite.Connection,
    *,
    nonce: str,
    expires_at: int,
) -> bool:
    """Durable single-use PoW nonce guard.

    Returns True and records the nonce if fresh; False if already spent.
    Uses INSERT … the PK conflict makes the second insert a no-op, and we
    detect freshness by the affected row count — atomic, no read-modify
    race.
    """
    cur = await conn.execute(
        "INSERT OR IGNORE INTO fabric_consumed_nonces (nonce, expires_at) "
        "VALUES (?, ?)",
        (nonce, int(expires_at)),
    )
    await conn.commit()
    return cur.rowcount > 0


async def prune_expired_nonces(conn: aiosqlite.Connection, *, now: int) -> int:
    """Delete consumed nonces whose challenge TTL has elapsed (they can
    never be presented again). Returns rows removed. Call periodically."""
    cur = await conn.execute(
        "DELETE FROM fabric_consumed_nonces WHERE expires_at > 0 AND expires_at < ?",
        (int(now),),
    )
    await conn.commit()
    removed = cur.rowcount
    if removed:
        log.info("fabric_consumed_nonces_pruned", removed=removed)
    return removed
