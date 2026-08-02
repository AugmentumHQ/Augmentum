"""Cross-device reading-position store.

Backs ``/api/sync/reading-positions`` (see ``sync_routes.py``), the server
side of the Android client's ``SyncRepository``. The browse-history half of
the sync surface reuses ``DiscoveryStore.upsert_history`` instead — it already
owns the multi-tenant ``browse_history`` upsert — so this module is only the
reading-position substrate.

Contract (mirrors the Android wire shapes in ``SyncModels.kt``):

  push: ``upsert_positions([{key, kind, position_fraction, position_detail,
        last_read_ms, device_id, title}], user_id=...)``
        → ``(accepted, rejected, conflicts)``
  pull: ``list_since(user_id=..., since_ms=..., exclude_device_id=...)``
        → ``[{key, kind, position_fraction, position_detail, last_read_ms,
              device_id, title}]``

Conflict resolution is last-write-wins by ``last_read_ms`` (device clock).
``updated_at_ms`` is the SERVER clock — it is the pull cursor and never
trusts the phone's clock.
"""
from __future__ import annotations

import secrets
import time

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class ReadingPositionStore:
    """User-scoped CRUD over the ``reading_positions`` table."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def upsert_positions(
        self,
        positions: list[dict],
        *,
        user_id: str = "",
    ) -> tuple[int, int, list[str]]:
        """Last-write-wins upsert of reading positions.

        Returns ``(accepted, rejected, conflicts)``:
          - accepted: rows written (new, or incoming last_read_ms >= stored)
          - rejected: rows dropped as malformed (missing/blank ``key``)
          - conflicts: keys where the incoming write was STALE (server holds a
            newer last_read_ms); the client should pull to pick up the winner.
        """
        if not user_id:
            raise ValueError("reading_positions upsert requires user_id")

        accepted = 0
        rejected = 0
        conflicts: list[str] = []

        for pos in positions:
            if not isinstance(pos, dict):
                rejected += 1
                continue
            key = str(pos.get("key") or "").strip()
            if not key:
                rejected += 1
                continue

            kind = str(pos.get("kind") or "book")
            frac = max(0.0, min(1.0, _safe_float(pos.get("position_fraction"))))
            detail = _safe_int(pos.get("position_detail"))
            last_read = _safe_int(pos.get("last_read_ms"))
            device_id = str(pos.get("device_id") or "")
            title = str(pos.get("title") or "")

            cursor = await self._conn.execute(
                "SELECT id, last_read_ms FROM reading_positions "
                "WHERE user_id = ? AND sync_key = ?",
                (user_id, key),
            )
            existing = await cursor.fetchone()
            now = _now_ms()

            if existing:
                existing_id, existing_last = existing
                if last_read < _safe_int(existing_last):
                    # Stale write — keep the newer server-side value and flag
                    # the key so the client knows to pull the winner.
                    conflicts.append(key)
                    continue
                # Preserve an existing title when the incoming row omits one.
                await self._conn.execute(
                    """UPDATE reading_positions
                       SET kind = ?, position_fraction = ?, position_detail = ?,
                           last_read_ms = ?, device_id = ?,
                           title = CASE WHEN ? != '' THEN ? ELSE title END,
                           updated_at_ms = ?
                       WHERE id = ?""",
                    (
                        kind, frac, detail, last_read, device_id,
                        title, title, now, existing_id,
                    ),
                )
                accepted += 1
            else:
                await self._conn.execute(
                    """INSERT INTO reading_positions
                       (id, user_id, sync_key, kind, position_fraction,
                        position_detail, last_read_ms, device_id, title,
                        updated_at_ms)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        secrets.token_hex(8), user_id, key, kind, frac,
                        detail, last_read, device_id, title, now,
                    ),
                )
                accepted += 1

        await self._conn.commit()
        return accepted, rejected, conflicts

    async def list_since(
        self,
        *,
        user_id: str = "",
        since_ms: int = 0,
        exclude_device_id: str = "",
        limit: int = 1000,
    ) -> list[dict]:
        """Positions updated (server clock) after ``since_ms``, newest first.

        ``exclude_device_id`` drops the caller's own rows so a device never
        re-pulls what it just pushed — pull is "what did OTHER devices do".
        """
        if not user_id:
            return []

        query = (
            "SELECT sync_key, kind, position_fraction, position_detail, "
            "last_read_ms, device_id, title "
            "FROM reading_positions WHERE user_id = ? AND updated_at_ms > ?"
        )
        params: list = [user_id, _safe_int(since_ms)]
        if exclude_device_id:
            query += " AND device_id != ?"
            params.append(exclude_device_id)
        query += " ORDER BY updated_at_ms DESC LIMIT ?"
        params.append(_safe_int(limit, 1000))

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "key": row[0],
                "kind": row[1],
                "position_fraction": row[2],
                "position_detail": row[3],
                "last_read_ms": row[4],
                "device_id": row[5],
                "title": row[6],
            }
            for row in rows
        ]
