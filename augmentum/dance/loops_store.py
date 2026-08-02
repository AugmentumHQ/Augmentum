"""SQLite store for curated animation loops (Phase C).

A loop is a named subset of animation ids that constrains the
conductor's selector when active. At most one loop is active per
user, enforced by a partial UNIQUE INDEX on the table.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import aiosqlite
import structlog

log = structlog.get_logger(__name__)


def _make_loop_id() -> str:
    ts = int(time.time())
    h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:10]
    return f"loop_{ts}_{h}"


def _row_to_dict(row: aiosqlite.Row | tuple, cols: list[str]) -> dict[str, Any]:
    d = dict(zip(cols, row))
    try:
        d["animation_ids"] = json.loads(d.get("animation_ids") or "[]")
    except Exception:
        d["animation_ids"] = []
    # SQLite returns the partial-index column as int — normalize to bool
    # at the boundary so the route response shape is consistent JSON.
    d["is_active"] = bool(d.get("is_active"))
    return d


class DanceLoopsStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def create(
        self,
        name: str,
        animation_ids: list[str] | None = None,
        notes: str | None = None,
        *,
        user_id: str = "",
    ) -> dict[str, Any]:
        if not user_id:
            raise ValueError("dance_loops create requires user_id")
        if not name.strip():
            raise ValueError("dance_loops create requires name")
        loop_id = _make_loop_id()
        now = time.time()
        await self._conn.execute(
            "INSERT INTO dance_loops "
            "(id, user_id, name, animation_ids, notes, is_active, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (loop_id, user_id, name.strip(),
             json.dumps(animation_ids or []),
             notes, now, now),
        )
        await self._conn.commit()
        result = await self.get(loop_id, user_id=user_id)
        if result is None:
            raise RuntimeError("dance_loops create: insert disappeared")
        return result

    async def get(
        self, loop_id: str, *, user_id: str = "",
    ) -> dict[str, Any] | None:
        # Isolation floor. dance_loops are inherently per-user (no shared
        # rows); an empty user_id historically dropped the filter and
        # returned any user's loop by id. Every sibling write already
        # raises on empty — match them.
        if not user_id:
            raise ValueError("dance_loops get requires user_id")
        cursor = await self._conn.execute(
            "SELECT * FROM dance_loops WHERE id = ? AND user_id = ?",
            (loop_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return _row_to_dict(row, cols)

    async def list_for_user(
        self, *, user_id: str = "",
    ) -> list[dict[str, Any]]:
        if not user_id:
            return []
        cursor = await self._conn.execute(
            "SELECT * FROM dance_loops WHERE user_id = ? "
            "ORDER BY updated_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [_row_to_dict(r, cols) for r in rows]

    async def get_active(
        self, *, user_id: str = "",
    ) -> dict[str, Any] | None:
        if not user_id:
            return None
        cursor = await self._conn.execute(
            "SELECT * FROM dance_loops WHERE user_id = ? AND is_active = 1 "
            "LIMIT 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return _row_to_dict(row, cols)

    async def update(
        self,
        loop_id: str,
        updates: dict[str, Any],
        *,
        user_id: str = "",
    ) -> dict[str, Any] | None:
        """Partial update of name / animation_ids / notes. is_active is
        managed exclusively via ``set_active`` so callers can't end up
        with two active loops by misuse."""
        if not user_id:
            raise ValueError("dance_loops update requires user_id")
        set_parts: list[str] = []
        params: list[Any] = []
        if "name" in updates:
            new_name = str(updates["name"]).strip()
            if not new_name:
                raise ValueError("dance_loops name cannot be empty")
            set_parts.append("name = ?")
            params.append(new_name)
        if "animation_ids" in updates:
            ids = updates["animation_ids"]
            if not isinstance(ids, list):
                raise ValueError("animation_ids must be a list")
            set_parts.append("animation_ids = ?")
            params.append(json.dumps(ids))
        if "notes" in updates:
            note = updates["notes"]
            set_parts.append("notes = ?")
            params.append(None if note is None else str(note))
        if not set_parts:
            return await self.get(loop_id, user_id=user_id)
        set_parts.append("updated_at = ?")
        params.append(time.time())
        params.extend([loop_id, user_id])
        await self._conn.execute(
            f"UPDATE dance_loops SET {', '.join(set_parts)} "
            "WHERE id = ? AND user_id = ?",
            params,
        )
        await self._conn.commit()
        return await self.get(loop_id, user_id=user_id)

    async def delete(
        self, loop_id: str, *, user_id: str = "",
    ) -> bool:
        if not user_id:
            raise ValueError("dance_loops delete requires user_id")
        cursor = await self._conn.execute(
            "DELETE FROM dance_loops WHERE id = ? AND user_id = ?",
            (loop_id, user_id),
        )
        await self._conn.commit()
        return (cursor.rowcount or 0) > 0

    async def set_active(
        self, loop_id: str | None, *, user_id: str = "",
    ) -> dict[str, Any] | None:
        """Set ``loop_id`` as the active loop for the user. Passing
        ``None`` clears any active loop (= unconstrained atlas).

        The two writes (clear-all-active, then mark the target active)
        are wrapped in a transaction so a crash mid-flight can't leave
        two rows active in violation of the partial unique index.
        """
        if not user_id:
            raise ValueError("dance_loops set_active requires user_id")
        async with self._conn.execute("BEGIN"):
            pass
        try:
            await self._conn.execute(
                "UPDATE dance_loops SET is_active = 0, updated_at = ? "
                "WHERE user_id = ? AND is_active = 1",
                (time.time(), user_id),
            )
            if loop_id is not None:
                await self._conn.execute(
                    "UPDATE dance_loops SET is_active = 1, updated_at = ? "
                    "WHERE id = ? AND user_id = ?",
                    (time.time(), loop_id, user_id),
                )
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise
        if loop_id is None:
            return None
        return await self.get(loop_id, user_id=user_id)
