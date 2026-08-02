"""Avatar CRUD operations against SQLite.

Every method takes ``user_id`` so tenants can't see or mutate each other's
avatars. Bundled avatars (``is_bundled = 1``) live under ``user_id IS NULL``
and are visible to every authenticated user as system-provided defaults.
"""
from __future__ import annotations

import json
import time
import hashlib
from typing import Any

import aiosqlite
import structlog

log = structlog.get_logger(__name__)


def _make_id() -> str:
    ts = int(time.time())
    h = hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:8]
    return f"avt_{ts}_{h}"


class AvatarStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def create(
        self,
        vrm_path: str,
        character_id: str | None = None,
        persona_id: str | None = None,
        source_image_id: str | None = None,
        thumbnail_path: str | None = None,
        mannerisms: dict | None = None,
        is_bundled: bool = False,
        avatar_type: str = "vrm",
        segmentation_data: str | None = None,
        *,
        user_id: str = "",
    ) -> dict[str, Any]:
        if not is_bundled and not user_id:
            raise ValueError("avatars insert requires user_id (non-bundled)")
        avatar_id = _make_id()
        mannerisms_json = json.dumps(mannerisms or {})
        # Bundled avatars stay under user_id NULL so every tenant sees them.
        effective_owner = None if is_bundled else user_id
        await self._conn.execute(
            """INSERT INTO avatars (id, character_id, persona_id, source_image_id,
               vrm_path, thumbnail_path, mannerisms, is_bundled, type,
               segmentation_data, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (avatar_id, character_id, persona_id, source_image_id,
             vrm_path, thumbnail_path, mannerisms_json, int(is_bundled),
             avatar_type, segmentation_data, effective_owner),
        )
        await self._conn.commit()
        return await self.get(avatar_id, user_id=user_id)

    async def get(
        self, avatar_id: str, *, user_id: str = "",
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM avatars WHERE id = ?"
        params: list = [avatar_id]
        if user_id:
            # Bundled avatars (user_id IS NULL) are visible to every tenant.
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
        else:
            # No scope: bundled rows only. Never fall through to an
            # unfiltered read — that would return another tenant's private
            # avatar by id. Live callers always pass a real user_id; this
            # is the isolation floor for any future/system caller.
            query += " AND user_id IS NULL"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row))

    async def get_by_character(
        self, character_id: str, *, user_id: str = "",
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM avatars WHERE character_id = ?"
        params: list = [character_id]
        if user_id:
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
        else:
            # No scope: bundled rows only (see get()).
            query += " AND user_id IS NULL"
        query += " ORDER BY created_at DESC LIMIT 1"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row))

    async def get_by_persona(
        self, persona_id: str, *, user_id: str = "",
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM avatars WHERE persona_id = ?"
        params: list = [persona_id]
        if user_id:
            query += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
        else:
            # No scope: bundled rows only (see get()).
            query += " AND user_id IS NULL"
        query += " ORDER BY created_at DESC LIMIT 1"
        cursor = await self._conn.execute(query, params)
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row))

    async def list_all(self, *, user_id: str = "") -> list[dict[str, Any]]:
        query = "SELECT * FROM avatars"
        params: list = []
        if user_id:
            query += " WHERE user_id = ? OR user_id IS NULL"
            params.append(user_id)
        query += " ORDER BY created_at DESC"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in rows]

    async def list_bundled(self) -> list[dict[str, Any]]:
        cursor = await self._conn.execute(
            "SELECT * FROM avatars WHERE is_bundled = 1 ORDER BY id"
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, r)) for r in rows]

    async def delete(self, avatar_id: str, *, user_id: str = "") -> None:
        # Bundled avatars (user_id IS NULL) are server-managed — only the
        # caller's own non-bundled rows are deletable here. Admin tooling
        # that needs to remove bundled rows should hit the DB directly.
        if not user_id:
            raise ValueError("avatars delete requires user_id")
        await self._conn.execute(
            "DELETE FROM avatars WHERE id = ? AND user_id = ?",
            (avatar_id, user_id),
        )
        await self._conn.commit()

    async def update_mannerisms(
        self, avatar_id: str, mannerisms: dict, *, user_id: str = "",
    ) -> None:
        if not user_id:
            raise ValueError("avatars update_mannerisms requires user_id")
        await self._conn.execute(
            "UPDATE avatars SET mannerisms = ?, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (json.dumps(mannerisms), avatar_id, user_id),
        )
        await self._conn.commit()

    async def assign_to_character(
        self, avatar_id: str, character_id: str, *, user_id: str = "",
    ) -> None:
        # OR user_id IS NULL lets a tenant claim a bundled avatar by
        # assigning it to one of their characters — the original use case
        # for the legacy admin-bypass clause.
        if not user_id:
            raise ValueError("avatars assign_to_character requires user_id")
        await self._conn.execute(
            "UPDATE avatars SET character_id = ?, updated_at = datetime('now') "
            "WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (character_id, avatar_id, user_id),
        )
        await self._conn.commit()

    async def assign_to_persona(
        self, avatar_id: str, persona_id: str, *, user_id: str = "",
    ) -> None:
        if not user_id:
            raise ValueError("avatars assign_to_persona requires user_id")
        await self._conn.execute(
            "UPDATE avatars SET persona_id = ?, updated_at = datetime('now') "
            "WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (persona_id, avatar_id, user_id),
        )
        await self._conn.commit()
