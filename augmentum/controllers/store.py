"""ControllerStore -- per-(user, system) remap CRUD over controller_remaps.

Small JSON blob per row; reads + writes are upsert-on-PK to keep the
caller-side merge logic minimal. The schema of bindings_json is
enforced at the service layer (defaults.py knows the action ids per
system); the store stays format-agnostic so adding a new system
doesn't require migrations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


VALID_PAD_ROUTING: frozenset[str] = frozenset({"index", "firstpress"})


@dataclass(frozen=True)
class ControllerRemap:
    user_id: str
    system_id: str
    bindings: dict[str, Any]            # partial override; merge-on-read with defaults
    pad_routing: str                    # 'index' or 'firstpress'
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "bindings": dict(self.bindings),
            "pad_routing": self.pad_routing,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ControllerStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(
        self, *, user_id: str, system_id: str,
    ) -> ControllerRemap | None:
        if not user_id or not system_id:
            return None
        cursor = await self._conn.execute(
            "SELECT * FROM controller_remaps "
            "WHERE user_id = ? AND system_id = ?",
            (user_id, system_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        return _row_to_remap(dict(zip(cols, row)))

    async def list_for_user(
        self, *, user_id: str,
    ) -> list[ControllerRemap]:
        if not user_id:
            return []
        cursor = await self._conn.execute(
            "SELECT * FROM controller_remaps "
            "WHERE user_id = ? ORDER BY system_id",
            (user_id,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [_row_to_remap(dict(zip(cols, r))) for r in rows]

    async def upsert(
        self,
        *,
        user_id: str,
        system_id: str,
        bindings: dict[str, Any] | None = None,
        pad_routing: str | None = None,
    ) -> ControllerRemap:
        """Upsert a remap. Empty bindings = "use defaults"; pass None for
        either field to keep the existing value (or default if no row).
        """
        if not user_id or not system_id:
            raise ValueError("user_id and system_id required")

        existing = await self.get(user_id=user_id, system_id=system_id)
        new_bindings = bindings if bindings is not None else (
            existing.bindings if existing else {}
        )
        new_pad_routing = pad_routing if pad_routing is not None else (
            existing.pad_routing if existing else "index"
        )
        if new_pad_routing not in VALID_PAD_ROUTING:
            raise ValueError(
                f"unknown pad_routing {new_pad_routing!r} "
                f"(known: {sorted(VALID_PAD_ROUTING)})"
            )

        # PRIMARY KEY (user_id, system_id) so ON CONFLICT keeps the
        # row tidy.
        await self._conn.execute(
            """INSERT INTO controller_remaps
               (user_id, system_id, bindings_json, pad_routing)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, system_id) DO UPDATE SET
                 bindings_json = excluded.bindings_json,
                 pad_routing = excluded.pad_routing,
                 updated_at = datetime('now')""",
            (
                user_id, system_id,
                json.dumps(new_bindings or {}),
                new_pad_routing,
            ),
        )
        await self._conn.commit()
        result = await self.get(user_id=user_id, system_id=system_id)
        # Just upserted -- assert for clarity
        assert result is not None
        return result

    async def delete(
        self, *, user_id: str, system_id: str,
    ) -> bool:
        if not user_id or not system_id:
            return False
        cursor = await self._conn.execute(
            "DELETE FROM controller_remaps "
            "WHERE user_id = ? AND system_id = ?",
            (user_id, system_id),
        )
        await self._conn.commit()
        return (cursor.rowcount or 0) > 0


def _row_to_remap(row: dict) -> ControllerRemap:
    raw = row.get("bindings_json") or "{}"
    bindings: dict = {}
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                bindings = decoded
        except json.JSONDecodeError:
            log.warning(
                "controller_remap_bad_json",
                user_id=row.get("user_id"),
                system_id=row.get("system_id"),
            )
    return ControllerRemap(
        user_id=str(row.get("user_id", "")),
        system_id=str(row.get("system_id", "")),
        bindings=bindings,
        pad_routing=str(row.get("pad_routing", "index")),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )
