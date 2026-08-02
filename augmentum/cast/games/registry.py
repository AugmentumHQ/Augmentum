"""SQLite-backed per-(user, title) cast profile store.

Mirrors :class:`augmentum.cast.cast_events.CastEventStore` in shape —
takes a live :class:`aiosqlite.Connection` and exposes async
get/upsert/delete operations.

Multi-tenant: every query joins on ``user_id``. A caller passing an
empty user_id reads/writes the anon row — same convention as the rest
of the codebase. Cross-user reads return None.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from augmentum.cast.games.models import (
    CLASSIFIED_DEFAULT,
    CLASSIFIED_MANUAL,
    CastProfile,
    KeymapProfile,
    STRATEGY_SHIM,
    _coerce_input_chain,
    _coerce_strategy,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


_SELECT_COLS = (
    "user_id, title_id, strategy, embed_url, container_profile_id, "
    "input_chain, keymap_json, quirks_json, classified_by, "
    "classified_at, failed_at, notes"
)


class CastProfileRegistry:
    """Async registry. Single-event-loop access; tolerates concurrent
    writes via the underlying aiosqlite serialisation.
    """

    def __init__(self, conn: "aiosqlite.Connection") -> None:
        self._conn = conn

    # ── reads ────────────────────────────────────────────────────

    async def get(
        self,
        title_id: str,
        *,
        user_id: str = "",
    ) -> CastProfile | None:
        if not title_id:
            return None
        cur = await self._conn.execute(
            f"SELECT {_SELECT_COLS} FROM cast_profiles "
            f"WHERE user_id = ? AND title_id = ?",
            (user_id, title_id),
        )
        try:
            row = await cur.fetchone()
        finally:
            await cur.close()
        if not row:
            return None
        return CastProfile.from_row(row)

    async def list_for_user(self, *, user_id: str = "") -> list[CastProfile]:
        cur = await self._conn.execute(
            f"SELECT {_SELECT_COLS} FROM cast_profiles "
            f"WHERE user_id = ? ORDER BY classified_at DESC",
            (user_id,),
        )
        try:
            rows = await cur.fetchall()
        finally:
            await cur.close()
        return [CastProfile.from_row(r) for r in rows]

    # ── writes ───────────────────────────────────────────────────

    async def upsert(
        self,
        profile: CastProfile,
        *,
        user_id: str = "",
    ) -> CastProfile:
        """Insert or replace ``profile`` for this user. The profile's
        own ``user_id`` is ignored — the explicit kwarg wins, matching
        the rest of the codebase's CRUD-with-user-id convention.

        Returns the persisted profile (with normalised fields).
        """
        if not profile.title_id:
            raise ValueError("CastProfile.title_id is required")
        chain = _coerce_input_chain(profile.input_chain)
        strategy = _coerce_strategy(profile.strategy)
        keymap_json = ""
        if profile.keymap and not profile.keymap.is_empty():
            keymap_json = json.dumps(profile.keymap.to_dict())
        quirks_json = json.dumps(profile.quirks or {})

        await self._conn.execute(
            "INSERT INTO cast_profiles ("
            "user_id, title_id, strategy, embed_url, container_profile_id, "
            "input_chain, keymap_json, quirks_json, "
            "classified_by, classified_at, failed_at, notes"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, title_id) DO UPDATE SET "
            "  strategy = excluded.strategy, "
            "  embed_url = excluded.embed_url, "
            "  container_profile_id = excluded.container_profile_id, "
            "  input_chain = excluded.input_chain, "
            "  keymap_json = excluded.keymap_json, "
            "  quirks_json = excluded.quirks_json, "
            "  classified_by = excluded.classified_by, "
            "  classified_at = excluded.classified_at, "
            "  failed_at = excluded.failed_at, "
            "  notes = excluded.notes",
            (
                user_id, profile.title_id, strategy, profile.embed_url,
                profile.container_profile_id,
                json.dumps(list(chain)),
                keymap_json, quirks_json,
                profile.classified_by or CLASSIFIED_DEFAULT,
                float(profile.classified_at or time.time()),
                float(profile.failed_at or 0.0),
                profile.notes or "",
            ),
        )
        await self._conn.commit()

        persisted = await self.get(profile.title_id, user_id=user_id)
        # get() never returns None right after a successful upsert under
        # this connection's serialised access pattern; the cast keeps
        # the type checker happy without adding a runtime branch.
        return persisted  # type: ignore[return-value]

    async def override(
        self,
        title_id: str,
        *,
        user_id: str = "",
        **fields: Any,
    ) -> CastProfile:
        """Manual override path — merges ``fields`` onto the existing
        profile (or a fresh default if none exists) and persists with
        ``classified_by='manual'``. Returns the resulting profile.

        Unknown adapter ids in ``input_chain`` are dropped; unknown
        strategies fall back to 'shim'.
        """
        existing = await self.get(title_id, user_id=user_id)
        base = existing or CastProfile(
            title_id=title_id,
            user_id=user_id,
            strategy=STRATEGY_SHIM,
        )
        # Normalise keymap field if a dict was passed (route handler
        # convenience — JSON in, dataclass internally).
        if "keymap" in fields and isinstance(fields["keymap"], dict):
            fields["keymap"] = KeymapProfile.from_dict(fields["keymap"])
        merged = base.merge_fields(
            **fields,
            classified_by=CLASSIFIED_MANUAL,
            classified_at=time.time(),
        )
        return await self.upsert(merged, user_id=user_id)

    async def mark_failed(
        self,
        title_id: str,
        *,
        user_id: str = "",
        when: float | None = None,
    ) -> None:
        """Stamp ``failed_at`` so the classifier promotes on next cast.
        No-op when no row exists (nothing to demote)."""
        await self._conn.execute(
            "UPDATE cast_profiles SET failed_at = ? "
            "WHERE user_id = ? AND title_id = ?",
            (float(when or time.time()), user_id, title_id),
        )
        await self._conn.commit()

    async def delete(self, title_id: str, *, user_id: str = "") -> bool:
        """Remove the profile so the next cast falls back to defaults.
        Returns True iff a row existed."""
        cur = await self._conn.execute(
            "DELETE FROM cast_profiles WHERE user_id = ? AND title_id = ?",
            (user_id, title_id),
        )
        await self._conn.commit()
        return (cur.rowcount or 0) > 0
