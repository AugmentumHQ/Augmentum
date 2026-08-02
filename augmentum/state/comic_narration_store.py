"""Comic ↔ voiced-narration pairing store.

CRUD over the ``comic_narrations`` table (migration 275). One row per
(user, comic_kind, comic_ref): a comic chapter has at most one narration; a
re-record resets the same row. The sibling of :class:`NarrationStore` — same
shape, plus a per-page checkpoint (``processed_pages``) and the ``timeline``
JSON blob that drives the cast pan-and-scan player.

Every method is user-scoped (``*, user_id: str = ""``) per the multi-tenant
invariant.
"""

from __future__ import annotations

import json
import uuid

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_COLS = (
    "id, user_id, comic_kind, comic_ref, narration_artifact_id, timeline, "
    "pages, voice, voice_male, voice_female, voice_cast, engine_id, "
    "reading_direction, status, error, job_id, processed_pages, total_pages, "
    "created_at, updated_at"
)


def _row(cur: aiosqlite.Cursor, row: tuple) -> dict:
    return {d[0]: row[i] for i, d in enumerate(cur.description)}


class ComicNarrationStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, comic_kind: str, comic_ref: str, *, user_id: str = "") -> dict | None:
        q = f"SELECT {_COLS} FROM comic_narrations WHERE comic_kind = ? AND comic_ref = ?"
        params: list = [comic_kind, comic_ref]
        if user_id:
            q += " AND user_id = ?"
            params.append(user_id)
        cur = await self._conn.execute(q, params)
        row = await cur.fetchone()
        return _row(cur, row) if row else None

    async def begin(
        self,
        comic_kind: str,
        comic_ref: str,
        voice: str,
        job_id: str,
        *,
        engine_id: str = "",
        reading_direction: str = "",
        voice_male: str = "",
        voice_female: str = "",
        voice_cast: str = "{}",
        user_id: str,
    ) -> str:
        """Create (or reset) the pairing row to a fresh 'pending' state.

        An empty ``reading_direction`` resolves to the install default rather
        than to left-to-right — the column's own ``DEFAULT 'ltr'`` is a schema
        artifact, not a product decision, and must never be what a row lands on.
        """
        from augmentum.ocr.reading_order import (
            default_reading_direction,
            normalize_reading_direction,
        )

        reading_direction = normalize_reading_direction(
            reading_direction, fallback=default_reading_direction(),
        )
        existing = await self.get(comic_kind, comic_ref, user_id=user_id)
        if existing:
            await self._conn.execute(
                "UPDATE comic_narrations SET voice = ?, voice_male = ?, "
                "voice_female = ?, voice_cast = ?, engine_id = ?, "
                "reading_direction = ?, job_id = ?, status = 'pending', error = '', "
                "narration_artifact_id = '', timeline = '[]', pages = '[]', "
                "processed_pages = 0, total_pages = 0, updated_at = datetime('now') "
                "WHERE id = ?",
                [voice, voice_male, voice_female, voice_cast, engine_id,
                 reading_direction, job_id, existing["id"]],
            )
            await self._conn.commit()
            return existing["id"]
        row_id = uuid.uuid4().hex[:16]
        await self._conn.execute(
            "INSERT INTO comic_narrations "
            "(id, user_id, comic_kind, comic_ref, voice, voice_male, voice_female, "
            "voice_cast, engine_id, reading_direction, job_id, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            [row_id, user_id, comic_kind, comic_ref, voice, voice_male,
             voice_female, voice_cast, engine_id, reading_direction, job_id],
        )
        await self._conn.commit()
        return row_id

    async def _set(self, row_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        await self._conn.execute(
            f"UPDATE comic_narrations SET {cols}, updated_at = datetime('now') WHERE id = ?",
            [*fields.values(), row_id],
        )
        await self._conn.commit()

    async def mark_running(self, row_id: str) -> None:
        await self._set(row_id, status="running", error="")

    async def set_progress(self, row_id: str, processed: int, total: int) -> None:
        """Checkpoint: pages done / total (for resume + UI progress)."""
        await self._set(row_id, processed_pages=int(processed), total_pages=int(total))

    async def set_timeline(self, row_id: str, timeline: list[dict]) -> None:
        """Persist the running bubble timeline (so a partial is castable)."""
        await self._set(row_id, timeline=json.dumps(timeline or []))

    async def set_pages(self, row_id: str, pages: list[dict]) -> None:
        """Persist the per-page streaming list (audio artifact + per-page lines).

        Called after each page's audio is synthesized so the player can begin
        on page 1 while later pages are still rendering.
        """
        await self._set(row_id, pages=json.dumps(pages or []))

    async def mark_done(
        self,
        row_id: str,
        narration_artifact_id: str = "",
        timeline: list[dict] | None = None,
        pages: list[dict] | None = None,
    ) -> None:
        """Mark the narration complete. In the streaming model the per-page
        ``pages`` list is the source of truth; ``narration_artifact_id`` /
        ``timeline`` are kept only for legacy/back-compat callers."""
        fields: dict = {"status": "done", "error": ""}
        if narration_artifact_id:
            fields["narration_artifact_id"] = narration_artifact_id
        if timeline is not None:
            fields["timeline"] = json.dumps(timeline)
        if pages is not None:
            fields["pages"] = json.dumps(pages)
        await self._set(row_id, **fields)

    async def mark_failed(self, row_id: str, error: str) -> None:
        await self._set(row_id, status="failed", error=(error or "")[:500])

    async def list_done(self, *, user_id: str = "") -> list[dict]:
        """All finished narrations for a user, most-recently-updated first.
        Used by the retention pass (prune oldest chapters beyond the cache
        cap after each synthesis)."""
        q = f"SELECT {_COLS} FROM comic_narrations WHERE status = 'done'"
        params: list = []
        if user_id:
            q += " AND user_id = ?"
            params.append(user_id)
        q += " ORDER BY updated_at DESC"
        cur = await self._conn.execute(q, params)
        rows = await cur.fetchall()
        return [_row(cur, r) for r in rows]

    async def delete(self, comic_kind: str, comic_ref: str, *, user_id: str = "") -> bool:
        q = "DELETE FROM comic_narrations WHERE comic_kind = ? AND comic_ref = ?"
        params: list = [comic_kind, comic_ref]
        if user_id:
            q += " AND user_id = ?"
            params.append(user_id)
        cur = await self._conn.execute(q, params)
        await self._conn.commit()
        return cur.rowcount > 0
