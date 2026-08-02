"""EPUB ↔ TTS-narration pairing store.

Tiny CRUD over the ``epub_narrations`` table (migration 148). One row per
(user, epub_kind, epub_ref): an EPUB can have at most one narration; a
re-record resets the same row. ``epub_kind`` is ``'artifact'`` (a Studio
artifact id) or ``'file'`` (a file-index row id).
"""

from __future__ import annotations

import uuid

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_COLS = (
    "id, user_id, epub_kind, epub_ref, narration_artifact_id, voice, "
    "status, error, job_id, processed_chunks, total_chunks, created_at, updated_at"
)


def _row(cur: aiosqlite.Cursor, row: tuple) -> dict:
    return {d[0]: row[i] for i, d in enumerate(cur.description)}


class NarrationStore:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def get(self, epub_kind: str, epub_ref: str, *, user_id: str = "") -> dict | None:
        q = f"SELECT {_COLS} FROM epub_narrations WHERE epub_kind = ? AND epub_ref = ?"
        params: list = [epub_kind, epub_ref]
        if user_id:
            q += " AND user_id = ?"
            params.append(user_id)
        cur = await self._conn.execute(q, params)
        row = await cur.fetchone()
        return _row(cur, row) if row else None

    async def begin(
        self, epub_kind: str, epub_ref: str, voice: str, job_id: str, *, user_id: str,
    ) -> str:
        """Create (or reset) the pairing row to a fresh 'pending' state.

        Returns the row id. Used right before enqueuing the synth job so
        the GET endpoint can report status immediately.
        """
        existing = await self.get(epub_kind, epub_ref, user_id=user_id)
        if existing:
            # Reset to a fresh run — including the checkpoint, so the synth
            # job re-derives the chunk plan from scratch rather than resuming
            # a partial that may no longer match.
            await self._conn.execute(
                "UPDATE epub_narrations SET voice = ?, job_id = ?, status = 'pending', "
                "error = '', narration_artifact_id = '', processed_chunks = 0, "
                "total_chunks = 0, updated_at = datetime('now') WHERE id = ?",
                [voice, job_id, existing["id"]],
            )
            await self._conn.commit()
            return existing["id"]
        row_id = uuid.uuid4().hex[:16]
        await self._conn.execute(
            "INSERT INTO epub_narrations "
            "(id, user_id, epub_kind, epub_ref, voice, job_id, status) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
            [row_id, user_id, epub_kind, epub_ref, voice, job_id],
        )
        await self._conn.commit()
        return row_id

    async def _set(self, row_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        await self._conn.execute(
            f"UPDATE epub_narrations SET {cols}, updated_at = datetime('now') WHERE id = ?",
            [*fields.values(), row_id],
        )
        await self._conn.commit()

    async def mark_running(self, row_id: str) -> None:
        await self._set(row_id, status="running", error="")

    async def set_progress(self, row_id: str, processed: int, total: int) -> None:
        """Checkpoint: how many chunks of how many are done (for resume + UI)."""
        await self._set(row_id, processed_chunks=int(processed), total_chunks=int(total))

    async def mark_done(self, row_id: str, narration_artifact_id: str) -> None:
        await self._set(row_id, status="done", narration_artifact_id=narration_artifact_id, error="")

    async def mark_failed(self, row_id: str, error: str) -> None:
        await self._set(row_id, status="failed", error=(error or "")[:500])
