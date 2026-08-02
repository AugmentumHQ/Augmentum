"""Toy multi-tenant note-store.

Augmentum-flavored: the project enforces ``user_id`` scoping on every
user-scoped table (see CLAUDE.md). A query that omits the filter leaks
data between tenants.
"""

from __future__ import annotations


class NoteStore:
    def __init__(self, conn) -> None:
        self._conn = conn

    async def get_note(self, note_id: str, *, user_id: str = "") -> dict | None:
        cur = await self._conn.execute(
            # BUG: no `AND user_id = ?` filter. User A can fetch User B's
            # note simply by knowing the note_id. The user_id parameter is
            # accepted but never used.
            "SELECT id, title, body FROM notes WHERE id = ?",
            (note_id,),
        )
        row = await cur.fetchone()
        return dict(row) if row else None
