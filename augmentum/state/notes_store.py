"""Browse notes persistence — individual SQLite rows replacing JSON blob."""

from __future__ import annotations

import json
from datetime import UTC

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class NotesStore:
    """Read/write browse notes as individual SQLite rows.

    Every read/write CRUD requires ``user_id`` so notes stay
    tenant-scoped. The legacy-migration path is the only sanctioned
    consumer of unscoped behaviour; it talks to the DB directly via
    its own delegate calls and surfaces the right ``user_id``.
    """

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def list_stubs(self, *, user_id: str) -> list[dict]:
        """List notes as metadata stubs (no full content) for ``user_id``.

        ``user_id`` is required — passing an empty string raises a
        ``ValueError`` so a route handler that forgets to thread the
        scope fails loudly instead of silently leaking every user's
        notes.
        """
        if not user_id:
            raise ValueError("notes_store.list_stubs requires user_id")
        cursor = await self._conn.execute(
            "SELECT id, title, tags, source_url, source_title, format, "
            "word_count, reading_time_min, origin, pinned, "
            "created_at, updated_at, substr(content, 1, 120) as preview "
            "FROM browse_notes WHERE user_id = ? "
            "ORDER BY pinned DESC, updated_at DESC",
            (user_id,),
        )
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        results = []
        for row in rows:
            d = dict(zip(cols, row))
            d["tags"] = json.loads(d.get("tags", "[]"))
            d["pinned"] = bool(d.get("pinned", 0))
            results.append(d)
        return results

    async def get(self, note_id: str, *, user_id: str) -> dict | None:
        """Get a single note with full content, scoped to the owning user.

        ``user_id`` is required — see :meth:`list_stubs` for rationale.
        """
        if not user_id:
            raise ValueError("notes_store.get requires user_id")
        cursor = await self._conn.execute(
            "SELECT * FROM browse_notes WHERE id = ? AND user_id = ?",
            (note_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cursor.description]
        d = dict(zip(cols, row))
        d["tags"] = json.loads(d.get("tags", "[]"))
        d["ai_blocks"] = json.loads(d.get("ai_blocks", "[]") or "[]")
        d["pinned"] = bool(d.get("pinned", 0))
        return d

    async def create(self, note: dict, *, user_id: str = "") -> dict:
        """Create a new note for ``user_id``. Returns the created note."""
        owner = user_id or note.get("user_id", "")
        if not owner:
            raise ValueError("browse_notes insert requires user_id")
        await self._conn.execute(
            """INSERT INTO browse_notes
               (id, title, content, tags, source_url, source_title,
                format, word_count, reading_time_min, ai_blocks,
                created_at, updated_at, user_id, origin)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                note["id"],
                note.get("title", "Untitled"),
                note.get("content", ""),
                json.dumps(note.get("tags", [])),
                note.get("source_url", ""),
                note.get("source_title", ""),
                note.get("format", "note"),
                int(note.get("word_count", 0) or 0),
                int(note.get("reading_time_min", 0) or 0),
                json.dumps(note.get("ai_blocks", [])),
                note.get("created_at", ""),
                note.get("updated_at", ""),
                owner,
                # Provenance, not silos: '' = user-created, 'companion'
                # when she creates it (note.create verb). Immutable —
                # update() deliberately never touches it.
                note.get("origin", ""),
            ),
        )
        await self._conn.commit()
        return note

    async def update(
        self, note_id: str, updates: dict, *, user_id: str = "",
    ) -> dict | None:
        """Update fields on an existing note. Returns the updated note or None."""
        if not user_id:
            raise ValueError("browse_notes update requires user_id")
        existing = await self.get(note_id, user_id=user_id)
        if not existing:
            return None

        if "title" in updates:
            existing["title"] = updates["title"]
        if "content" in updates:
            existing["content"] = updates["content"]
        if "tags" in updates:
            existing["tags"] = updates["tags"]
        if "format" in updates:
            existing["format"] = updates["format"] or "note"
        if "word_count" in updates:
            existing["word_count"] = int(updates["word_count"] or 0)
        if "reading_time_min" in updates:
            existing["reading_time_min"] = int(updates["reading_time_min"] or 0)
        if "ai_blocks" in updates:
            existing["ai_blocks"] = updates["ai_blocks"] or []
        if "pinned" in updates:
            existing["pinned"] = 1 if updates["pinned"] else 0

        # A pure pin/unpin toggle shouldn't read as "edited" — only bump
        # updated_at when a content-bearing field actually changed, so
        # pinning doesn't reshuffle the recents order.
        content_fields = {
            "title", "content", "tags", "format",
            "word_count", "reading_time_min", "ai_blocks",
        }
        if content_fields & updates.keys():
            from datetime import datetime
            existing["updated_at"] = datetime.now(UTC).isoformat()

        await self._conn.execute(
            "UPDATE browse_notes SET "
            "title = ?, content = ?, tags = ?, "
            "format = ?, word_count = ?, reading_time_min = ?, ai_blocks = ?, "
            "pinned = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (
                existing["title"],
                existing["content"],
                json.dumps(existing["tags"]),
                existing.get("format", "note"),
                int(existing.get("word_count", 0) or 0),
                int(existing.get("reading_time_min", 0) or 0),
                json.dumps(existing.get("ai_blocks", [])),
                1 if existing.get("pinned") else 0,
                existing["updated_at"],
                note_id,
                user_id,
            ),
        )
        await self._conn.commit()
        existing["pinned"] = bool(existing.get("pinned"))
        return existing

    async def delete(self, note_id: str, *, user_id: str = "") -> bool:
        """Delete a note. Returns True if a row was removed."""
        if not user_id:
            raise ValueError("browse_notes delete requires user_id")
        cursor = await self._conn.execute(
            "DELETE FROM browse_notes WHERE id = ? AND user_id = ?",
            (note_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def migrate_from_json(
        self, settings_store, *, user_id: str,
    ) -> int:
        """One-shot migration: move notes from the legacy JSON blob to rows.

        Called once on startup. Legacy blob notes are claimed for
        ``user_id`` (typically the oldest active user / initial admin).
        ``user_id`` must be non-empty — pre-auth migrations are no
        longer supported; if no users exist yet, defer the migration.
        """
        if not user_id:
            log.info("notes_migration_deferred", reason="no_user")
            return 0
        old_key = "browse_notes"
        raw = await settings_store.get(old_key)
        if not raw:
            return 0

        try:
            notes = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return 0

        if not isinstance(notes, list) or not notes:
            return 0

        migrated = 0
        for note in notes:
            if not isinstance(note, dict) or not note.get("id"):
                continue
            try:
                existing = await self.get(note["id"], user_id=user_id)
                if existing:
                    continue
                await self.create(note, user_id=user_id)
                migrated += 1
            except Exception:
                log.debug("note_migration_failed", note_id=note.get("id"), exc_info=True)
                continue

        if migrated > 0:
            await settings_store.set(old_key, None)
            log.info("notes_migrated_to_sqlite", count=migrated, user_id=user_id)

        return migrated
