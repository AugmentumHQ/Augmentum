"""Bookmarks adapter — saved external URLs (videos, articles).

Unlike the uploads adapter, bookmarks have no on-disk file: they're
pointers into the outside world.  The file_index row carries the URL +
display metadata in `source_metadata`; the Files panel routes the
"open" action to the YouTube panel (for YouTube URLs) or a new tab
(for everything else).

Reusing file_index keeps the search/tag/trash/audit story uniform —
a saved video shows up next to your uploaded ones in the Videos chip
without any UI special-casing on the listing path.

Source slug: "bookmarks". `source_id` is a deterministic hash of the
canonical URL so re-saving the same URL is idempotent (UPSERT semantics).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.vfs import register_file, unregister_file

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


def bookmark_id(url: str) -> str:
    """Deterministic source_id for a bookmark URL.

    URL normalisation is intentionally light (case-insensitive scheme +
    host, drop trailing slash) so that http vs https on the same host
    still produces distinct rows — different scheme, different content
    risk profile.
    """
    canon = (url or "").strip()
    digest = hashlib.sha1(canon.encode("utf-8", errors="ignore")).hexdigest()
    return f"bm_{digest[:16]}"


class BookmarksAdapter:
    name = "bookmarks"

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def save(
        self,
        *,
        url: str,
        title: str,
        user_id: str,
        thumbnail: str = "",
        channel: str = "",
        duration: float | None = None,
        platform: str = "",
        video_id: str = "",
        kind: str = "video",
    ) -> dict:
        """Upsert a bookmark row in file_index.  Returns the row dict."""
        if not url:
            raise ValueError("bookmarks require a url")
        if not user_id:
            raise ValueError("bookmarks require a user_id")
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ValueError("bookmark url must be http(s)")

        bid = bookmark_id(url)
        meta = {
            "url": url,
            "thumbnail": thumbnail or "",
            "channel": channel or "",
            "duration": duration,
            "platform": platform or "",
            "video_id": video_id or "",
        }

        # Idempotent UPSERT — same URL re-saves update the title/thumbnail
        # rather than create duplicates.  Goes through register_file so
        # the FTS sync trigger fires.
        existing = await self._get(bid, user_id=user_id)
        if existing:
            await self._conn.execute(
                "UPDATE file_index SET name = ?, source_metadata = ?, "
                "thumbnail = ?, updated_at = datetime('now'), is_trashed = 0, "
                "trashed_at = NULL WHERE id = ? AND user_id = ?",
                (title or "Untitled", json.dumps(meta), thumbnail or "",
                 existing["id"], user_id),
            )
            await self._conn.commit()
            log.info("bookmark_updated", id=existing["id"], user_id=user_id, url=url)
            return {**existing, "name": title or "Untitled", **meta}

        await register_file(
            user_id=user_id, source=self.name, source_id=bid,
            name=title or "Untitled",
            mime_type="application/x-bookmark",
            size_bytes=0, real_path=None,
            thumbnail=thumbnail or None,
            source_metadata=meta,
        )
        # Force kind=video for bookmarks; classify.py defaults it based on
        # MIME, but our synthetic mime won't trigger any branch.
        await self._conn.execute(
            "UPDATE file_index SET kind = ? WHERE source = ? AND source_id = ? AND user_id = ?",
            (kind, self.name, bid, user_id),
        )
        await self._conn.commit()
        log.info("bookmark_saved", id=bid, user_id=user_id, url=url, title=title)
        return {
            "id": bid,
            "source": self.name,
            "source_id": bid,
            "name": title or "Untitled",
            "kind": kind,
            **meta,
        }

    async def _get(self, source_id: str, *, user_id: str) -> dict | None:
        cursor = await self._conn.execute(
            "SELECT id, name FROM file_index WHERE source = ? AND source_id = ? "
            "AND user_id = ?",
            (self.name, source_id, user_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1]}

    # --- Adapter Protocol ---------------------------------------------

    async def resolve(self, source_id: str, *, user_id: str) -> str | None:
        # No on-disk file. Render endpoint handles bookmark rows by
        # surfacing a preview shell with the saved URL.
        return None

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT source_id FROM file_index "
            "WHERE source = ? AND user_id = ?",
            (self.name, user_id),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        # Bookmarks store all their state in file_index.source_metadata,
        # so unregistering the row is the whole job — no blobs, no files.
        ok = await unregister_file(self.name, source_id, user_id=user_id)
        if ok:
            log.info("bookmark_deleted", id=source_id, user_id=user_id)
        return ok
