"""Media-server VFS adapter — rows point at external streams, not blobs.

Follows the bookmarks pattern: all state lives in ``file_index`` via
``source_metadata``; ``resolve()`` always returns None because there's
no on-disk file. The streaming proxy (``/api/media/stream/{file_id}``)
takes over for playback and generates a token-bearing upstream URL
from the row's metadata on every request.

One adapter instance handles every media provider (audiobookshelf,
emby, ...). Each provider gets a distinct ``source`` slug — we register
this adapter under all of them so the unified pipeline (trash, search,
source-chip filtering) treats each consistently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.vfs import unregister_file

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class MediaServerAdapter:
    """One adapter object, bound to a specific provider slug."""

    def __init__(self, provider_slug: str, conn: aiosqlite.Connection) -> None:
        self.name = provider_slug
        self._conn = conn

    # --- Adapter Protocol ---------------------------------------------

    async def resolve(self, source_id: str, *, user_id: str) -> str | None:
        # Streams are not filesystem-resolvable. The /api/media/stream
        # route handles playback; the download endpoint deliberately
        # returns nothing for media rows.
        return None

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT source_id FROM file_index "
            "WHERE source = ? AND user_id = ?",
            (self.name, user_id),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        ok = await unregister_file(self.name, source_id, user_id=user_id)
        if ok:
            log.info(
                "media_file_deleted", source=self.name,
                source_id=source_id, user_id=user_id,
            )
        return ok
