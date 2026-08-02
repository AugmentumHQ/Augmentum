"""Chat-images adapter — VL attachment blobs stored inline in chat_images.

Chat image bytes live in the DB (no on-disk file), so delete is just a
row drop plus the usual file_index unregister. Mirrors the cascade
performed by ``DELETE /api/chat-images/{id}``; without this adapter,
purging a trashed chat image from the Files panel would leave
`chat_images` orphaned.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.vfs import purge_thumbnails, unregister_file

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class ChatImagesAdapter:
    name = "chat_images"

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def resolve(self, source_id: str, *, user_id: str) -> bytes | None:
        cursor = await self._conn.execute(
            "SELECT data FROM chat_images WHERE id = ? AND user_id = ?",
            (source_id, user_id),
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return None
        return row[0]

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT id FROM chat_images WHERE user_id = ?", (user_id,),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        cursor = await self._conn.execute(
            "DELETE FROM chat_images WHERE id = ? AND user_id = ?",
            (source_id, user_id),
        )
        await self._conn.commit()
        if not cursor.rowcount:
            return False
        await unregister_file(self.name, source_id, user_id=user_id)
        purge_thumbnails(self.name, source_id)
        log.info("chat_image_deleted_via_adapter", id=source_id, user_id=user_id)
        return True
