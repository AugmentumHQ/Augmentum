"""Images adapter — gallery-generated images stored in image_generations.

Delete path delegates to :class:`ImagePersistence.delete_generation`, which
drops the row, clears `image_cache`, and unregisters from the file index.
This adapter also removes the on-disk PNG so purging a trashed image from
the Files panel matches the blast radius of deleting it from the gallery.

Without this adapter, trashed rows with ``source="images"`` would drop
from `file_index` via the bulk purge but leave `image_generations` (and
the PNG) orphaned — the asymmetric cascade bug.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class ImagesAdapter:
    name = "images"

    def __init__(self, conn: aiosqlite.Connection, output_dir: str = "") -> None:
        self._conn = conn
        self._output_dir = output_dir

    async def resolve(self, source_id: str, *, user_id: str) -> str | None:
        cursor = await self._conn.execute(
            "SELECT file_path FROM image_generations "
            "WHERE image_id = ? AND user_id = ?",
            (source_id, user_id),
        )
        row = await cursor.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            return row[0]
        if self._output_dir:
            fallback = os.path.join(self._output_dir, f"{source_id}.png")
            if os.path.exists(fallback):
                return fallback
        return None

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT image_id FROM image_generations WHERE user_id = ?",
            (user_id,),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        from augmentum.image.persistence import ImagePersistence
        from augmentum.vfs import purge_thumbnails

        persistence = ImagePersistence(self._conn)
        file_path = await persistence.delete_generation(source_id, user_id=user_id)
        if file_path is None:
            return False
        if file_path and os.path.exists(file_path):
            with contextlib.suppress(OSError):
                os.remove(file_path)
        purge_thumbnails(self.name, source_id)
        log.info("image_deleted_via_adapter", id=source_id, user_id=user_id)
        return True
