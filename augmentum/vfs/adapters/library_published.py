"""Library publications adapter — coder-built artifacts saved to Library.

Surfaces publications in the unified file index so they show up in the
Files panel alongside uploads / documents / artifacts. Resolve returns
the publication's entry file (the same one the launcher loads) so a
quick-look preview shows the playable artifact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

    from augmentum.library.publications import PublicationStore


log = get_logger(__name__)


class LibraryPublishedAdapter:
    name = "library_published"

    def __init__(
        self,
        conn: aiosqlite.Connection,
        store: PublicationStore,
    ) -> None:
        self._conn = conn
        self._store = store

    async def resolve(self, source_id: str, *, user_id: str) -> str | None:
        row = await self._store.get(source_id, user_id=user_id)
        if not row:
            return None
        entry_point = row.get("entry_point", "index.html")
        fs_path = self._store.storage.asset_path(
            user_id=user_id, publication_id=source_id, rel_path=entry_point,
        )
        return str(fs_path) if fs_path else None

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT id FROM library_publications WHERE user_id = ?",
            (user_id,),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        ok = await self._store.delete(source_id, user_id=user_id)
        if ok:
            log.info(
                "library_publication_deleted_via_adapter",
                id=source_id, user_id=user_id,
            )
        return ok
