"""Artifacts adapter — agentic-mode generated documents / sheets / charts.

Delete path delegates to :meth:`ArtifactStore.delete`, which drops the
row, removes the on-disk file, and unregisters from file_index. Without
this adapter, trashing an artifact via the Files panel would drop the
file_index row on purge but leave the artifact record + disk file
orphaned — the same asymmetric-cascade bug the image adapters fix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


class ArtifactsAdapter:
    name = "artifacts"

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    def _store(self):
        # Lazy import + instantiate: ArtifactStore is a thin wrapper over
        # the shared db conn. Doing this at call time keeps the adapter
        # independent of the image/artifact subsystem wiring order in
        # server.py.
        from augmentum.tools.artifact_storage import ArtifactStore
        return ArtifactStore(self._conn)

    async def resolve(self, source_id: str, *, user_id: str) -> str | None:
        store = self._store()
        info = await store.get(source_id, user_id=user_id)
        if not info or not info.get("path"):
            return None
        full = store.get_file_path(info["path"])
        if not full or not full.is_file():
            return None
        return str(full)

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT id FROM artifacts WHERE user_id = ?", (user_id,),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        ok = await self._store().delete(source_id, user_id=user_id)
        if ok:
            log.info("artifact_deleted_via_adapter", id=source_id, user_id=user_id)
        return ok
