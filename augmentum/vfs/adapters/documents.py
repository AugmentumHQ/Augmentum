"""Documents adapter — ingested RAG documents + their chunks.

Delete path delegates to :meth:`DocumentStore.delete_document`, which
drops the document row, its chunks, vec entries, and the file_index
row in one ownership-checked transaction. Without this adapter,
trashing a document via the Files panel would leave the `documents`
and `document_chunks` tables (and their vec entries) orphaned while
only the file_index row dropped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.sqlite_backend import SQLiteBackend

log = get_logger(__name__)


class DocumentsAdapter:
    name = "documents"

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend
        self._conn = backend.conn

    def _store(self):
        # Lazy construct — DocumentStore is a stateless wrapper over the
        # backend, same trick as ImagesAdapter/ArtifactsAdapter. Keeps the
        # adapter independent of document_store wiring order in server.py.
        from augmentum.documents.store import DocumentStore
        return DocumentStore(self._backend)

    async def resolve(self, source_id: str, *, user_id: str) -> bytes | None:
        # Documents are chunked in the DB with no reassembly path, matching
        # the hardcoded branch in files_routes._resolve_by_source.
        return None

    async def list_source_ids(self, *, user_id: str) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT id FROM documents WHERE user_id = ?", (user_id,),
        )
        return [row[0] for row in await cursor.fetchall()]

    async def delete(self, source_id: str, *, user_id: str) -> bool:
        ok = await self._store().delete_document(source_id, user_id=user_id)
        if ok:
            log.info("document_deleted_via_adapter", id=source_id, user_id=user_id)
        return ok
