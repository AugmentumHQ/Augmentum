"""Document store — CRUD and retrieval for document RAG chunks."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from augmentum.documents.chunker import chunk_with_parents, extract_text
from augmentum.memory.embeddings import EmbeddingService
from augmentum.utils.fts import tokenize_fts_query as _tokenize_fts_query
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


class DocumentStore:
    """Document storage and retrieval backed by SQLite + vec0 + FTS5."""

    def __init__(self, backend: SQLiteBackend) -> None:
        self._backend = backend
        self._conn = backend.conn
        self._vec_enabled = backend.vec_enabled
        self._last_search_dual_source: bool = True  # updated each search() call

    async def ingest(
        self,
        data: bytes,
        filename: str,
        mime_type: str,
        user_id: str = "default",
        scope: str | None = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        backend: object | None = None,
    ) -> dict:
        """Extract text, chunk, embed, and store a document.

        When ``backend`` is provided and contextual retrieval is enabled,
        an LLM generates short context preambles for each chunk before
        embedding — Anthropic's contextual retrieval pattern.

        Returns metadata dict with document id, chunk count, and filename.
        """
        from augmentum.config import settings

        # Extract text pages
        pages = extract_text(data, mime_type, filename)
        if not pages:
            raise ValueError(f"Could not extract text from {filename}")

        # Create parent-child chunk pairs for precision search + context retrieval
        child_chunks, parent_chunks = chunk_with_parents(
            pages, child_size=chunk_size, chunk_overlap=chunk_overlap, filename=filename,
        )
        if not child_chunks:
            raise ValueError(f"No content to chunk from {filename}")

        doc_id = uuid.uuid4().hex

        # Insert document record
        await self._conn.execute(
            "INSERT INTO documents (id, user_id, filename, mime_type, file_size, chunk_count, scope) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, user_id, filename, mime_type, len(data), len(child_chunks), scope),
        )

        # Store parent chunks (large context windows, not vectorized)
        parent_id_map: dict[int, str] = {}
        for parent in parent_chunks:
            pid = uuid.uuid4().hex
            parent_id_map[parent.index] = pid
            await self._conn.execute(
                "INSERT INTO document_chunks "
                "(id, document_id, chunk_index, content, page_num, char_offset, token_count, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    pid, doc_id, -(parent.index + 1),  # negative index = parent chunk
                    parent.text, parent.page_num, parent.char_offset,
                    len(parent.text) // 4,
                ),
            )

        # Build embed texts — enriched with filename/section headers
        embed_texts = [c.enriched_text or c.text for c in child_chunks]

        # Contextual retrieval: LLM-generated chunk context (Anthropic pattern)
        if settings.document_rag_contextual_retrieval and backend is not None:
            from augmentum.documents.contextual import generate_chunk_contexts, prepend_context

            full_text = "\n\n".join(text for text, _ in pages)
            raw_texts = [c.text for c in child_chunks]
            contexts = await generate_chunk_contexts(raw_texts, full_text, backend)
            embed_texts = [
                prepend_context(et, ctx)
                for et, ctx in zip(embed_texts, contexts, strict=False)
            ]

        embeddings = await asyncio.to_thread(EmbeddingService.embed, embed_texts)

        # Insert child chunks with parent_id reference
        for chunk, embedding in zip(child_chunks, embeddings, strict=False):
            chunk_id = uuid.uuid4().hex
            blob = EmbeddingService.to_blob(embedding)

            # Resolve parent ID
            parent_db_id = parent_id_map.get(chunk.parent_index) if chunk.parent_index is not None else None

            await self._conn.execute(
                "INSERT INTO document_chunks "
                "(id, document_id, chunk_index, content, page_num, char_offset, token_count, embedding, parent_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk_id, doc_id, chunk.index, chunk.text,
                    chunk.page_num, chunk.char_offset,
                    len(chunk.text) // 4,
                    blob, parent_db_id,
                ),
            )

            # Insert into vec table (child chunks only — used for search)
            if self._vec_enabled:
                try:
                    await self._conn.execute(
                        "INSERT INTO doc_chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                        (chunk_id, blob),
                    )
                except Exception:
                    log.warning("doc_vec_insert_failed", chunk_id=chunk_id, exc_info=True)

        await self._conn.commit()

        from augmentum.vfs import register_file
        await register_file(
            user_id=user_id, source="documents", source_id=doc_id,
            name=filename, mime_type=mime_type, size_bytes=len(data),
            description=f"{filename} ({len(child_chunks)} chunks)",
            source_metadata={"chunk_count": len(child_chunks), "scope": scope or ""},
        )

        log.info("document_ingested", doc_id=doc_id, filename=filename,
                 children=len(child_chunks), parents=len(parent_chunks), size=len(data))

        return {
            "id": doc_id,
            "filename": filename,
            "mime_type": mime_type,
            "chunk_count": len(child_chunks),
            "file_size": len(data),
        }

    async def search(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
        scope: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """Hybrid search: vector + FTS5 with RRF merge + optional reranking.

        Returns list of dicts with chunk content, document filename, page, and score.
        """
        from augmentum.config import settings

        # Widen candidate pool when reranking is enabled (fetch more, filter later)
        candidate_multiplier = 10 if settings.reranker_enabled else 2
        candidate_limit = limit * candidate_multiplier

        vec_results = await self._vector_search(
            query, user_id, limit=candidate_limit, scope=scope, document_id=document_id,
        )
        fts_results = await self._fts_search(
            query, user_id, limit=candidate_limit, scope=scope, document_id=document_id,
        )

        self._last_search_dual_source = bool(vec_results) and bool(fts_results)

        # RRF merge
        merged = self._rrf_merge(vec_results, fts_results, k=60)

        # Cross-encoder reranking for precision. Off-loaded to a thread —
        # the cross-encoder is CPU/GPU-heavy and was stalling the loop.
        if settings.reranker_enabled and merged:
            merged = await asyncio.to_thread(
                self._rerank_results, query, merged, limit,
            )
        else:
            merged = merged[:limit]

        return merged

    @staticmethod
    def _rerank_results(
        query: str, results: list[dict], top_k: int,
    ) -> list[dict]:
        """Rerank RRF results with cross-encoder for higher precision."""
        from augmentum.memory.reranker import RerankService

        try:
            return RerankService.rerank_dicts(
                query, results, content_key="content", top_k=top_k,
            )
        except Exception:
            log.debug("doc_rerank_failed_using_rrf_order", exc_info=True)
            return results[:top_k]

    async def search_for_recall(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
        scope: str | None = None,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        """Search optimized for injection into recall pipeline.

        Returns matched child chunks directly — parent expansion is disabled
        because testing showed it dilutes precision on narrative content
        (46% → 75% without parents) while making zero difference on
        structured content (95% either way).

        When ``document_ids`` is provided, only searches within those documents.

        Returns dicts with 'content', 'source' label, and 'score'.
        """
        # When scoped to specific documents, search each and merge
        if document_ids:
            all_results: list[dict] = []
            for doc_id in document_ids:
                results = await self.search(
                    query, user_id=user_id, limit=limit, scope=scope,
                    document_id=doc_id,
                )
                all_results.extend(results)
            # Sort by score and take top N
            all_results.sort(key=lambda r: r.get("score", 0), reverse=True)
            results = all_results[:limit]
        else:
            results = await self.search(query, user_id=user_id, limit=limit, scope=scope)

        enriched: list[dict] = []

        for r in results:
            enriched.append({
                "content": r["content"],
                "source": f"[Document: {r['filename']}"
                          + (f" p.{r['page_num']}" if r.get("page_num") else "")
                          + "]",
                "score": r["score"],
            })

        return enriched

    async def get_full_content(self, doc_id: str, *, user_id: str) -> dict | None:
        """Get the full concatenated content of a document.

        Returns {filename, content, page_count} or None.
        """
        try:
            # Get document metadata (scoped to the owning user)
            cursor = await self._conn.execute(
                "SELECT filename FROM documents WHERE id = ? AND user_id = ?",
                (doc_id, user_id),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            filename = row["filename"]

            # Get all child chunks in order (positive chunk_index)
            # document_chunks.document_id was ownership-gated above, so the
            # chunks read is implicitly scoped.
            cursor = await self._conn.execute(
                "SELECT content, page_num FROM document_chunks "
                "WHERE document_id = ? AND chunk_index >= 0 "
                "ORDER BY chunk_index",
                (doc_id,),
            )
            rows = await cursor.fetchall()
            if not rows:
                return None

            content = "\n".join(r["content"] for r in rows)
            page_nums = {r["page_num"] for r in rows if r["page_num"]}

            return {
                "filename": filename,
                "content": content,
                "page_count": len(page_nums) if page_nums else 1,
            }
        except Exception:
            log.debug("get_full_content_failed", doc_id=doc_id, exc_info=True)
            return None

    async def _get_parent_content(
        self, doc_id: str, child_chunk_index: int, child_chunk_id: str | None = None,
    ) -> str | None:
        """Look up the parent chunk for a given child chunk.

        Uses the parent_id FK when a child_chunk_id is available (preferred).
        Falls back to mapping child_chunk_index to the closest parent chunk
        via the parent_index relationship stored during ingestion.
        """
        try:
            # Preferred: follow the parent_id FK from the child chunk
            if child_chunk_id:
                cursor = await self._conn.execute(
                    "SELECT p.content FROM document_chunks c "
                    "JOIN document_chunks p ON p.id = c.parent_id "
                    "WHERE c.id = ? AND c.parent_id IS NOT NULL",
                    (child_chunk_id,),
                )
                row = await cursor.fetchone()
                if row:
                    return row["content"]

            # Fallback: find the parent chunk that covers this child index
            # Parent chunks have negative chunk_index = -(parent_index+1)
            # Child chunks reference parent_index during ingestion
            cursor = await self._conn.execute(
                "SELECT p.content FROM document_chunks c "
                "JOIN document_chunks p ON p.id = c.parent_id "
                "WHERE c.document_id = ? AND c.chunk_index = ? AND c.parent_id IS NOT NULL",
                (doc_id, child_chunk_index),
            )
            row = await cursor.fetchone()
            if row:
                return row["content"]
        except Exception:
            log.debug("parent_chunk_lookup_failed", exc_info=True)
        return None

    async def list_documents(self, user_id: str = "default", scope: str | None = None) -> list[dict]:
        """List all documents for a user."""
        sql = "SELECT id, filename, mime_type, file_size, chunk_count, scope, created_at FROM documents WHERE user_id = ?"
        params: list = [user_id]
        if scope is not None:
            sql += " AND (scope = ? OR scope IS NULL)"
            params.append(scope)
        sql += " ORDER BY created_at DESC"

        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def delete_document(self, doc_id: str, *, user_id: str) -> bool:
        """Delete a document (and its chunks) if owned by ``user_id``."""
        # Verify ownership BEFORE any cascade — otherwise a non-owner could
        # wipe the vec entries via the subquery even if the documents DELETE
        # is ultimately rejected.
        try:
            cur = await self._conn.execute(
                "SELECT user_id FROM documents WHERE id = ? AND user_id = ?",
                (doc_id, user_id),
            )
            row = await cur.fetchone()
        except Exception:
            row = None
        if not row:
            return False

        # Remove vec entries first
        if self._vec_enabled:
            try:
                await self._conn.execute(
                    "DELETE FROM doc_chunks_vec WHERE chunk_id IN "
                    "(SELECT id FROM document_chunks WHERE document_id = ?)",
                    (doc_id,),
                )
            except Exception:
                log.debug("doc_vec_delete_failed", doc_id=doc_id, exc_info=True)

        cursor = await self._conn.execute(
            "DELETE FROM documents WHERE id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        await self._conn.commit()

        # Cascade into file_index so the files panel doesn't strand the row
        if cursor.rowcount > 0:
            from augmentum.vfs import unregister_file
            await unregister_file("documents", doc_id, user_id=user_id)

        return cursor.rowcount > 0

    async def get_document(self, doc_id: str, *, user_id: str) -> dict | None:
        """Get document metadata by ID, scoped to the owning user."""
        cursor = await self._conn.execute(
            "SELECT id, user_id, filename, mime_type, file_size, chunk_count, scope, created_at "
            "FROM documents WHERE id = ? AND user_id = ?",
            (doc_id, user_id),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Internal search methods
    # ------------------------------------------------------------------

    async def _vector_search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        scope: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """Vector similarity search via vec0."""
        if not self._vec_enabled:
            return []

        embedding = await asyncio.to_thread(EmbeddingService.embed_query, query)
        blob = EmbeddingService.to_blob(embedding)

        try:
            cursor = await self._conn.execute(
                "SELECT v.chunk_id, v.distance, c.content, c.page_num, c.chunk_index, "
                "d.filename, d.id as doc_id "
                "FROM doc_chunks_vec v "
                "JOIN document_chunks c ON c.id = v.chunk_id "
                "JOIN documents d ON d.id = c.document_id "
                "WHERE v.embedding MATCH ? AND k = ? "
                "AND d.user_id = ? "
                + ("AND d.scope = ? " if scope else "")
                + ("AND d.id = ? " if document_id else ""),
                (blob, limit, user_id)
                + ((scope,) if scope else ())
                + ((document_id,) if document_id else ()),
            )
            rows = await cursor.fetchall()
        except Exception:
            log.debug("doc_vector_search_failed", exc_info=True)
            return []

        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "filename": row["filename"],
                "doc_id": row["doc_id"],
                "page_num": row["page_num"],
                "chunk_index": row["chunk_index"],
                "distance": row["distance"],
            }
            for row in rows
        ]

    async def _fts_search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        scope: str | None = None,
        document_id: str | None = None,
    ) -> list[dict]:
        """Full-text keyword search via FTS5 with AND-first/OR-fallback."""
        tokenized = _tokenize_fts_query(query)

        # Determine expression(s) to try
        if isinstance(tokenized, tuple):
            expressions = [tokenized[0], tokenized[1]]  # AND first, OR fallback
        else:
            expressions = [tokenized]

        for fts_expr in expressions:
            if not fts_expr.strip():
                continue

            sql = (
                "SELECT c.id as chunk_id, c.content, c.page_num, c.chunk_index, "
                "d.filename, d.id as doc_id, "
                "rank "
                "FROM document_chunks_fts f "
                "JOIN document_chunks c ON c.rowid = f.rowid "
                "JOIN documents d ON d.id = c.document_id "
                "WHERE document_chunks_fts MATCH ? "
                "AND d.user_id = ? "
            )
            params: list = [fts_expr, user_id]
            if scope:
                sql += "AND d.scope = ? "
                params.append(scope)
            if document_id:
                sql += "AND d.id = ? "
                params.append(document_id)
            sql += f"ORDER BY rank LIMIT {limit}"

            try:
                cursor = await self._conn.execute(sql, params)
                rows = await cursor.fetchall()
            except Exception:
                log.debug("doc_fts_search_failed", expr=fts_expr[:50], exc_info=True)
                continue  # Try next expression

            if rows:
                return [
                    {
                        "chunk_id": row["chunk_id"],
                        "content": row["content"],
                        "filename": row["filename"],
                        "doc_id": row["doc_id"],
                        "page_num": row["page_num"],
                        "chunk_index": row["chunk_index"],
                        "rank": abs(row["rank"]),
                    }
                    for row in rows
                ]
            # No rows with AND — try OR fallback

        return []

    @staticmethod
    def _rrf_merge(
        vec_results: list[dict],
        fts_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion merge of vector and FTS results."""
        scores: dict[str, float] = {}
        items: dict[str, dict] = {}

        for rank_idx, item in enumerate(vec_results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank_idx + 1)
            items[cid] = item

        for rank_idx, item in enumerate(fts_results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank_idx + 1)
            items[cid] = item

        merged = []
        for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            result = {**items[cid], "score": score}
            # Remove internal fields
            result.pop("distance", None)
            result.pop("rank", None)
            merged.append(result)

        return merged
