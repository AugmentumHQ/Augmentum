"""Embedding + retrieval layer over the turn archive — Phase 2 LTM.

Phase 1 (``augmentum/coder/turn_archive.py``) writes every completed
turn to ``coder_turn_archive``. This module sits on top:

* **Embed pending rows** — after each append (or on a backfill sweep)
  produce a float32 embedding of ``user_goal + outcome + summary`` and
  insert into ``coder_turn_archive_vec``. Flip ``embedding_status``
  from ``pending`` to ``embedded`` on success, ``skipped`` on
  terminal failure.
* **Semantic search** — given a query string, return top-k archive
  entries by L2 distance. The recall tool wraps this.

Why ``embedded`` vs ``skipped`` as terminal states: the embedding
service can fail transiently (model loading, ONNX hiccup) or
terminally (text too long, encoding error). Transient failures stay
``pending`` so a future sweep retries; terminal failures move to
``skipped`` so we don't loop on them.

Best-effort throughout: any error here is logged and absorbed; the
write path never blocks the agent loop on embedding. The Phase 1
archive remains useful (timeline browsing) even when this layer is
fully degraded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


# Hard limits to keep embedding load bounded under fan-out turns.
_EMBED_TEXT_MAX_CHARS = 4_000  # ~1000 tokens; nomic 768-dim handles this comfortably
_BACKFILL_BATCH_LIMIT = 32     # max rows per backfill sweep call


@dataclass(slots=True)
class RecallHit:
    """One semantic-search match."""
    archive_id: str
    turn_index: int
    user_goal: str
    outcome: str
    summary: str
    event_time: int
    distance: float
    files_edited_count: int = 0
    files_read_count: int = 0


def _compose_embedding_text(
    *,
    user_goal: str,
    outcome: str,
    summary: str,
) -> str:
    """Build the canonical text to embed for one archive row.

    Lead with ``user_goal`` so the embedding's high-weight dimensions
    reflect the user's intent — the most stable signal across turns.
    ``outcome`` and ``summary`` follow for context. We cap the
    composite at _EMBED_TEXT_MAX_CHARS to keep the embedder's
    attention scratch bounded (see EmbeddingService.MAX_SEQ_TOKENS).
    """
    parts: list[str] = []
    g = (user_goal or "").strip()
    if g:
        parts.append(g)
    o = (outcome or "").strip()
    if o:
        parts.append(f"outcome: {o}")
    s = (summary or "").strip()
    if s:
        parts.append(s)
    blob = " · ".join(parts)
    if len(blob) > _EMBED_TEXT_MAX_CHARS:
        blob = blob[:_EMBED_TEXT_MAX_CHARS] + "…"
    return blob


async def _set_embedding_status(
    conn: aiosqlite.Connection,
    *,
    archive_id: str,
    status: str,
) -> None:
    """Flip ``embedding_status`` for one row. Caller commits."""
    try:
        await conn.execute(
            "UPDATE coder_turn_archive SET embedding_status = ? "
            "WHERE archive_id = ?",
            (status, archive_id),
        )
    except Exception as exc:
        log.debug(
            "coder_turn_archive_embed.status_update_failed",
            archive_id=archive_id, error=str(exc)[:160],
        )


async def embed_one(
    conn: aiosqlite.Connection,
    *,
    archive_id: str,
) -> bool:
    """Embed a single archive row and write to the vec0 mirror.

    Returns True on success (status now ``embedded``), False on any
    failure. Failures NOT marked ``skipped`` here — that's the caller's
    decision based on whether the failure is transient or terminal.
    """
    try:
        cursor = await conn.execute(
            "SELECT user_goal, outcome, summary, embedding_status "
            "FROM coder_turn_archive WHERE archive_id = ?",
            (archive_id,),
        )
        row = await cursor.fetchone()
    except Exception as exc:
        log.debug(
            "coder_turn_archive_embed.fetch_failed",
            archive_id=archive_id, error=str(exc)[:160],
        )
        return False

    if not row:
        return False
    if (row[3] or "") == "embedded":
        return True  # already done; idempotent

    text = _compose_embedding_text(
        user_goal=row[0] or "", outcome=row[1] or "", summary=row[2] or "",
    )
    if not text:
        # Nothing meaningful to embed — mark skipped so we don't loop.
        await _set_embedding_status(conn, archive_id=archive_id, status="skipped")
        await conn.commit()
        return False

    try:
        import asyncio

        from augmentum.memory.embeddings import (
            EmbeddingService,
            EmbeddingUnavailable,
        )
        try:
            # Run on the threadpool so first-call model load (~30s on
            # cold cache) doesn't freeze the event loop. Knowledge
            # packs use the same pattern (packs.py::search).
            vec = await asyncio.to_thread(EmbeddingService.embed_one, text)
        except EmbeddingUnavailable as exc:
            # Embedder broken on this node — leave the row PENDING so
            # a future sweep retries when the service is back.
            log.debug(
                "coder_turn_archive_embed.embedder_unavailable",
                error=str(exc)[:160],
            )
            return False
        blob = EmbeddingService.to_blob(vec)
    except Exception as exc:
        log.debug(
            "coder_turn_archive_embed.embed_failed",
            archive_id=archive_id, error=str(exc)[:160],
        )
        return False

    try:
        # vec0 PRIMARY KEY upsert pattern — DELETE-then-INSERT because
        # vec0 doesn't support UPDATE on the embedding column. Matches
        # the discovery_store.py::upsert_cluster_vec convention.
        await conn.execute(
            "DELETE FROM coder_turn_archive_vec WHERE archive_id = ?",
            (archive_id,),
        )
        await conn.execute(
            "INSERT INTO coder_turn_archive_vec(archive_id, embedding) "
            "VALUES (?, ?)",
            (archive_id, blob),
        )
        await _set_embedding_status(
            conn, archive_id=archive_id, status="embedded",
        )
        await conn.commit()
        return True
    except Exception as exc:
        log.debug(
            "coder_turn_archive_embed.write_failed",
            archive_id=archive_id, error=str(exc)[:160],
        )
        return False


async def embed_pending(
    conn: aiosqlite.Connection,
    *,
    user_id: str = "",
    workspace_id: str = "",
    limit: int | None = None,
) -> int:
    """Backfill sweep: embed up to ``limit`` pending rows.

    Scoped by ``(user_id, workspace_id)`` when provided so a workspace
    open doesn't pull a global backlog. Returns count of rows
    successfully embedded.

    Called from:
      * The post-write hook in ``append_turn`` (limit=1, scope=this turn)
      * A future periodic sweep (limit=batch, scope=workspace)
    """
    cap = max(1, min(int(limit or _BACKFILL_BATCH_LIMIT), _BACKFILL_BATCH_LIMIT))

    where = "embedding_status = 'pending'"
    params: list = []
    if user_id:
        where += " AND user_id = ?"
        params.append(user_id)
    if workspace_id:
        where += " AND workspace_id = ?"
        params.append(workspace_id)
    params.append(cap)

    try:
        cursor = await conn.execute(
            f"SELECT archive_id FROM coder_turn_archive WHERE {where} "
            f"ORDER BY recorded_at ASC LIMIT ?",
            tuple(params),
        )
        rows = await cursor.fetchall()
    except Exception as exc:
        log.debug(
            "coder_turn_archive_embed.list_pending_failed",
            error=str(exc)[:160],
        )
        return 0

    succeeded = 0
    for (aid,) in rows or []:
        if await embed_one(conn, archive_id=aid):
            succeeded += 1
    return succeeded


async def search_similar(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    workspace_id: str,
    query: str,
    k: int = 5,
    similarity_threshold: float = 0.0,
) -> list[RecallHit]:
    """Top-k semantic matches for ``query`` within a workspace.

    Returns hits sorted by ascending L2 distance (closest first). The
    ``similarity_threshold`` is a distance ceiling — hits with distance
    > threshold are dropped. Set 0 to disable (return raw top-k).

    Best-effort: returns ``[]`` on any failure (embedder unavailable,
    empty vec table, query encoding error). The recall tool wraps
    this so the agent sees an empty result rather than an error.
    """
    if not user_id or not workspace_id:
        return []
    if not query or not query.strip():
        return []
    k = max(1, min(int(k or 5), 32))

    try:
        import asyncio

        from augmentum.memory.embeddings import (
            EmbeddingService,
            EmbeddingUnavailable,
        )
        try:
            # Threadpool same as embed_one — keeps recall queries
            # non-blocking even on cold-cache nodes.
            qvec = await asyncio.to_thread(
                EmbeddingService.embed_query, query.strip(),
            )
        except EmbeddingUnavailable:
            return []
        qblob = EmbeddingService.to_blob(qvec)
    except Exception as exc:
        log.debug(
            "coder_turn_archive_embed.query_embed_failed",
            error=str(exc)[:160],
        )
        return []

    try:
        # Joined search: vec table for ranking, base table for content.
        # ``MATCH`` is sqlite-vec's KNN operator; ``distance`` is L2.
        # The LIMIT inside the subquery is what bounds the vec scan;
        # the outer WHERE filters by workspace, which matters because
        # the vec table is not workspace-partitioned (PK is archive_id
        # alone) — same pattern as file_index_vec / discovery_clusters.
        cursor = await conn.execute(
            """
            SELECT v.archive_id, v.distance,
                   a.turn_index, a.user_goal, a.outcome, a.summary,
                   a.event_time, a.files_read, a.files_edited
            FROM coder_turn_archive_vec v
            INNER JOIN coder_turn_archive a ON a.archive_id = v.archive_id
            WHERE v.embedding MATCH ? AND k = ?
              AND a.user_id = ? AND a.workspace_id = ?
            ORDER BY v.distance ASC
            """,
            (qblob, k * 4, user_id, workspace_id),
        )
        rows = await cursor.fetchall()
    except Exception as exc:
        log.debug(
            "coder_turn_archive_embed.search_failed",
            error=str(exc)[:160],
        )
        return []

    import json as _json
    hits: list[RecallHit] = []
    for row in rows or []:
        distance = float(row[1] or 0.0)
        if similarity_threshold > 0 and distance > similarity_threshold:
            continue
        files_read = []
        files_edited = []
        try:
            files_read = _json.loads(row[7] or "[]") or []
        except Exception as exc:
            log.warning(
                "turn_archive_files_read_parse_failed",
                archive_id=row[0], error=str(exc)[:160],
            )
        try:
            files_edited = _json.loads(row[8] or "[]") or []
        except Exception as exc:
            log.warning(
                "turn_archive_files_edited_parse_failed",
                archive_id=row[0], error=str(exc)[:160],
            )
        hits.append(RecallHit(
            archive_id=row[0] or "",
            turn_index=int(row[2] or 0),
            user_goal=row[3] or "",
            outcome=row[4] or "",
            summary=row[5] or "",
            event_time=int(row[6] or 0),
            distance=distance,
            files_read_count=len(files_read) if isinstance(files_read, list) else 0,
            files_edited_count=len(files_edited) if isinstance(files_edited, list) else 0,
        ))
        if len(hits) >= k:
            break
    return hits


__all__ = [
    "RecallHit",
    "embed_one",
    "embed_pending",
    "search_similar",
]
