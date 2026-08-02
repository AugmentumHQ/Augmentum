-- 175_file_index_vec.sql
-- Vector index for file_index, complementing the existing FTS5 index.
--
-- File embeddings are already stored as BLOB in file_index.embedding by
-- the background enrichment loop (vfs/enrichment.py::enrich_pending).
-- Before this migration those BLOBs were dead weight: stored but never
-- queryable. This vec0 mirror exposes them to sqlite-vec L2 distance,
-- which is the missing leg for the Reference Resolver hybrid retrieval
-- (vec + FTS5 + RRF + cross-encoder rerank).
--
-- Inserts/updates are handled from Python (FileIndexService) because
-- vec0 virtual-table writes are easier to reason about outside triggers,
-- and the existing pattern in state/discovery_store.py::upsert_cluster_vec
-- works that way. Orphaned vec rows from bulk DELETEs are harmless because
-- search_by_embedding INNER JOINs file_index — orphans naturally drop.

CREATE VIRTUAL TABLE IF NOT EXISTS file_index_vec USING vec0(
    file_id TEXT PRIMARY KEY,
    embedding float[768]
);
