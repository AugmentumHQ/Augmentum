-- Vector index for coder_turn_archive — Phase 2 of the LTM substrate.
--
-- Mirrors the file_index_vec pattern (migration 175): a vec0 virtual
-- table holding the float32 embedding per archive_id. The base
-- table's ``embedding_status`` column transitions:
--   pending → embedded   (after this table gets the row)
--   pending → skipped    (when embedding fails terminally — caller decides)
--
-- Inserts/updates are handled from Python (turn_archive_embed.py) per
-- the same rationale as migration 175 — vec0 writes are easier to
-- reason about outside triggers. Orphaned vec rows from base-table
-- deletes are harmless: search_similar INNER JOINs back to
-- coder_turn_archive on archive_id, so orphans naturally drop out.
--
-- Dimension matches the EmbeddingService.DIMENSION (768) used by
-- knowledge packs, memory, dream journal, and file_index — so the
-- same nomic-embed-text-v1.5 model that's already loaded serves the
-- turn archive too. No second embedding model in memory.

CREATE VIRTUAL TABLE IF NOT EXISTS coder_turn_archive_vec USING vec0(
    archive_id TEXT PRIMARY KEY,
    embedding float[768]
);
