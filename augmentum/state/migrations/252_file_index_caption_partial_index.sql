-- 252_file_index_caption_partial_index.sql
--
-- Add a partial index covering the B4 (image caption) leg of
-- enrich_pending's UNION query in augmentum/vfs/enrichment.py.
--
-- Pre-fix: B4 — `WHERE mime_type LIKE 'image/%' AND (description
-- IS NULL OR description = '') AND is_trashed = 0 AND
-- (last_enrichment_attempt IS NULL OR last_enrichment_attempt < ?)` —
-- forced a scan because no existing index covered the image-caption
-- predicate. On a 64k-row table the raw query took ~60ms; under
-- production WAL contention this surfaced as 1.5-3s slow_db_op
-- warnings (101 hits in 12h on 2026-06-08).
--
-- The existing partial index `idx_file_index_needs_enrichment` covers
-- B1 (embedding IS NULL) and B2 (epub thumbnail). B4 was added later
-- (Piece 3 backfill) and never got matching index coverage.
--
-- A partial index on `last_enrichment_attempt WHERE <B4 predicate>`
-- keeps the index small (rows leave it as soon as a caption is
-- written) and matches the WHERE-clause exactly so the optimizer
-- picks it for the lookup.

CREATE INDEX IF NOT EXISTS idx_file_index_needs_caption
    ON file_index(last_enrichment_attempt)
    WHERE mime_type LIKE 'image/%'
      AND (description IS NULL OR description = '')
      AND is_trashed = 0;

-- ANALYZE is required for SQLite's optimizer to pick a partial index
-- over an overlapping general index. Without this, the planner stays
-- on `idx_file_index_enrichment_attempt` and the new index goes
-- unused (verified 2026-06-08: 34ms → 0.26ms after ANALYZE).
ANALYZE file_index;
