-- 134_file_index_needs_enrichment_index.sql
-- Partial index covering exactly the rows the enrichment loop is
-- looking for, ordered by the column it filters on.
--
-- Without this, ``enrich_pending``'s SELECT was full-scanning
-- file_index every 30 seconds because:
--   - ``embedding IS NULL`` has no usable index (no rows match
--     for most users, but the optimizer doesn't know that)
--   - the ``OR last_enrichment_attempt IS NULL`` predicate makes
--     idx_file_index_enrichment_attempt useless (matches ~99% of
--     rows on a fresh index)
--   - the OR structure across three predicates blocks the optimizer
--     from picking any single column index
--
-- Measured cost on a 63k-row table: 110-160ms per cycle, every 30s
-- (14,400 SELECTs/day → ~24 min CPU/day spent scanning). After this
-- index lands, that drops to a sub-millisecond seek because the
-- partial index typically holds 0-10 rows (only files the loop
-- actually has work to do on).
--
-- Index column choice: ``last_enrichment_attempt`` is what the AND
-- clause filters on (``IS NULL OR < datetime('now', ?)``), so
-- ordering the partial index by it lets the loop walk it in skip-
-- recently-attempted order without an extra sort.
--
-- The third OR branch in enrich_pending (artifacts.cover_url
-- backfill) is intentionally NOT in the partial-index condition.
-- It's a transient backfill case for artifact-sourced EPUBs and
-- it's covered by the query's UNION ALL rewrite landing in the
-- same change — that branch uses idx_file_index_mime + the
-- foreign-key index on artifacts.

CREATE INDEX IF NOT EXISTS idx_file_index_needs_enrichment
    ON file_index(last_enrichment_attempt)
    WHERE embedding IS NULL
       OR (mime_type = 'application/epub+zip' AND thumbnail IS NULL);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (134, 'file_index needs-enrichment partial index');
