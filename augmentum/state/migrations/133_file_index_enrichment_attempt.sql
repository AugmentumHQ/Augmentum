-- 133_file_index_enrichment_attempt.sql
-- Track when the VFS enrichment loop last touched each file so a
-- repeatedly-failing enrichment doesn't requeue the same row every
-- 30 seconds forever.
--
-- Symptom this fixes (observed 2026-05-09): 5 EPUB files in the
-- Library kept hitting `OperationalError: database is locked` on
-- their cover/source_metadata UPDATE; the enrichment loop's bare
-- `except Exception` swallowed the error and the next 30-second
-- pass picked the same files again. 600+ lock errors logged in
-- 30 minutes for the same 5 file_ids.
--
-- After this migration the enrich_pending SELECT filters out rows
-- whose last attempt was less than ENRICHMENT_RETRY_INTERVAL ago
-- (default 1 hour). Each per-file pass sets the column in a
-- finally block, so failures back off without manual intervention.
-- A successful pass clears the row from the pending set on its own
-- (embedding/thumbnail predicates stop matching).

ALTER TABLE file_index ADD COLUMN last_enrichment_attempt TEXT;

CREATE INDEX IF NOT EXISTS idx_file_index_enrichment_attempt
    ON file_index(last_enrichment_attempt);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (133, 'file_index.last_enrichment_attempt');
