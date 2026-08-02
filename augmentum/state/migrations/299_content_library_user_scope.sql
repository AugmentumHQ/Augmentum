-- 299_content_library_user_scope.sql
-- Scope content_library by user_id.
--
-- Migration 093 added user_id to the sibling discovery-engine tables
-- (interaction_signals, browse_history, interest_clusters) but MISSED
-- content_library. This closes the gap: user-distilled knowledge chunks
-- must be tenant-isolated like their siblings.

ALTER TABLE content_library ADD COLUMN user_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_content_library_user
    ON content_library (user_id, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (299, 'content_library += user_id: close the tenant-isolation gap missed in migration 093');
