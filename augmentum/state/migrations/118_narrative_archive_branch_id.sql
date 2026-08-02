-- 118_narrative_archive_branch_id.sql
-- Add branch_id to narrative_archive so vector retrieval can filter by branch
-- ancestry instead of relying on destructive prune-on-switch. Existing rows
-- default to 'main' — safe because alternate-branch archive content was never
-- persisted in this table previously (the bug this work fixes destroyed it
-- on every branch switch).

ALTER TABLE narrative_archive ADD COLUMN branch_id TEXT NOT NULL DEFAULT 'main';

CREATE INDEX IF NOT EXISTS idx_narrative_archive_branch
    ON narrative_archive(session_id, branch_id, turn_number);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (118, 'narrative_archive branch_id column');
