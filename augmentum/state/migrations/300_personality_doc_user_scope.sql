-- 300_personality_doc_user_scope.sql
-- Scope personality_doc_candidates by user_id.
--
-- The table was created in migration 192 with companion_id but no user_id.
-- Since companion_id is shared ("becca" for all users), one user's
-- consolidation proposals were visible/actionable by another.

ALTER TABLE personality_doc_candidates ADD COLUMN user_id TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (300, 'personality_doc_candidates += user_id: tenant-isolate consolidation proposals');
