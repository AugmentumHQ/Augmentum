-- 301_ab_test_votes_user_scope.sql
-- Scope ab_test_votes by user_id so preference signals are tenant-isolated.

ALTER TABLE ab_test_votes ADD COLUMN user_id TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (301, 'ab_test_votes += user_id: tenant-isolate model preference votes');
