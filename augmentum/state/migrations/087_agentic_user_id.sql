-- 087_agentic_user_id.sql
-- Add user_id scoping to agentic mode tables.
--
-- Follows the pattern from migration 072: add the column, FK to users(id),
-- composite index on (user_id, session_id) for the common lookup path.
-- Legacy rows stay NULL until the backfill below claims them for the
-- oldest user — matches migration 083's approach so single-user installs
-- upgrade cleanly.
--
-- Covered tables:
--   agentic_tasks            — one row per task, queried by (user, session)
--   agentic_tool_call_cache  — one row per cached tool call, queried by task

ALTER TABLE agentic_tasks ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE agentic_tool_call_cache ADD COLUMN user_id TEXT REFERENCES users(id);

CREATE INDEX IF NOT EXISTS idx_agentic_tasks_user_session
    ON agentic_tasks(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_tool_call_cache_user_task
    ON agentic_tool_call_cache(user_id, task_id);

-- Backfill legacy rows to the oldest user (single-user install convention).
UPDATE agentic_tasks
   SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1)
 WHERE user_id IS NULL;

UPDATE agentic_tool_call_cache
   SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1)
 WHERE user_id IS NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (87, 'agentic_user_id_scoping');
