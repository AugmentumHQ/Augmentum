-- 239_companion_standing_tasks.sql
-- Standing tasks: recurring jobs Becca runs on the user's behalf.
--
-- The relational complement to companion_tracked_topics (migration 238):
-- where a topic is "watch this subject for things I'd want to know," a
-- standing task is "do this specific thing on a cadence and tell me."
--
-- Kinds (initial):
--   feed_digest      — periodic digest of curator items for a topic
--   github_releases  — poll a GitHub repo for new releases
--   url_watch        — fetch a URL, hash compare, surface if changed
--   recurring_search — periodic SearXNG query, surface novel results
--
-- params is JSON, shape depends on kind. last_result is the JSON of the
-- most recent run's surfaced findings (for re-display in the UI without
-- re-running).

CREATE TABLE IF NOT EXISTS companion_standing_tasks (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id TEXT NOT NULL,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    interval_seconds INTEGER NOT NULL DEFAULT 86400,    -- daily default
    last_run_at TEXT,
    next_run_at TEXT,
    last_result TEXT,                                    -- JSON
    last_result_summary TEXT,
    last_error TEXT,
    consecutive_error_count INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_standing_tasks_user
    ON companion_standing_tasks(user_id, companion_id);

CREATE INDEX IF NOT EXISTS idx_standing_tasks_due
    ON companion_standing_tasks(enabled, next_run_at);
