-- First-class persisted build runs for Build Mode.
--
-- ACTIVE_BUILDS remains the in-process execution cache, but this table is the
-- durable user/session/task/artifact spine the UI can reload after refresh.

CREATE TABLE IF NOT EXISTS build_runs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    session_id TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    artifact_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'application',
    status TEXT NOT NULL DEFAULT 'queued',
    name TEXT NOT NULL DEFAULT '',
    request_json TEXT NOT NULL DEFAULT '{}',
    progress_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_build_runs_user_session_updated
    ON build_runs(user_id, session_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_build_runs_user_task
    ON build_runs(user_id, task_id);

CREATE INDEX IF NOT EXISTS idx_build_runs_user_artifact
    ON build_runs(user_id, artifact_id);

