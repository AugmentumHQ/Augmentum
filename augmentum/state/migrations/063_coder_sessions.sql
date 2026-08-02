-- Coder mode: agent session state
CREATE TABLE IF NOT EXISTS coder_sessions (
    session_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT 'waiting',
    plan TEXT DEFAULT '',
    plan_steps TEXT DEFAULT '[]',
    current_step INTEGER DEFAULT 0,
    step_outputs TEXT DEFAULT '{}',
    working_set TEXT DEFAULT '[]',
    files_read TEXT DEFAULT '[]',
    tool_calls_made INTEGER DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

INSERT INTO schema_version (version, applied_at) VALUES (63, strftime('%s', 'now'));
