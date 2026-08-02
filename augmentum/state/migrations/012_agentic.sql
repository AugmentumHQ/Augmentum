-- Agentic mode: task tracking and artifact storage

CREATE TABLE IF NOT EXISTS agentic_tasks (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL DEFAULT '',
    flow_id         TEXT,
    status          TEXT NOT NULL DEFAULT 'planning',   -- planning, running, paused, approval_pending, completed, failed
    autonomy_level  INTEGER NOT NULL DEFAULT 2,         -- 1=suggest, 2=ask, 3=inform, 4=autonomous
    title           TEXT NOT NULL DEFAULT '',
    plan_md         TEXT NOT NULL DEFAULT '',            -- running plan markdown (attention anchor)
    current_step    INTEGER NOT NULL DEFAULT 0,
    total_steps     INTEGER NOT NULL DEFAULT 0,
    step_outputs    TEXT NOT NULL DEFAULT '{}',          -- JSON: step_index → output
    tool_calls_made INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at    TEXT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',
    filename        TEXT NOT NULL,
    display_name    TEXT NOT NULL DEFAULT '',
    format          TEXT NOT NULL,                      -- pdf, docx, pptx, xlsx, png, md
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    path            TEXT NOT NULL,                      -- relative path under artifact dir
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    metadata        TEXT NOT NULL DEFAULT '{}'           -- JSON: page_count, slide_count, etc.
);

CREATE INDEX IF NOT EXISTS idx_artifacts_task ON artifacts(task_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_agentic_tasks_session ON agentic_tasks(session_id);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (12, 'Agentic mode task tracking and artifacts');
