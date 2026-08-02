-- Coder run ledger, workspace services, and model capability hints.

CREATE TABLE IF NOT EXISTS coder_turn_runs (
    id                     TEXT PRIMARY KEY,
    user_id                TEXT NOT NULL DEFAULT '',
    workspace_id           TEXT NOT NULL DEFAULT '',
    session_id             TEXT NOT NULL DEFAULT '',
    strategy               TEXT NOT NULL DEFAULT '',
    model                  TEXT NOT NULL DEFAULT '',
    provider               TEXT NOT NULL DEFAULT '',
    prompt_profile         TEXT NOT NULL DEFAULT '',
    tooling_profile        TEXT NOT NULL DEFAULT '',
    status                 TEXT NOT NULL DEFAULT 'running',
    started_at             REAL NOT NULL,
    first_event_at         REAL,
    first_useful_action_at REAL,
    completed_at           REAL,
    updated_at             REAL NOT NULL,
    iterations             INTEGER NOT NULL DEFAULT 0,
    tool_calls             INTEGER NOT NULL DEFAULT 0,
    parallel_waves         INTEGER NOT NULL DEFAULT 0,
    retries                INTEGER NOT NULL DEFAULT 0,
    no_response_events     INTEGER NOT NULL DEFAULT 0,
    empty_native_content   INTEGER NOT NULL DEFAULT 0,
    malformed_tool_calls   INTEGER NOT NULL DEFAULT 0,
    commands_run           TEXT NOT NULL DEFAULT '[]',
    files_touched          TEXT NOT NULL DEFAULT '[]',
    tests_run              TEXT NOT NULL DEFAULT '[]',
    browser_checks         TEXT NOT NULL DEFAULT '[]',
    finish_reason          TEXT NOT NULL DEFAULT '',
    fallback_reason        TEXT NOT NULL DEFAULT '',
    checkpoint_id          TEXT NOT NULL DEFAULT '',
    changed_files          TEXT NOT NULL DEFAULT '[]',
    closeout_json          TEXT NOT NULL DEFAULT '{}',
    metrics_json           TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_coder_turn_runs_user_workspace
    ON coder_turn_runs(user_id, workspace_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_coder_turn_runs_session
    ON coder_turn_runs(session_id, started_at DESC);

CREATE TABLE IF NOT EXISTS coder_turn_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    type      TEXT NOT NULL,
    phase     TEXT NOT NULL DEFAULT '',
    status    TEXT NOT NULL DEFAULT '',
    payload   TEXT NOT NULL DEFAULT '{}',
    UNIQUE(run_id, seq),
    FOREIGN KEY(run_id) REFERENCES coder_turn_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_coder_turn_events_run_seq
    ON coder_turn_events(run_id, seq);

CREATE TABLE IF NOT EXISTS coder_workspace_services (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL DEFAULT '',
    workspace_id  TEXT NOT NULL,
    name          TEXT NOT NULL,
    command       TEXT NOT NULL,
    cwd           TEXT NOT NULL DEFAULT '/workspace',
    env_json      TEXT NOT NULL DEFAULT '{}',
    pid           INTEGER NOT NULL DEFAULT 0,
    ports_json    TEXT NOT NULL DEFAULT '[]',
    log_path      TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'unknown',
    last_probe    TEXT NOT NULL DEFAULT '{}',
    error         TEXT NOT NULL DEFAULT '',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_coder_workspace_services_workspace
    ON coder_workspace_services(user_id, workspace_id);

CREATE TABLE IF NOT EXISTS coder_model_capabilities (
    model                    TEXT PRIMARY KEY,
    provider                 TEXT NOT NULL DEFAULT '',
    native_tool_support      INTEGER NOT NULL DEFAULT 0,
    tool_call_quirks         TEXT NOT NULL DEFAULT '[]',
    thinking_tag_behavior    TEXT NOT NULL DEFAULT '',
    empty_response_tendency  REAL NOT NULL DEFAULT 0.0,
    preferred_strategy       TEXT NOT NULL DEFAULT '',
    max_useful_tools         INTEGER NOT NULL DEFAULT 0,
    qwen_no_thinking         INTEGER NOT NULL DEFAULT 0,
    force_hybrid_supervision INTEGER NOT NULL DEFAULT 0,
    health_json              TEXT NOT NULL DEFAULT '{}',
    updated_at               REAL NOT NULL
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (141, 'Coder turn ledger, workspace services, model capabilities');
