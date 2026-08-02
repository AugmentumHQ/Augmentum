-- Cross-session subagent run audit. One row per task_dispatch spawn:
-- parent run, role, resolved model (often "@provider:peer" form),
-- prompt, budget usage, final output, structured tool-call log.
--
-- Cascade-deleted with the parent coder run; user-scoped per the
-- multi-tenant pattern (CLAUDE.md). Indices cover the two common reads:
-- list-by-user (history sidebar) and list-by-parent (nested UI cards
-- under a turn).

CREATE TABLE IF NOT EXISTS coder_subagent_runs (
    subagent_id     TEXT PRIMARY KEY,
    parent_run_id   TEXT NOT NULL DEFAULT '',
    parent_turn_id  TEXT NOT NULL DEFAULT '',
    user_id         TEXT NOT NULL DEFAULT '',
    workspace_id    TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',

    role            TEXT NOT NULL,
    model_spec      TEXT NOT NULL DEFAULT '',  -- "claude-sonnet-4-6@anthropic"
    model_resolved  TEXT NOT NULL DEFAULT '',  -- "claude-sonnet-4-6" (clean id)
    backend_key     TEXT NOT NULL DEFAULT '',  -- "anthropic" | "engine" | "fabric:tower"

    prompt          TEXT NOT NULL DEFAULT '',
    context_mode    TEXT NOT NULL DEFAULT 'workspace',

    started_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    completed_at    INTEGER,

    stop_reason     TEXT NOT NULL DEFAULT 'running',
    stop_detail     TEXT NOT NULL DEFAULT '',
    stuck_pattern   TEXT NOT NULL DEFAULT '',

    iterations      INTEGER NOT NULL DEFAULT 0,
    tool_calls      INTEGER NOT NULL DEFAULT 0,
    tokens_in       INTEGER NOT NULL DEFAULT 0,
    tokens_out      INTEGER NOT NULL DEFAULT 0,
    wallclock_ms    INTEGER NOT NULL DEFAULT 0,

    output_text     TEXT NOT NULL DEFAULT '',
    tool_call_log   TEXT NOT NULL DEFAULT '[]'  -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_coder_subagent_runs_user
    ON coder_subagent_runs(user_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_coder_subagent_runs_parent
    ON coder_subagent_runs(parent_run_id, started_at);

CREATE INDEX IF NOT EXISTS idx_coder_subagent_runs_session
    ON coder_subagent_runs(session_id, started_at);
