-- 311_pi_runs.sql
-- Persisted pi (pi.dev terminal coding agent) session mirrors.
--
-- Unlike claude_runs (migration 287), these rows are NOT server-executed
-- runs: the pi CLI runs on the user's host and PUSHES a normalized mirror
-- of its session here (opt-in, /sync on) so terminal sessions become
-- first-class Augmentum objects — visible in the web coder Agents panel,
-- with the host session file path recorded as the resume affordance
-- (`pi --session <file>` on the host owns actual resumption).
--
-- Deliberately a separate table pair rather than a generalization of
-- claude_runs: claude_runs rows are owned by the server-side RunManager,
-- whose orphan-reconciliation marks stale "running" rows failed — that
-- lifecycle would wrongly kill live host-side pi sessions. No raw_jsonl
-- either: the host session file IS the raw record (session_file column).
--
-- Both tables are user-scoped per CLAUDE.md (`user_id` column).

CREATE TABLE IF NOT EXISTS pi_runs (
    id            TEXT PRIMARY KEY,                  -- pi session id (uuid from the session file)
    user_id       TEXT NOT NULL,
    project       TEXT NOT NULL DEFAULT '',          -- cwd basename on the host
    session_file  TEXT NOT NULL DEFAULT '',          -- host path (resume affordance)
    title         TEXT NOT NULL DEFAULT '',          -- pi session name / first prompt
    model         TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'running',   -- running | done | failed | detached
    outcome       TEXT NOT NULL DEFAULT '',
    error         TEXT NOT NULL DEFAULT '',
    files_changed TEXT NOT NULL DEFAULT '[]',        -- json array of paths
    num_turns     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pi_runs_user_proj
    ON pi_runs(user_id, project, created_at DESC);

CREATE TABLE IF NOT EXISTS pi_run_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    kind       TEXT NOT NULL DEFAULT '',             -- message|tool_call|file_change|status
    text       TEXT NOT NULL DEFAULT '',
    tool       TEXT NOT NULL DEFAULT '',
    path       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Unique on (run_id, seq) so the host pusher can retry batches idempotently
-- (INSERT OR IGNORE dedupes replays after a network blip).
CREATE UNIQUE INDEX IF NOT EXISTS idx_pi_run_events_run_seq
    ON pi_run_events(run_id, seq);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (311, 'pi_runs + pi_run_events: pushed pi terminal-session mirrors (host-owned, opt-in sync)');
