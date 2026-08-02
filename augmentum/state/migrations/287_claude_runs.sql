-- 287_claude_runs.sql
-- Persisted Claude Code run history for the external-coder ("Run with Claude")
-- surface.
--
-- Until now a run streamed live into the Agents panel and then vanished: the
-- transcript was pure DOM (lost on refresh/modal close) and only a coarse
-- one-line outcome reached the companion's engineering journal. This stores
-- each run durably so the panel can show a per-workspace history, reopen a
-- transcript, and CONTINUE a run via Claude Code's NATIVE `--resume <session_id>`.
--
-- Fidelity mirrors Claude Code's own model (see design discussion 2026-06-22):
--   * raw_jsonl  — the verbatim stream-json the CLI emitted (full fidelity,
--                  future-proof; written once at finish from the run buffer).
--   * claude_run_events — the normalized events we display (one row each),
--                  appended live during the run so the history view survives a
--                  mid-run refresh.
-- session_id is Claude's own session UUID (captured from the stream-json
-- `system/init` event); it + the stable `/workspace` exec CWD are what make
-- `claude --resume <session_id>` work across runs.
--
-- Both tables are user-scoped per CLAUDE.md (`user_id` column).

CREATE TABLE IF NOT EXISTS claude_runs (
    id            TEXT PRIMARY KEY,                 -- our run id (uuid hex)
    user_id       TEXT NOT NULL,
    workspace_id  TEXT NOT NULL,
    session_id    TEXT NOT NULL DEFAULT '',         -- Claude's session UUID (for --resume)
    task          TEXT NOT NULL DEFAULT '',
    permission    TEXT NOT NULL DEFAULT 'auto',
    status        TEXT NOT NULL DEFAULT 'running',   -- running | done | failed
    outcome       TEXT NOT NULL DEFAULT '',          -- summary one-liner
    error         TEXT NOT NULL DEFAULT '',
    files_changed TEXT NOT NULL DEFAULT '[]',        -- json array of paths
    raw_jsonl     TEXT NOT NULL DEFAULT '',          -- verbatim stream-json (full fidelity)
    cost_usd      REAL NOT NULL DEFAULT 0,
    num_turns     INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER NOT NULL DEFAULT 0,
    resumed_from  TEXT NOT NULL DEFAULT '',          -- prior run id when this continues one
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_claude_runs_user_ws
    ON claude_runs(user_id, workspace_id, created_at DESC);

CREATE TABLE IF NOT EXISTS claude_run_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    kind       TEXT NOT NULL DEFAULT '',             -- started|message|thinking|file_change|...
    text       TEXT NOT NULL DEFAULT '',
    tool       TEXT NOT NULL DEFAULT '',
    path       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_claude_run_events_run
    ON claude_run_events(run_id, seq);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (287, 'claude_runs + claude_run_events: persisted Claude Code run history (raw + normalized) with session_id for native resume');
