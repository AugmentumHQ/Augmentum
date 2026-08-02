-- 321_claude_runs_model.sql
-- Record which model an external Claude Code run actually used.
--
-- Until now claude_runs captured cost/turns/duration/session but NOT the model,
-- so the unified Coding Agents history hardcoded a "claude-code" placeholder and
-- there was no way to see (or, upstream, to target) a specific model. The model
-- is present in Claude's stream-json `system/init` event; we now capture it in
-- ClaudeStreamCollector and persist it here so every run is labelled with the
-- real model — even when the user left the choice on "Account default" (we still
-- record what Claude picked). pi_runs already has this column; coding_runs
-- stores the user-chosen model; this brings claude_runs to parity.
--
-- Provider-neutral by design: any future in-container engine (Codex) writing to
-- its own runs table follows the same "capture from the init/handshake event"
-- pattern. See augmentum/coder/external/providers.py.

ALTER TABLE claude_runs ADD COLUMN model TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (321, 'claude_runs.model: record the model an external Claude run used (parity with pi_runs/coding_runs)');
