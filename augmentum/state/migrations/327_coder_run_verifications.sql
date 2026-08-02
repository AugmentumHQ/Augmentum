-- 327_coder_run_verifications.sql
-- Durable home for a completed coder run's independent verification verdict
-- (augmentum/coder/run_verifier.py). The verdict already rides the completion
-- envelope for the LIVE brief-open path (companion perception / __previewBrief),
-- but a brief opened COLD later — e.g. from a stale "coder.run.complete"
-- notification deep-link — has no envelope. This row is what that cold open
-- reads via GET /api/coder/runs/{run_id}/verification.
--
-- User-scoped (one row per broker run_id). Keyed by run_id because a delegated
-- run enqueues coder_background_run directly and may have no coding_runs row.

CREATE TABLE IF NOT EXISTS coder_run_verifications (
    run_id         TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL REFERENCES users(id),
    workspace_id   TEXT NOT NULL DEFAULT '',
    tier           TEXT NOT NULL DEFAULT 'unchecked',   -- verified|probable|failed|human_required|unchecked
    oracle         TEXT NOT NULL DEFAULT 'none',        -- mechanical|judgment|none
    reason         TEXT NOT NULL DEFAULT '',
    verifier_model TEXT NOT NULL DEFAULT '',
    self_verified  INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_coder_run_verifications_user
    ON coder_run_verifications (user_id);
