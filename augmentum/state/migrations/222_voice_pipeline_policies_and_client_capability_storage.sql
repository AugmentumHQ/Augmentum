-- 222_voice_pipeline_policies_and_client_capability_storage.sql
-- Per-user voice pipeline policy (auto / local / server / custom) for each
-- consumer surface. Falls through to install-wide defaults from config.py
-- when no row exists for (user_id, surface). Surface values are validated
-- at the route layer: 'call' | 'companion' | 'narration' | 'readaloud'.
-- Mode values: 'auto' | 'local' | 'server' | 'custom'.

CREATE TABLE IF NOT EXISTS voice_pipeline_policies (
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    surface TEXT NOT NULL,
    mode TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, surface)
);

CREATE INDEX IF NOT EXISTS idx_voice_pipeline_policies_user
    ON voice_pipeline_policies(user_id);
