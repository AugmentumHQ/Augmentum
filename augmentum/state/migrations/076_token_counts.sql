CREATE TABLE IF NOT EXISTS token_counts (
    id          TEXT PRIMARY KEY,
    model_id    TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    last_used   REAL NOT NULL,
    use_count   INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tc_model ON token_counts(model_id);
CREATE INDEX IF NOT EXISTS idx_tc_last_used ON token_counts(last_used);

INSERT INTO schema_version (version, applied_at, description)
VALUES (76, datetime('now'), 'Token count cache for engine v2');
