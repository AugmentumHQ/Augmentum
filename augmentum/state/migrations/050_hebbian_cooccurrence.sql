-- Hebbian co-occurrence tracking
-- NOTE: source_type already exists from 006_memory.sql; not re-added here.
ALTER TABLE memories ADD COLUMN retrieval_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN last_accessed_at TEXT;

CREATE TABLE IF NOT EXISTS memory_cooccurrence (
    user_id TEXT NOT NULL,
    id_a TEXT NOT NULL,
    id_b TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    last_updated TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, id_a, id_b)
);

CREATE INDEX IF NOT EXISTS idx_mem_cooccur_a
    ON memory_cooccurrence(user_id, id_a, count DESC);
CREATE INDEX IF NOT EXISTS idx_mem_cooccur_b
    ON memory_cooccurrence(user_id, id_b, count DESC);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (50, 'hebbian_cooccurrence');
