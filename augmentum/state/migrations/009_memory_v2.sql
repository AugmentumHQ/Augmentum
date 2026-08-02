-- 009_memory_v2.sql: Add tier system and compaction tracking to memories

ALTER TABLE memories ADD COLUMN tier TEXT NOT NULL DEFAULT 'active';
ALTER TABLE memories ADD COLUMN last_compacted_at TEXT;

CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(user_id, tier, valid_until);
CREATE INDEX IF NOT EXISTS idx_memories_compaction ON memories(
    user_id, valid_until, access_count, importance, updated_at
);

INSERT INTO schema_version (version, description) VALUES (9, 'Memory tier system');
