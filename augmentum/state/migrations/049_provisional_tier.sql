-- Add provisional support
ALTER TABLE memories ADD COLUMN provisional_expires_at TEXT;
ALTER TABLE memories ADD COLUMN evidence TEXT DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_memories_provisional
    ON memories(tier, provisional_expires_at)
    WHERE tier = 'provisional';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (49, 'provisional_tier_and_evidence');
