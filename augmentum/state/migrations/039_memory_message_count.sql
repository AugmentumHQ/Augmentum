-- Persist message_count directly in narrative_memory.
-- Previously recovered from facts/plots/contradictions which are empty
-- when state tracking is disabled (default config).
ALTER TABLE narrative_memory ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (39, 'memory_message_count');
