-- Three-layer narrative memory: STATE snapshot + MEMORY ledger
-- Adds two new columns to narrative_memory for the new architecture.
-- Old columns (memory_summary, overflow_summaries, archived_messages) kept for backward compat.

ALTER TABLE narrative_memory ADD COLUMN state_snapshot TEXT NOT NULL DEFAULT '{}';
ALTER TABLE narrative_memory ADD COLUMN memory_ledger TEXT NOT NULL DEFAULT '[]';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (36, 'three_layer_memory');
