-- Per-branch state cache for narrative branch swap/restore.
-- Stores saved state snapshots + memory ledgers keyed by branch ID
-- so users can swap between conversation branches without losing context.
ALTER TABLE narrative_memory ADD COLUMN branch_states TEXT NOT NULL DEFAULT '{}';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (38, 'branch_states');
