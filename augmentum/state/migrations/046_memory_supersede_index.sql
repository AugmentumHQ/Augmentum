-- Index on superseded_by for efficient version history chain walking.
CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded_by);

INSERT INTO schema_version (version, description) VALUES (46, 'Memory superseded_by index');
