-- 032_missing_indexes.sql: Add missing indexes for common query patterns

-- memories.superseded_by — used by version history chain lookups
CREATE INDEX IF NOT EXISTS idx_memories_superseded
    ON memories(superseded_by);

-- memories(updated_at, access_count, importance) — composite for compaction
-- candidate queries that filter/sort on these columns outside the
-- existing idx_memories_compaction (which leads with user_id, valid_until)
CREATE INDEX IF NOT EXISTS idx_memories_compaction_sort
    ON memories(updated_at, access_count, importance);

-- Refresh query-planner statistics on all tables
ANALYZE;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (53, 'Missing indexes');
