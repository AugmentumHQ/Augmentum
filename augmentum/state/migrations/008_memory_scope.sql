-- 008_memory_scope.sql: Add scope tagging to memories for cross-project isolation

ALTER TABLE memories ADD COLUMN scope TEXT;

CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(user_id, scope);

INSERT INTO schema_version (version, description) VALUES (8, 'Memory scope tagging');
