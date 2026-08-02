-- Reflection support
ALTER TABLE memories ADD COLUMN source_memory_ids TEXT DEFAULT '[]';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (51, 'reflection_support');
