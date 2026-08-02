ALTER TABLE narrative_memory ADD COLUMN memory_settings TEXT DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (48, 'narrative_memory_settings');
