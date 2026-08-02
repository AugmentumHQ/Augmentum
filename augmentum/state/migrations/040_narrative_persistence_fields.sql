-- Additional narrative persistence fields:
-- message_history: engine's internal message list for summary refresh after restart
-- graph_summary: cached KG summary to avoid expensive rebuild on restart
-- needs_compaction: resume interrupted compaction
ALTER TABLE narrative_memory ADD COLUMN message_history TEXT NOT NULL DEFAULT '[]';
ALTER TABLE narrative_memory ADD COLUMN graph_summary TEXT NOT NULL DEFAULT '';
ALTER TABLE narrative_memory ADD COLUMN needs_compaction INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (40, 'narrative_persistence_fields');
