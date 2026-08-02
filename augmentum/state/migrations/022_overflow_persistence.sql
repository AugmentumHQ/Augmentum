-- 022_overflow_persistence.sql: Persist overflow summaries and archived messages

ALTER TABLE narrative_memory ADD COLUMN overflow_summaries TEXT NOT NULL DEFAULT '[]';
ALTER TABLE narrative_memory ADD COLUMN archived_messages TEXT NOT NULL DEFAULT '[]';

INSERT INTO schema_version (version, description) VALUES (22, 'Overflow summaries and archived message persistence');
