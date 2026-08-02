-- Add last_request_log column to narrative_memory for context viewer persistence
ALTER TABLE narrative_memory ADD COLUMN last_request_log TEXT NOT NULL DEFAULT '{}';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (47, 'Narrative request log persistence');
