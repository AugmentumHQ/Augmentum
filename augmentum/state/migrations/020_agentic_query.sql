-- Add original_query column to agentic_tasks for resume support
ALTER TABLE agentic_tasks ADD COLUMN original_query TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (20, 'Agentic task original query for resume');
