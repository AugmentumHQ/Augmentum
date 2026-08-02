ALTER TABLE reasoning_flow_steps ADD COLUMN model_override TEXT DEFAULT '';
INSERT OR IGNORE INTO schema_version (version, description) VALUES (33, 'Per-step model override');
