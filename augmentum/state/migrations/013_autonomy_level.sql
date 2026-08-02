-- Add autonomy_level to reasoning flows for agentic mode control.
-- 1=suggest, 2=ask (default), 3=inform, 4=autonomous

ALTER TABLE reasoning_flows ADD COLUMN autonomy_level INTEGER DEFAULT 2;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (13, 'Autonomy level on reasoning flows');
