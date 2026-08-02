-- Add missing columns to reasoning_flows for autonomy and escalation support.

ALTER TABLE reasoning_flows ADD COLUMN autonomy_level INTEGER DEFAULT 2;
ALTER TABLE reasoning_flows ADD COLUMN escalation_flow TEXT DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (26, 'Add flow autonomy_level and escalation_flow columns');
