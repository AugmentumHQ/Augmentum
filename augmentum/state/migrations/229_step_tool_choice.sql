ALTER TABLE reasoning_flow_steps ADD COLUMN tool_choice TEXT DEFAULT '';
INSERT OR IGNORE INTO schema_version (version, description) VALUES (229, 'Per-step tool_choice (auto/required/none/tool name)');
