-- Add conversation history column to coder_sessions.
-- Stores the full message array as JSON: [{id, role, content, tool, input, result, metadata}]
ALTER TABLE coder_sessions ADD COLUMN conversation TEXT DEFAULT '[]';

INSERT INTO schema_version (version, applied_at) VALUES (77, strftime('%s', 'now'));
