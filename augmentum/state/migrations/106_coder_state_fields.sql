-- Coder mode: fields that CoderState has serialized for a while but the
-- backing table never fully grew to match. Without these columns the live
-- persistence path either drops useful state or route-level cleanup queries
-- explode on upgraded databases.
ALTER TABLE coder_sessions ADD COLUMN tasks TEXT DEFAULT '[]';
ALTER TABLE coder_sessions ADD COLUMN recent_validation_errors TEXT DEFAULT '[]';
ALTER TABLE coder_sessions ADD COLUMN recent_tool_calls TEXT DEFAULT '[]';
ALTER TABLE coder_sessions ADD COLUMN background_processes TEXT DEFAULT '[]';
ALTER TABLE coder_sessions ADD COLUMN iterations_remaining INTEGER DEFAULT 20;
ALTER TABLE coder_sessions ADD COLUMN iterations_ceiling INTEGER DEFAULT 75;
ALTER TABLE coder_sessions ADD COLUMN iterations_since_progress INTEGER DEFAULT 0;
ALTER TABLE coder_sessions ADD COLUMN fanout_limit INTEGER DEFAULT 5;
ALTER TABLE coder_sessions ADD COLUMN consecutive_failures INTEGER DEFAULT 0;

INSERT INTO schema_version (version, applied_at) VALUES (106, strftime('%s', 'now'));
