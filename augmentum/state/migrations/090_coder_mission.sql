-- Coder mode: structured mission (list of Promises with verification specs).
-- Supersedes the free-text `plan_steps` for new sessions; existing rows
-- default to an empty mission and use the legacy plan_steps path.
ALTER TABLE coder_sessions ADD COLUMN mission TEXT DEFAULT '[]';

INSERT INTO schema_version (version, applied_at) VALUES (90, strftime('%s', 'now'));
