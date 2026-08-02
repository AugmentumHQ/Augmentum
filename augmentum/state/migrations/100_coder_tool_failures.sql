-- Coder mode: recent soft-failure ring buffer, rendered in the sticky
-- reminder under "Recent repeated failures". Dedup key is (tool, target)
-- so retries on the same path/command get counted together instead of
-- flooding the reminder. Example row: {tool: "code_edit", target:
-- "/snake.html", error: "stale read", count: 5, last_at: ...}.
-- Capped at 4 entries FIFO in application code.
ALTER TABLE coder_sessions ADD COLUMN recent_tool_failures TEXT DEFAULT '[]';

INSERT INTO schema_version (version, applied_at) VALUES (100, strftime('%s', 'now'));
