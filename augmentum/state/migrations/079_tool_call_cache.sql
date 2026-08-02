-- 079_tool_call_cache.sql
-- Persistent cache of tool execution results keyed by
-- (task_id, step_idx, call_hash). On agentic task resume, completed tool
-- calls are replayed from this table instead of re-running the tool.
--
-- call_hash = sha256(tool_name + canonical_json(args))
-- step_idx  = flow-step index at which the call was made (for scoping)
-- output    = tool's textual output (truncated at write-time if needed)
-- metadata  = JSON-serialized metadata dict (artifact URLs, etc.)
-- success   = 1 if the tool returned success, 0 otherwise
-- ts        = write time (ISO8601)

CREATE TABLE IF NOT EXISTS agentic_tool_call_cache (
    task_id    TEXT NOT NULL,
    step_idx   INTEGER NOT NULL,
    call_hash  TEXT NOT NULL,
    tool_name  TEXT NOT NULL,
    output     TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}',
    success    INTEGER NOT NULL DEFAULT 1,
    ts         TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (task_id, step_idx, call_hash)
);

CREATE INDEX IF NOT EXISTS idx_tool_call_cache_task
    ON agentic_tool_call_cache(task_id);
