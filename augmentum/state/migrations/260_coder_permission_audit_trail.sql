-- 260_coder_permission_audit_trail.sql
-- Durable audit trail for coder tool-permission decisions.
--
-- Before this table, an approval lived only in structlog output: once
-- logs rotate there is no record that the user approved the shell_exec
-- that ran. One row per DECISION (not per tool call):
--   decided_by = 'user'        — Allow/Deny clicked in the approval modal
--   decided_by = 'timeout'     — modal ignored until the registry timeout
--                                fired (decision is always 'denied')
--   decided_by = 'disconnect'  — client went away mid-request (denied)
--   decided_by = 'policy'      — permissions.toml rule matched
--
-- Deliberately NOT recorded: plan-mode 'auto' approvals — auto mode
-- approves every mutating call in a turn (hundreds of rows of noise),
-- and coder_turn_events already records each executed tool call. The
-- audit table answers "who allowed this", not "what ran".
--
-- tool_input is a truncated JSON preview (capped at write time by the
-- store), not the full payload — a file_write's content can be hundreds
-- of KB and the full call record already lives in the turn ledger.

CREATE TABLE IF NOT EXISTS coder_permission_audit (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    workspace_id TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL DEFAULT '',
    tool_input TEXT NOT NULL DEFAULT '{}',
    decision TEXT NOT NULL DEFAULT '',
    decided_by TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_coder_permission_audit_user_ws_time
    ON coder_permission_audit(user_id, workspace_id, created_at);
