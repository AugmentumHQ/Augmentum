-- 153_companion_state_log.sql
-- Append-only state transition history.
--
-- Every accepted transition on any of the three axes (state, role, focus)
-- writes a row here. This is the audit trail for state-machine behavior:
-- it lets the trigger model in Sprint 4a recompose "what was she doing
-- when" without keeping the full event log in memory.
--
-- For role transitions (3-vector deltas), from_value and to_value carry
-- the vector as a JSON string. For discrete axes (state, focus) they
-- carry the value directly. The runtime never modifies past rows.

CREATE TABLE IF NOT EXISTS companion_state_log (
    id             INTEGER PRIMARY KEY,
    companion_id   TEXT NOT NULL,
    ts             TEXT NOT NULL DEFAULT (datetime('now')),
    axis           TEXT NOT NULL,                -- state|role|focus
    from_value     TEXT,                          -- prior value; JSON for role
    to_value       TEXT NOT NULL,                 -- new value; JSON for role
    reason         TEXT NOT NULL DEFAULT ''       -- transition_reason
);

CREATE INDEX IF NOT EXISTS idx_cstate_log_companion_ts
    ON companion_state_log(companion_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_cstate_log_axis_ts
    ON companion_state_log(companion_id, axis, ts DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (153, 'companion_state_log: append-only state transition history');
