-- 116_narrative_state_snapshots.sql
-- Per-branch STATE history (no overwrite). Each refresh appends a new row tagged
-- (branch_id, message_index). On rollback_to(N), engine reads the most recent
-- snapshot with branch_id IN ancestry(B) AND message_index < N. Eliminates the
-- empty-STATE-on-first-turn-after-branch bug — the model always gets the prior
-- snapshot rounded down to the nearest refresh.

CREATE TABLE IF NOT EXISTS narrative_state_snapshots (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    branch_id       TEXT NOT NULL,
    message_index   INTEGER NOT NULL,
    snapshot_data   TEXT NOT NULL,                          -- JSON of StateSnapshot
    user_id         TEXT REFERENCES users(id),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_state_snapshots_lookup
    ON narrative_state_snapshots(session_id, branch_id, message_index);

CREATE INDEX IF NOT EXISTS idx_state_snapshots_user
    ON narrative_state_snapshots(user_id, session_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (116, 'narrative_state_snapshots table');
