-- 115_narrative_branches.sql
-- First-class branch metadata + ancestry + lifecycle state.
-- Replaces the legacy `branch_states` JSON blob on narrative_memory by promoting
-- branches to their own rows. Ancestry walks (B -> parent -> ... -> main) are done
-- via repeated parent_branch_id lookups; status is a cosmetic UI hint
-- ('active' | 'stale' | 'archived'). NO status causes deletion — explicit DELETE
-- /branches/{id} is the only way data leaves.

CREATE TABLE IF NOT EXISTS narrative_branches (
    branch_id          TEXT NOT NULL,
    session_id         TEXT NOT NULL,
    parent_branch_id   TEXT,                                  -- NULL only for 'main'
    branch_point       INTEGER NOT NULL DEFAULT 0,            -- divergence message_index
    status             TEXT NOT NULL DEFAULT 'active',        -- active | stale | archived
    user_id            TEXT REFERENCES users(id),
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    last_visited_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, branch_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_narrative_branches_user
    ON narrative_branches(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_narrative_branches_parent
    ON narrative_branches(session_id, parent_branch_id);

CREATE INDEX IF NOT EXISTS idx_narrative_branches_status
    ON narrative_branches(session_id, status);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (115, 'narrative_branches table');
