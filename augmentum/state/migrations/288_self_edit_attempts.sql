-- 288_self_edit_attempts.sql
-- The lineage/archive for Augmentum editing ITSELF (self-improvement).
--
-- Pillar (Westworld-rooted): grow by remembering mistakes; the archive is
-- sacred; a rollback restores CODE but never erases the LESSON. So this table
-- has NO prune/delete path by design — unlike companion_journal (which
-- soft/hard-deletes on a TTL), every self-edit attempt and what was learned
-- from it is kept permanently. The whole point of self-improvement is the
-- accumulated record of what was tried, what passed the fitness gate, what
-- failed, and why.
--
-- One row per ATTEMPT (propose -> isolate candidate -> agent edits -> fitness
-- gate -> promote|reject|rollback). The detailed edit transcript lives in the
-- linked Claude run (claude_runs/claude_run_events via run_id, migration 287);
-- this table is the higher-level decision/outcome ledger.
--
-- User-scoped per CLAUDE.md (`user_id` column).

CREATE TABLE IF NOT EXISTS self_edit_attempts (
    id              TEXT PRIMARY KEY,                 -- uuid hex
    user_id         TEXT NOT NULL,
    objective       TEXT NOT NULL DEFAULT '',         -- what was asked / proposed
    surface         TEXT NOT NULL DEFAULT '',         -- frontend | backend | prompt | config | ...
    tier            TEXT NOT NULL DEFAULT 'green',     -- green | yellow | red (autonomy gradient)
    status          TEXT NOT NULL DEFAULT 'proposed',  -- proposed|editing|gated|promoted|rejected|rolled_back|failed
    base_ref        TEXT NOT NULL DEFAULT '',         -- commit SHA the candidate branched from
    candidate_ref   TEXT NOT NULL DEFAULT '',         -- candidate branch / worktree name
    run_id          TEXT NOT NULL DEFAULT '',         -- linked claude_runs.id (the editing transcript)
    gate_passed     INTEGER NOT NULL DEFAULT 0,        -- 1 iff the fitness gate passed
    gate_verdict    TEXT NOT NULL DEFAULT '{}',        -- json: per-check results + evidence
    files_changed   TEXT NOT NULL DEFAULT '[]',        -- json array of paths
    outcome         TEXT NOT NULL DEFAULT '',          -- one-line result
    -- THE pillar field: what was learned, especially on failure/rollback.
    -- Never wiped — survives even when the code change is reverted.
    lesson          TEXT NOT NULL DEFAULT '',
    promoted_commit TEXT NOT NULL DEFAULT '',          -- SHA promoted to the live branch, if any
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_self_edit_user_created
    ON self_edit_attempts(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_self_edit_status
    ON self_edit_attempts(user_id, status);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (288, 'self_edit_attempts: permanent never-pruned lineage of Augmentum self-edits (objective/candidate/gate/outcome/lesson)');
