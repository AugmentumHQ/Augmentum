-- 143_coder_workspaces_bug_finder.sql
-- Workspace kind + optional Bug Finder verifier model.
--
-- ``kind`` is the workspace type:
--   * ``regular`` (default) — standard coder workspace; terminal + files +
--     preview + the AI coding agent.
--   * ``bug_finder`` — workspace dedicated to autonomous bug hunting;
--     the coder UI surfaces a Bug Finder tab in the workbench and the
--     primary surface is the audit report. All other coder features
--     (terminal, files, preview) still work in the workspace, but Bug
--     Finder is the home view on entry.
--
-- ``bug_finder_verifier_model`` is the optional per-workspace verifier
-- model override. Nullable on purpose: the local-hardware default is
-- single-model self-verification (planner/detector/verifier/fixer all
-- use the user's currently-selected model), because swapping a second
-- model onto the GPU costs more in real-world thrashing than the
-- correlated-error risk it would mitigate. Users opt in to a different
-- verifier per workspace when they have the resources for it.

ALTER TABLE coder_workspaces
    ADD COLUMN kind TEXT NOT NULL DEFAULT 'regular';

ALTER TABLE coder_workspaces
    ADD COLUMN bug_finder_verifier_model TEXT;

CREATE INDEX IF NOT EXISTS idx_coder_workspaces_kind
    ON coder_workspaces(kind) WHERE kind != 'regular';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (143, 'coder_workspaces: kind + bug_finder_verifier_model');
