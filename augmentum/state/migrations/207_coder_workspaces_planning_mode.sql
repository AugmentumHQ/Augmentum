-- 207_coder_workspaces_planning_mode.sql
-- Per-workspace planning-mode toggle for the coder agent.
--
-- Lives on project_checkouts (the renamed-as-of-migration-200 table
-- that holds workspace container metadata). Naming note: the file
-- name preserves "coder_workspaces" for grep/searchability with the
-- earlier 138_coder_workspaces_safeguards.sql pattern even though
-- the live table is project_checkouts.
--
-- Values match the converged Shift+Tab cycle from Claude Code:
--
--   "default" (DEFAULT) — current behavior. Plan + Act with per-tool
--     permission prompts. The legacy shape; preserves expectations
--     for existing workspaces post-migration.
--
--   "plan" — read-only exploration. The tool allowlist filter
--     excludes write/shell tools (file_write, code_edit, multi_edit,
--     shell_exec). System prompt nudges the model to gather context
--     and propose a plan rather than start editing. Used as the
--     "explore first" checkpoint before unleashing edits.
--
--   "auto" — auto-approve mode. The permission policy resolver
--     short-circuits to "allow" for every tool, skipping the modal
--     entirely. Used during trusted long-form work after the user
--     has validated the agent's behavior. Equivalent to CC's
--     "auto-accept edits" mode.
--
-- Toggle via PUT /api/coder/workspaces/{id}/planning-mode or the
-- Shift+Tab keybinding in the composer UI. Persists per-workspace
-- so a user's preference survives container restart.

ALTER TABLE project_checkouts
    ADD COLUMN planning_mode TEXT NOT NULL DEFAULT 'default';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (207, 'Per-workspace coder planning mode (default | plan | auto)');
