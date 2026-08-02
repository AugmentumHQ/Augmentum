-- 200_project_checkouts.sql
-- Phase 1, PR-1.2 of the Integrated Coding Nervous System.
-- See docs/superpowers/specs/2026-05-29-integrated-coding-nervous-system.md.
--
-- Migration 199 introduced the durable `Project` substrate. This migration
-- retargets the disposable container layer onto it:
--
--   1. Rename `coder_workspaces` -> `project_checkouts`. The container
--      is the checkout; the bare repo on host is the source of truth.
--   2. Add `project_checkouts.project_id` FK to projects(id).
--   3. Backfill: one Project per existing workspace. The backfill REUSES
--      the existing workspace id as the new project id — this trivializes
--      the link step + keeps every sidecar table whose `workspace_id`
--      column we are NOT renaming (coder_sessions, coder_workspace_services,
--      bug_finder_runs, coder_profile) continuing to resolve correctly,
--      since the same opaque ID now identifies a project.
--   4. Rename `coder_turn_runs.workspace_id` -> `project_id`. Per-turn
--      commit attribution is project-scoped per the spec; the rename
--      reflects that.
--
-- Why reuse the workspace ID as the project ID?
--   * No additional join needed in the linking UPDATE
--   * Sidecar tables that point at workspace_id keep working unchanged
--   * delete_user's runtime user_id sweep already covers every row
--   * Cosmetic loss: backfilled IDs don't carry the 'prj_' prefix that
--     new projects do. Acceptable inconsistency; logs distinguish via
--     the origin='legacy_workspace' tag set below.
--
-- Slug generation
--   * Lowercase, ASCII alnum + dash only is enforced at write time by
--     ProjectStore. For backfill we apply a minimal sanitiser
--     (lowercase, replace space/underscore/slash/dot with dash) plus an
--     ID-suffix tiebreaker to guarantee per-user uniqueness.
--
-- Risk register (from the spec)
--   * "Migration 200 (rename coder_workspaces) breaks tests + audit
--     scripts" — mitigated by updating every code reference in this PR.
--     A view alias is intentionally NOT created: a clean rename is
--     simpler than maintaining two names through one release.
--
-- Idempotence
--   * The migration runner skips files whose version <= max(schema_version),
--     so re-runs do not happen on a healthy install.
--   * Within one pass, the backfill INSERT uses OR IGNORE on the
--     uniq_projects_user_slug constraint and `DROP INDEX IF EXISTS` +
--     `CREATE INDEX IF NOT EXISTS` are guarded.
--   * ALTER TABLE RENAME (TO + COLUMN) is NOT IF-EXISTS-able in SQLite.
--     If a previous attempt failed AFTER step 2 but BEFORE bumping
--     schema_version, retry will error with "no such table:
--     coder_workspaces". This is rare (one ALTER between two
--     guarded steps) and recovery is a one-line manual SQL — better
--     than the alternative of CTAS-and-drop which would silently
--     lose constraints/indexes.


-- 1. Backfill projects FROM the still-named coder_workspaces.
--    Skip rows with NULL/empty user_id (pre-093 legacy, never migrated).
--    Use the workspace id directly as project id; tag origin so we can
--    distinguish backfilled-from-workspace projects from manually-created
--    ones in later analysis.
INSERT OR IGNORE INTO projects (
    id, user_id, slug, name, description, kind, origin,
    default_branch, created_at, updated_at, last_activity_at
)
SELECT
    cw.id,
    cw.user_id,
    -- Slug: best-effort sanitise + workspace-id suffix as a tiebreaker.
    -- The store treats backfilled slugs as opaque; user can rename later.
    LOWER(
        REPLACE(
            REPLACE(
                REPLACE(
                    REPLACE(COALESCE(cw.name, 'workspace'), ' ', '-'),
                    '_', '-'),
                '/', '-'),
            '.', '-')
    ) || '-' || substr(cw.id, -6),
    COALESCE(cw.name, 'Workspace'),
    '',
    'coder',
    'legacy_workspace',
    'main',
    COALESCE(cw.created_at, strftime('%s', 'now')),
    COALESCE(cw.last_active, cw.created_at, strftime('%s', 'now')),
    COALESCE(cw.last_active, cw.created_at, strftime('%s', 'now'))
FROM coder_workspaces cw
WHERE cw.user_id IS NOT NULL AND cw.user_id != '';


-- 2. Rename coder_workspaces -> project_checkouts.
ALTER TABLE coder_workspaces RENAME TO project_checkouts;


-- 3. Add project_id column with SET NULL on delete.
--    SET NULL (not CASCADE) per spec: a checkout might briefly exist
--    without a project during the materialization window, and the
--    delete_user user-scoped sweep handles the row removal anyway.
ALTER TABLE project_checkouts ADD COLUMN project_id TEXT
    REFERENCES projects(id) ON DELETE SET NULL;


-- 4. Link backfilled checkouts to their projects (id == id).
UPDATE project_checkouts SET project_id = id
    WHERE project_id IS NULL AND user_id IS NOT NULL AND user_id != '';


-- 5. Rename the old workspace-named index for hygiene. Index entries
--    still resolve since they were carried by the table rename, but
--    leaving the old name in pragma listings is confusing and the
--    audit script picks up the drift.
DROP INDEX IF EXISTS idx_coder_workspaces_user;
CREATE INDEX IF NOT EXISTS idx_project_checkouts_user
    ON project_checkouts(user_id);

DROP INDEX IF EXISTS idx_coder_workspaces_kind;
CREATE INDEX IF NOT EXISTS idx_project_checkouts_kind
    ON project_checkouts(kind) WHERE kind != 'regular';

CREATE INDEX IF NOT EXISTS idx_project_checkouts_project
    ON project_checkouts(project_id);


-- 6a. Sidecar memory repointing: add project_id to coder_sessions.
--     turn_summaries persist on coder_sessions; tracking project_id
--     here lets a recycled workspace look up prior turn_summaries by
--     project (Phase 2 wires the lookup). Backfilled to the checkout's
--     project_id when available. Nullable for legacy rows.
ALTER TABLE coder_sessions ADD COLUMN project_id TEXT
    REFERENCES projects(id) ON DELETE SET NULL;

UPDATE coder_sessions
SET project_id = (
    SELECT pc.project_id FROM project_checkouts pc
    WHERE pc.id = coder_sessions.workspace_id
)
WHERE project_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_coder_sessions_project
    ON coder_sessions(project_id);


-- 6. Rename coder_turn_runs.workspace_id -> project_id.
--    Per-turn commit attribution is project-scoped: the bare repo
--    survives container recycle and the commit lives in it.
--    Sidecar tables that stay workspace-scoped (coder_sessions,
--    coder_workspace_services, bug_finder_runs, coder_profile) keep
--    workspace_id — their semantics genuinely follow the checkout,
--    not the durable project.
ALTER TABLE coder_turn_runs RENAME COLUMN workspace_id TO project_id;

-- Recreate the index under the new column name. SQLite's RENAME COLUMN
-- updates index definitions automatically, but the name remains the
-- workspace-flavoured one; rename it for hygiene.
DROP INDEX IF EXISTS idx_coder_turn_runs_user_workspace;
CREATE INDEX IF NOT EXISTS idx_coder_turn_runs_user_project
    ON coder_turn_runs(user_id, project_id, started_at DESC);


INSERT OR IGNORE INTO schema_version (version, description)
VALUES (200, 'rename coder_workspaces -> project_checkouts + backfill projects + rename coder_turn_runs.workspace_id');
