-- 199_projects.sql
-- Phase 1, PR-1.1 of the Integrated Coding Nervous System.
-- See docs/superpowers/specs/2026-05-29-integrated-coding-nervous-system.md.
--
-- Introduces `Project` as the canonical noun for user-owned coding work.
-- A Project is the durable substrate behind every coding interaction:
-- chat code blocks, App Builder runs, and Coder workspaces all become
-- views over the same Project. The container is a disposable checkout;
-- the bare git repo at {data_dir}/projects/{user_id}/{project_id}.git/
-- is the source of truth and survives every container recycle.
--
-- This PR is purely additive: tables only, no behavior change. PR-1.2
-- (migration 200) renames coder_workspaces -> project_checkouts and
-- repoints the container path through the bare repo. PR-1.3 (migration
-- 205) wires library_publications onto project_refs.
--
-- Tables:
--   projects        — user-owned coding artifact identity
--   project_repos   — 1:1 sidecar describing the on-disk bare repo
--   project_refs    — every named pointer (branch/tag/savepoint/...)
--
-- Conventions:
--   - REAL epoch seconds for timestamps (matches 197/198, ledger family)
--   - ON DELETE CASCADE on user_id matches the user-deletion strands fix
--     in [[project_user_deletion_strands_data]]
--   - kind / origin / ref kind are open strings; v1 enum is documented
--     here, future kinds add without a migration
--   - slug uniqueness is per-user, NOT global. Marketplace handles
--     cross-user dedup separately via display name + author.
--
-- ON-DISK NOTE:
--   delete_user() sweeps every table with a user_id column, which
--   covers the rows above. The bare repo dir at
--   {data_dir}/projects/{user_id}/ lives outside the DB and must be
--   rmtree'd separately — wired in PR-1.1 task #79.

CREATE TABLE IF NOT EXISTS projects (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    slug                TEXT NOT NULL,
    name                TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    -- kind: 'scratchpad' | 'app' | 'coder' (v1)
    kind                TEXT NOT NULL,
    -- origin: 'chat:<msg_id>' | 'builder:<run_id>' | 'manual' |
    --         'clone:<url>' | 'fork:<project_id>'
    origin              TEXT NOT NULL DEFAULT 'manual',
    default_branch      TEXT NOT NULL DEFAULT 'main',
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    last_activity_at    REAL NOT NULL,
    archived_at         REAL
);

-- Listing path: user's projects newest-active first.
CREATE INDEX IF NOT EXISTS idx_projects_user_activity
    ON projects(user_id, last_activity_at DESC);

-- Slug-collision path: (user_id, slug) is the soft uniqueness key.
-- A UNIQUE index — not constraint — keeps the slug query fast and
-- lets store.py raise a friendly error before insert.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_projects_user_slug
    ON projects(user_id, slug);


CREATE TABLE IF NOT EXISTS project_repos (
    project_id          TEXT PRIMARY KEY
                        REFERENCES projects(id) ON DELETE CASCADE,
    -- repo_path is absolute on host. Stored explicitly (rather than
    -- derived) so a data-dir relocation can be reconciled by rewriting
    -- this column instead of re-deriving every path at read time.
    repo_path           TEXT NOT NULL,
    head_ref            TEXT NOT NULL DEFAULT 'refs/heads/main',
    sha_count           INTEGER NOT NULL DEFAULT 0,
    size_bytes          INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
);


CREATE TABLE IF NOT EXISTS project_refs (
    id                  TEXT PRIMARY KEY,
    project_id          TEXT NOT NULL
                        REFERENCES projects(id) ON DELETE CASCADE,
    -- kind: 'branch' | 'tag' | 'savepoint' | 'publication' | 'share'
    kind                TEXT NOT NULL,
    -- ref_name is the full git ref, e.g.
    --   'refs/heads/main', 'refs/savepoints/<uuid>',
    --   'refs/published/<pub_id>'.
    ref_name            TEXT NOT NULL,
    sha                 TEXT NOT NULL,
    label               TEXT NOT NULL DEFAULT '',
    created_at          REAL NOT NULL,
    -- created_by_message_id ties a save-point to the chat turn that
    -- made it. NULL for user-driven refs and migration backfill.
    created_by_message_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_project_refs_project_kind
    ON project_refs(project_id, kind);

CREATE UNIQUE INDEX IF NOT EXISTS uniq_project_refs_project_ref
    ON project_refs(project_id, ref_name);


INSERT OR IGNORE INTO schema_version (version, description)
VALUES (199, 'projects + project_repos + project_refs — Phase 1 PR-1.1');
