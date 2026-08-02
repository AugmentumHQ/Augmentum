-- 211_coder_workspaces_always_on.sql
-- Per-workspace container lifecycle policy: always_on vs on-demand.
--
-- Lives on project_checkouts (the renamed-as-of-migration-200 table
-- that holds workspace container metadata). Filename keeps the
-- "coder_workspaces" prefix for grep-continuity with the earlier
-- 138 / 207 / 208 / 209 patterns even though the live table is
-- project_checkouts.
--
-- Before this migration:
--   Coder workspaces had no idle reaper. Containers stayed running
--   indefinitely once created, until the user explicitly stopped or
--   deleted them. This was effectively "always on, manually managed."
--   The downside: forgotten workspaces accumulate, holding RAM + ports
--   + (when the workspace is hosting a dev server) network sockets.
--
-- After this migration:
--   ``always_on=0`` (default for NEW workspaces): the workspace
--     participates in the idle reaper. A background sweep stops the
--     container after CODER_IDLE_TIMEOUT seconds of no activity
--     (default 600s / 10 min). The DB row + volume survive — only the
--     container process is stopped. Restarting is one click / next chat.
--     This is the right default for transient "try a thing" workspaces.
--
--   ``always_on=1``: the workspace is exempt from the reaper. The
--     container stays running even when no client is connected.
--     Use this for workspaces hosting a long-running dev server,
--     test harness, or daemon that the user wants to leave up while
--     they switch tabs / step away. The workspace remains disposable
--     via the new "Stop now" button or the existing delete path.
--
-- Backfill: every EXISTING row is set to always_on=1. This preserves
-- the pre-migration behavior (containers don't auto-stop) for
-- workspaces the user already trusts, and avoids a surprise where
-- their daemons get reaped after a server restart. New workspaces
-- created after this migration default to always_on=0 (on-demand),
-- matching the ALTER's DEFAULT clause.
--
-- Toggle via PUT /api/coder/workspaces/{id}/always-on. Persists per-
-- workspace so the choice survives container restart.

ALTER TABLE project_checkouts
    ADD COLUMN always_on INTEGER NOT NULL DEFAULT 0;

-- Preserve existing behavior for rows created before the reaper
-- existed. Without this UPDATE, every legacy workspace would start
-- being auto-stopped on the next idle window.
UPDATE project_checkouts SET always_on = 1;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (211, 'Per-workspace coder container always_on flag (0=on-demand reaper, 1=persist)');
