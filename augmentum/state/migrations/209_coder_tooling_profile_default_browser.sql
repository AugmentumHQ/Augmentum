-- 209_coder_tooling_profile_default_browser.sql
-- Shift the workspace tooling-profile default from "standard" to "browser".
--
-- Background: migration 140 added the column with DEFAULT 'standard'.
-- Since then, the manager (ContainerManager.create_workspace),
-- config (coder_default_tooling_profile), and UI all converged on
-- "browser" as the right default — the browser profile installs
-- Playwright + Chromium which the coder agent's browser_evaluate /
-- screenshot / verify-preview tools depend on. The route schema
-- default still said "standard", so any client that POSTed without
-- explicitly setting tooling_profile (including older UI builds and
-- non-UI callers) landed on "standard" and silently lost the
-- Playwright tooling.
--
-- The recreate path preserves whatever was persisted at creation
-- time, so a workspace stuck on "standard" never self-heals across
-- container restarts. This backfill flips legacy rows so the next
-- recreate runs the browser setup block and pip-installs Playwright.
-- Running containers are NOT mutated in place — the install happens
-- the next time the container is recreated or restarted.
--
-- Rows that have been intentionally moved off "standard" (to "power"
-- or already to "browser") are untouched. The route default and
-- column default in 140 are aligned by this migration; no future
-- workspace should land on the legacy sentinel unless a caller
-- explicitly passes tooling_profile="standard".

UPDATE project_checkouts
SET tooling_profile = 'browser'
WHERE tooling_profile = 'standard'
   OR tooling_profile IS NULL
   OR tooling_profile = '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (209, 'Flip workspace tooling_profile default to browser');
