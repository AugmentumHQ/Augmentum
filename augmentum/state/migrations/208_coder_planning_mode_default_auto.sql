-- 208_coder_planning_mode_default_auto.sql
-- Shift the planning-mode default from "default" to "auto".
--
-- Background: migration 207 added the planning_mode column with
-- DEFAULT 'default', which gated every tool call behind a per-tool
-- permission modal. UX is friction-heavy out of the box and doesn't
-- match the "trust the model" pattern that modern coding agents
-- (Cursor Composer, Aider) ship by default. New stance:
--
--   "auto" (default) — model runs freely, no confirmation modals.
--                      Permission policy file still applies if the
--                      operator has authored explicit deny rules.
--   "default"        — per-tool permission prompts on mutations.
--                      Renamed to "approve" in the UI label, keeping
--                      the legacy string here to avoid a vocabulary
--                      churn migration.
--   "plan"           — soft planning guidance. Model is nudged to
--                      outline an approach before editing, but ALL
--                      tools remain available. The model decides
--                      when to propose-and-wait vs proceed. Hard
--                      tool filter was removed alongside this
--                      migration; the previous "plan = read-only by
--                      construction" semantics live on as an opt-in
--                      via future ``planning_mode='plan_strict'`` if
--                      requested.
--
-- Per-workspace cycle stays Shift+Tab in the composer; users who
-- intentionally selected one of the other modes (rare given 207
-- shipped earlier today) keep their choice.

-- Bulk-update rows that still hold the column-default sentinel.
-- Rows the user has intentionally set to "plan" or already to "auto"
-- are untouched. Safe because no UX surface gave users a way to
-- distinguish "intentionally default" from "never touched" — the
-- former wouldn't exist in practice given 207 just landed.
UPDATE project_checkouts
SET planning_mode = 'auto'
WHERE planning_mode = 'default'
   OR planning_mode IS NULL
   OR planning_mode = '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (208, 'Flip planning_mode default to auto + soft plan mode');
