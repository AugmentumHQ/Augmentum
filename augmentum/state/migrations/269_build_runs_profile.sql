-- 269_build_runs_profile.sql
-- Builder Profiles foundation (spec:
-- docs/superpowers/specs/2026-06-15-builder-profiles-system-synthesizer-design.md).
--
-- The builder is moving from a fixed multi-pass pipeline to the coder
-- agentic harness driven by a Builder Power. A build is now dual-resident:
-- a live workspace (project_checkout) AND a library artifact. These
-- additive columns let a build_run record which profile produced it, how
-- it's delivered, which OS capabilities it was granted, and which
-- workspace backs it (so the library Play surface can preview the live
-- dev server while the build is still running).
--
--   profile_id        — the Builder Profile / Power id ('static','game',…)
--   target            — 'inline' (also emits a static bundle) | 'workspace'
--   capabilities_json — granted substrate capabilities (P4 bridge); '[]' now
--   workspace_id      — the project_checkout backing this build, when any
--
-- All additive, all back-compat: existing rows default to the historical
-- static/inline shape.

ALTER TABLE build_runs ADD COLUMN profile_id TEXT NOT NULL DEFAULT 'static';
ALTER TABLE build_runs ADD COLUMN target TEXT NOT NULL DEFAULT 'inline';
ALTER TABLE build_runs ADD COLUMN capabilities_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE build_runs ADD COLUMN workspace_id TEXT;

CREATE INDEX IF NOT EXISTS idx_build_runs_user_profile
    ON build_runs(user_id, profile_id);
