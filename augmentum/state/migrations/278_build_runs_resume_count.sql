-- 278_build_runs_resume_count.sql
-- Resumable / re-promptable builds: track how many times a build run has
-- been continued.
--
-- A build that stops (budget exhausted mid-verify, stuck, cancelled, or a
-- finished build the user wants to extend) can now be RESUMED on its existing
-- workspace instead of restarted from scratch. Each resume re-enters the
-- autonomous loop on the same build_runs row; this counter records how many
-- times that happened so the surface can show "continued N×" and a future
-- guard can cap runaway resume loops.
--
-- All additive, back-compat: existing rows default to 0 (never resumed).
ALTER TABLE build_runs
    ADD COLUMN resume_count INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (278, 'build_runs: resume_count for resumable / re-promptable builds');
