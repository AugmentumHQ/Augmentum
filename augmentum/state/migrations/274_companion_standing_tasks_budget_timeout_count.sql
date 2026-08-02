-- 274_companion_standing_tasks_budget_timeout_count.sql
-- Separate a "ran out of its time budget" cancellation from a real
-- error in the standing-task auto-pause logic (audit 2026-06-17).
--
-- Background: a standing task that merely runs slow (its verb's
-- asyncio.wait_for wallclock budget cancels it) used to increment the
-- SAME consecutive_error_count as a genuinely-broken task, so 5 slow
-- runs auto-paused a perfectly-healthy task — the 2026-06-08 incident
-- where a slow briefing paused tick_scheduler for ~7 hours.
--
-- This additive column lets the runner track budget-timeout cancels on
-- their own counter with a looser threshold, so a transiently-slow task
-- recovers while a permanently-stuck one still eventually pauses. Both
-- counters reset to 0 on any successful run.
--
-- All additive, back-compat: existing rows default to 0.
ALTER TABLE companion_standing_tasks
    ADD COLUMN consecutive_budget_timeout_count INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (274, 'companion_standing_tasks: separate consecutive_budget_timeout_count from error count (audit 2026-06-17)');
