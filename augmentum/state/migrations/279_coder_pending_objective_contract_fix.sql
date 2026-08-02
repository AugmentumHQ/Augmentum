-- 279_coder_pending_objective_contract_fix.sql
-- Repair: re-add coder_sessions.pending_objective_contract.
--
-- Migration 107 added this column, and 107 is RECORDED as applied — but on at
-- least one install the column is absent from coder_sessions, so every coder
-- turn logs `coder_state_persist_failed: table coder_sessions has no column
-- named pending_objective_contract` and that field silently fails to persist.
-- The likely cause is the migration_files_applied backfill marking 107 as
-- applied from the schema_version watermark WITHOUT re-running its ALTER on a
-- DB where the column never actually landed.
--
-- This re-runs the ALTER under a NEW version (> watermark) so it executes. The
-- migration runner catches "duplicate column" individually (see
-- sqlite.py::_run_migrations), so on installs where 107 took correctly this is
-- a graceful no-op. Additive + back-compat: existing rows default to '{}'.
ALTER TABLE coder_sessions
    ADD COLUMN pending_objective_contract TEXT DEFAULT '{}';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (279, 'coder_sessions: re-add pending_objective_contract (107 drift repair)');
