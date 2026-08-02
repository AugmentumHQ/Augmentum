-- 138_coder_workspaces_safeguards.sql
-- Per-workspace toggle for coder soft circuit-breakers.
--
-- When 1 (default), the hybrid loop's soft breakers fire as before
-- (action_stagnation, test_failure_streak, same_file_edit, etc.).
-- When 0, those breakers are bypassed and only the hard iteration
-- ceiling (raised to 500 in phase_act.py) remains as runaway
-- protection. Intended for strong API-backed or strong local models
-- where the breakers cut off legitimate work.

ALTER TABLE coder_workspaces
    ADD COLUMN safeguards_enabled INTEGER NOT NULL DEFAULT 1;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (138, 'Per-workspace coder safeguards toggle');
