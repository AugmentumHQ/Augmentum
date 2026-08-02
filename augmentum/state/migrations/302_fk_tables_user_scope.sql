-- 302_fk_tables_user_scope.sql
-- Denormalize user_id onto 4 FK-only tables for defense-in-depth.
--
-- These tables are always accessed via a user-scoped parent (projects,
-- coder_turn_runs, companion_skill_instances), but direct queries would
-- be unscoped. Adding user_id allows direct WHERE clauses without JOINs
-- and closes the tenant-isolation gap.

ALTER TABLE companion_skill_outcomes ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
ALTER TABLE project_repos ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
ALTER TABLE project_refs ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
ALTER TABLE coder_turn_events ADD COLUMN user_id TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (302, 'companion_skill_outcomes/project_repos/project_refs/coder_turn_events += user_id: defense-in-depth tenant isolation');
