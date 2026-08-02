-- 322_harness_assignment_run_link.sql
-- Link a bridge assignment to the coding_runs row that tracks it, so the
-- "My machine" External history advances from queued → working → done as the
-- agent on the user's machine picks it up and reports back.
--
-- Tier-1 "My machine" pickup (2026-07-21): the composer's harness dispatch
-- already creates BOTH a harness_agent_requests row (kind='assignment', the
-- work handed to a live agent) AND a coding_runs row (driver='harness', the
-- status record shown in the Agents history). They weren't linked, so the row
-- sat at 'queued' forever even after the agent ran the task. This column ties
-- them: on delivery at check-in we flip the run to 'working'; on the session's
-- done/failed check-in we finalize it. Provider-neutral — claude-aug and pi
-- share the same check-in lifecycle. See docs/superpowers/specs/
-- 2026-07-21-my-machine-assignment-pickup.md.

ALTER TABLE harness_agent_requests ADD COLUMN linked_run_id TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (322, 'harness_agent_requests.linked_run_id: tie an assignment to its coding_runs status row for My-machine pickup lifecycle');
