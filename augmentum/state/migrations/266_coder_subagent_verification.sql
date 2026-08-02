-- Subagent return-path verification (agents/verify.py).
-- When a subagent stops cleanly and the lead handed down success_criteria,
-- an independent judge checks the output satisfies each criterion before the
-- stop is honored. Persist the outcome so the lead, the subagent history
-- sidebar, and the eval harness can tell a verified completion from an
-- unverified (or failed) one.
--
-- verification: 'unchecked' (gate off / no criteria) | 'passed' | 'failed'
--   (criteria unmet after re-entries exhausted) | 'error' (judge gave no
--   signal — failed open, treat as unchecked for trust).
-- verification_reason: the judge's one-line reason when failed; '' otherwise.
ALTER TABLE coder_subagent_runs ADD COLUMN verification TEXT NOT NULL DEFAULT 'unchecked';
ALTER TABLE coder_subagent_runs ADD COLUMN verification_reason TEXT NOT NULL DEFAULT '';
