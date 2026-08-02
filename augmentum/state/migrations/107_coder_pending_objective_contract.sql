-- Persist unresolved completion requirements across coder turns.
-- Used for cross-turn continuation grounding when the model still
-- needs to prove or state something plainly before the task is done.

ALTER TABLE coder_sessions
ADD COLUMN pending_objective_contract TEXT DEFAULT '{}';
