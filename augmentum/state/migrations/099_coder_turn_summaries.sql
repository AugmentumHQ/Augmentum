-- Coder mode: per-turn summaries that persist across user turns.
-- Solves the "model re-runs the same file_read every turn because it
-- has no memory of what it did before" problem. Each entry is a small
-- JSON dict (goal + files read + files edited + outcome + blockers)
-- generated algorithmically at the end of every _act_hybrid /
-- _act_canonical turn. Capped at 10 entries FIFO in application code.
ALTER TABLE coder_sessions ADD COLUMN turn_summaries TEXT DEFAULT '[]';

INSERT INTO schema_version (version, applied_at) VALUES (99, strftime('%s', 'now'));
