-- Per-turn priming telemetry — branch-level token counts captured at
-- system-prompt build time. Lets us verify the priming tree (intent
-- exemplar, tool shortlist, profile facts, sticky reminder, powers)
-- is actually saving budget without losing quality.
--
-- Stored as JSON of the shape:
--   {
--     "intent": "DEBUG",
--     "tier":   "native",
--     "branches": {
--       "rules": 634, "exemplar": 281, "tool_short": 412,
--       "profile": 118, "sticky": 384, "powers": 0
--     },
--     "total_priming_tokens": 1829,
--     "exemplar_loaded": true
--   }
-- Empty default ('{}') keeps backfill on existing rows trivial.

ALTER TABLE coder_turn_runs ADD COLUMN priming_telemetry TEXT NOT NULL DEFAULT '{}';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (196, 'Coder priming-tree per-branch token telemetry');
