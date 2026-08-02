-- Per-turn cost capture for the coder inspector's Cost section.
--
-- Three columns added to coder_turn_runs:
--   input_cost_usd   — prompt-side USD (lookup_cost(model) * prompt_tokens)
--   output_cost_usd  — completion-side USD (lookup_cost(model) * completion_tokens)
--   cost_model_id    — the model name we costed against (for display + audit)
--
-- Local-hosted models lookup to (0, 0) in cost_table.py → row records $0 cost.
-- Cloud-hosted (Anthropic / OpenAI / Groq / OpenRouter via Fabric) records
-- real spend. The inspector's /inspector-state endpoint sums these columns
-- per (workspace_id, session_id, user_id).
--
-- See docs/superpowers/specs/2026-05-28-coder-inspector-design.md §6.

ALTER TABLE coder_turn_runs ADD COLUMN input_cost_usd REAL NOT NULL DEFAULT 0;
ALTER TABLE coder_turn_runs ADD COLUMN output_cost_usd REAL NOT NULL DEFAULT 0;
ALTER TABLE coder_turn_runs ADD COLUMN cost_model_id TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (198, 'Coder turn cost capture columns for inspector panel');
