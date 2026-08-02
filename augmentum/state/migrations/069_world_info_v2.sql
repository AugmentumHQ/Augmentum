-- World Info V2 — extend lorebook_entries and global_lorebook_entries with
-- secondary keywords, inclusion groups, probability, budget, matching,
-- recursion control, scanning scope, timed effects, outlet, and comment.

-- lorebook_entries
ALTER TABLE lorebook_entries ADD COLUMN secondary_keywords TEXT NOT NULL DEFAULT '[]';
ALTER TABLE lorebook_entries ADD COLUMN selective INTEGER NOT NULL DEFAULT 1;
ALTER TABLE lorebook_entries ADD COLUMN selective_logic INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN group_name TEXT NOT NULL DEFAULT '';
ALTER TABLE lorebook_entries ADD COLUMN group_override INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN group_weight INTEGER NOT NULL DEFAULT 100;
ALTER TABLE lorebook_entries ADD COLUMN probability INTEGER NOT NULL DEFAULT 100;
ALTER TABLE lorebook_entries ADD COLUMN use_probability INTEGER NOT NULL DEFAULT 1;
ALTER TABLE lorebook_entries ADD COLUMN ignore_budget INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN match_whole_words INTEGER;
ALTER TABLE lorebook_entries ADD COLUMN use_group_scoring INTEGER;
ALTER TABLE lorebook_entries ADD COLUMN exclude_recursion INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN prevent_recursion INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN delay_until_recursion INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN match_persona INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN match_char_description INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN match_char_personality INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN match_scenario INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN match_creator_notes INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN delay_turns INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN outlet_name TEXT NOT NULL DEFAULT '';
ALTER TABLE lorebook_entries ADD COLUMN comment TEXT NOT NULL DEFAULT '';

-- global_lorebook_entries
ALTER TABLE global_lorebook_entries ADD COLUMN secondary_keywords TEXT NOT NULL DEFAULT '[]';
ALTER TABLE global_lorebook_entries ADD COLUMN selective INTEGER NOT NULL DEFAULT 1;
ALTER TABLE global_lorebook_entries ADD COLUMN selective_logic INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN group_name TEXT NOT NULL DEFAULT '';
ALTER TABLE global_lorebook_entries ADD COLUMN group_override INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN group_weight INTEGER NOT NULL DEFAULT 100;
ALTER TABLE global_lorebook_entries ADD COLUMN probability INTEGER NOT NULL DEFAULT 100;
ALTER TABLE global_lorebook_entries ADD COLUMN use_probability INTEGER NOT NULL DEFAULT 1;
ALTER TABLE global_lorebook_entries ADD COLUMN ignore_budget INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN match_whole_words INTEGER;
ALTER TABLE global_lorebook_entries ADD COLUMN use_group_scoring INTEGER;
ALTER TABLE global_lorebook_entries ADD COLUMN exclude_recursion INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN prevent_recursion INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN delay_until_recursion INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN match_persona INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN match_char_description INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN match_char_personality INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN match_scenario INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN match_creator_notes INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN delay_turns INTEGER NOT NULL DEFAULT 0;
ALTER TABLE global_lorebook_entries ADD COLUMN outlet_name TEXT NOT NULL DEFAULT '';
ALTER TABLE global_lorebook_entries ADD COLUMN comment TEXT NOT NULL DEFAULT '';
ALTER TABLE global_lorebook_entries ADD COLUMN scan_depth INTEGER NOT NULL DEFAULT 5;
ALTER TABLE global_lorebook_entries ADD COLUMN case_sensitive INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (69, 'World Info V2 fields');
