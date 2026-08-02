-- Add member_summaries column to character_groups for user-editable
-- per-character compact summaries used in group chat prompts.
-- JSON object: {"CharName": "custom summary text", ...}

ALTER TABLE character_groups ADD COLUMN member_summaries TEXT NOT NULL DEFAULT '{}';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (43, 'group_member_summaries');
