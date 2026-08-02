-- Add avatar column to character_groups for group portrait storage.
-- Stored as base64 data URL (same format as character avatars).

ALTER TABLE character_groups ADD COLUMN avatar TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (44, 'group_avatar');
