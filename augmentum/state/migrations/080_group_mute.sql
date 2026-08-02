-- Group chat Tier 1: per-member mute support.
-- muted_names is a JSON array of character names to exclude from the speaker
-- rotation while still including their card as context (present in scene,
-- just silent). Defaults to empty — existing groups keep current behavior.

ALTER TABLE character_groups ADD COLUMN muted_names TEXT NOT NULL DEFAULT '[]';
