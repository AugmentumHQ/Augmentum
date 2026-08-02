-- Add user_id to character_relationships.
--
-- Migration 072 added user_id to the other user-scoped narrative tables
-- (facts, entities, plot_threads, contradictions, lorebook_entries,
-- assumptions, character_cards, narrative_memory) but missed this one.
-- Persistence code in narrative_persistence._save_relationships writes
-- "AND user_id = ?" when a user_id is supplied, which raised
-- OperationalError ("no such column: user_id") on every narrative save
-- with relationships present. The error was caught and logged as
-- relationship_save_failed, so relationships silently stopped persisting.
--
-- Nullable for backward compat, backfilled below (matches 072/083 pattern).

ALTER TABLE character_relationships ADD COLUMN user_id TEXT REFERENCES users(id);

CREATE INDEX IF NOT EXISTS idx_charrel_user ON character_relationships(user_id);

UPDATE character_relationships
SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1)
WHERE user_id IS NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (84, 'character_relationships_user_id');
