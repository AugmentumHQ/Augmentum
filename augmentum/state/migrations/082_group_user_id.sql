-- Scope character_groups by user. Before this migration, a client could
-- load any user's group via X-Augmentum-Group-Id because GroupStore.get_group
-- had no user_id predicate. With the column added and queries updated, group
-- access follows the same tenancy boundary as every other narrative table.
--
-- Backfill: attribute any existing groups to the oldest active user. This is
-- the safe default for single-user deployments (the overwhelming majority
-- today). Multi-user installs can reassign via the group editor after
-- deployment. Groups with NULL user_id after this are treated as unowned
-- and become invisible to every user — the admin intentionally orphaning
-- them is better than silent cross-tenant exposure.

ALTER TABLE character_groups ADD COLUMN user_id TEXT REFERENCES users(id);

UPDATE character_groups
   SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1)
 WHERE user_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_character_groups_user ON character_groups(user_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (82, 'character_groups_user_id');
