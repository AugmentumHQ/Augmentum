-- 173_companion_owner_user_id.sql
-- Resolves the dream gate at sleep_wake.invoke_dream:67.
--
-- CompanionRuntime is companion-scoped (companion_id) but the DreamEngine
-- pipeline filters every step by user_id. Without a resolvable owner_user_id
-- runtime-driven dreams silently no-op (memories_count=0 → silent exit).
--
-- Single-companion phase: one row in companion_identities, owned by one
-- user. Sprint 7+ household phase introduces multiple companions per user
-- and may flip this to NOT NULL with a backfill rule from a future
-- companion_owners join table; for now NULLABLE preserves a clean
-- "unowned" sentinel for tests and bare-fixture installs.

ALTER TABLE companion_identities ADD COLUMN owner_user_id TEXT
    REFERENCES users(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_companion_identities_owner
    ON companion_identities(owner_user_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (173, 'companion_owner_user_id: resolves dream + tick user-scoping');
