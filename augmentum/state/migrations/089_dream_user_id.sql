-- Add user_id to dream-system tables.
-- Migration 072 scoped the primary user-data tables but skipped the dream
-- system, which was landed in 058 before multi-tenancy existed. Dream
-- content is among the most personal data the system produces (synthetic
-- autobiographical reflections), so scoping it is required before the
-- dream system is exposed in a multi-user deployment.
--
-- Nullable for backward compat (SQLite can't ADD NOT NULL to existing).
-- Existing rows stay NULL; application code treats an empty user_id as
-- "do not filter" so single-user installs continue to work.

ALTER TABLE dream_entries ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE dream_portraits ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE dream_cycles ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE dream_memory_log ADD COLUMN user_id TEXT REFERENCES users(id);

-- Indexes for the primary (user_id, persona_id) and (user_id, created_at)
-- access patterns used by the journal and portrait managers.
CREATE INDEX IF NOT EXISTS idx_dream_entries_user_persona
    ON dream_entries(user_id, persona_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dream_portraits_user_persona
    ON dream_portraits(user_id, persona_id, is_current, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dream_cycles_user_persona
    ON dream_cycles(user_id, persona_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_dream_memory_log_user_persona
    ON dream_memory_log(user_id, persona_id, memory_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (89, 'Dream system: add user_id to dream_entries, dream_portraits, dream_cycles, dream_memory_log');
