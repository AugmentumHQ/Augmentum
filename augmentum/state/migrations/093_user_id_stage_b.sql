-- Stage B of the multi-tenancy rollout: add user_id to the long tail of
-- tables that predated migration 072 (or were landed afterward without
-- per-feature scoping). A single migration keeps the backfill atomic and
-- readable — every row that existed before this migration is claimed by
-- the oldest active user, matching the convention from migrations
-- 083/087/089.
--
-- Covered categories:
--   Personal data (user owns every row):
--     user_personas, prompt_presets, regex_scripts, voice_enrollments,
--     browse_notes, avatars, coder_workspaces, voice_mixes,
--     interaction_signals, browse_history, interest_clusters,
--     narrative_archive
--
--   Transitively-scoped (ownership derives from a parent row; we still
--   denormalize user_id so queries don't need to JOIN back to the parent
--   for filtering, matching the pattern CLAUDE.md documents):
--     session_messages, fact_tags, entity_state_history,
--     reasoning_flow_steps, session_knowledge_packs
--
--   Server-level and intentionally NOT scoped:
--     coder_templates      — shared starter environments
--     token_counts         — content-addressable tokenizer cache
--     domain_reputation    — shared knowledge about URL reputation
--
-- A couple of unique constraints (voice_mixes.name PK, browse_history.url
-- UNIQUE) are legitimate single-tenant leftovers that two tenants can now
-- collide on. Rewriting a PK in SQLite requires a full table rebuild, so
-- for now the application layer filters by user_id and surface-level
-- collisions raise an INSERT error that's easy to UX around. A follow-up
-- migration will rebuild those tables with tenant-widened keys.

-- ---------------------------------------------------------------------------
-- 1. ADD COLUMN — personal data
-- ---------------------------------------------------------------------------

ALTER TABLE user_personas        ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE prompt_presets       ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE regex_scripts        ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE voice_enrollments    ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE browse_notes         ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE avatars              ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE coder_workspaces     ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE voice_mixes          ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE interaction_signals  ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE browse_history       ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE interest_clusters    ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE narrative_archive    ADD COLUMN user_id TEXT REFERENCES users(id);

-- ---------------------------------------------------------------------------
-- 2. ADD COLUMN — transitively-scoped (denormalized for query ergonomics)
-- ---------------------------------------------------------------------------

ALTER TABLE session_messages        ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE fact_tags               ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE entity_state_history    ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE reasoning_flow_steps    ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE session_knowledge_packs ADD COLUMN user_id TEXT REFERENCES users(id);

-- ---------------------------------------------------------------------------
-- 3. BACKFILL — claim every existing row for the oldest active user.
-- The subquery returns NULL on a fresh install (no users yet), which
-- leaves user_id IS NULL and matches the "create admin, then backfill"
-- flow that auth_setup already expects.
-- ---------------------------------------------------------------------------

UPDATE user_personas         SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE prompt_presets        SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE regex_scripts         SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE voice_enrollments     SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE browse_notes          SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE avatars               SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE coder_workspaces      SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE voice_mixes           SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE interaction_signals   SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE browse_history        SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE interest_clusters     SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE narrative_archive     SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;

UPDATE session_messages        SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE fact_tags               SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE entity_state_history    SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE reasoning_flow_steps    SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE session_knowledge_packs SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;

-- ---------------------------------------------------------------------------
-- 4. INDEXES — only on tables whose primary access pattern is per-user.
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_user_personas_user        ON user_personas(user_id);
CREATE INDEX IF NOT EXISTS idx_prompt_presets_user       ON prompt_presets(user_id);
CREATE INDEX IF NOT EXISTS idx_regex_scripts_user        ON regex_scripts(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_enrollments_user    ON voice_enrollments(user_id);
CREATE INDEX IF NOT EXISTS idx_browse_notes_user         ON browse_notes(user_id);
CREATE INDEX IF NOT EXISTS idx_avatars_user              ON avatars(user_id);
CREATE INDEX IF NOT EXISTS idx_coder_workspaces_user     ON coder_workspaces(user_id);
CREATE INDEX IF NOT EXISTS idx_voice_mixes_user          ON voice_mixes(user_id);
CREATE INDEX IF NOT EXISTS idx_interaction_signals_user  ON interaction_signals(user_id);
CREATE INDEX IF NOT EXISTS idx_browse_history_user       ON browse_history(user_id);
CREATE INDEX IF NOT EXISTS idx_interest_clusters_user    ON interest_clusters(user_id);
CREATE INDEX IF NOT EXISTS idx_narrative_archive_user    ON narrative_archive(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_session_messages_user     ON session_messages(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_session_knowledge_packs_user ON session_knowledge_packs(user_id, session_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (93, 'stage_b_user_id_backfill');
