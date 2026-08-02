-- Backfill user_id on rows that predate migration 072.
--
-- Migration 072 added user_id columns but did not populate existing rows —
-- those rows have user_id IS NULL and became invisible to authenticated
-- queries once we enforced "AND user_id = ?" filtering end-to-end.
--
-- For single-user installs (the common case today) the correct action is
-- to assign all orphaned rows to the oldest user. Multi-user installs with
-- pre-072 data will need to reassign via the UI after this backfill.
--
-- The subquery `(SELECT id FROM users ORDER BY created_at ASC LIMIT 1)`
-- returns NULL if no users exist yet (fresh install that hasn't registered
-- a first user). In that case every UPDATE is a no-op — safe to run before
-- auth has been configured.

-- Core
UPDATE sessions        SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE ui_sessions     SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE ui_characters   SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;

-- Narrative
UPDATE facts              SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE entities           SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE plot_threads       SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE contradictions     SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE lorebook_entries   SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE assumptions        SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE character_cards    SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE narrative_memory   SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;

-- Memory
UPDATE memories             SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE kg_nodes             SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE kg_edges             SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE memory_cooccurrence  SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE memory_events        SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;

-- Image
UPDATE image_generations  SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE chat_images        SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE image_cache        SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;

-- Documents
UPDATE documents          SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE document_chunks    SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE session_documents  SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;

-- Tools / Workflows
UPDATE artifacts         SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE custom_flows      SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE reasoning_flows   SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;
UPDATE coder_sessions    SET user_id = (SELECT id FROM users ORDER BY created_at ASC LIMIT 1) WHERE user_id IS NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (83, 'backfill_user_id_on_legacy_rows');
