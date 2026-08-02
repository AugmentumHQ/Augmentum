-- Add user_id to all user-scoped tables.
-- Nullable for backward compat (SQLite can't ADD NOT NULL to existing).
-- Backfilled on first admin creation; enforced in application code.

-- Core
ALTER TABLE sessions ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE ui_sessions ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE ui_characters ADD COLUMN user_id TEXT REFERENCES users(id);

-- Narrative
ALTER TABLE facts ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE entities ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE plot_threads ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE contradictions ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE lorebook_entries ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE assumptions ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE character_cards ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE narrative_memory ADD COLUMN user_id TEXT REFERENCES users(id);

-- Memory
ALTER TABLE memories ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE kg_nodes ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE kg_edges ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE memory_cooccurrence ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE memory_events ADD COLUMN user_id TEXT REFERENCES users(id);

-- Image
ALTER TABLE image_generations ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE chat_images ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE image_cache ADD COLUMN user_id TEXT REFERENCES users(id);

-- Documents
ALTER TABLE documents ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE document_chunks ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE session_documents ADD COLUMN user_id TEXT REFERENCES users(id);

-- Tools/Workflows
ALTER TABLE artifacts ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE custom_flows ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE reasoning_flows ADD COLUMN user_id TEXT REFERENCES users(id);
ALTER TABLE coder_sessions ADD COLUMN user_id TEXT REFERENCES users(id);

-- Indexes for all
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_ui_sessions_user ON ui_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_ui_characters_user ON ui_characters(user_id);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
CREATE INDEX IF NOT EXISTS idx_entities_user ON entities(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_image_generations_user ON image_generations(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_user ON artifacts(user_id);
CREATE INDEX IF NOT EXISTS idx_custom_flows_user ON custom_flows(user_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_flows_user ON reasoning_flows(user_id);
CREATE INDEX IF NOT EXISTS idx_coder_sessions_user ON coder_sessions(user_id);
