CREATE TABLE IF NOT EXISTS avatars (
    id TEXT PRIMARY KEY,
    character_id TEXT,
    persona_id TEXT,
    source_image_id TEXT,
    vrm_path TEXT NOT NULL,
    thumbnail_path TEXT,
    mannerisms TEXT NOT NULL DEFAULT '{}',
    is_bundled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_avatars_character ON avatars(character_id);
CREATE INDEX IF NOT EXISTS idx_avatars_persona ON avatars(persona_id);
