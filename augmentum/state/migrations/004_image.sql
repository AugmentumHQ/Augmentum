-- 004_image.sql: Image generation subsystem tables

CREATE TABLE IF NOT EXISTS image_models (
    name TEXT PRIMARY KEY,
    pipeline_type TEXT NOT NULL DEFAULT 'sd15',
    path TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'huggingface',
    size_bytes INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS image_generations (
    image_id TEXT PRIMARY KEY,
    session_id TEXT DEFAULT '',
    prompt TEXT NOT NULL,
    negative_prompt TEXT DEFAULT '',
    model TEXT NOT NULL,
    seed INTEGER DEFAULT -1,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    cfg_scale REAL NOT NULL DEFAULT 7.0,
    preset TEXT DEFAULT '',
    loras TEXT DEFAULT '[]',
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_image_generations_session ON image_generations(session_id);
CREATE INDEX IF NOT EXISTS idx_image_generations_created ON image_generations(created_at);

CREATE TABLE IF NOT EXISTS image_cache (
    cache_key TEXT PRIMARY KEY,
    image_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (image_id) REFERENCES image_generations(image_id)
);

CREATE TABLE IF NOT EXISTS image_presets (
    name TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT DEFAULT '',
    positive_tags TEXT DEFAULT '',
    negative_tags TEXT DEFAULT '',
    recommended_model TEXT DEFAULT '',
    cfg_scale REAL DEFAULT 7.0,
    steps INTEGER DEFAULT 20,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS image_loras (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    trigger_words TEXT DEFAULT '[]',
    size_bytes INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO schema_version (version, description) VALUES (4, 'Image generation subsystem');
