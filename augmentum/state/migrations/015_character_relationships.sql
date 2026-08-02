-- 015_character_relationships.sql: Structured character relationship tracking

CREATE TABLE IF NOT EXISTS character_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    trust REAL NOT NULL DEFAULT 0.0,
    affection REAL NOT NULL DEFAULT 0.0,
    tension REAL NOT NULL DEFAULT 0.0,
    label TEXT NOT NULL DEFAULT '',
    last_updated_at INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, source_entity, target_entity)
);
CREATE INDEX IF NOT EXISTS idx_charrel_session ON character_relationships(session_id);
CREATE INDEX IF NOT EXISTS idx_charrel_source ON character_relationships(session_id, source_entity);

INSERT INTO schema_version (version, description) VALUES (15, 'Character relationship graph');
