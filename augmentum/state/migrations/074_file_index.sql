-- Unified file index across all subsystems
CREATE TABLE IF NOT EXISTS file_index (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    name TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT '',
    size_bytes INTEGER NOT NULL DEFAULT 0,
    real_path TEXT,
    description TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    thumbnail TEXT,
    embedding BLOB,
    is_directory INTEGER NOT NULL DEFAULT 0,
    parent_id TEXT,
    source_metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_file_index_user ON file_index(user_id);
CREATE INDEX IF NOT EXISTS idx_file_index_source ON file_index(source, source_id);
CREATE INDEX IF NOT EXISTS idx_file_index_mime ON file_index(mime_type);
CREATE INDEX IF NOT EXISTS idx_file_index_created ON file_index(created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_file_index_source_unique
    ON file_index(user_id, source, source_id);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS file_index_fts USING fts5(
    name, description, tags,
    content=file_index, content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS file_index_fts_insert AFTER INSERT ON file_index BEGIN
    INSERT INTO file_index_fts(rowid, name, description, tags)
    VALUES (new.rowid, new.name, new.description, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS file_index_fts_delete AFTER DELETE ON file_index BEGIN
    INSERT INTO file_index_fts(file_index_fts, rowid, name, description, tags)
    VALUES ('delete', old.rowid, old.name, old.description, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS file_index_fts_update AFTER UPDATE ON file_index BEGIN
    INSERT INTO file_index_fts(file_index_fts, rowid, name, description, tags)
    VALUES ('delete', old.rowid, old.name, old.description, old.tags);
    INSERT INTO file_index_fts(rowid, name, description, tags)
    VALUES (new.rowid, new.name, new.description, new.tags);
END;
