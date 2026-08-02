-- 006_memory.sql: Cross-session memory system with vector search and FTS5

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,  -- 'preference', 'fact', 'entity', 'narrative', 'analysis'
    importance REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.8,
    embedding BLOB,  -- float32 vector (384-dim for bge-small-en-v1.5)
    valid_from TEXT NOT NULL DEFAULT (datetime('now')),
    valid_until TEXT,  -- NULL = still current
    superseded_by TEXT REFERENCES memories(id),
    source_type TEXT,  -- 'extracted', 'user_manual', 'system'
    source_context TEXT,  -- JSON: session_id, message_index, etc.
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(user_id, memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_valid ON memories(user_id, valid_until);

-- FTS5 index for keyword search
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    memory_type,
    content=memories,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, memory_type)
    VALUES (new.rowid, new.content, new.memory_type);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
    VALUES ('delete', old.rowid, old.content, old.memory_type);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
    VALUES ('delete', old.rowid, old.content, old.memory_type);
    INSERT INTO memories_fts(rowid, content, memory_type)
    VALUES (new.rowid, new.content, new.memory_type);
END;

INSERT INTO schema_version (version, description) VALUES (6, 'Cross-session memory system with FTS5');
