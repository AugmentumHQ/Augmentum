-- 058_dream_system.sql
-- Dream System: Persona Introspection Engine — schema for synthetic autobiographical memory

-- Dream entries: individual outputs of a dream cycle (reflections, voice notes, threads, impressions)
CREATE TABLE IF NOT EXISTS dream_entries (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL DEFAULT 'default',
    content TEXT NOT NULL,
    entry_type TEXT NOT NULL DEFAULT 'reflection',
    source_memories TEXT NOT NULL DEFAULT '[]',
    source_sessions TEXT NOT NULL DEFAULT '[]',
    context_window TEXT NOT NULL DEFAULT '{}',
    embedding BLOB,
    weight REAL NOT NULL DEFAULT 1.0,
    pinned INTEGER NOT NULL DEFAULT 0,
    dream_cycle_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_dream_entries_persona_created
    ON dream_entries(persona_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_dream_entries_cycle
    ON dream_entries(dream_cycle_id);

CREATE INDEX IF NOT EXISTS idx_dream_entries_persona_type
    ON dream_entries(persona_id, entry_type);

-- Dream portraits: synthesized snapshots of persona state (voice, threads, impressions)
CREATE TABLE IF NOT EXISTS dream_portraits (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL DEFAULT 'default',
    voice_notes TEXT NOT NULL DEFAULT '',
    active_threads TEXT NOT NULL DEFAULT '',
    impressions TEXT NOT NULL DEFAULT '',
    source_entries TEXT NOT NULL DEFAULT '[]',
    is_current INTEGER NOT NULL DEFAULT 1,
    checkpoint_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dream_portraits_persona_current_created
    ON dream_portraits(persona_id, is_current, created_at DESC);

-- Dream cycles: metadata about each dream run
CREATE TABLE IF NOT EXISTS dream_cycles (
    id TEXT PRIMARY KEY,
    persona_id TEXT NOT NULL DEFAULT 'default',
    trigger_reason TEXT NOT NULL,
    memories_count INTEGER NOT NULL DEFAULT 0,
    entries_count INTEGER NOT NULL DEFAULT 0,
    model_used TEXT,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_dream_cycles_persona_started
    ON dream_cycles(persona_id, started_at DESC);

-- Dream memory log: tracks which memories have been processed in which cycle
CREATE TABLE IF NOT EXISTS dream_memory_log (
    memory_id TEXT NOT NULL,
    dream_cycle_id TEXT NOT NULL,
    persona_id TEXT NOT NULL DEFAULT 'default',
    dreamed_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (memory_id, dream_cycle_id)
);

CREATE INDEX IF NOT EXISTS idx_dream_memory_log_persona_memory
    ON dream_memory_log(persona_id, memory_id);

-- FTS5 index for keyword search over dream entries
CREATE VIRTUAL TABLE IF NOT EXISTS dream_entries_fts USING fts5(
    content,
    entry_type,
    content=dream_entries,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync with dream_entries
CREATE TRIGGER IF NOT EXISTS dream_entries_ai AFTER INSERT ON dream_entries BEGIN
    INSERT INTO dream_entries_fts(rowid, content, entry_type)
    VALUES (new.rowid, new.content, new.entry_type);
END;

CREATE TRIGGER IF NOT EXISTS dream_entries_ad AFTER DELETE ON dream_entries BEGIN
    INSERT INTO dream_entries_fts(dream_entries_fts, rowid, content, entry_type)
    VALUES ('delete', old.rowid, old.content, old.entry_type);
END;

CREATE TRIGGER IF NOT EXISTS dream_entries_au AFTER UPDATE ON dream_entries BEGIN
    INSERT INTO dream_entries_fts(dream_entries_fts, rowid, content, entry_type)
    VALUES ('delete', old.rowid, old.content, old.entry_type);
    INSERT INTO dream_entries_fts(rowid, content, entry_type)
    VALUES (new.rowid, new.content, new.entry_type);
END;

-- Extend memories table for dream system integration
ALTER TABLE memories ADD COLUMN source_message_id TEXT;
ALTER TABLE memories ADD COLUMN user_approved INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (58, 'Dream System: persona introspection engine schema');
