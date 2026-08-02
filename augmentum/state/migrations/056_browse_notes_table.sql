-- 056_browse_notes_table.sql
-- Individual rows for browse notes (replaces JSON blob in app_settings).
-- Enables proper indexing, faster queries, and scalable storage.

CREATE TABLE IF NOT EXISTS browse_notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'Untitled',
    content TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    source_url TEXT NOT NULL DEFAULT '',
    source_title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_browse_notes_updated ON browse_notes(updated_at DESC);
