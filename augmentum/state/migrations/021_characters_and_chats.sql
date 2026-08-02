-- Migration 021: Server-side character cards and chat sessions
-- Moves data from client localStorage to persistent server storage.

CREATE TABLE IF NOT EXISTS ui_characters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',   -- Full character JSON blob
    avatar TEXT NOT NULL DEFAULT '',   -- Base64 data URL or empty
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ui_characters_name ON ui_characters(name);

CREATE TABLE IF NOT EXISTS ui_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Chat',
    mode TEXT NOT NULL DEFAULT 'passthrough',
    data TEXT NOT NULL DEFAULT '{}',   -- Full session JSON blob (tree, metadata, etc.)
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ui_sessions_mode ON ui_sessions(mode);
CREATE INDEX IF NOT EXISTS idx_ui_sessions_updated ON ui_sessions(updated_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (21, 'Server-side character cards and chat sessions');
