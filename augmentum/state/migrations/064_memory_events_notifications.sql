-- 064_memory_events_notifications.sql
-- Memory events log (tier changes, promotions, consolidations, dream cycles, extractions)
-- Persistent notifications (replaces in-memory dict)

CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    event_type TEXT NOT NULL,
    memory_id TEXT,
    detail TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_memory_events_user_created
    ON memory_events(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_events_type
    ON memory_events(user_id, event_type);

CREATE TABLE IF NOT EXISTS memory_notifications (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    content TEXT NOT NULL,
    evidence TEXT,
    tier TEXT NOT NULL DEFAULT 'provisional',
    confidence REAL NOT NULL DEFAULT 0.5,
    memory_type TEXT NOT NULL DEFAULT 'fact',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_notifications_user_status
    ON memory_notifications(user_id, status, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (64, 'Memory events log and persistent notifications');
