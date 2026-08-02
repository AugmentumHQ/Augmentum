-- Add avatar column to user_personas table
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT DEFAULT (datetime('now')));
INSERT OR IGNORE INTO schema_version (version) VALUES (17);

ALTER TABLE user_personas ADD COLUMN avatar TEXT NOT NULL DEFAULT '';
