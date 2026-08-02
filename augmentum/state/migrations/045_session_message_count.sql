-- Add message_count column to ui_sessions for fast metadata queries.
-- Computed from the tree on save, avoids fragile regex counting on read.

ALTER TABLE ui_sessions ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0;

-- Backfill: count tree nodes in existing sessions by parsing the JSON.
-- SQLite json_each can iterate the tree object keys.
UPDATE ui_sessions
SET message_count = (
    SELECT count(*)
    FROM json_each(json_extract(data, '$.tree'))
)
WHERE json_valid(data) AND json_extract(data, '$.tree') IS NOT NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (45, 'Session message count column for fast metadata');
