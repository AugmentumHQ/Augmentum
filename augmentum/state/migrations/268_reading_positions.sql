-- 268_reading_positions.sql
-- Cross-device reading-position sync substrate. Backs the Augmentum Android
-- client's SyncRepository (POST/GET /api/sync/reading-positions): the phone
-- pushes book/article positions and pulls what other devices recorded, so a
-- book started on the phone resumes on the desktop and vice-versa.
--
-- One row per (user_id, sync_key). Conflict resolution is last-write-wins by
-- last_read_ms (DEVICE clock). updated_at_ms is the SERVER clock (epoch ms)
-- and is the pull cursor — the client passes the server's echoed now_ms back
-- as since_ms, so pull is immune to phone-clock skew.
CREATE TABLE IF NOT EXISTS reading_positions (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    sync_key TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'book',
    position_fraction REAL NOT NULL DEFAULT 0.0,
    position_detail INTEGER NOT NULL DEFAULT 0,
    last_read_ms INTEGER NOT NULL DEFAULT 0,
    device_id TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One position per (user, key). The upsert path looks up this pair, so the
-- unique index both enforces the invariant and accelerates the lookup.
CREATE UNIQUE INDEX IF NOT EXISTS idx_reading_positions_user_key
    ON reading_positions(user_id, sync_key);

-- Pull query is WHERE user_id = ? AND updated_at_ms > ? ORDER BY updated_at_ms.
CREATE INDEX IF NOT EXISTS idx_reading_positions_user_updated
    ON reading_positions(user_id, updated_at_ms);
