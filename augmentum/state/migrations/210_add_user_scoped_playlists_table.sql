-- 210_add_user_scoped_playlists_table.sql
--
-- User-scoped playlists. Items are stored as a JSON array to keep v1 minimal
-- (single round-trip CRUD, no join table). Each item is a typed reference
-- to media that lives elsewhere — we don't store audio bytes.
--
-- Item schema (frontend contract):
--   { type: 'youtube', videoId, title, channel?, thumbnail? }
--   { type: 'file',    fileId,  name,  kind: 'audio'|'video', thumbnail? }
--
-- Playback dispatch follows the item type — YouTube routes through
-- grove-ambient.loadVideo(), file items mount an <audio>/<video> element into
-- the same orb via loadMediaVideo(). Stations are intentionally excluded;
-- they never end and so don't chain.

CREATE TABLE IF NOT EXISTS playlists (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    items_json  TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_playlists_user
    ON playlists(user_id, updated_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (210, 'playlists - user-scoped media queues (youtube + file items)');
