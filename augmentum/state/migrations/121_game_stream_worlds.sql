-- 121_game_stream_worlds.sql
-- Persistent world records for AGSP. One row per user-owned world (e.g. a
-- Luanti world). The on-disk world data lives in a per-user volume; this row
-- is the metadata + access-control layer.
--
-- whitelist_user_ids is a JSON array of users allowed to join via Augmentum
-- (Multiplayer is opt-in -- empty array means owner-only).

CREATE TABLE IF NOT EXISTS game_stream_worlds (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id),
    profile_id          TEXT NOT NULL,                         -- 'luanti', etc.
    name                TEXT NOT NULL,
    settings_json       TEXT NOT NULL DEFAULT '{}',            -- gamemode, seed, mod set, etc.
    whitelist_user_ids  TEXT NOT NULL DEFAULT '[]',            -- JSON array of additional allowed users
    storage_path        TEXT NOT NULL DEFAULT '',              -- relative path under user data dir
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    last_played_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_game_stream_worlds_user
    ON game_stream_worlds(user_id, last_played_at DESC);

CREATE INDEX IF NOT EXISTS idx_game_stream_worlds_profile
    ON game_stream_worlds(profile_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (121, 'game_stream_worlds table');
