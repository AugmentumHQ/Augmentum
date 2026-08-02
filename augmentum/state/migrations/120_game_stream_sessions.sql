-- 120_game_stream_sessions.sql
-- Active streaming sessions for the Augmentum Game Streaming Platform (AGSP).
-- One row per running game container; survives server restart so the lifecycle
-- manager can reconcile container state on startup. Terminal rows aren't deleted
-- here -- a separate retention job trims them so we keep telemetry context.

CREATE TABLE IF NOT EXISTS game_stream_sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    world_id        TEXT,                                  -- nullable for ephemeral sessions
    profile_id      TEXT NOT NULL,                         -- e.g. 'luanti'
    status          TEXT NOT NULL DEFAULT 'stopped',       -- stopped|starting|ready|connected|idle|stopping
    container_id    TEXT,                                  -- docker container id (running rows only)
    stream_port     INTEGER,                               -- WebRTC signaling port (allocated from pool)
    game_port       INTEGER,                               -- game server port (allocated from pool)
    bitrate_mbps    INTEGER NOT NULL DEFAULT 4,
    resolution      TEXT NOT NULL DEFAULT '1280x720',
    encoder         TEXT NOT NULL DEFAULT 'auto',          -- auto|nvenc|vaapi|x264
    exit_reason     TEXT,                                  -- clean|idle|crash|cancelled (terminal rows)
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_game_stream_sessions_user
    ON game_stream_sessions(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_game_stream_sessions_status
    ON game_stream_sessions(status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_game_stream_sessions_world
    ON game_stream_sessions(world_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (120, 'game_stream_sessions table');
