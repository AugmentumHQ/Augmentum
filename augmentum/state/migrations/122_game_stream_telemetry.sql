-- 122_game_stream_telemetry.sql
-- Streaming-quality telemetry samples (RTT/jitter/loss/bitrate/fps).
-- Used for adaptive bitrate decisions, debugging, and a future quality
-- dashboard. user_id is denormalized for fast user-scoped queries; the
-- session_id FK does the cleanup on session deletion.

CREATE TABLE IF NOT EXISTS game_stream_telemetry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    user_id         TEXT NOT NULL REFERENCES users(id),
    ts              TEXT NOT NULL DEFAULT (datetime('now')),
    rtt_ms          REAL,
    jitter_ms       REAL,
    packet_loss     REAL,                                  -- 0..1
    bitrate_kbps    INTEGER,
    fps             REAL,
    FOREIGN KEY (session_id) REFERENCES game_stream_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_game_stream_telemetry_session
    ON game_stream_telemetry(session_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_game_stream_telemetry_user
    ON game_stream_telemetry(user_id, ts DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (122, 'game_stream_telemetry table');
