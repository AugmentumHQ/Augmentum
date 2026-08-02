-- WebXR / Quest-style immersive app session state.
--
-- This is intentionally user-scoped. XR state includes room placement,
-- transcript/panel resume snapshots, input preferences, and device hints;
-- none of that should bleed between users on a shared headset/browser.

CREATE TABLE IF NOT EXISTS xr_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    voice_session_id TEXT NOT NULL DEFAULT '',
    surface TEXT NOT NULL DEFAULT 'voice',
    room_id TEXT NOT NULL DEFAULT 'modern-room',
    seat_id TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL DEFAULT 'preflight',
    device_hint_json TEXT NOT NULL DEFAULT '{}',
    room_state_json TEXT NOT NULL DEFAULT '{}',
    input_preferences_json TEXT NOT NULL DEFAULT '{}',
    performance_profile_json TEXT NOT NULL DEFAULT '{}',
    last_snapshot_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_xr_sessions_user_updated
    ON xr_sessions(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_xr_sessions_user_status
    ON xr_sessions(user_id, status);

CREATE TABLE IF NOT EXISTS xr_session_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_xr_session_events_session
    ON xr_session_events(session_id, id);

CREATE INDEX IF NOT EXISTS idx_xr_session_events_user_created
    ON xr_session_events(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS xr_seats (
    id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT 'Default seat',
    x REAL NOT NULL DEFAULT -0.30,
    y REAL NOT NULL DEFAULT 0.0,
    z REAL NOT NULL DEFAULT 2.30,
    rot_y REAL NOT NULL DEFAULT 3.141592653589793,
    env_id TEXT NOT NULL DEFAULT 'modern-room',
    avatar_x REAL,
    avatar_y REAL,
    avatar_z REAL,
    avatar_rot_y REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, id)
);
