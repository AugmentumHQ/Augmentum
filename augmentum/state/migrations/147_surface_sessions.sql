-- Durable surface sessions for paired browsers, TVs, phones, game streams,
-- comic readers, XR rooms, and Augmentum-native panels.
--
-- Surface sessions are intentionally user-scoped. Public TV/browser access
-- is delegated through short-lived in-memory tokens, never by exposing these
-- tables without a user boundary.

CREATE TABLE IF NOT EXISTS surface_sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    kind TEXT NOT NULL DEFAULT 'surface.generic',
    title TEXT NOT NULL DEFAULT '',
    content_ref_json TEXT NOT NULL DEFAULT '{}',
    state_json TEXT NOT NULL DEFAULT '{}',
    participants_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    revision INTEGER NOT NULL DEFAULT 0,
    pairing_code TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_surface_sessions_user_status_updated
    ON surface_sessions(user_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_surface_sessions_user_kind_updated
    ON surface_sessions(user_id, kind, updated_at DESC);

CREATE TABLE IF NOT EXISTS surface_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES surface_sessions(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id),
    revision INTEGER NOT NULL DEFAULT 0,
    event_type TEXT NOT NULL,
    source_participant_id TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_surface_events_session_revision
    ON surface_events(user_id, session_id, revision, id);

CREATE INDEX IF NOT EXISTS idx_surface_events_user_created
    ON surface_events(user_id, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (147, 'surface sessions and event trace');
