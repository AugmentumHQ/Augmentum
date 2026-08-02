-- 249_mobile_pairing.sql
--
-- Android/mobile pairing substrate.
--
-- auth_sessions.source lets the server distinguish ordinary browser
-- sessions from Android, cast receiver, and future device-pair sessions.
-- source_device_id binds a session to a durable trusted device row so
-- revoking that device can revoke its active sessions too.

ALTER TABLE auth_sessions ADD COLUMN source TEXT NOT NULL DEFAULT 'web';
ALTER TABLE auth_sessions ADD COLUMN source_device_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_auth_sessions_source_device
    ON auth_sessions(user_id, source, source_device_id);

CREATE TABLE IF NOT EXISTS trusted_mobile_devices (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id           TEXT NOT NULL DEFAULT '',
    label               TEXT NOT NULL DEFAULT '',
    platform            TEXT NOT NULL DEFAULT 'android',
    app_version         TEXT NOT NULL DEFAULT '',
    public_key          TEXT NOT NULL DEFAULT '',
    key_alg             TEXT NOT NULL DEFAULT '',
    scopes_json         TEXT NOT NULL DEFAULT '[]',
    capabilities_json   TEXT NOT NULL DEFAULT '[]',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at        TEXT NOT NULL DEFAULT '',
    revoked_at          TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_trusted_mobile_devices_user
    ON trusted_mobile_devices(user_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_trusted_mobile_devices_user_device
    ON trusted_mobile_devices(user_id, device_id)
    WHERE device_id != '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (249, 'mobile pairing: auth session source + trusted mobile devices');

