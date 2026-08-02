-- Pairing state for drivers that need durable pairing tokens
-- (Hue bridge tokens, Matter CASE session keys, AirPlay 2 HAP pairings).
--
-- Most drivers don't need this — DLNA, browser, augmentum_surface, and
-- webhook all run unauthenticated within the LAN. The pairings table
-- exists for the minority that do.
--
-- `pairing_data` is encrypted at the persistence layer same as
-- `saved_devices.auth`; the column itself is opaque text.
--
-- ON DELETE CASCADE on `device_id` keeps pairings in sync with their
-- device — removing a device tears down its pairing.

CREATE TABLE IF NOT EXISTS device_pairings (
    id             TEXT PRIMARY KEY,                              -- 'pair_<sha1-12>'
    device_id      TEXT NOT NULL REFERENCES saved_devices(id) ON DELETE CASCADE,
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    state          TEXT NOT NULL DEFAULT 'pending',               -- 'pending' | 'active' | 'expired' | 'failed'
    pairing_data   TEXT NOT NULL DEFAULT '{}',                    -- encrypted JSON
    expires_at     TEXT NOT NULL DEFAULT '',                      -- ISO timestamp; '' = no expiry
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_device_pairings_user
    ON device_pairings(user_id);

CREATE INDEX IF NOT EXISTS idx_device_pairings_device
    ON device_pairings(device_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (130, 'device_pairings durable pairing state for Hue/Matter/AirPlay');
