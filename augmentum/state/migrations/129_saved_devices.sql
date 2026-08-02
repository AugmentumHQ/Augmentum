-- Saved devices for the device substrate (TVs, speakers, lights, sensors,
-- augmentum's own UI surfaces).
--
-- Replaces the per-feature receiver concept (Cast/DLNA/AirPlay TVs only)
-- with a unified, driver-agnostic registry. Every external thing
-- Augmentum can talk to lives here: smart TVs via DLNA/Cast/AirPlay
-- drivers; smart bulbs via Hue/Matter/MQTT; sensors and switches; phones
-- via the companion driver; and Augmentum's own UI panels via the
-- internal surface driver.
--
-- See `docs/superpowers/specs/2026-05-07-device-substrate-design.md` for
-- the architectural rationale.
--
-- The `auth` blob stores driver-specific credentials (Hue token, Cast
-- pairing ID, OAuth tokens). It is encrypted at the persistence layer
-- before insert and decrypted on read; the column itself is opaque text.
-- Drivers never see ciphertext.
--
-- `bindings` lets one canonical Device collapse multiple driver
-- bindings — e.g. the same Sony Bravia advertised on both Cast and
-- DLNA collapses to one row with two driver bindings, and capability
-- routing picks the better driver per call.

CREATE TABLE IF NOT EXISTS saved_devices (
    id            TEXT PRIMARY KEY,                           -- 'dev_<sha1-12>'
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    driver        TEXT NOT NULL,                              -- 'dlna' | 'cast' | 'hue' | ...
    native_id     TEXT NOT NULL,                              -- driver-specific stable ID
    label         TEXT NOT NULL,                              -- user-visible name
    capabilities  TEXT NOT NULL DEFAULT '[]',                 -- JSON array of capability IDs
    address       TEXT NOT NULL DEFAULT '{}',                 -- JSON: {host, port, descriptor_url, ...}
    auth          TEXT NOT NULL DEFAULT '{}',                 -- encrypted JSON blob
    status        TEXT NOT NULL DEFAULT 'unverified',         -- 'online' | 'offline' | 'unverified' | 'paired'
    last_seen_at  TEXT NOT NULL DEFAULT '',                   -- ISO timestamp; updated on successful snapshot
    metadata      TEXT NOT NULL DEFAULT '{}',                 -- JSON: {model, manufacturer, icon_url}
    config        TEXT NOT NULL DEFAULT '{}',                 -- JSON: user prefs (alias, room, etc)
    bindings      TEXT NOT NULL DEFAULT '[]',                 -- JSON: secondary {driver, native_id, capabilities} pairs
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_saved_devices_user
    ON saved_devices(user_id);

CREATE INDEX IF NOT EXISTS idx_saved_devices_driver
    ON saved_devices(user_id, driver);

-- One device per (user, driver, native_id) — re-discovery doesn't create
-- duplicates. Cross-driver dedup (same physical device on Cast and DLNA)
-- is handled in code via the bindings field, not the unique index.
CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_devices_unique
    ON saved_devices(user_id, driver, native_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (129, 'saved_devices registry for device substrate (TVs, lights, sensors, internal surfaces)');
