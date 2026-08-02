-- 187_trusted_receivers_and_cast_events.sql
--
-- Persistent identity + audit log for cast receivers.
--
-- Today's ReceiverRegistry tracks only in-memory connections. Once a
-- receiver disconnects (TV reboot, network blip) its registration_id
-- is lost — the next connect generates a fresh one. That breaks every
-- "my TVs" management feature: no stable name, no revocation, no
-- audit. These tables fix that.
--
-- trusted_receivers — long-lived row per (user, device). Bound to a
--   runtime ConnectedReceiver via device_id (provided by the receiver
--   app on its ready event). Survives reboots; the user names it once
--   and the name persists.
--
-- receiver_cast_events — audit log. Recorded server-side every time
--   a surface is dispatched to a receiver, closed when the surface
--   ends or the receiver disconnects. Powers Recent Activity UI +
--   "currently showing" status.
--
-- Both follow the multi-tenant rule (user_id REFERENCES users(id)
-- ON DELETE CASCADE) and align with the user-scoped tables list in
-- CLAUDE.md.

CREATE TABLE IF NOT EXISTS trusted_receivers (
    id              TEXT PRIMARY KEY,                              -- 'tr_<sha1-12>'
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label           TEXT NOT NULL DEFAULT '',                      -- user-visible name
    platform        TEXT NOT NULL DEFAULT '',                      -- 'android-tv' | 'browser' | 'tizen' | ...
    device_id       TEXT NOT NULL DEFAULT '',                      -- stable per-device id from receiver
    info            TEXT NOT NULL DEFAULT '{}',                    -- JSON snapshot of capabilities/screen
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at    TEXT NOT NULL DEFAULT '',                      -- bumped on every connect
    last_cast_at    TEXT NOT NULL DEFAULT '',                      -- bumped on every surface dispatch
    revoked_at      TEXT NOT NULL DEFAULT ''                       -- non-empty = revoked
);

CREATE INDEX IF NOT EXISTS idx_trusted_receivers_user
    ON trusted_receivers(user_id);

-- One row per (user, device_id) — receivers that send the same
-- device_id across reconnects rebind to the same trusted row.
-- The WHERE clause excludes browser-tab receivers that don't ship
-- a device_id (they each get a fresh row on connect).
CREATE UNIQUE INDEX IF NOT EXISTS idx_trusted_receivers_device_unique
    ON trusted_receivers(user_id, device_id)
    WHERE device_id != '';


CREATE TABLE IF NOT EXISTS receiver_cast_events (
    id              TEXT PRIMARY KEY,                              -- 'cev_<sha1-12>'
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    trusted_id      TEXT NOT NULL DEFAULT '',                      -- references trusted_receivers.id; '' for unbound
    registration_id TEXT NOT NULL DEFAULT '',                      -- runtime WS id at the time of cast
    surface_id      TEXT NOT NULL DEFAULT '',
    surface_kind    TEXT NOT NULL DEFAULT '',
    surface_url     TEXT NOT NULL DEFAULT '',
    slot            TEXT NOT NULL DEFAULT 'main',
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT NOT NULL DEFAULT '',                      -- '' = active
    end_reason      TEXT NOT NULL DEFAULT ''                       -- 'user_stop' | 'replaced' | 'disconnected' | 'ended'
);

CREATE INDEX IF NOT EXISTS idx_receiver_cast_events_user_started
    ON receiver_cast_events(user_id, started_at DESC);

-- Fast lookup for "what's currently showing on this receiver" — index
-- the active subset (ended_at = '') by trusted_id.
CREATE INDEX IF NOT EXISTS idx_receiver_cast_events_active
    ON receiver_cast_events(trusted_id, started_at DESC)
    WHERE ended_at = '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (
    187,
    'trusted_receivers + receiver_cast_events for persistent receiver identity + audit log'
);
