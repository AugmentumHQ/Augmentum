-- 283_connect_presence.sql
-- Persistent presence (comms platform, Phase 1 — discovery/presence).
--
-- Today presence is purely in-memory in ConnectHub (online iff a WS is
-- attached), so a server restart or a page reload loses everyone's status and
-- "last seen" is impossible. This table durably records each user's last
-- online/offline transition so the directory can show "last seen 2h ago" for
-- offline peers and presence survives restarts.
--
-- The ConnectHub writes here via an optional presence-sink hook (it stays
-- DB-agnostic; the route wiring installs the sink). One row per user
-- (`user_id` PK) — this is the user's GLOBAL presence, not a per-viewer view.
-- User-scoped per CLAUDE.md (`user_id` column).
CREATE TABLE IF NOT EXISTS connect_presence (
    user_id      TEXT PRIMARY KEY,
    state        TEXT NOT NULL DEFAULT 'offline',  -- online | offline
    last_seen_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (283, 'connect_presence: durable last-seen / online state');
