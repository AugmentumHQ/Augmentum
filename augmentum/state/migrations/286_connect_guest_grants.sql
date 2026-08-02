-- 286_connect_guest_grants.sql
-- Durable guest pass (Connect comms platform, Phase 3a — the grant spine).
--
-- A grant is the durable, REVOCABLE relationship behind an external guest: it
-- links a host (the inviter) to a scoped role='guest' account, carries the
-- scopes the guest is allowed (text/call), and holds the hash of a durable
-- surface token the saved homescreen PWA presents to re-establish a scoped
-- session. The host revokes the grant to cut access (revoked_at) — the single,
-- legible kill-switch.
--
-- User-scoped per CLAUDE.md: `user_id` is the HOST who owns/controls the grant
-- (so it joins the user-scoped table list). The guest is referenced by
-- `guest_user_id` (a separate role='guest' users row). Only the SHA-256 hash of
-- the durable token is stored; the raw token is shown once and lives in the PWA.
-- See docs/superpowers/specs/2026-06-21-connect-durable-guest-surface-design.md.
CREATE TABLE IF NOT EXISTS connect_guest_grants (
    grant_id      TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,                 -- host (inviter) controlling the grant
    host_did      TEXT NOT NULL DEFAULT '',      -- denormalized for the guest surface
    guest_user_id TEXT NOT NULL,                 -- the scoped role='guest' account
    guest_did     TEXT NOT NULL DEFAULT '',
    token_hash    TEXT NOT NULL UNIQUE,          -- durable surface credential (hash only)
    scopes        TEXT NOT NULL DEFAULT 'text',  -- comma list: text[,call] — call is opt-in
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_used_at  TEXT NOT NULL DEFAULT '',
    revoked_at    TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_guest_grants_host
    ON connect_guest_grants (user_id);
CREATE INDEX IF NOT EXISTS idx_guest_grants_guest
    ON connect_guest_grants (guest_user_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (286, 'connect_guest_grants: durable revocable guest pass');
