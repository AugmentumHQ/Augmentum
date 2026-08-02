-- 282_auth_invites.sql
-- Open internal access (Connect comms platform, Phase 1).
--
-- An invite is a link a host operator (or, later, a permitted user) mints so
-- a person can self-claim an account: they open /ui/connect-join/?token=...,
-- choose a username + their OWN password, and land in Connect with the
-- inviter already a contact. Replaces the only path that exists today —
-- admin creates the account AND relays a password out-of-band.
--
-- Server-level table (NOT user-scoped): an invite is managed at the install
-- level and CLAIMED by someone who has no account yet, so there is no owning
-- `user_id` to scope by. `inviter_user_id` records who created it (named so
-- the doc-facts auto-classifier does NOT treat this as a user-scoped table).
--
-- Security: only the SHA-256 hash of the token is stored; the raw token is
-- shown to the creator exactly once and travels in the link. Expiry,
-- max_uses, and revocation are all enforced at claim time.
CREATE TABLE IF NOT EXISTS auth_invites (
    id              TEXT PRIMARY KEY,
    token_hash      TEXT NOT NULL UNIQUE,
    inviter_user_id TEXT NOT NULL,
    -- account_claim → a full user account; external_guest → a scoped guest
    -- account (Phase 3). role is the account role the claim provisions.
    kind            TEXT NOT NULL DEFAULT 'account_claim',
    role            TEXT NOT NULL DEFAULT 'user',
    invitee_email   TEXT NOT NULL DEFAULT '',
    handle_hint     TEXT NOT NULL DEFAULT '',
    max_uses        INTEGER NOT NULL DEFAULT 1,
    use_count       INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    -- Empty string = never expires; otherwise an ISO datetime compared with
    -- datetime('now') at claim time.
    expires_at      TEXT NOT NULL DEFAULT '',
    claimed_at      TEXT NOT NULL DEFAULT '',
    claimed_user_id TEXT NOT NULL DEFAULT '',
    revoked_at      TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_auth_invites_inviter
    ON auth_invites (inviter_user_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (282, 'auth_invites: self-claim onboarding for Connect open access');
