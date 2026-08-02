-- 296_guest_portal.sql
-- Guest comms portal: admin-confirmed registration + IP allowlist.
--
-- The invited guest experience the operator asked for:
--   1. invitee lands on the portal and REGISTERS (username/display/password)
--      -> a scoped role='guest' account is created, but PENDING.
--   2. the admin CONFIRMS the registration as a final step -> this creates
--      the guest grant (existing connect_guest_grants) AND allowlists the
--      IP they registered from, so they can return.
--   3. the guest returns to an installable mini-app (messenger/dialer) and
--      may text/call the inviter (+ whoever the inviter's scopes allow).
--
-- These two tables add the pending-confirm gate + the IP allowlist on top
-- of the existing guest-grant ACL machinery. Server-level admin surface;
-- rows reference the guest + inviter user ids.

CREATE TABLE IF NOT EXISTS guest_registrations (
    registration_id   TEXT PRIMARY KEY,                 -- uuid hex
    invite_token_hash TEXT NOT NULL DEFAULT '',         -- the invite used (provenance)
    inviter_user_id   TEXT NOT NULL,                    -- host/admin who must confirm
    guest_user_id     TEXT NOT NULL,                    -- the pending role='guest' account
    display_name      TEXT NOT NULL DEFAULT '',
    requested_ip      TEXT NOT NULL DEFAULT '',         -- IP they registered from
    scopes            TEXT NOT NULL DEFAULT 'text',     -- what the inviter allows: text[,call,video]
    status            TEXT NOT NULL DEFAULT 'pending',  -- pending | confirmed | denied
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at        TEXT NOT NULL DEFAULT '',
    decided_by        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_guest_reg_inviter_status
    ON guest_registrations (inviter_user_id, status);
CREATE INDEX IF NOT EXISTS idx_guest_reg_guest
    ON guest_registrations (guest_user_id);

-- Per-guest IP allowlist. A confirmed guest may reach the portal only from
-- an allowlisted address (the one they registered from, plus any the admin
-- later adds). Mobile IPs change, so the admin can confirm new addresses.
CREATE TABLE IF NOT EXISTS guest_ip_allowlist (
    guest_user_id TEXT NOT NULL,
    ip            TEXT NOT NULL,
    added_by      TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (guest_user_id, ip)
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (296, 'guest_portal: admin-confirmed guest registrations + per-guest IP allowlist');
