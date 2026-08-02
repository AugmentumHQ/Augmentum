-- 285_auth_invites_claimed_ip.sql
-- IP-whitelisted re-access (Connect comms platform, Phase 3).
--
-- (Renumbered from 284 to avoid a collision with another in-flight migration.)
--
-- Record the IP an invite was claimed from. The admin uses it to "reconnect" a
-- recipient who's off-network: mint a fresh public link PINNED to this IP
-- (allowed_ips), so the recipient regains access while a leaked URL is dead
-- from any other address. Through cloudflared the real client IP arrives in the
-- Cf-Connecting-Ip header.
--
-- Additive / back-compat: existing rows default to '' (claimed before capture,
-- or claimed locally where no public IP applies).
ALTER TABLE auth_invites
    ADD COLUMN claimed_ip TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (285, 'auth_invites: claimed_ip for IP-whitelisted re-access');
